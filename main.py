import torch
import os
import torch.nn as nn
from tqdm import tqdm
import json
from modules import *
from clearml import Task, Dataset


# Efficient/flash attention не поддерживают backward через backward (нужен для d2v в PINN).
# Math backend поддерживает произвольный порядок градиентов.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

LOCAL = False

if not LOCAL:
    task = Task.init()

    dataset = Dataset.get(dataset_name='SimVascDataset', dataset_project='kornaeva-rnf/GA_PINN_3D')
    DATASET_PATH = dataset.get_local_copy()

    dataset = Dataset.get(dataset_name='trained_models', dataset_project='kornaeva-rnf/GA_PINN_3D')
    MODELS_PATH = dataset.get_local_copy()

    del dataset

else:
    MODELS_PATH = 'trained_models'
    DATASET_PATH = 'SimVascDataset'

INTERIOR_SIZE = 500
WALLS_SIZE    = 250
INLET_SIZE    = 100
OUTLET_SIZE   = 100
OUTERIOR_SIZE = 250

Q = 1.5e-6

# Границы срезов, вычисленные из констант датасета
_BND_START = INTERIOR_SIZE                                             # начало boundary (walls)
_BND_END   = INTERIOR_SIZE + WALLS_SIZE + INLET_SIZE + OUTLET_SIZE    # конец non-outerior

B = 2

TRAIN_PINN = True
RESUME_PINN = False
GEN_INTERIOR_POINTS = False

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = []
        for dir in os.listdir(path):
            for file in os.listdir(os.path.join(path, dir)):
                if file.count('_') == 1 and file.split('_')[-1] != '-1.stl':
                    self.data.append(load_stl(os.path.join(path, dir, file), device='cuda', gen_int_p=GEN_INTERIOR_POINTS))

    def __getitem__(self, index):
        agg = self.data[index]

        idx_int   = torch.randperm(len(agg['x_dict']['interior']))[:INTERIOR_SIZE]
        idx_w     = torch.randperm(len(agg['x_dict']['walls']))[:WALLS_SIZE]
        idx_in    = torch.randperm(len(agg['x_dict']['inlet']))[:INLET_SIZE]
        idx_out   = torch.randperm(len(agg['x_dict']['outlet']))[:OUTLET_SIZE]
        idx_outer = torch.randperm(len(agg['x_dict']['outerior']))[:OUTERIOR_SIZE]

        x = torch.cat([
            agg['x_dict']['interior'][idx_int],
            agg['x_dict']['walls'][idx_w],
            agg['x_dict']['inlet'][idx_in],
            agg['x_dict']['outlet'][idx_out],
            agg['x_dict']['outerior'][idx_outer],
        ], dim=0)

        # 0=interior, 1=walls, 2=inlet, 3=outlet, 4=outerior
        x_label = torch.cat([
            torch.zeros(INTERIOR_SIZE,             dtype=torch.long),
            torch.ones(WALLS_SIZE,                 dtype=torch.long),
            torch.full((INLET_SIZE,),    2,        dtype=torch.long),
            torch.full((OUTLET_SIZE,),   3,        dtype=torch.long),
            torch.full((OUTERIOR_SIZE,), 4,        dtype=torch.long),
        ])

        phi_w = torch.cat([
            agg['phi_w_dict']['interior'][idx_int],
            agg['phi_w_dict']['walls'][idx_w],
            agg['phi_w_dict']['inlet'][idx_in],
            agg['phi_w_dict']['outlet'][idx_out],
            agg['phi_w_dict']['outerior'][idx_outer],
        ])

        phi_in = torch.cat([
            agg['phi_in_dict']['interior'][idx_int],
            agg['phi_in_dict']['walls'][idx_w],
            agg['phi_in_dict']['inlet'][idx_in],
            agg['phi_in_dict']['outlet'][idx_out],
            agg['phi_in_dict']['outerior'][idx_outer],
        ])

        norm_in    = agg['n_dict']['inlet']
        norm_out   = agg['n_dict']['outlet']
        center_out = agg['n_dict']['outlet_center']

        l = agg['l']
        s = agg['s']
        v_mean = agg['v_mean']

        return x, torch.stack((phi_w, phi_in), 1), torch.cat((norm_out, center_out)), norm_in.repeat(len(x), 1), norm_out.repeat(len(x), 1), center_out.repeat(len(x), 1), l.repeat(len(x), 1), s.repeat(len(x), 1), v_mean.repeat(len(x), 1), x_label

    def __len__(self):
        return len(self.data)
    

class GAPinn(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_enc_layers=6, num_dec_layers=4,
                 dim_ff=2048, dropout=0.1):
        super().__init__()

        self.projector = nn.Linear(3, d_model)
        self.embedding = nn.Embedding(5, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        self.out_head = nn.Linear(d_model, 6)

        dec_layer_phi = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True, activation='gelu')
        self.transformer_decoder_phi = nn.TransformerDecoder(dec_layer_phi, num_layers=num_dec_layers)

        dec_layer_flow = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_decoder_flow = nn.TransformerDecoder(dec_layer_flow, num_layers=num_dec_layers)

        self.phi_head = nn.Linear(d_model, 2)
        self.vel_head = nn.Linear(d_model, 3)
        self.p_head   = nn.Linear(d_model, 1)

    def forward(self, x, norm_in, norm_out, center_out, l, s, v_mean, x_label):
        # x:          (B, N, 3)   N = _BND_END + OUTERIOR_SIZE
        # x_label:    (B, N)  long  {0=interior, 1=walls, 2=inlet, 3=outlet, 4=outerior}
        # norm_in:    (B, 3)
        # norm_out:   (B, 3)
        # center_out: (B, 3)
        B = x.shape[0]

        x_proj = self.projector(x)                 # (B, N, d_model)
        label_emb = self.embedding(x_label)        # (B, N, d_model)
        x_emb  = x_proj + label_emb  # (B, N, d_model)

        # --- Encoder: walls + inlet + outlet  [:, _BND_START:_BND_END] ---
        cls_tokens = self.cls_token.expand(B, -1, -1)                              # (B, 1, d_model)
        enc_in     = torch.cat([cls_tokens, x_emb[:, _BND_START:_BND_END]], dim=1) # (B, N_bnd+1, d_model)
        enc_out    = self.transformer_encoder(enc_in)                               # (B, N_bnd+1, d_model)
        x_cls      = enc_out[:, :1]                                                 # (B, 1, d_model)

        out_pred = self.out_head(x_cls.squeeze(1))                                  # (B, 6)

        # --- Decoder phi: all points ---
        phi_seq  = self.transformer_decoder_phi(x_emb, x_cls)            # (B, N, d_model)
        # phi_seq  = self.transformer_decoder_phi(x_emb + x_cls, enc_out)            # (B, N, d_model)
        phi_pred = self.phi_head(phi_seq)                                            # (B, N, 2)

        # --- Decoder flow: interior + walls + inlet + outlet  [:, :_BND_END] ---
        x_grad = x[:, :_BND_END] * 2 * l[:, :_BND_END]
        x_grad.requires_grad_(True)
        x_proj_grad = self.projector(x_grad / l[:, :_BND_END] / 2)                 # (B, N, d_model)
        x_emb  = x_proj_grad + label_emb[:, :_BND_END]  # (B, N, d_model)

        flow_seq = self.transformer_decoder_flow(
            x_emb[:, :_BND_END], x_cls)                                  # (B, _BND_END, d_model)
            # x_emb[:, :_BND_END] + x_cls, enc_out)                                  # (B, _BND_END, d_model)

        v = (self.vel_head(flow_seq) * phi_pred[:, :_BND_END, 1:2]
                  + phi_pred[:, :_BND_END, 0:1] * norm_in[:, :_BND_END] * ((1 / v_mean[:, :_BND_END])*(Q / s[:, :_BND_END])))      # (B, _BND_END, 3)

        signed_dist = ((x[:, :_BND_END] - center_out[:, :_BND_END])
                       * norm_out[:, :_BND_END]).sum(-1, keepdim=True)
        p = self.p_head(flow_seq) * signed_dist * 10                       # (B, _BND_END, 1)

        return out_pred, phi_pred, v[..., 0:1], v[..., 1:2], v[..., 2:3], p, x_grad
    

ga_pinn = GAPinn().cuda()

if TRAIN_PINN:
    if RESUME_PINN:
        ga_pinn.load_state_dict(torch.load(f'{MODELS_PATH}/mlp_dist.pth'))
    else:
        ga_pinn.load_state_dict(torch.load(f'{MODELS_PATH}/mlp_pinn.pth'))

dataset = Dataset(DATASET_PATH)

loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=True)

if TRAIN_PINN:
    if RESUME_PINN:
        optimizer = torch.optim.Adam(ga_pinn.transformer_decoder_flow.parameters(), lr=1e-4)
        optimizer.load_state_dict(torch.load(f'{MODELS_PATH}/optimizer_pinn.pth'))
    else:
        optimizer = torch.optim.Adam(ga_pinn.transformer_decoder_flow.parameters(), lr=1e-4)
     
else:
    optimizer = torch.optim.Adam(ga_pinn.parameters(), lr=5e-4)

if TRAIN_PINN:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, 0.97,
                                                        last_epoch=- 1)
else:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.97,
                                                        last_epoch=- 1)

loss_fcn = torch.nn.MSELoss()

if TRAIN_PINN:
    with open(f'{MODELS_PATH}/history_pinn.json', 'r') as fp:
        history = json.load(fp)
else:
    history = {'res_1': [], 'res_2': [], 'res_3': [], 'res_4': [], 'mse_out': [], 'mse_phi': []}

for i in tqdm(range(10000)):
    ga_pinn.train()
    for x, phi, out, norm_in, norm_out, center_out, l, s, v_mean, x_label in tqdm(loader):
        optimizer.zero_grad()

        x, phi, out, norm_in, norm_out, center_out, l, s, v_mean, x_label = \
        x.to('cuda'), phi.to('cuda'), out.to('cuda'), norm_in.to('cuda'), norm_out.to('cuda'), center_out.to('cuda'), l.to('cuda'), s.to('cuda'), v_mean.to('cuda'), x_label.to('cuda')

        out_pred, phi_pred, v1, v2, v3, p, x_grad = ga_pinn(x, norm_in, norm_out, center_out, l, s, v_mean, x_label)

        if TRAIN_PINN:
            dv1, dv2, dv3, d2v1, d2v2, d2v3, dp = calc_grad(v1, v2, v3, p, x_grad)

            res = calc_res(v1, v2, v3, p, dv1, dv2, dv3, d2v1, d2v2, d2v3, dp)

            loss_res = zero_loss(res)
            
            loss = loss_res
        else:
            loss_out = loss_fcn(out_pred, out)

            loss_phi = loss_fcn(phi_pred, phi)

            loss = loss_out + loss_phi

        loss.backward()

        optimizer.step()

        if TRAIN_PINN:
            history['res_1'].append(mse_zero_loss(res[0].detach().cpu()).item())
            history['res_2'].append(mse_zero_loss(res[1].detach().cpu()).item())
            history['res_3'].append(mse_zero_loss(res[2].detach().cpu()).item())
            history['res_4'].append(mse_zero_loss(res[3].detach().cpu()).item())
        else:
            history['mse_out'].append(loss_out.detach().cpu().item())
            history['mse_phi'].append(loss_phi.detach().cpu().item())
    
    if TRAIN_PINN:
        torch.save(ga_pinn.state_dict(), f'mlp_pinn.pth')
        torch.save(optimizer.state_dict(), f'optimizer_pinn.pth')
        with open('history_pinn.json', 'w') as fp:
            json.dump(history, fp)
    else:
        torch.save(ga_pinn.state_dict(), f'mlp_dist.pth')
        torch.save(optimizer.state_dict(), f'optimizer_dist.pth')

        with open('history_dist.json', 'w') as fp:
            json.dump(history, fp)

    lr_scheduler.step()
