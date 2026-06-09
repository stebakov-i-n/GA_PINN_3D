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

LOCAL = True

if not LOCAL:
    task = Task.init(auto_connect_frameworks=False)

    dataset = Dataset.get(dataset_name='SimVascDataset', dataset_project='kornaeva-rnf/GA_PINN_3D')
    DATASET_PATH = dataset.get_local_copy()

    dataset = Dataset.get(dataset_name='trained_models', dataset_project='kornaeva-rnf/GA_PINN_3D')
    MODELS_PATH = dataset.get_local_copy()

    del dataset

else:
    task = Task.init(project_name='kornaeva-rnf/GA_PINN_3D', task_name='Train_only_div_v', auto_connect_frameworks=False)
    MODELS_PATH = './'
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

B = 8

TRAIN_PINN = True
RESUME_PINN = False
GEN_INTERIOR_POINTS = False

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = []
        for dir in os.listdir(path):
            for file in os.listdir(os.path.join(path, dir)):
                if (file.count('_') == 1) and (file.split('_')[-1] != '-1.stl') and ('.stl' in file):
                    if file.replace('.stl', '.pt') in os.listdir(os.path.join(path, dir)):
                        self.data.append(load_stl(os.path.join(path, dir, file), odd=False, device='cuda', gen_int_p=GEN_INTERIOR_POINTS))
                        self.data.append(load_stl(os.path.join(path, dir, file), odd=True, device='cuda', gen_int_p=GEN_INTERIOR_POINTS))
                        # break
            # break


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
        center_in = agg['n_dict']['inlet_center']
        center_out = agg['n_dict']['outlet_center']

        l = agg['l']
        s = agg['s']
        v_mean = agg['v_mean']

        return x, torch.stack((phi_w, phi_in), 1), torch.cat((center_in, center_out)), norm_in.repeat(len(x), 1), norm_out.repeat(len(x), 1), center_out.repeat(len(x), 1), l.repeat(len(x), 1), s.repeat(len(x), 1), v_mean.repeat(len(x), 1), x_label

    def __len__(self):
        return len(self.data)


def create_layers(input_size, hidden_layer_size, hidden_layers, output_size):
    layers = []
    for i in range(hidden_layers):
        if not i:
            layers += [nn.Linear(input_size, hidden_layer_size, bias=True),
                       nn.Tanh()]
            torch.nn.init.xavier_normal_(layers[-2].weight)
        else:
            layers += [nn.Linear(hidden_layer_size, hidden_layer_size, bias=True),
                       nn.Tanh()]
            torch.nn.init.xavier_normal_(layers[-2].weight)
            
    layers.append(nn.Linear(hidden_layer_size, output_size, bias=True))
    torch.nn.init.xavier_normal_(layers[-1].weight)
    
    return layers


# class GAPinn(nn.Module):
#     def __init__(self, d_model=512, nhead=8, num_enc_layers=6, num_dec_layers=4,
#                  dim_ff=2048, dropout=0.0):
#         super().__init__()

#         self.d_model = d_model

#         self.projector = nn.Linear(3, d_model)
#         self.embedding = nn.Embedding(5, d_model)
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
#             dropout=dropout, batch_first=True, norm_first=True, activation='gelu')
#         self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

#         self.out_head = nn.Linear(d_model, 6)

#         # dec_layer_phi = nn.TransformerDecoderLayer(
#         #     d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
#         #     dropout=dropout, batch_first=True, norm_first=True, activation='gelu')
#         # self.transformer_decoder_phi = nn.TransformerDecoder(dec_layer_phi, num_layers=num_dec_layers)

#         # dec_layer_flow = nn.TransformerDecoderLayer(
#         #     d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
#         #     dropout=dropout, batch_first=True, norm_first=True)
#         # self.transformer_decoder_flow = nn.TransformerDecoder(dec_layer_flow, num_layers=num_dec_layers)

#         self.transformer_decoder_phi = nn.Sequential(*create_layers(d_model + 3, 20, 10, d_model))

#         self.transformer_decoder_flow = nn.Sequential(*create_layers(d_model + 3, 20, 10, d_model))

#         self.phi_head = nn.Linear(d_model, 2)
#         self.vel_head = nn.Linear(d_model, 3)
#         self.p_head   = nn.Linear(d_model, 1)

#     def forward(self, x, norm_in, norm_out, center_out, l, s, v_mean, x_label):
#         # x:          (B, N, 3)   N = _BND_END + OUTERIOR_SIZE
#         # x_label:    (B, N)  long  {0=interior, 1=walls, 2=inlet, 3=outlet, 4=outerior}
#         # norm_in:    (B, 3)
#         # norm_out:   (B, 3)
#         # center_out: (B, 3)
#         B = x.shape[0]

#         x_proj = self.projector(x)

#         # --- Encoder: walls + inlet + outlet  [:, _BND_START:_BND_END] ---
#         cls_tokens = self.cls_token.expand(B, -1, -1)                              # (B, 1, d_model)
#         enc_in     = torch.cat([cls_tokens, x_proj[:, _BND_START:_BND_END]], dim=1) # (B, N_bnd+1, d_model)
#         enc_out    = self.transformer_encoder(enc_in)                               # (B, N_bnd+1, d_model)
#         x_cls      = enc_out[:, :1]                                                 # (B, 1, d_model)

#         out_pred = self.out_head(x_cls.squeeze(1))                                  # (B, 6)

#         if TRAIN_PINN:
#             x_grad = x * 2 * l
#             x_grad.requires_grad_(True)
#             x_dec = (x_grad / l / 2)              # (B, N, d_model)
#             x_cls = x_cls.repeat(1, x.shape[1], 1)
#             x_dec = torch.cat((x_dec, x_cls), -1)

#         else:
#             x_grad = None
#             x_dec = (x)
#             x_cls = x_cls.repeat(1, x.shape[1], 1)
#             x_dec = torch.cat((x_dec, x_cls), -1)

#         # --- Decoder phi: all points ---
#         phi_seq  = self.transformer_decoder_phi(x_dec)            # (B, N, d_model)
#         phi_pred = self.phi_head(phi_seq)                                            # (B, N, 2)
#         # --- Decoder flow: interior + walls + inlet + outlet  [:, :_BND_END] ---
#         flow_seq = self.transformer_decoder_flow(x_dec[:, :_BND_END]) # (B, _BND_END, d_model)

#         v = (self.vel_head(flow_seq) * phi_pred[:, :_BND_END, 1:2]
#                   + phi_pred[:, :_BND_END, 0:1] * norm_in[:, :_BND_END] * ((1 / v_mean[:, :_BND_END])*(Q / s[:, :_BND_END])))      # (B, _BND_END, 3)

#         signed_dist = ((x[:, :_BND_END] - center_out[:, :_BND_END])
#                        * norm_out[:, :_BND_END]).sum(-1, keepdim=True)
#         p = self.p_head(flow_seq) * signed_dist * 10                       # (B, _BND_END, 1)

#         return out_pred, phi_pred, v[..., 0:1], v[..., 1:2], v[..., 2:3], p, x_grad

# class GAPinn(nn.Module):
#     def __init__(self, d_model=512, nhead=8, num_enc_layers=6, num_dec_layers=4,
#                  dim_ff=2048, dropout=0.0):
#         super().__init__()

#         self.projector = nn.Linear(3, d_model)
#         self.embedding = nn.Embedding(5, d_model)
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
#             dropout=dropout, batch_first=True, norm_first=True, activation='gelu')
#         self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

#         self.out_head = nn.Linear(d_model, 6)

#         dec_layer_phi = nn.TransformerDecoderLayer(
#             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
#             dropout=dropout, batch_first=True, norm_first=True, activation='gelu')
#         self.transformer_decoder_phi = nn.TransformerDecoder(dec_layer_phi, num_layers=num_dec_layers)

#         dec_layer_flow = nn.TransformerDecoderLayer(
#             d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
#             dropout=dropout, batch_first=True, norm_first=True)
#         self.transformer_decoder_flow = nn.TransformerDecoder(dec_layer_flow, num_layers=num_dec_layers)

#         self.phi_head = nn.Linear(d_model, 2)
#         self.vel_head = nn.Linear(d_model, 3)
#         self.p_head   = nn.Linear(d_model, 1)

#     def forward(self, x, norm_in, norm_out, center_out, l, s, v_mean, x_label):
#         # x:          (B, N, 3)   N = _BND_END + OUTERIOR_SIZE
#         # x_label:    (B, N)  long  {0=interior, 1=walls, 2=inlet, 3=outlet, 4=outerior}
#         # norm_in:    (B, 3)
#         # norm_out:   (B, 3)
#         # center_out: (B, 3)
#         B = x.shape[0]

#         if TRAIN_PINN:
#             x_grad = x * 2 * l
#             x_grad.requires_grad_(True)
#             x_proj = self.projector(x_grad / l / 2)                 # (B, N, d_model)

#         else:
#             x_grad = None
#             x_proj = self.projector(x)

#         # --- Encoder: walls + inlet + outlet  [:, _BND_START:_BND_END] ---
#         cls_tokens = self.cls_token.expand(B, -1, -1)                              # (B, 1, d_model)
#         enc_in     = torch.cat([cls_tokens, x_proj[:, _BND_START:_BND_END]], dim=1) # (B, N_bnd+1, d_model)
#         enc_out    = self.transformer_encoder(enc_in)                               # (B, N_bnd+1, d_model)
#         x_cls      = enc_out[:, :1]                                                 # (B, 1, d_model)

#         out_pred = self.out_head(x_cls.squeeze(1))                                  # (B, 6)

#         # --- Decoder phi: all points ---
#         phi_seq  = self.transformer_decoder_phi(x_proj, x_cls)            # (B, N, d_model)
#         # phi_seq  = self.transformer_decoder_phi(x_proj + x_cls, enc_out)            # (B, N, d_model)
#         phi_pred = self.phi_head(phi_seq)                                            # (B, N, 2)

#         # --- Decoder flow: interior + walls + inlet + outlet  [:, :_BND_END] ---
        
#         flow_seq = self.transformer_decoder_flow(
#             x_proj[:, :_BND_END], x_cls)                                  # (B, _BND_END, d_model)
#             # x_proj[:, :_BND_END] + x_cls, enc_out)                                  # (B, _BND_END, d_model)

#         v = (self.vel_head(flow_seq) * phi_pred[:, :_BND_END, 1:2]
#                   + phi_pred[:, :_BND_END, 0:1] * norm_in[:, :_BND_END] * ((1 / v_mean[:, :_BND_END])*(Q / s[:, :_BND_END])))      # (B, _BND_END, 3)

#         signed_dist = ((x[:, :_BND_END] - center_out[:, :_BND_END])
#                        * norm_out[:, :_BND_END]).sum(-1, keepdim=True)
#         p = self.p_head(flow_seq) * signed_dist * 10                       # (B, _BND_END, 1)

#         return out_pred, phi_pred, v[..., 0:1], v[..., 1:2], v[..., 2:3], p, x_grad

import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__() 
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super(FeedForward, self).__init__() 
        self.linear = nn.Sequential(*[
            nn.Linear(d_model, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_model)
        ])

    def forward(self, x):
        return self.linear(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()
        
    def forward(self, x):
        x2 = self.act1(x)
        x = x + self.attn(x2,x2,x2)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x, e_outputs): 
        x2 = self.act1(x)
        x = x + self.attn(x2, e_outputs, e_outputs)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.act = WaveAct()

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return self.act(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.act = WaveAct()
        
    def forward(self, x, e_outputs):
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.act(x)


class GAPinn(nn.Module):
    def __init__(self, d_hidden=512, d_model=32, N=1, heads=2):
        super(GAPinn, self).__init__()

        self.linear_emb = nn.Linear(9, d_model)

        self.encoder = Encoder(d_model, N, heads)
        self.decoder_phi = Decoder(d_model, N, heads)
        self.decoder_pflow= Decoder(d_model, N, heads)

        self.linear_out_phi = nn.Sequential(*[
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, 2)
        ])

        self.linear_out_v = nn.Sequential(*[
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, 3)
        ])

        self.linear_out_p = nn.Sequential(*[
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, 1)
        ])

    def forward(self, x, norm_in, norm_out, center_out, l, s, v_mean, x_label):

        B = x.shape[0]

        if TRAIN_PINN:
            x_grad = x * 2 * l
            x_grad.requires_grad_(True)
            x_proj = self.linear_emb(torch.cat((x_grad / l / 2, out.unsqueeze(1).repeat(1, x.shape[1], 1)), -1))                 # (B, N, d_model)

        else:
            x_grad = None
            x_proj = self.linear_emb(torch.cat((x, out.unsqueeze(1).repeat(1, x.shape[1], 1)), -1))

        e_outputs = self.encoder(x_proj)

        d_output_phi = self.decoder_phi(x_proj, e_outputs)
        d_output_flow = self.decoder_flow(x_proj[:, :_BND_END], e_outputs[:, :_BND_END])

        phi_pred = self.linear_out_phi(d_output_phi)
        v = self.linear_out_v(d_output_flow)
        p = self.linear_out_p(d_output_flow)

        v = (v * phi_pred[:, :_BND_END, 1:2]
                  + phi_pred[:, :_BND_END, 0:1] * norm_in[:, :_BND_END] * ((1 / v_mean[:, :_BND_END])*(Q / s[:, :_BND_END])))      # (B, _BND_END, 3)

        signed_dist = ((x[:, :_BND_END] - center_out[:, :_BND_END])
                       * norm_out[:, :_BND_END]).sum(-1, keepdim=True)
        p = p * signed_dist * 10                       # (B, _BND_END, 1)

        return None, phi_pred, v[..., 0:1], v[..., 1:2], v[..., 2:3], p, x_grad
    

ga_pinn = GAPinn().cuda()

if TRAIN_PINN:
    if RESUME_PINN:
        if not LOCAL:
            resume_task = Task.get_task(task_id='14b9427c14da4b1f8daf8ee1cd988daf',
                project_name='kornaeva-rnf/GA_PINN_3D'
            )

            history_path = resume_task.artifacts['history'].get_local_copy()
            model_path = resume_task.artifacts['model'].get_local_copy()    
            optimizer_path = resume_task.artifacts['optimizer'].get_local_copy()
            ga_pinn.load_state_dict(torch.load(model_path))
        else:
            ga_pinn.load_state_dict(torch.load(f'{MODELS_PATH}/mlp_pinn.pth'))
    else:
        ga_pinn.load_state_dict(torch.load(f'{MODELS_PATH}/mlp_dist.pth'))

dataset = Dataset(DATASET_PATH)

loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=True)

if TRAIN_PINN:
    if RESUME_PINN:
        # optimizer = torch.optim.Adam([*ga_pinn.transformer_decoder_flow.parameters(), *ga_pinn.vel_head.parameters(), *ga_pinn.p_head.parameters()], lr=5e-5)
        optimizer = torch.optim.Adam([*ga_pinn.decoder_phi.parameters(), *ga_pinn.linear_out_v.parameters(), *ga_pinn.linear_out_p.parameters()], lr=1e-3)
        if not LOCAL:
            optimizer.load_state_dict(torch.load(optimizer_path))
        else:
            optimizer.load_state_dict(torch.load(f'{MODELS_PATH}/optimizer_pinn.pth'))
    else:
        # optimizer = torch.optim.Adam([*ga_pinn.transformer_decoder_flow.parameters(), *ga_pinn.vel_head.parameters(), *ga_pinn.p_head.parameters()], lr=1e-3)
        optimizer = torch.optim.Adam([*ga_pinn.decoder_phi.parameters(), *ga_pinn.linear_out_v.parameters(), *ga_pinn.linear_out_p.parameters()], lr=1e-3)
     
else:
    optimizer = torch.optim.Adam(ga_pinn.parameters(), lr=1e-3)

if TRAIN_PINN:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, 0.97,
                                                        last_epoch=- 1)
else:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.97,
                                                        last_epoch=- 1)

loss_fcn = torch.nn.MSELoss()

if TRAIN_PINN:
    if RESUME_PINN:
        if not LOCAL:
            with open(history_path, 'r') as fp:
               history = json.load(fp)
        else:
            with open(f'{MODELS_PATH}/history_pinn.json', 'r') as fp:
               history = json.load(fp)
    else:
        history = {'res_1': [], 'res_2': [], 'res_3': [], 'res_4': [], 'mse_out': [], 'mse_phi': []}
else:
    history = {'res_1': [], 'res_2': [], 'res_3': [], 'res_4': [], 'mse_out': [], 'mse_phi': []}

logger = task.get_logger()

# for i in range(10000):
#     for key in ('res_1', 'res_2', 'res_3', 'res_4'):
#         logger.report_scalar(title='Residuals', series=key, value=history[key][10000 + i], iteration=i)


for i in tqdm(range(10000)):
    ga_pinn.train()

    epoch_sums = {'res_1': 0., 'res_2': 0., 'res_3': 0., 'res_4': 0.,
                  'mse_out': 0., 'mse_phi': 0.}
    n_batches = 0

    for x, phi, out, norm_in, norm_out, center_out, l, s, v_mean, x_label in tqdm(loader):
        x, phi, out, norm_in, norm_out, center_out, l, s, v_mean, x_label = \
            x.to('cuda'), phi.to('cuda'), out.to('cuda'), norm_in.to('cuda'), norm_out.to('cuda'), center_out.to('cuda'), l.to('cuda'), s.to('cuda'), v_mean.to('cuda'), x_label.to('cuda')
        
        def closure():
            out_pred, phi_pred, v1, v2, v3, p, x_grad = ga_pinn(x, norm_in, norm_out, center_out, l, s, v_mean, x_label)

            if TRAIN_PINN:
                dv1, dv2, dv3, d2v1, d2v2, d2v3, dp = calc_grad(v1, v2, v3, p, x_grad)

                res = calc_res(v1, v2, v3, p, dv1, dv2, dv3, d2v1, d2v2, d2v3, dp)

                loss_res = zero_loss(res)

                loss = loss_res
            else:
                # loss_out = loss_fcn(out_pred, out)

                loss_phi = loss_fcn(phi_pred, phi)

                loss = loss_phi
            optimizer.zero_grad()
            loss.backward()

            if TRAIN_PINN:
                epoch_sums['res_1'] += mse_zero_loss(res[0].detach().cpu()).item()
                # epoch_sums['res_2'] += mse_zero_loss(res[1].detach().cpu()).item()
                # epoch_sums['res_3'] += mse_zero_loss(res[2].detach().cpu()).item()
                # epoch_sums['res_4'] += mse_zero_loss(res[3].detach().cpu()).item()
            else:
                # epoch_sums['mse_out'] += loss_out.detach().cpu().item()
                epoch_sums['mse_phi'] += loss_phi.detach().cpu().item()

            return loss
        
        optimizer.step(closure)

        n_batches += 1

    if TRAIN_PINN:
        for key in ('res_1', 'res_2', 'res_3', 'res_4'):
            mean_val = epoch_sums[key] / n_batches
            history[key].append(mean_val)
            logger.report_scalar(title='Residuals', series=key, value=mean_val, iteration=i)
    else:
        for key in ('mse_out', 'mse_phi'):
            mean_val = epoch_sums[key] / n_batches
            history[key].append(mean_val)
            logger.report_scalar(title='Losses', series=key, value=mean_val, iteration=i)

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


if TRAIN_PINN:
    task.upload_artifact(f'model', artifact_object='mlp_pinn.pth')
    task.upload_artifact(f'history', artifact_object='history_pinn.json')
    task.upload_artifact(f'optimizer', artifact_object='optimizer_pinn.pth')
else:
    task.upload_artifact(f'model', artifact_object='mlp_dist.pth')
    task.upload_artifact(f'history', artifact_object='history_dist.json')
    task.upload_artifact(f'optimizer', artifact_object='optimizer_dist.pth')