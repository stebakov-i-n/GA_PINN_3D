import torch
import os
import torch.nn as nn
from tqdm import tqdm
import json
import copy
from modules import *
from clearml import Task, Dataset
import os
from sklearn.model_selection import train_test_split


# Efficient/flash attention не поддерживают backward через backward (нужен для d2v в PINN).
# Math backend поддерживает произвольный порядок градиентов.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

LOCAL = True
FULL_TRAIN = False

USE_EMB = True
ACT = 'wave'

INTERIOR_SIZE = 500
WALLS_SIZE    = 250
INLET_SIZE    = 100
OUTLET_SIZE   = 100
OUTERIOR_SIZE = 250

Q = 1.5e-6
S = 2e-6

# Границы срезов, вычисленные из констант датасета
_BND_START = INTERIOR_SIZE                                             # начало boundary (walls)
_BND_END   = INTERIOR_SIZE + WALLS_SIZE + INLET_SIZE + OUTLET_SIZE    # конец non-outerior

PHI_EPOCHS = 10000
EPOCHS = 11000
DIV_POR = 5
VAL_EVERY = 50
B = 16

RESUME_PINN = True
RESUME_TASK = 'a6c0b9d0b4b44b918462644b3c35e3af'
GEN_POINTS = False

if not LOCAL:
    task = Task.init(auto_connect_frameworks=False)

    dataset_train = Dataset.get(dataset_name='SimVascDataset', dataset_project='kornaeva-rnf/GA_PINN_3D')
    DATASET_PATH = dataset_train.get_local_copy()

    dataset_train = Dataset.get(dataset_name='trained_models', dataset_project='kornaeva-rnf/GA_PINN_3D')
    MODELS_PATH = dataset_train.get_local_copy()

    del dataset_train

else:
    task = Task.init(project_name='kornaeva-rnf/GA_PINN_3D', task_name='Test_1', auto_connect_frameworks=False)
    MODELS_PATH = './'
    DATASET_PATH = 'SimVascDataset'

if FULL_TRAIN:
    SPLIT = {'train': [], 'val': [], 'test': []}

    SPLIT['train'], SPLIT['val'] = train_test_split(os.listdir(DATASET_PATH), test_size=0.222, random_state=13)
    SPLIT['val'], SPLIT['test'] = train_test_split(SPLIT['val'], test_size=0.5, random_state=13)
else:
    SPLIT = {'train': ["0145_H_CORO_KD", "0151_H_AO_H"], 'val': ["0096_A_AO_COA"], 'test': ["0209_H_CERE_CA"]}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split="train"):
        self.data = []
        for dir in os.listdir(path):
            if dir not in SPLIT[split]:
                continue
            for file in os.listdir(os.path.join(path, dir)):
                if (file.count('_') == 1) and (file.split('_')[-1] != '-1.stl') and ('.stl' in file):
                    if (GEN_POINTS) or (file.replace('.stl', '_interior.pt') in os.listdir(os.path.join(path, dir))):
                        self.data.append(load_stl(os.path.join(path, dir, file), odd=False, device='cuda', gen_p=GEN_POINTS))
                        torch.cuda.empty_cache()

        self.phi_stage = True


    def __getitem__(self, index):
        if index < len(self.data):
            inlet = 'inlet'
            outlet = 'outlet'
        else:
            index = index - len(self.data)
            inlet = 'outlet'
            outlet = 'inlet'

        agg = self.data[index]

        idx_int   = torch.randperm(len(agg['x_dict']['interior'][:(100000 if self.phi_stage else None)]))[:INTERIOR_SIZE]
        idx_w     = torch.randperm(len(agg['x_dict']['walls']))[:WALLS_SIZE]
        idx_in    = torch.randperm(len(agg['x_dict'][inlet]))[:INLET_SIZE]
        idx_out   = torch.randperm(len(agg['x_dict'][outlet]))[:OUTLET_SIZE]
        idx_outer = torch.randperm(len(agg['x_dict']['outerior']))[:OUTERIOR_SIZE]

        x = torch.cat([
            agg['x_dict']['interior'][idx_int],
            agg['x_dict']['walls'][idx_w],
            agg['x_dict'][inlet][idx_in],
            agg['x_dict'][outlet][idx_out],
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
            agg['phi_w_dict']['interior'][idx_int] if self.phi_stage else agg['phi_w_dict']['interior'][:INTERIOR_SIZE],
            agg['phi_w_dict']['walls'][idx_w],
            agg['phi_w_dict'][inlet][idx_in],
            agg['phi_w_dict'][outlet][idx_out],
            agg['phi_w_dict']['outerior'][idx_outer],
        ])

        phi_key = "in" if inlet == 'inlet' else 'out'

        phi_in = torch.cat([
                agg[f'phi_{phi_key}_dict']['interior'][idx_int],
                agg[f'phi_{phi_key}_dict']['walls'][idx_w],
                agg[f'phi_{phi_key}_dict']['inlet'][idx_in],
                agg[f'phi_{phi_key}_dict']['outlet'][idx_out],
                agg[f'phi_{phi_key}_dict']['outerior'][idx_outer],
            ])

        norm_in    = agg['n_dict'][inlet]
        norm_out   = agg['n_dict'][outlet]
        center_in = agg['n_dict'][f'{inlet}_center']
        center_out = agg['n_dict'][f'{outlet}_center']

        l = agg['l']
        if inlet  == 'inlet':
            s = agg['s_in']
            v_mean = agg['v_mean_in']
        else:
            s = agg['s_out']
            v_mean = agg['v_mean_out']

        return x, torch.stack((phi_w, phi_in), 1), torch.cat((center_in, center_out)), norm_in.repeat(len(x), 1), norm_out.repeat(len(x), 1), center_out.repeat(len(x), 1), l.repeat(len(x), 1), s.repeat(len(x), 1), v_mean.repeat(len(x), 1), x_label

    def __len__(self):
        return len(self.data) * 2

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_act():
    if ACT == 'wave':
        return WaveAct()
    elif ACT == 'tanh':
        return torch.nn.Tanh()
    elif ACT == 'silu':
        return nn.SiLU(inplace=True)

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
            get_act(),
            nn.Linear(d_ff, d_ff),
            get_act(),
            nn.Linear(d_ff, d_model)
        ])

    def forward(self, x):
        return self.linear(x)
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = get_act()
        self.act2 = get_act()
        
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
        self.act1 = get_act()
        self.act2 = get_act()

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
        self.act = get_act()

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return self.act(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.act = get_act()
        
    def forward(self, x, e_outputs):
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.act(x)


class GAPinn(nn.Module):
    def __init__(self, d_hidden=512, d_model=256, N=4, heads=8):
        super(GAPinn, self).__init__()

        if USE_EMB:
            self.embedding = nn.Embedding(5, d_model)

        self.linear_emb = nn.Linear(9, d_model)

        self.encoder = Encoder(d_model, N, heads)
        self.decoder_phi = Decoder(d_model, N, heads)
        # self.decoder_flow = Decoder(d_model, N, heads)

        self.linear_out_phi = nn.Sequential(*[
            nn.Linear(d_model, d_hidden),
            get_act(),
            nn.Linear(d_hidden, d_hidden),
            get_act(),
            nn.Linear(d_hidden, 2)
        ])

        self.linear_out_v = nn.Sequential(*[
            nn.Linear(d_model, d_hidden),
            get_act(),
            nn.Linear(d_hidden, d_hidden),
            get_act(),
            nn.Linear(d_hidden, d_hidden),
            get_act(),
            nn.Linear(d_hidden, d_hidden),
            get_act(),
            nn.Linear(d_hidden, d_hidden),
            get_act(),
            nn.Linear(d_hidden, d_hidden),
            get_act(),
            nn.Linear(d_hidden, d_hidden),
            get_act(),
            nn.Linear(d_hidden, d_hidden),
            get_act(),
            nn.Linear(d_hidden, d_hidden),
            get_act(),
            nn.Linear(d_hidden, d_hidden),
            get_act(),
            nn.Linear(d_hidden, 3)
        ])

        self.linear_out_p = nn.Sequential(*[
            nn.Linear(d_model, d_hidden),
            get_act(),
            nn.Linear(d_hidden, d_hidden),
            get_act(),
            nn.Linear(d_hidden, 1)
        ])

    def forward(self, x, norm_in, norm_out, center_out, l, s, v_mean, x_label, pinn=False):

        B = x.shape[0]

        if pinn:
            x_grad = x * 2 * l
            # x_grad = x
            x_grad.requires_grad_(True)
            x_proj_enc = self.linear_emb(torch.cat((x_grad.clone().detach() / l / 2, out.unsqueeze(1).repeat(1, x.shape[1], 1)), -1))                 # (B, N, d_model)
            if USE_EMB:
                label_emb = self.embedding(x_label)        # (B, N, d_model)
                x_proj_enc = x_proj_enc + label_emb
            
            x_proj_dec = self.linear_emb(torch.cat((x_grad / l / 2, out.unsqueeze(1).repeat(1, x.shape[1], 1)), -1))                 # (B, N, d_model)
            # x_proj_enc = self.linear_emb(torch.cat((x_grad.clone().detach(), out.unsqueeze(1).repeat(1, x.shape[1], 1)), -1))                 # (B, N, d_model)
            # x_proj_dec = self.linear_emb(torch.cat((x_grad, out.unsqueeze(1).repeat(1, x.shape[1], 1)), -1))                 # (B, N, d_model)
        else:
            x_grad = None
            x_proj_enc = x_proj_dec = self.linear_emb(torch.cat((x, out.unsqueeze(1).repeat(1, x.shape[1], 1)), -1))
            if USE_EMB:
                label_emb = self.embedding(x_label)        # (B, N, d_model)
                x_proj_enc = x_proj_enc + label_emb
            

        e_outputs = self.encoder(x_proj_enc)

        d_output_phi = self.decoder_phi(x_proj_dec, e_outputs)
        d_output_flow = self.decoder_phi(x_proj_dec[:, :_BND_END], e_outputs[:, :_BND_END])

        phi_pred = self.linear_out_phi(d_output_phi)
        v = self.linear_out_v(d_output_flow)
        p = self.linear_out_p(d_output_flow)

        v = (v * phi_pred[:, :_BND_END, 1:2]
                  + phi_pred[:, :_BND_END, 0:1] * norm_in[:, :_BND_END] * ((1 / v_mean[:, :_BND_END])*((Q * (s[:, :_BND_END] / S)) / s[:, :_BND_END])))      # (B, _BND_END, 3)

        signed_dist = ((x[:, :_BND_END] - center_out[:, :_BND_END])
                       * norm_out[:, :_BND_END]).sum(-1, keepdim=True)
        p = p * signed_dist * 10                       # (B, _BND_END, 1)

        return None, phi_pred, v[..., 0:1], v[..., 1:2], v[..., 2:3], p, x_grad
    

ga_pinn = GAPinn().cuda()

if RESUME_PINN:
    resume_task = Task.get_task(task_id=RESUME_TASK,
        project_name='kornaeva-rnf/GA_PINN_3D'
    )

    history_path = resume_task.artifacts['history'].get_local_copy()
    model_path = resume_task.artifacts['model'].get_local_copy()    
    optimizer_path = resume_task.artifacts['optimizer'].get_local_copy()
    ga_pinn.load_state_dict(torch.load(model_path))


dataset_train = Dataset(DATASET_PATH, 'train')
dataset_val = Dataset(DATASET_PATH, 'val')

loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=B, shuffle=True)
loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=B, shuffle=False)

optimizer_phi = torch.optim.Adam([*ga_pinn.linear_emb.parameters(), *ga_pinn.encoder.parameters(), *ga_pinn.decoder_phi.parameters(), *ga_pinn.linear_out_phi.parameters()], lr=5e-4)
optimizer_flow = torch.optim.Adam([*ga_pinn.linear_out_v.parameters(), *ga_pinn.linear_out_p.parameters()], lr=5e-4)

if RESUME_PINN:
    optimizer_flow.load_state_dict(torch.load(optimizer_path))

lr_scheduler_phi = torch.optim.lr_scheduler.StepLR(optimizer_phi, 200, 0.97,
                                                    last_epoch=- 1)
lr_scheduler_flow = torch.optim.lr_scheduler.StepLR(optimizer_flow, 200, 0.97,
                                                    last_epoch=- 1)
loss_fcn = torch.nn.MSELoss()

history = {'res_1': [], 'res_2': [], 'res_3': [], 'res_4': [], 'res_sum': [], 'mse_phi': [], 'lr_flow': [], 'lr_phi': []}
history_val = {'res_1': [], 'res_2': [], 'res_3': [], 'res_4': [], 'res_sum': [], 'mse_phi': []}

if RESUME_PINN:
    with open(history_path, 'r') as fp:
        history = json.load(fp)
    if 'history_val' in resume_task.artifacts:
        history_val_path = resume_task.artifacts['history_val'].get_local_copy()
        with open(history_val_path, 'r') as fp:
            history_val = json.load(fp)

logger = task.get_logger()

complete_epochs = len(history['res_1']) + len(history['mse_phi'])

flag = True

save_callback = SaveBest('res_sum', "model_best.pth", 'min')
save_callback.start(history, ga_pinn)

def run_batches(loader, i, train):
    """Проход по одному loader'у: forward/residuals для каждого батча.

    При train=False веса не обновляются (нет zero_grad/backward/step),
    но autograd-граф по x всё равно нужен calc_grad для residual-лоссов.
    """
    ga_pinn.train() if train else ga_pinn.eval()

    epoch_sums = {'res_1': 0., 'res_2': 0., 'res_3': 0., 'res_4': 0.,
                  'mse_out': 0., 'mse_phi': 0.}
    epoch_counts = {key: 0 for key in epoch_sums}
    n_batches = 0

    for x, phi, out, norm_in, norm_out, center_out, l, s, v_mean, x_label in tqdm(loader):
        x, phi, out, norm_in, norm_out, center_out, l, s, v_mean, x_label = \
            x.to('cuda'), phi.to('cuda'), out.to('cuda'), norm_in.to('cuda'), norm_out.to('cuda'), center_out.to('cuda'), l.to('cuda'), s.to('cuda'), v_mean.to('cuda'), x_label.to('cuda')

        if train:
            optimizer_flow.zero_grad()
            optimizer_phi.zero_grad()

        is_res_batch = (not (n_batches % DIV_POR)) or (i >= PHI_EPOCHS)

        out_pred, phi_pred, v1, v2, v3, p, x_grad = ga_pinn(x, norm_in, norm_out, center_out, l, s, v_mean, x_label, pinn=bool(is_res_batch))

        if is_res_batch:
            dv1, dv2, dv3, d2v1, d2v2, d2v3, dp = calc_grad(v1, v2, v3, p, x_grad, div_v_only=i >= PHI_EPOCHS)

            res = calc_res(v1, v2, v3, p, dv1, dv2, dv3, d2v1, d2v2, d2v3, dp, div_v_only=i >= PHI_EPOCHS)

            loss = zero_loss(res)
        else:
            loss_phi = loss_fcn(phi_pred, phi)

            loss = loss_phi

        if train:
            loss.backward()

        if is_res_batch:
            epoch_sums['res_4'] += mse_zero_loss(res[0].detach().cpu()).item()
            epoch_counts['res_4'] += 1
            if i >= PHI_EPOCHS:
                epoch_sums['res_1'] += mse_zero_loss(res[1].detach().cpu()).item()
                epoch_sums['res_2'] += mse_zero_loss(res[2].detach().cpu()).item()
                epoch_sums['res_3'] += mse_zero_loss(res[3].detach().cpu()).item()
                epoch_counts['res_1'] += 1
                epoch_counts['res_2'] += 1
                epoch_counts['res_3'] += 1
        else:
            epoch_sums['mse_phi'] += loss_phi.detach().cpu().item()
            epoch_counts['mse_phi'] += 1

        if train:
            if is_res_batch:
                optimizer_flow.step()
            else:
                optimizer_phi.step()

        n_batches += 1

    is_flow_epoch = (not (n_batches % DIV_POR)) or (i >= PHI_EPOCHS)
    return epoch_sums, epoch_counts, is_flow_epoch


def do_train(i):
    global flag, lr_scheduler_flow

    epoch_sums, epoch_counts, is_flow_epoch = run_batches(loader_train, i, train=True)

    if is_flow_epoch:
        mean_val_sum = 0
        for key in ['res_4', 'res_1', 'res_2', 'res_3'][:1 if i >= PHI_EPOCHS else 4]:
            mean_val = epoch_sums[key] / epoch_counts[key]
            history[key].append(mean_val)
            mean_val_sum += mean_val
            logger.report_scalar(title='Residuals', series=key, value=mean_val, iteration=i)
        if i >= PHI_EPOCHS:
            history['res_sum'].append(mean_val_sum)
            save_callback.step()
    else:
        for key in ['mse_phi']:
            mean_val = epoch_sums[key] / epoch_counts[key]
            history[key].append(mean_val)
            logger.report_scalar(title='Losses', series=key, value=mean_val, iteration=i)

    torch.save(ga_pinn.state_dict(), f'model.pth')
    torch.save(optimizer_flow.state_dict(), f'optimizer_flow.pth')
    torch.save(optimizer_phi.state_dict(), f'optimizer_phi.pth')
    with open('history.json', 'w') as fp:
        json.dump(history, fp)

    if is_flow_epoch:
        if (i >= PHI_EPOCHS) and flag:
            flag = False
            lr_scheduler_flow = torch.optim.lr_scheduler.StepLR(optimizer_flow, 400, 0.97,
                                                    last_epoch=- 1)

        lr_scheduler_flow.step()
        lr_flow = optimizer_flow.param_groups[0]['lr']
        logger.report_scalar(title='Learning Rate', series='lr_flow', value=lr_flow, iteration=i)
        history['lr_flow'].append(lr_flow)
    else:
        lr_scheduler_phi.step()
        lr_phi = optimizer_phi.param_groups[0]['lr']
        logger.report_scalar(title='Learning Rate', series='lr_phi', value=lr_phi, iteration=i)
        history['lr_phi'].append(lr_phi)


def do_val(i):
    epoch_sums, epoch_counts, is_flow_epoch = run_batches(loader_val, i, train=False)

    if is_flow_epoch:
        mean_val_sum = 0
        for key in ['res_4', 'res_1', 'res_2', 'res_3'][:1 if i >= PHI_EPOCHS else 4]:
            mean_val = epoch_sums[key] / epoch_counts[key]
            history_val[key].append(mean_val)
            mean_val_sum += mean_val
            logger.report_scalar(title='Residuals (val)', series=key, value=mean_val, iteration=i)
        if i >= PHI_EPOCHS:
            history_val['res_sum'].append(mean_val_sum)
    else:
        for key in ['mse_phi']:
            mean_val = epoch_sums[key] / epoch_counts[key]
            history_val[key].append(mean_val)
            logger.report_scalar(title='Losses (val)', series=key, value=mean_val, iteration=i)

    with open('history_val.json', 'w') as fp:
        json.dump(history_val, fp)


for i in tqdm(range(complete_epochs, complete_epochs + EPOCHS)):
    do_train(i)

    if i % VAL_EVERY == 0:
        do_val(i)

task.upload_artifact(f'model', artifact_object='model.pth')
task.upload_artifact(f'history', artifact_object='history.json')
task.upload_artifact(f'history_val', artifact_object='history_val.json')
task.upload_artifact(f'optimizer_phi', artifact_object='optimizer_phi.pth')
task.upload_artifact(f'optimizer_flow', artifact_object='optimizer_flow.pth')