import torch
import os
import torch.nn as nn
from tqdm import tqdm
import json
import copy
from datetime import datetime
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
USE_CLEARML = False

USE_EMB = True
ACT_TF = 'relu'
ACT_DEC = 'wave'
USE_CLS_TOKEN = True   # False = старая архитектура (decoder_phi + cross-attention);
                        # True = CLS-токен: MLP phi/flow берут на вход [cls_output, x_grad]

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

PHI_EPOCHS = 40000
EPOCHS = 20000
DIV_POR = 5
VAL_EVERY = 50
B = 10 if LOCAL else 24

END_TO_END = True   # True — без разделения на phi/flow: один optimizer_all на все
                      # веса, каждый батч считает и суммирует loss_phi + loss_res,
                      # оба логируются каждую эпоху; PHI_EPOCHS/DIV_POR/freeze/enter_flow_stage игнорируются

RESUME_PINN = True
RESUME_TASK = '6ecaa0f3899141a2b3c6a10abcfa434a'
RESUME_PATH = 'trained_models/end2end_flow'
RESUME_SOURCE = 'local'
GEN_POINTS = False

AUGMENT_ROTATION = True
AUGMENT_PERMUTE = True
AUGMENT_REFLECT = True

if not LOCAL:
    dataset_train = Dataset.get(dataset_name='SimVascDatasetFull', dataset_project='kornaeva-rnf/GA_PINN_3D')
    DATASET_PATH = dataset_train.get_local_copy()

    del dataset_train

else:
    DATASET_PATH = 'SimVascDataset'

RUN_NAME = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
SAVE_DIR = os.path.join('trained_models', RUN_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

if USE_CLEARML:
    if not LOCAL:
        task = Task.init(auto_connect_frameworks=False)
    else:
        task = Task.init(project_name='kornaeva-rnf/GA_PINN_3D', task_name='Test_6', auto_connect_frameworks=False)
else:
    task = None


class _NullLogger:
    """Заглушка логгера при USE_CLEARML=False, чтобы не ветвить вызовы report_scalar по коду."""
    def report_scalar(self, *args, **kwargs):
        pass

if FULL_TRAIN:
    with open('full_split.json', 'r') as fp:
        SPLIT = json.load(fp)    
else:
    with open('debug_split.json', 'r') as fp:
        SPLIT = json.load(fp) 

CONSTANTS = {
    'LOCAL': LOCAL,
    'FULL_TRAIN': FULL_TRAIN,
    'USE_CLEARML': USE_CLEARML,
    'USE_EMB': USE_EMB,
    'USE_CLS_TOKEN': USE_CLS_TOKEN,
    'ACT_TF': ACT_TF,
    'ACT_DEC': ACT_DEC,
    'INTERIOR_SIZE': INTERIOR_SIZE,
    'WALLS_SIZE': WALLS_SIZE,
    'INLET_SIZE': INLET_SIZE,
    'OUTLET_SIZE': OUTLET_SIZE,
    'OUTERIOR_SIZE': OUTERIOR_SIZE,
    'Q': Q,
    'S': S,
    'PHI_EPOCHS': PHI_EPOCHS,
    'EPOCHS': EPOCHS,
    'DIV_POR': DIV_POR,
    'END_TO_END': END_TO_END,
    'VAL_EVERY': VAL_EVERY,
    'B': B,
    'RESUME_PINN': RESUME_PINN,
    'RESUME_TASK': RESUME_TASK,
    'RESUME_PATH': RESUME_PATH,
    'GEN_POINTS': GEN_POINTS,
    'AUGMENT_ROTATION': AUGMENT_ROTATION,
    'AUGMENT_PERMUTE': AUGMENT_PERMUTE,
    'AUGMENT_REFLECT': AUGMENT_REFLECT,
    'DATASET_PATH': DATASET_PATH,
    'SPLIT': SPLIT,
}

with open(os.path.join(SAVE_DIR, 'config.json'), 'w') as fp:
    json.dump(CONSTANTS, fp, indent=2)

GEOMETRY_SELECTION_PATH = 'geometry_selection.json'
if os.path.exists(GEOMETRY_SELECTION_PATH):
    with open(GEOMETRY_SELECTION_PATH, 'r', encoding='utf-8') as fp:
        GEOMETRY_SELECTION = json.load(fp)
else:
    GEOMETRY_SELECTION = {}

# ключи вида "case_dir/file.stl", как их пишет select_geometries.py
ACCEPTED_GEOMETRIES = {k for k, v in GEOMETRY_SELECTION.items() if v == 'accept'}
print(f'Loaded {len(ACCEPTED_GEOMETRIES)} accepted geometries from {GEOMETRY_SELECTION_PATH}')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split="train"):
        self.data = []
        self.keys = []
        for dir in os.listdir(path):
            if dir not in SPLIT[split]:
                continue
            for file in os.listdir(os.path.join(path, dir)):
                if (file.count('_') == 1) and (file.split('_')[-1] != '-1.stl') and ('.stl' in file):
                    if (GEN_POINTS) or (file.replace('.stl', '_interior.pt') in os.listdir(os.path.join(path, dir))):
                        self.data.append(load_stl(os.path.join(path, dir, file), odd=False, device='cuda', gen_p=GEN_POINTS))
                        self.keys.append(os.path.join(dir, file).replace("/", "\\"))
                        torch.cuda.empty_cache()

        self.phi_stage = True

    def enter_flow_stage(self):
        """Переход на 2-й этап (flow): оставляет только принятые в geometry_selection.json
        геометрии и выключает phi_stage. __len__ сам подхватывает укороченный self.data."""
        if not self.phi_stage:
            return
        
        keep = [i for i, k in enumerate(self.keys) if k in ACCEPTED_GEOMETRIES]

        if not keep:
            raise RuntimeError(
                f"Ни одна геометрия не помечена как 'accept' в {GEOMETRY_SELECTION_PATH} "
                "для этого сплита. Запустите select_geometries.py перед переходом на 2-й этап."
            )
        self.data = [self.data[i] for i in keep]
        self.keys = [self.keys[i] for i in keep]
        self.phi_stage = False


    def __getitem__(self, index):
        if index < len(self.data):
            inlet = 'inlet'
            outlet = 'outlet'
        else:
            index = index - len(self.data)
            inlet = 'outlet'
            outlet = 'inlet'

        agg = self.data[index]

        idx_int   = torch.randperm(len(agg['x_dict']['interior'][:(100000 if (self.phi_stage or END_TO_END) else None)]))[:INTERIOR_SIZE]
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
            agg['phi_w_dict']['interior'][idx_int] if (self.phi_stage or END_TO_END) else agg['phi_w_dict']['interior'][:INTERIOR_SIZE],
            agg['phi_w_dict']['walls'][idx_w],
            agg['phi_w_dict'][inlet][idx_in],
            agg['phi_w_dict'][outlet][idx_out],
            agg['phi_w_dict']['outerior'][idx_outer],
        ])

        phi_key = "in" if inlet == 'inlet' else 'out'

        phi_in = torch.cat([
                agg[f'phi_{phi_key}_dict']['interior'][idx_int] if (self.phi_stage or END_TO_END) else agg[f'phi_{phi_key}_dict']['interior'][:INTERIOR_SIZE],
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

def get_act(act):
    if act == 'wave':
        return WaveAct()
    elif act == 'tanh':
        return torch.nn.Tanh()
    elif act == 'silu':
        return nn.SiLU(inplace=True)
    elif act == 'relu':
            return nn.ReLU(inplace=True)

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
            get_act(ACT_TF),
            nn.Linear(d_ff, d_ff),
            get_act(ACT_TF),
            nn.Linear(d_ff, d_model)
        ])

    def forward(self, x):
        return self.linear(x)
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = get_act(ACT_TF)
        self.act2 = get_act(ACT_TF)
        
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
        self.act1 = get_act(ACT_TF)
        self.act2 = get_act(ACT_TF)

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
        self.act = get_act(ACT_TF)

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return self.act(x)


class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.act = get_act(ACT_TF)
        
    def forward(self, x, e_outputs):
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.act(x)


class ResBlock(nn.Module):
    def __init__(self, d, act):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), get_act(act), nn.Linear(d, d))
        self.act = get_act(act)
    def forward(self, x):
        return self.act(x + self.net(x))


class GAPinn(nn.Module):
    def __init__(self, d_hidden_phi=256, d_hidden_v=256, d_model=256, N=4, heads=8):
        super(GAPinn, self).__init__()

        if USE_EMB:
            self.embedding = nn.Embedding(5, d_model)

        self.linear_emb = nn.Linear(9, d_model)

        self.encoder = Encoder(d_model, N, heads)

        self.linear_out_phi = nn.Sequential(*[
                        nn.Linear(d_model + (9 if USE_CLS_TOKEN else 0), d_hidden_phi),
                        ResBlock(d_hidden_phi, ACT_DEC),
                        ResBlock(d_hidden_phi, ACT_DEC),
                        ResBlock(d_hidden_phi, ACT_DEC),
                        nn.Linear(d_hidden_phi, 2)
                    ])

        self.linear_out_flow = nn.Sequential(*[
            nn.Linear(d_model + (9 if USE_CLS_TOKEN else 0), d_hidden_v),
            ResBlock(d_hidden_v, ACT_DEC),
            ResBlock(d_hidden_v, ACT_DEC),
            ResBlock(d_hidden_v, ACT_DEC),
            nn.Linear(d_hidden_v, 4)
        ])

        if USE_CLS_TOKEN:
            # обучаемый CLS-токен: добавляется к последовательности точек перед
            # энкодером, его выход после self-attention — общий вектор геометрии,
            # который вместе с координатами точки напрямую подаётся в MLP phi/flow
            # (без decoder_phi/cross-attention).
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)       
        else:
            self.decoder_phi = Decoder(d_model, N, heads)
            # self.decoder_flow = Decoder(d_model, N, heads)


    def forward(self, x, out, norm_in, norm_out, center_out, l, s, v_mean, x_label, pinn=False):
        if pinn:
            x_grad = x * 2 * l
            # x_grad = x
            x_grad.requires_grad_(True)
            x_proj_enc = self.linear_emb(torch.cat((x_grad.clone().detach() / l / 2, out.unsqueeze(1).repeat(1, x.shape[1], 1)), -1))                 # (B, N, d_model)
            if USE_EMB:
                label_emb = self.embedding(x_label)        # (B, N, d_model)
                x_proj_enc = x_proj_enc + label_emb

            if not USE_CLS_TOKEN:
                x_proj_dec = self.linear_emb(torch.cat((x_grad / l / 2, out.unsqueeze(1).repeat(1, x.shape[1], 1)), -1))                 # (B, N, d_model)
            coord = x_grad
        else:
            x_grad = None
            x_proj_enc = self.linear_emb(torch.cat((x, out.unsqueeze(1).repeat(1, x.shape[1], 1)), -1))
            if USE_EMB:
                label_emb = self.embedding(x_label)        # (B, N, d_model)
                x_proj_enc = x_proj_enc + label_emb
            if not USE_CLS_TOKEN:
                x_proj_dec = x_proj_enc
            coord = x

        if USE_CLS_TOKEN:
            cls_tok = self.cls_token.expand(x_proj_enc.shape[0], -1, -1)                        # (B, 1, d_model)
            enc_output = self.encoder(torch.cat((cls_tok, x_proj_enc), dim=1))
            cls_out = enc_output[:, 0:1]                                                        # (B, 1, d_model)

            coord_norm = coord / ((l * 2) if pinn else 1)

            phi_pred = self.linear_out_phi(torch.cat((
                cls_out.expand(-1, coord.shape[1], -1), coord_norm,
                out.unsqueeze(1).repeat(1, coord.shape[1], 1)), -1))

            pred = self.linear_out_flow(torch.cat((
                cls_out.expand(-1, _BND_END, -1), coord_norm[:, :_BND_END],
                out.unsqueeze(1).repeat(1, _BND_END, 1)), -1))
        else:
            e_outputs = self.encoder(x_proj_enc)

            d_output_phi = self.decoder_phi(x_proj_dec, e_outputs)
            d_output_flow = self.decoder_phi(x_proj_dec[:, :_BND_END], e_outputs[:, :_BND_END])

            phi_pred = self.linear_out_phi(d_output_phi)
            pred = self.linear_out_flow(d_output_flow)

        v = pred[..., :3]
        p = pred[..., 3:]

        v = (v * phi_pred[:, :_BND_END, 1:2]
                  + phi_pred[:, :_BND_END, 0:1] * norm_in[:, :_BND_END] * ((1 / v_mean[:, :_BND_END])*((Q * (s[:, :_BND_END] / S)) / s[:, :_BND_END])))      # (B, _BND_END, 3)

        signed_dist = ((x[:, :_BND_END] - center_out[:, :_BND_END])
                       * norm_out[:, :_BND_END]).sum(-1, keepdim=True)
        
        p = p * signed_dist * 10                       # (B, _BND_END, 1)

        return None, phi_pred, v[..., 0:1], v[..., 1:2], v[..., 2:3], p, x_grad


@torch.no_grad()
def reset_weights(m):
    """Resets weights to PyTorch defaults."""
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

ga_pinn = GAPinn().cuda()

if RESUME_PINN:
    if RESUME_SOURCE == 'clearml':
        resume_task = Task.get_task(task_id=RESUME_TASK,
            project_name='kornaeva-rnf/GA_PINN_3D'
        )

        history_path = resume_task.artifacts['history'].get_local_copy()
        history_val_path = resume_task.artifacts['history_val'].get_local_copy()
        model_path = resume_task.artifacts['model'].get_local_copy()
        if END_TO_END:
            optimizer_all_path = resume_task.artifacts['optimizer_all'].get_local_copy()
        else:
            optimizer_flow_path = resume_task.artifacts['optimizer_flow'].get_local_copy()
            optimizer_phi_path = resume_task.artifacts['optimizer_phi'].get_local_copy()
        ga_pinn.load_state_dict(torch.load(model_path))
    else:
        history_path = f"{RESUME_PATH}/history.json"
        history_val_path = f"{RESUME_PATH}/history_val.json"
        model_path = f"{RESUME_PATH}/model.pth"
        
        
        if END_TO_END:
            optimizer_all_path = f"{RESUME_PATH}/optimizer_all.pth"
        else:
            optimizer_flow_path = f"{RESUME_PATH}/optimizer_flow.pth"
            optimizer_phi_path = f"{RESUME_PATH}/optimizer_phi.pth"
        ga_pinn.load_state_dict(torch.load(model_path))

    # ga_pinn.linear_out_flow = nn.Sequential(*[
    #         nn.Linear(256 + (9 if USE_CLS_TOKEN else 0), 256),
    #         ResBlock(256, ACT_DEC),
    #         ResBlock(256, ACT_DEC),
    #         ResBlock(256, ACT_DEC),
    #         nn.Linear(256, 4)
    #     ]).to('cuda')


dataset_train = Dataset(DATASET_PATH, 'train')
dataset_val = Dataset(DATASET_PATH, 'val')

dataset_train.enter_flow_stage()
dataset_val.enter_flow_stage()

loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=B, shuffle=True)
loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=B, shuffle=False)

phi_trunk_params = [*ga_pinn.linear_emb.parameters(), *ga_pinn.encoder.parameters(), *ga_pinn.linear_out_phi.parameters()]
if USE_CLS_TOKEN:
    phi_trunk_params += [ga_pinn.cls_token]
else:
    phi_trunk_params += [*ga_pinn.decoder_phi.parameters()]

if END_TO_END:
    optimizer_all = torch.optim.Adam(ga_pinn.parameters(), lr=5e-4)
else:
    optimizer_phi = torch.optim.Adam(phi_trunk_params, lr=5e-4)
    optimizer_flow = torch.optim.Adam([*ga_pinn.linear_out_flow.parameters()], lr=5e-4)

if RESUME_PINN:
    if END_TO_END:
        optimizer_all.load_state_dict(torch.load(optimizer_all_path))
    else:
        # optimizer_flow.load_state_dict(torch.load(optimizer_flow_path))
        optimizer_phi.load_state_dict(torch.load(optimizer_phi_path))

if END_TO_END:
    lr_scheduler_all = torch.optim.lr_scheduler.StepLR(optimizer_all, 200, 0.97,
                                                        last_epoch=- 1)
else:
    lr_scheduler_phi = torch.optim.lr_scheduler.StepLR(optimizer_phi, 400, 0.97,
                                                        last_epoch=- 1)
    lr_scheduler_flow = torch.optim.lr_scheduler.StepLR(optimizer_flow, 400, 0.97,
                                                        last_epoch=- 1)
loss_fcn = torch.nn.MSELoss()

history = {'res_1': [], 'res_2': [], 'res_3': [], 'res_4': [], 'res_sum': [], 'mse_phi': [], 'lr_flow': [], 'lr_phi': []}
history_val = {'res_1': [], 'res_2': [], 'res_3': [], 'res_4': [], 'res_sum': [], 'mse_phi': []}

if RESUME_PINN:
    with open(history_path, 'r') as fp:
        history = json.load(fp)

    with open(history_val_path, 'r') as fp:
        history_val = json.load(fp)
        

logger = task.get_logger() if USE_CLEARML else _NullLogger()

complete_epochs = len(history['res_4']) if END_TO_END else len(history['res_4']) + len(history['mse_phi'])

flag = True

save_callback = SaveBest('res_sum', os.path.join(SAVE_DIR, "model_best.pth"), 'min')
save_callback.start(history, ga_pinn)

def run_batches(loader, i, train):
    """Проход по одному loader'у: forward/residuals для каждого батча.

    При train=False веса не обновляются (нет zero_grad/backward/step),
    но autograd-граф по x всё равно нужен calc_grad для residual-лоссов.
    """
    ga_pinn.train() if train else ga_pinn.eval()

    epoch_sums = {'res_1': 0., 'res_2': 0., 'res_3': 0., 'res_4': 0.,
                  'mse_out': 0., 'mse_phi': 0.}

    is_flow_epoch = END_TO_END or ((not (i % DIV_POR)) and train and (i != 0)) or (i >= PHI_EPOCHS)

    for x, phi, out, norm_in, norm_out, center_out, l, s, v_mean, x_label in tqdm(loader):
        x, phi, out, norm_in, norm_out, center_out, l, s, v_mean, x_label = \
            x.to('cuda'), phi.to('cuda'), out.to('cuda'), norm_in.to('cuda'), norm_out.to('cuda'), center_out.to('cuda'), l.to('cuda'), s.to('cuda'), v_mean.to('cuda'), x_label.to('cuda')

        if train and (AUGMENT_ROTATION or AUGMENT_PERMUTE or AUGMENT_REFLECT):
            T = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(x.shape[0], 1, 1)
            if AUGMENT_ROTATION:
                T = random_rotation_matrices(x.shape[0], x.device, x.dtype) @ T
            if AUGMENT_REFLECT:
                T = random_reflection_matrices(x.shape[0], x.device, x.dtype) @ T
            if AUGMENT_PERMUTE:
                T = random_axis_permutation_matrices(x.shape[0], x.device, x.dtype) @ T

            x, norm_in, norm_out, center_out = apply_orthogonal_transform(T, x, norm_in, norm_out, center_out)
            center_in_aug, center_out_aug = apply_orthogonal_transform(T, out[:, :3], out[:, 3:])
            out = torch.cat((center_in_aug, center_out_aug), dim=-1)

        if train:
            if END_TO_END:
                optimizer_all.zero_grad()
            elif is_flow_epoch:
                optimizer_flow.zero_grad()
            else:
                optimizer_phi.zero_grad()

        out_pred, phi_pred, v1, v2, v3, p, x_grad = ga_pinn(x, out, norm_in, norm_out, center_out, l, s, v_mean, x_label, pinn=bool(is_flow_epoch))

        # END_TO_END: обе части лосса считаются и суммируются каждый батч (одни веса,
        # один optimizer_all) — is_flow_epoch/END_TO_END ниже не exclusive-ветки,
        # а независимые добавки к loss, как раньше при разделении на фазы.
        loss = 0
        if is_flow_epoch:
            dv1, dv2, dv3, d2v1, d2v2, d2v3, dp = calc_grad(v1, v2, v3, p, x_grad, div_v_only=i < PHI_EPOCHS * 0)

            res = calc_res(v1, v2, v3, p, dv1[:, :_BND_END], dv2[:, :_BND_END], dv3[:, :_BND_END], d2v1[:, :_BND_END], d2v2[:, :_BND_END], d2v3[:, :_BND_END], dp[:, :_BND_END], div_v_only=i < PHI_EPOCHS * 0)

            loss = loss + zero_loss(res) * 0.000001
        if (not is_flow_epoch) or END_TO_END:
            loss_phi = loss_fcn(phi_pred, phi)

            loss = loss + loss_phi

        if train:
            loss.backward()

        if is_flow_epoch:
            # calc_res возвращает [res1, res2, res3, res4] (x/y/z-импульс, несжимаемость)
            # при div_v_only=False, и [res4] при div_v_only=True — несжимаемость всегда
            # последняя, поэтому res[-1], а не res[0].
            epoch_sums['res_4'] += mse_zero_loss(res[-1].detach().cpu()).item()
            if i >= PHI_EPOCHS * 0:
                epoch_sums['res_1'] += mse_zero_loss(res[0].detach().cpu()).item()
                epoch_sums['res_2'] += mse_zero_loss(res[1].detach().cpu()).item()
                epoch_sums['res_3'] += mse_zero_loss(res[2].detach().cpu()).item()
        if (not is_flow_epoch) or END_TO_END:
            epoch_sums['mse_phi'] += loss_phi.detach().cpu().item()

        if train:
            if END_TO_END:
                optimizer_all.step()
            elif is_flow_epoch:
                optimizer_flow.step()
            else:
                optimizer_phi.step()

    return epoch_sums, is_flow_epoch


def do_train(i):
    epoch_sums, is_flow_epoch = run_batches(loader_train, i, train=True)

    if is_flow_epoch:
        mean_val_sum = 0
        for key in ['res_4', 'res_1', 'res_2', 'res_3'][:1 if i < PHI_EPOCHS * 0 else 4]:
            mean_val = epoch_sums[key] / len(loader_train)
            history[key].append(mean_val)
            mean_val_sum += mean_val
            logger.report_scalar(title='Residuals', series=key, value=mean_val, iteration=i)
        if i >= PHI_EPOCHS * 0:
            history['res_sum'].append(mean_val_sum)
            save_callback.step()
    if (not is_flow_epoch) or END_TO_END:
        mean_val = epoch_sums['mse_phi'] / len(loader_train)
        history['mse_phi'].append(mean_val)
        logger.report_scalar(title='Losses', series='mse_phi', value=mean_val, iteration=i)

    torch.save(ga_pinn.state_dict(), os.path.join(SAVE_DIR, 'model.pth'))
    
    if END_TO_END:
        torch.save(optimizer_all.state_dict(), os.path.join(SAVE_DIR, 'optimizer_all.pth'))
    else:
        torch.save(optimizer_flow.state_dict(), os.path.join(SAVE_DIR, 'optimizer_flow.pth'))
        torch.save(optimizer_phi.state_dict(), os.path.join(SAVE_DIR, 'optimizer_phi.pth'))
    
    with open(os.path.join(SAVE_DIR, 'history.json'), 'w') as fp:
        json.dump(history, fp)

    if END_TO_END:
        lr_scheduler_all.step()
        lr_all = optimizer_all.param_groups[0]['lr']
        logger.report_scalar(title='Learning Rate', series='lr_flow', value=lr_all, iteration=i)
        logger.report_scalar(title='Learning Rate', series='lr_phi', value=lr_all, iteration=i)
        history['lr_flow'].append(lr_all)
        history['lr_phi'].append(lr_all)
    elif is_flow_epoch:
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
    epoch_sums, is_flow_epoch = run_batches(loader_val, i, train=False)

    if is_flow_epoch:
        mean_val_sum = 0
        for key in ['res_4', 'res_1', 'res_2', 'res_3'][:1 if i < PHI_EPOCHS * 0 else 4]:
            mean_val = epoch_sums[key] / len(loader_val)
            history_val[key].append(mean_val)
            mean_val_sum += mean_val
            logger.report_scalar(title='Residuals (val)', series=key, value=mean_val, iteration=i)
        if i >= PHI_EPOCHS * 0:
            history_val['res_sum'].append(mean_val_sum)
    if (not is_flow_epoch) or END_TO_END:
        mean_val = epoch_sums['mse_phi'] / len(loader_val)
        history_val['mse_phi'].append(mean_val)
        logger.report_scalar(title='Losses (val)', series='mse_phi', value=mean_val, iteration=i)

    with open(os.path.join(SAVE_DIR, 'history_val.json'), 'w') as fp:
        json.dump(history_val, fp)


def freeze_phi_trunk(model):
    """На flow-этапе optimizer_phi.step() больше не вызывается — веса phi-ветки
    (linear_emb, embedding, encoder, decoder_phi/cls_token, linear_out_phi) больше
    не обучаются. Выключаем им requires_grad, чтобы autograd не сохранял активации,
    нужные только для градиента по этим весам. На градиент по x (нужен для
    residual-лосса — calc_grad дифференцирует именно по x_grad) это не влияет:
    d(output)/d(x) не зависит от того, требуют ли веса по пути градиент сами."""
    modules = [model.linear_emb, model.encoder, model.linear_out_phi]
    if USE_EMB:
        modules.append(model.embedding)
    if USE_CLS_TOKEN:
        model.cls_token.requires_grad_(False)
    else:
        modules.append(model.decoder_phi)
    for module in modules:
        for param in module.parameters():
            param.requires_grad_(False)


for i in tqdm(range(complete_epochs, complete_epochs + EPOCHS)):
    if (i >= PHI_EPOCHS) and flag and not END_TO_END:
        flag = False
        lr_scheduler_flow = torch.optim.lr_scheduler.StepLR(optimizer_flow, 400, 0.97,
                                                last_epoch=- 1)

        dataset_train.enter_flow_stage()
        dataset_val.enter_flow_stage()
        freeze_phi_trunk(ga_pinn)

        # ga_pinn.linear_out_flow = ga_pinn.linear_out_flow.apply(reset_weights)

        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=B, shuffle=True)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=B, shuffle=False)

    do_train(i)

    if i % VAL_EVERY == 0:
        do_val(i)

if USE_CLEARML:
    task.upload_artifact(f'model', artifact_object=os.path.join(SAVE_DIR, 'model.pth'))
    task.upload_artifact(f'model_best', artifact_object=os.path.join(SAVE_DIR, 'model_best.pth'))
    task.upload_artifact(f'history', artifact_object=os.path.join(SAVE_DIR, 'history.json'))
    task.upload_artifact(f'history_val', artifact_object=os.path.join(SAVE_DIR, 'history_val.json'))
    
    if END_TO_END:
        task.upload_artifact(f'optimizer_all', artifact_object=os.path.join(SAVE_DIR, 'optimizer_all.pth'))
    else:
        task.upload_artifact(f'optimizer_phi', artifact_object=os.path.join(SAVE_DIR, 'optimizer_phi.pth'))
        task.upload_artifact(f'optimizer_flow', artifact_object=os.path.join(SAVE_DIR, 'optimizer_flow.pth'))