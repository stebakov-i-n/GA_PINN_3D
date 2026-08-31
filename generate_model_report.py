"""
Формирует PDF-отчёт по всем вариантам MODE из test_model1.ipynb: прогоняет
каждый чекпоинт на одном и том же тестовом сплите/сэмпле и собирает те же
визуализации (phi pred vs gt, векторное поле v, давление p, MAE по всей
тестовой выборке, инвариантность CLS-эмбеддинга к аугментациям) в один PDF,
по разделу на модель.

Датасет грузится один раз и переиспользуется для всех моделей (сам датасет от
MODE не зависит — меняются только веса/архитектура decoder-head).

Запуск (нужно окружение с CUDA, torch, matplotlib, clearml, sklearn — conda env
PINN в этом проекте):
    python generate_model_report.py
"""
import copy
import json
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from modules import *

plt.show = lambda *a, **kw: None   # headless — фигуры сохраняем в PDF вручную

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {DEVICE}')

# --- конфигурация теста (как в test_model1.ipynb) ---
DATASET_PATH = 'SimVascDataset'
TEST_SPLIT = 'train'
TEST_STAGE = 'flow'
FULL_TRAIN = False
CHECKPOINT_NAME = 'model.pth'
IDX = 2                  # тот же тестовый сэмпл для всех моделей — для сравнимости отчёта
N_RANDOM_AUGS = 10
OUTPUT_PDF = 'model_comparison_report.pdf'

MODES = {
    'baseline':     dict(LOCAL_RUN_DIR='trained_models/baseline',            ACT_TF='wave', USE_IN_OUT=False),
    'aug':          dict(LOCAL_RUN_DIR='trained_models/2026-08-14_22-37-04', ACT_TF='wave', USE_IN_OUT=False),
    'aug_full':     dict(LOCAL_RUN_DIR='trained_models/aug_full',            ACT_TF='relu', USE_IN_OUT=True),
    'end2end_phi':  dict(LOCAL_RUN_DIR='trained_models/2026-08-26_06-45-14', ACT_TF='relu', USE_IN_OUT=True),
    'end2end_flow': dict(LOCAL_RUN_DIR='trained_models/2026-08-30_20-50-18', ACT_TF='relu', USE_IN_OUT=True),
}

# --- константы, общие для всех MODE (main.py / test_model1.ipynb) ---
USE_EMB = True
ACT_DEC = 'wave'

INTERIOR_SIZE = 500
WALLS_SIZE    = 250
INLET_SIZE    = 100
OUTLET_SIZE   = 100
OUTERIOR_SIZE = 250

Q = 1.5e-6
S = 2e-6

_BND_START = INTERIOR_SIZE
_BND_END   = INTERIOR_SIZE + WALLS_SIZE + INLET_SIZE + OUTLET_SIZE

GEN_POINTS = False
LABEL_NAMES = ['interior', 'walls', 'inlet', 'outlet', 'outerior']

USE_CLS_TOKEN = True

if FULL_TRAIN:
    with open('full_split.json', 'r') as fp:
        SPLIT = json.load(fp)
else:
    with open('debug_split.json', 'r') as fp:
        SPLIT = json.load(fp)

GEOMETRY_SELECTION_PATH = 'geometry_selection.json'
if os.path.exists(GEOMETRY_SELECTION_PATH):
    with open(GEOMETRY_SELECTION_PATH, 'r', encoding='utf-8') as fp:
        GEOMETRY_SELECTION = json.load(fp)
else:
    GEOMETRY_SELECTION = {}
ACCEPTED_GEOMETRIES = {k for k, v in GEOMETRY_SELECTION.items() if v == 'accept'}
print(f'{len(SPLIT[TEST_SPLIT])} case-директорий в сплите "{TEST_SPLIT}", '
      f'{len(ACCEPTED_GEOMETRIES)} принятых геометрий всего')


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

        # GT phi_w/phi_in/phi_out по интерьеру посчитан (см. load_stl) только для
        # первых n_interior_phi закэшированных точек, независимо от phi_stage.
        # idx_int всегда должен лежать в этом диапазоне, иначе x и phi (gt) по
        # интерьеру относятся к разным точкам (main.py на flow-этапе с этим мирится,
        # т.к. там phi вообще не участвует в лоссе — здесь же сравниваем pred vs gt).
        n_interior_phi = len(agg['phi_w_dict']['interior'])
        idx_int   = torch.randperm(min(len(agg['x_dict']['interior']), n_interior_phi))[:INTERIOR_SIZE]
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
            agg['phi_w_dict']['interior'][idx_int],
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

        if USE_IN_OUT:
            if 'aug_full' in model_path:
                self.linear_out_phi = nn.Sequential(*[
                                        nn.Linear(d_model + (9 if USE_CLS_TOKEN else 0), d_hidden_phi),
                                        get_act(ACT_DEC),
                                        nn.Linear(d_hidden_v, d_hidden_v),
                                        get_act(ACT_DEC),
                                        nn.Linear(d_hidden_phi, 2)
                                    ])

                self.linear_out_flow = nn.Sequential(*[
                                        nn.Linear(d_model + (9 if USE_CLS_TOKEN else 0), d_hidden_v),
                                        get_act(ACT_DEC),
                                        nn.Linear(d_hidden_v, d_hidden_v),
                                        get_act(ACT_DEC),
                                        nn.Linear(d_hidden_v, 4)
                                    ])
            else:
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
        else:
            self.linear_out_phi = nn.Sequential(*[
                        nn.Linear(d_model + (3 if USE_CLS_TOKEN else 0), d_hidden_phi),
                        get_act(ACT_DEC),
                        nn.Linear(d_hidden_v, d_hidden_v),
                        get_act(ACT_DEC),
                        nn.Linear(d_hidden_phi, 2)
                    ])

            self.linear_out_flow = nn.Sequential(*[
                        nn.Linear(d_model + (3 if USE_CLS_TOKEN else 0), d_hidden_v),
                        get_act(ACT_DEC),
                        nn.Linear(d_hidden_v, d_hidden_v),
                        get_act(ACT_DEC),
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

    def forward(self, x, out, norm_in, norm_out, center_out, l, s, v_mean, x_label, pinn=False):
        if pinn:
            x_grad = x * 2 * l
            x_grad.requires_grad_(True)
            x_proj_enc = self.linear_emb(torch.cat((x_grad.clone().detach() / l / 2, out.unsqueeze(1).repeat(1, x.shape[1], 1)), -1))
            if USE_EMB:
                label_emb = self.embedding(x_label)
                x_proj_enc = x_proj_enc + label_emb

            if not USE_CLS_TOKEN:
                x_proj_dec = self.linear_emb(torch.cat((x_grad / l / 2, out.unsqueeze(1).repeat(1, x.shape[1], 1)), -1))
            coord = x_grad
        else:
            x_grad = None
            x_proj_enc = self.linear_emb(torch.cat((x, out.unsqueeze(1).repeat(1, x.shape[1], 1)), -1))
            if USE_EMB:
                label_emb = self.embedding(x_label)
                x_proj_enc = x_proj_enc + label_emb
            if not USE_CLS_TOKEN:
                x_proj_dec = x_proj_enc
            coord = x

        if USE_CLS_TOKEN:
            cls_tok = self.cls_token.expand(x_proj_enc.shape[0], -1, -1)
            enc_output = self.encoder(torch.cat((cls_tok, x_proj_enc), dim=1))
            cls_out = enc_output[:, 0:1]

            coord_norm = coord / ((l * 2) if pinn else 1)
            if USE_IN_OUT:
                phi_pred = self.linear_out_phi(torch.cat((
                    cls_out.expand(-1, coord.shape[1], -1), coord_norm,
                    out.unsqueeze(1).repeat(1, coord.shape[1], 1)), -1))

                pred = self.linear_out_flow(torch.cat((
                    cls_out.expand(-1, _BND_END, -1), coord_norm[:, :_BND_END],
                    out.unsqueeze(1).repeat(1, _BND_END, 1)), -1))
            else:
                phi_pred = self.linear_out_phi(torch.cat((
                                    cls_out.expand(-1, coord.shape[1], -1), coord_norm), -1))

                pred = self.linear_out_flow(torch.cat((
                    cls_out.expand(-1, _BND_END, -1), coord_norm[:, :_BND_END]), -1))
        else:
            e_outputs = self.encoder(x_proj_enc)

            d_output_phi = self.decoder_phi(x_proj_dec, e_outputs)
            d_output_flow = self.decoder_phi(x_proj_dec[:, :_BND_END], e_outputs[:, :_BND_END])

            phi_pred = self.linear_out_phi(d_output_phi)
            pred = self.linear_out_flow(d_output_flow)

        v = pred[..., :3]
        p = pred[..., 3:]

        v = (v * phi_pred[:, :_BND_END, 1:2]
                  + phi_pred[:, :_BND_END, 0:1] * norm_in[:, :_BND_END] * ((1 / v_mean[:, :_BND_END])*((Q * (s[:, :_BND_END] / S)) / s[:, :_BND_END])))

        signed_dist = ((x[:, :_BND_END] - center_out[:, :_BND_END])
                       * norm_out[:, :_BND_END]).sum(-1, keepdim=True)
        p = p * signed_dist * 10

        return None, phi_pred, v[..., 0:1], v[..., 1:2], v[..., 2:3], p, x_grad


def set_axes_equal(ax):
    """mplot3d не выравнивает оси по умолчанию — без этого форма визуально искажается."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.5 * max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def run_inference(dataset, idx):
    """Прогоняет один сэмпл датасета через модель (без градиентов, pinn=True —
    как в test_model1.ipynb)."""
    subset = torch.utils.data.Subset(dataset, [idx])
    loader = torch.utils.data.DataLoader(subset, batch_size=1)
    x, phi, out, norm_in, norm_out, center_out, l, s, v_mean, x_label = next(iter(loader))
    x, phi, out, norm_in, norm_out, center_out, l, s, v_mean, x_label = \
        x.to(DEVICE), phi.to(DEVICE), out.to(DEVICE), norm_in.to(DEVICE), norm_out.to(DEVICE), \
        center_out.to(DEVICE), l.to(DEVICE), s.to(DEVICE), v_mean.to(DEVICE), x_label.to(DEVICE)

    with torch.no_grad():
        _, phi_pred, v1, v2, v3, p, _ = ga_pinn(x, out, norm_in, norm_out, center_out, l, s, v_mean, x_label, pinn=True)

    return dict(x=x[0], phi=phi[0], phi_pred=phi_pred[0], x_label=x_label[0],
                v1=v1[0, :, 0], v2=v2[0, :, 0], v3=v3[0, :, 0], p=p[0, :, 0])


def scatter_scalar(ax, points, values, title, vmin=None, vmax=None):
    points = points.detach().cpu().numpy()
    values = values.detach().cpu().numpy()
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=values, cmap='jet',
                     s=8, vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    set_axes_equal(ax)
    return sc


def plot_phi_channel(result, channel, channel_name):
    x = result['x']
    label = result['x_label']
    gt = result['phi'][:, channel]
    pred = result['phi_pred'][:, channel]

    fig, axes = plt.subplots(len(LABEL_NAMES), 2, figsize=(7, 3 * len(LABEL_NAMES)),
                              subplot_kw={'projection': '3d'})
    for row, name in enumerate(LABEL_NAMES):
        mask = label == row
        sc = scatter_scalar(axes[row, 0], x[mask], gt[mask], f'{name} — gt', gt[mask].min(), gt[mask].max())
        scatter_scalar(axes[row, 1], x[mask], pred[mask], f'{name} — pred', gt[mask].min(), gt[mask].max())
        fig.colorbar(sc, ax=axes[row, :], shrink=0.7, pad=0.02)

    fig.suptitle(f'{channel_name} (канал {channel})', fontsize=13)


def plot_phi_curve(result, channel, channel_name):
    label = result['x_label'].cpu().numpy()
    gt = result['phi'][:, channel].detach().cpu().numpy()
    pred = result['phi_pred'][:, channel].detach().cpu().numpy()

    counts = np.bincount(label, minlength=len(LABEL_NAMES))
    boundaries = np.cumsum(counts)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(gt, label='gt', color='#2a78d6', linewidth=1.2)
    ax.plot(pred, label='pred', color='#e34948', linewidth=1.2, linestyle='--')

    prev = 0
    for name, boundary, count in zip(LABEL_NAMES, boundaries, counts):
        if count == 0:
            continue
        if prev > 0:
            ax.axvline(prev, color='#c3c2b7', linewidth=0.8, linestyle=':')
        ax.text(prev + count / 2, 1.02, name, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=8, color='#52514e')
        prev = boundary

    ax.set_xlim(0, len(label))
    ax.set_xlabel('индекс точки')
    ax.set_ylabel(channel_name)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.set_title(f'{channel_name} (канал {channel}) — gt vs pred по индексу точки', fontsize=11)


def plot_velocity_field(result, step=1):
    x = result['x'][:_BND_END].detach().cpu().numpy()
    u = result['v1'].detach().cpu().numpy()
    v = result['v2'].detach().cpu().numpy()
    w = result['v3'].detach().cpu().numpy()

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')
    scale = float(np.abs(x).max()) * 0.08

    ax.quiver(x[::step, 0], x[::step, 1], x[::step, 2],
              u[::step], v[::step], w[::step],
              length=scale, normalize=True, color="#e92c0b",
              linewidth=0.6, arrow_length_ratio=0.6, alpha=0.6)
    ax.set_title('Векторное поле скорости (pred)')
    set_axes_equal(ax)


def plot_scalar_by_surface(result, values, title, surfaces=('interior', 'walls', 'inlet', 'outlet')):
    x = result['x'][:_BND_END]
    label = result['x_label'][:_BND_END]
    vmin, vmax = float(values.min()), float(values.max())

    fig, axes = plt.subplots(1, len(surfaces), figsize=(4 * len(surfaces), 4),
                              subplot_kw={'projection': '3d'})
    for ax, name in zip(np.atleast_1d(axes), surfaces):
        row = LABEL_NAMES.index(name)
        mask = label == row
        sc = scatter_scalar(ax, x[mask], values[mask], name, vmin, vmax)
    fig.colorbar(sc, ax=axes, shrink=0.7, pad=0.02)
    fig.suptitle(title, fontsize=13)


def load_sample(dataset, idx):
    subset = torch.utils.data.Subset(dataset, [idx])
    loader = torch.utils.data.DataLoader(subset, batch_size=1)
    x, phi, out, norm_in, norm_out, center_out, l, s, v_mean, x_label = next(iter(loader))
    return [t.to(DEVICE) for t in (x, phi, out, norm_in, norm_out, center_out, l, s, v_mean, x_label)]


def augment_sample(x, out, norm_in, norm_out, center_out, rotate=True, reflect=True, permute=True):
    B_ = x.shape[0]
    T = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(B_, 1, 1)
    if rotate:
        T = random_rotation_matrices(B_, x.device, x.dtype) @ T
    if reflect:
        T = random_reflection_matrices(B_, x.device, x.dtype) @ T
    if permute:
        T = random_axis_permutation_matrices(B_, x.device, x.dtype) @ T

    x_t, norm_in_t, norm_out_t, center_out_t = apply_orthogonal_transform(T, x, norm_in, norm_out, center_out)
    center_in_aug, center_out_aug = apply_orthogonal_transform(T, out[:, :3], out[:, 3:])
    out_t = torch.cat((center_in_aug, center_out_aug), dim=-1)

    return x_t, out_t, norm_in_t, norm_out_t, center_out_t


_embedding_capture = {}

def _capture_encoder_output(module, inputs, output):
    _embedding_capture['enc_output'] = output.detach()


def get_cls_embedding(x, out, norm_in, norm_out, center_out, l, s, v_mean, x_label):
    with torch.no_grad():
        ga_pinn(x, out, norm_in, norm_out, center_out, l, s, v_mean, x_label, pinn=True)
    return _embedding_capture['enc_output'][:, 0].clone()


def cosine_similarity(a, b):
    a, b = a.flatten(), b.flatten()
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def relative_l2_distance(a, b):
    a, b = a.flatten(), b.flatten()
    return float((a - b).norm().item() / (a.norm().item() + 1e-12))


_FIXED_VARIANT_NAMES = ['identity', 'rotation only', 'reflection only', 'permutation only', 'combined']
_FIXED_PALETTE = ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7']


def plot_similarity_bars(records):
    names = [r[0] for r in records]
    cos_vals = [r[1] for r in records]
    colors = [_FIXED_PALETTE[_FIXED_VARIANT_NAMES.index(n)] if n in _FIXED_VARIANT_NAMES else '#898781'
              for n in names]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 4))
    ax.bar(names, cos_vals, color=colors)
    ax.axhline(1.0, color='#c3c2b7', linewidth=0.8, linestyle=':')
    ax.set_ylim(min(0.0, min(cos_vals) - 0.05), 1.05)
    ax.set_ylabel('cosine similarity к baseline')
    ax.set_title('Похожесть CLS-эмбеддинга при аугментации геометрии')
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()


def plot_embedding_curves(embeddings_dict):
    names = list(embeddings_dict.keys())
    fixed = {'baseline': '#0b0b0b', **dict(zip(_FIXED_VARIANT_NAMES, _FIXED_PALETTE))}

    fig, ax = plt.subplots(figsize=(12, 4))
    random_labelled = False
    for name in names:
        vals = embeddings_dict[name]
        if name == 'baseline':
            ax.plot(vals, color=fixed[name], linewidth=1.2, linestyle='-', label=name, zorder=4)
        elif name in fixed:
            ax.plot(vals, color=fixed[name], linewidth=1.2, linestyle='--', label=name, zorder=3)
        else:
            ax.plot(vals, color='#898781', linewidth=1.0, alpha=0.5, zorder=2,
                    label=None if random_labelled else 'случайные аугментации')
            random_labelled = True

    ax.set_xlabel('компонента эмбеддинга')
    ax.set_ylabel('значение')
    ax.set_title('CLS-эмбеддинг по компонентам: baseline vs аугментации')
    ax.legend(frameon=False, fontsize=8, loc='upper right', ncol=2)
    plt.tight_layout()


def save_current_figure(pdf):
    fig = plt.gcf()
    pdf.savefig(fig)
    plt.close(fig)


def add_text_page(pdf, lines, title=None):
    fig = plt.figure(figsize=(8.5, 11))
    if title:
        fig.text(0.08, 0.95, title, fontsize=16, weight='bold', va='top')
    fig.text(0.08, 0.88, '\n'.join(lines), fontsize=10, va='top', family='monospace')
    plt.axis('off')
    pdf.savefig(fig)
    plt.close(fig)


print('Загрузка тестового датасета (общий для всех моделей)...')
dataset_test = Dataset(DATASET_PATH, TEST_SPLIT)
if TEST_STAGE == 'flow':
    dataset_test.enter_flow_stage()
print(f'{len(dataset_test)} тестовых сэмплов ({len(dataset_test.data)} геометрий x2), '
      f'phi_stage={dataset_test.phi_stage}')


with PdfPages(OUTPUT_PDF) as pdf:
    add_text_page(pdf, [
        f'Тестовый сплит: {TEST_SPLIT} ({TEST_STAGE})',
        f'Датасет: {DATASET_PATH}',
        f'Тестовый образец: IDX={IDX}',
        f'Сравниваемые модели: {", ".join(MODES.keys())}',
    ], title='Сравнение моделей GAPinn')

    for mode_name, cfg in MODES.items():
        print(f'\n=== MODE = {mode_name} ===')
        ACT_TF = cfg['ACT_TF']
        USE_IN_OUT = cfg['USE_IN_OUT']
        LOCAL_RUN_DIR = cfg['LOCAL_RUN_DIR']
        model_path = os.path.join(LOCAL_RUN_DIR, CHECKPOINT_NAME)

        if not os.path.exists(model_path):
            print(f'  чекпоинт не найден: {model_path} — раздел пропущен')
            add_text_page(pdf, [f'Чекпоинт не найден: {model_path}', 'Раздел пропущен.'], title=f'MODE = {mode_name}')
            continue

        ga_pinn = GAPinn().to(DEVICE)
        ga_pinn.load_state_dict(torch.load(model_path, map_location=DEVICE))
        ga_pinn.eval()
        print(f'  Загружены веса: {model_path}')

        add_text_page(pdf, [
            f'Чекпоинт: {model_path}',
            f'ACT_TF: {ACT_TF}   USE_IN_OUT: {USE_IN_OUT}',
        ], title=f'Модель: {mode_name}')

        result = run_inference(dataset_test, IDX)

        plot_phi_channel(result, 0, 'phi_1 — расстояние до стенок'); save_current_figure(pdf)
        plot_phi_channel(result, 1, 'phi_2 — расстояние до стенок и входа'); save_current_figure(pdf)
        plot_phi_curve(result, 0, 'phi_1'); save_current_figure(pdf)
        plot_phi_curve(result, 1, 'phi_2'); save_current_figure(pdf)

        plot_velocity_field(result); save_current_figure(pdf)
        speed = (result['v1'] ** 2 + result['v2'] ** 2 + result['v3'] ** 2) ** 0.5
        plot_scalar_by_surface(result, speed, f'{mode_name}: |v| (pred) по поверхностям'); save_current_figure(pdf)
        plot_scalar_by_surface(result, result['p'], f'{mode_name}: p (pred) по поверхностям'); save_current_figure(pdf)

        print(f'  Считаю MAE по phi на всей тестовой выборке...')
        mae_phi1, mae_phi2 = [], []
        for idx in tqdm(range(len(dataset_test)), desc=f'{mode_name}: MAE'):
            r = run_inference(dataset_test, idx)
            err = (r['phi_pred'] - r['phi']).abs().mean(dim=0)
            mae_phi1.append(err[0].item())
            mae_phi2.append(err[1].item())

        add_text_page(pdf, [
            f'phi_1 MAE: {np.mean(mae_phi1):.4f} +/- {np.std(mae_phi1):.4f}  (по {len(mae_phi1)} сэмплам)',
            f'phi_2 MAE: {np.mean(mae_phi2):.4f} +/- {np.std(mae_phi2):.4f}',
        ], title=f'{mode_name}: MAE по phi (тестовая выборка)')

        print('  Оцениваю инвариантность CLS-эмбеддинга к аугментациям...')
        try:
            _encoder_hook.remove()
        except NameError:
            pass
        _encoder_hook = ga_pinn.encoder.register_forward_hook(_capture_encoder_output)

        x0, phi0, out0, norm_in0, norm_out0, center_out0, l0, s0, v_mean0, x_label0 = load_sample(dataset_test, IDX)
        baseline_emb = get_cls_embedding(x0, out0, norm_in0, norm_out0, center_out0, l0, s0, v_mean0, x_label0)

        variants = {
            'identity':          dict(rotate=False, reflect=False, permute=False),
            'rotation only':     dict(rotate=True,  reflect=False, permute=False),
            'reflection only':   dict(rotate=False, reflect=True,  permute=False),
            'permutation only':  dict(rotate=False, reflect=False, permute=True),
            'combined':          dict(rotate=True,  reflect=True,  permute=True),
        }

        torch.manual_seed(0)
        records = []
        embeddings = {'baseline': baseline_emb.cpu().numpy().flatten()}

        for name, flags in variants.items():
            x_t, out_t, norm_in_t, norm_out_t, center_out_t = augment_sample(
                x0, out0, norm_in0, norm_out0, center_out0, **flags)
            emb = get_cls_embedding(x_t, out_t, norm_in_t, norm_out_t, center_out_t, l0, s0, v_mean0, x_label0)
            records.append((name, cosine_similarity(baseline_emb, emb), relative_l2_distance(baseline_emb, emb)))
            embeddings[name] = emb.cpu().numpy().flatten()

        for i in range(N_RANDOM_AUGS):
            x_t, out_t, norm_in_t, norm_out_t, center_out_t = augment_sample(x0, out0, norm_in0, norm_out0, center_out0)
            emb = get_cls_embedding(x_t, out_t, norm_in_t, norm_out_t, center_out_t, l0, s0, v_mean0, x_label0)
            name = f'random #{i + 1}'
            records.append((name, cosine_similarity(baseline_emb, emb), relative_l2_distance(baseline_emb, emb)))
            embeddings[name] = emb.cpu().numpy().flatten()

        add_text_page(pdf, [f'{n:<18} cos={c:+.4f}  relL2={r:.4f}' for n, c, r in records],
                      title=f'{mode_name}: инвариантность CLS-эмбеддинга к аугментациям')

        plot_similarity_bars(records); save_current_figure(pdf)
        plot_embedding_curves(embeddings); save_current_figure(pdf)

        del ga_pinn
        torch.cuda.empty_cache()

print(f'\nГотово: {OUTPUT_PDF}')
