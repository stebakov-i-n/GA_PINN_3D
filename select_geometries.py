"""
Интерактивный отбор геометрий датасета для 2-го этапа обучения (flow-стадия).

Показывает по одной геометрии за раз (3D scatter точек + нормали входа/выхода),
кнопки "Принять"/"Отклонить" (или клавиши a/r) сохраняют решение в JSON сразу
после клика — скрипт можно прерывать и перезапускать в любой момент, уже
размеченные геометрии автоматически пропускаются.

Запуск: python select_geometries.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button

from modules import load_geometry_light

# 'a' (keymap.all_axes) и 'r' (keymap.home) по умолчанию перехвачены тулбаром
# matplotlib (сброс вида и т.п.) и конфликтуют с нашими accept/reject-хоткеями.
for _bindings in plt.rcParams:
    if _bindings.startswith('keymap.') and isinstance(plt.rcParams[_bindings], list):
        for _key in ('a', 'r'):
            if _key in plt.rcParams[_bindings]:
                plt.rcParams[_bindings].remove(_key)

DATASET_PATH = 'SimVascDataset'
SELECTION_PATH = 'geometry_selection.json'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# точек меньше, чем в augmentation_viz.ipynb — тут важна скорость просмотра, не точность
N_WALLS, N_INLET, N_OUTLET, N_INTERIOR, N_OUTERIOR = 800, 200, 200, 800, 800

LABEL_COLORS = ['#2a78d6', '#008300', '#eda100', '#4a3aa7', '#898781']
LABEL_NAMES = ['interior', 'walls', 'inlet', 'outlet', 'outerior']
NORMAL_COLOR = '#e34948'


def find_all_geometries(dataset_path):
    """Те же файлы, что видит Dataset.__init__ в main.py: *_N.stl"""
    paths = []
    for case_dir in sorted(os.listdir(dataset_path)):
        case_path = os.path.join(dataset_path, case_dir)
        if not os.path.isdir(case_path):
            continue
        for file in sorted(os.listdir(case_path)):
            if (file.count('_') == 1) and (file.split('_')[-1] != '-1.stl') and ('.stl' in file):
                paths.append(os.path.join(case_dir, file).replace("/", "\\"))
    return paths


def load_selection():
    if os.path.exists(SELECTION_PATH):
        with open(SELECTION_PATH, 'r', encoding='utf-8') as fp:
            return json.load(fp)
    return {}


def save_selection(selection):
    with open(SELECTION_PATH, 'w', encoding='utf-8') as fp:
        json.dump(selection, fp, indent=2, ensure_ascii=False)


def set_axes_equal(ax):
    """mplot3d не выравнивает оси по умолчанию — без этого форма визуально искажается."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    radius = 0.5 * max(limits[:, 1] - limits[:, 0])
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def plot_geometry(ax, x, x_label, norm_in, norm_out, center_in, center_out, title):
    ax.cla()
    x_np = x.cpu().numpy()
    label_np = x_label.cpu().numpy()

    for lbl in range(5):
        mask = label_np == lbl
        if mask.any():
            ax.scatter(x_np[mask, 0], x_np[mask, 1], x_np[mask, 2],
                        s=4, color=LABEL_COLORS[lbl], label=LABEL_NAMES[lbl],
                        depthshade=False, linewidths=0)

    for center, norm in [(center_in, norm_in), (center_out, norm_out)]:
        c = center.cpu().numpy()
        n = norm.cpu().numpy()
        ax.quiver(c[0], c[1], c[2], n[0], n[1], n[2],
                   length=0.2, color=NORMAL_COLOR, linewidth=2, normalize=True)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.legend(loc='upper left', fontsize=8, frameon=False)
    set_axes_equal(ax)


class Reviewer:
    def __init__(self, dataset_path, selection_path):
        self.dataset_path = dataset_path
        self.selection = load_selection()

        all_paths = find_all_geometries(dataset_path)
        self.queue = [p for p in all_paths if p not in self.selection]

        print(f'{len(all_paths)} геометрий всего, {len(self.selection)} уже размечено, '
              f'{len(self.queue)} осталось.')

        self.fig = plt.figure(figsize=(9, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.subplots_adjust(bottom=0.15)

        ax_reject = self.fig.add_axes([0.56, 0.02, 0.18, 0.07])
        ax_accept = self.fig.add_axes([0.76, 0.02, 0.18, 0.07])
        self.btn_reject = Button(ax_reject, 'Отклонить (r)', color='#f7d9d8', hovercolor='#f0b3b1')
        self.btn_accept = Button(ax_accept, 'Принять (a)', color='#d3ecd9', hovercolor='#a8dab6')
        self.btn_reject.on_clicked(lambda event: self.decide('reject'))
        self.btn_accept.on_clicked(lambda event: self.decide('accept'))

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.current_path = None
        self.show_next()

    def on_key(self, event):
        if event.key == 'a':
            self.decide('accept')
        elif event.key == 'r':
            self.decide('reject')

    def show_next(self):
        if not self.queue:
            self.ax.cla()
            self.ax.set_title('Готово: все геометрии размечены', fontsize=13)
            self.fig.canvas.draw_idle()
            self.current_path = None
            return

        self.current_path = self.queue[0]
        full_path = os.path.join(self.dataset_path, self.current_path)

        x_dict, norm_in, norm_out, center_in, center_out = load_geometry_light(
            full_path, device=DEVICE, n_walls=N_WALLS, n_inlet=N_INLET, n_outlet=N_OUTLET,
            n_interior=N_INTERIOR, n_outerior=N_OUTERIOR)
        x = torch.cat([x_dict['interior'], x_dict['walls'], x_dict['inlet'],
                       x_dict['outlet'], x_dict['outerior']], dim=0)
        x_label = torch.cat([
            torch.full((len(x_dict['interior']),), 0),
            torch.full((len(x_dict['walls']),), 1),
            torch.full((len(x_dict['inlet']),), 2),
            torch.full((len(x_dict['outlet']),), 3),
            torch.full((len(x_dict['outerior']),), 4),
        ])

        decided = len(self.selection)
        total = decided + len(self.queue)
        title = f'[{decided + 1}/{total}]  {self.current_path}'
        plot_geometry(self.ax, x, x_label, norm_in, norm_out, center_in, center_out, title)
        self.fig.canvas.draw_idle()

    def decide(self, decision):
        if self.current_path is None:
            return
        self.selection[self.current_path] = decision
        save_selection(self.selection)
        print(f'{decision:>7}: {self.current_path}')
        self.queue.pop(0)
        self.show_next()


if __name__ == '__main__':
    Reviewer(DATASET_PATH, SELECTION_PATH)
    plt.show()
