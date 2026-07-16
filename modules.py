import numpy as np
import torch
from stl import mesh
from torch import autograd

def calc_grad(v1, v2, v3, p, x, div_v_only=True):
    dv1 = autograd.grad(v1.sum(), x, create_graph=True)[0]
    dv2 = autograd.grad(v2.sum(), x, create_graph=True)[0]
    dv3 = autograd.grad(v3.sum(), x, create_graph=True)[0]
    dp = autograd.grad(p.sum(), x, create_graph=True)[0]
    if div_v_only:
        return dv1, dv2, dv3, dv1, dv2, dv3, dp
    d2v1 = autograd.grad(dv1.sum(), x, create_graph=True)[0]
    d2v2 = autograd.grad(dv2.sum(), x, create_graph=True)[0]
    d2v3 = autograd.grad(dv3.sum(), x, create_graph=True)[0]
    return dv1, dv2, dv3, d2v1, d2v2, d2v3, dp

def calc_res(v1, v2, v3, p, dv1, dv2, dv3, d2v1, d2v2, d2v3, dp, div_v_only=True):
    mu = 3e-3
    rho = 1050
    if not div_v_only:
        res1 = (v1 * dv1[..., 0:1] + v2 * dv1[..., 1:2] + v3 * dv1[..., 2:3]) - mu * (d2v1[..., 0:1] + d2v1[..., 1:2] + d2v1[..., 2:3]) / rho + dp[..., 0:1] / rho
        res2 = (v1 * dv2[..., 0:1] + v2 * dv2[..., 1:2] + v3 * dv2[..., 2:3]) - mu * (d2v2[..., 0:1] + d2v2[..., 1:2] + d2v2[..., 2:3]) / rho + dp[..., 1:2] / rho
        res3 = (v1 * dv3[..., 0:1] + v2 * dv3[..., 1:2] + v3 * dv3[..., 2:3]) - mu * (d2v3[..., 0:1] + d2v3[..., 1:2] + d2v3[..., 2:3]) / rho + dp[..., 2:3] / rho
    res4 = dv1[..., 0:1] + dv2[..., 1:2] + dv3[..., 2:3]
    if not div_v_only:
        return [res1, res2, res3, res4]    
    return [res4]

def mse_zero_loss(f):
    return (f ** 2).mean()

def zero_loss(outputs):
    loss = 0
    for i in range(len(outputs)):
        loss += mse_zero_loss(outputs[i])
    loss = loss / len(outputs)
    return loss

def random_rotation_matrices(batch_size, device, dtype=torch.float32):
    """Равномерно случайные матрицы поворота SO(3), по одной на элемент батча.

    Метод Меццадри: QR-разложение случайной гауссовой матрицы даёт Q, равномерно
    распределённую на O(3); фиксация знака диагонали R делает распределение Q
    хаар-равномерной, после чего при необходимости меняется знак одного столбца,
    чтобы гарантировать det(Q) = +1 (собственно поворот, без отражения).
    """
    A = torch.randn(batch_size, 3, 3, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    d = torch.diagonal(R, dim1=-2, dim2=-1)
    Q = Q * torch.sign(d).unsqueeze(-2)
    neg_det = torch.linalg.det(Q) < 0
    Q[neg_det, :, 0] *= -1
    return Q

def random_axis_permutation_matrices(batch_size, device, dtype=torch.float32):
    """Случайные перестановки порядка координатных осей (напр. x1 x2 x3 -> x2 x1 x3),
    по одной перестановочной матрице на элемент батча."""
    perms = torch.stack([torch.randperm(3, device=device) for _ in range(batch_size)])
    return torch.eye(3, device=device, dtype=dtype)[perms]

def random_reflection_matrices(batch_size, device, dtype=torch.float32):
    """Случайные отражения по координатным осям (каждая из x1,x2,x3 независимо
    с вероятностью 1/2), по одной диагональной матрице ±1 на элемент батча."""
    signs = torch.randint(0, 2, (batch_size, 3), device=device, dtype=dtype) * 2 - 1
    return torch.diag_embed(signs)

def apply_orthogonal_transform(T, *tensors):
    """Применяет батч ортогональных преобразований T (B, 3, 3) к позициям/направлениям.

    Каждый тензор в tensors — либо (B, N, 3) (точки/направления на каждую точку),
    либо (B, 3) (одиночный вектор на элемент батча, напр. центр входа/выхода).
    """
    result = []
    for t in tensors:
        if t.dim() == 3:
            result.append(torch.einsum('bij,bnj->bni', T, t))
        else:
            result.append(torch.einsum('bij,bj->bi', T, t))
    return result

def point_to_triangles_distance(
    points,               # (P, 3)
    triangles,            # (T, 3, 3)
    point_chunk_size = 200,
    triangle_chunk_size = 10000
):
    """
    Returns:
        sum_dist: (P,) — сумма расстояний от каждой точки до всех треугольников
    """
    P = points.shape[0]
    T = triangles.shape[0]

    p_chunk = point_chunk_size    or P
    t_chunk = triangle_chunk_size or T

    sum_dist = torch.zeros(P, dtype=points.dtype, device=points.device)

    for p_start in range(0, P, p_chunk):
        p_end = min(p_start + p_chunk, P)
        p = points[p_start:p_end].unsqueeze(1)   # (Cp, 1, 3)

        for t_start in range(0, T, t_chunk):
            t_end = min(t_start + t_chunk, T)
            tri   = triangles[t_start:t_end]      # (Ct, 3, 3)

            a  = tri[:, 0].unsqueeze(0)
            b  = tri[:, 1].unsqueeze(0)
            c  = tri[:, 2].unsqueeze(0)
            ab = b - a
            ac = c - a
            ap = p - a

            d1 = (ab * ap).sum(-1)
            d2 = (ac * ap).sum(-1)
            bp = p - b
            d3 = (ab * bp).sum(-1)
            d4 = (ac * bp).sum(-1)
            cp = p - c
            d5 = (ab * cp).sum(-1)
            d6 = (ac * cp).sum(-1)

            va = d3 * d6 - d5 * d4
            vb = d5 * d2 - d1 * d6
            vc = d1 * d4 - d3 * d2

            mask_a  = (d1 <= 0) & (d2 <= 0)
            mask_b  = (d3 >= 0) & (d4 <= d3)
            mask_c  = (d6 >= 0) & (d5 <= d6)
            mask_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
            mask_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
            mask_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)

            t_ab = (d1 / (d1 - d3).clamp(min=1e-10)).clamp(0.0, 1.0)
            t_ac = (d2 / (d2 - d6).clamp(min=1e-10)).clamp(0.0, 1.0)
            t_bc = ((d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp(min=1e-10)).clamp(0.0, 1.0)

            denom   = (va + vb + vc).clamp(min=1e-10)
            closest = a + (vb / denom).unsqueeze(-1) * ab + (vc / denom).unsqueeze(-1) * ac

            Cp = p_end - p_start
            Ct = t_end - t_start
            closest = torch.where(mask_bc.unsqueeze(-1), b + t_bc.unsqueeze(-1) * (c - b), closest)
            closest = torch.where(mask_ac.unsqueeze(-1), a + t_ac.unsqueeze(-1) * ac,       closest)
            closest = torch.where(mask_ab.unsqueeze(-1), a + t_ab.unsqueeze(-1) * ab,       closest)
            closest = torch.where(mask_c.unsqueeze(-1),  c.expand(Cp, Ct, 3),               closest)
            closest = torch.where(mask_b.unsqueeze(-1),  b.expand(Cp, Ct, 3),               closest)
            closest = torch.where(mask_a.unsqueeze(-1),  a.expand(Cp, Ct, 3),               closest)

            # Суммируем по треугольникам прямо здесь — (Cp,)
            sum_dist[p_start:p_end] += (1.0 / (p - closest).norm(dim=-1).pow(2).clamp(min=1e-10)).sum(dim=-1)

            del a, b, c, ab, ac, ap, bp, cp
            del d1, d2, d3, d4, d5, d6
            del va, vb, vc
            del mask_a, mask_b, mask_c, mask_ab, mask_ac, mask_bc
            del t_ab, t_ac, t_bc, denom, closest

    return sum_dist


def phi(x, segments, m=2.):
    tmp = point_to_triangles_distance(x, segments)
    x = None
    segments = None
    return 1 / (tmp ** (1 / m))


def calc_phi(x, segments):
    phi_seg = phi(x, segments).cpu()
    x = None
    segments = None
    return phi_seg


def get_point_from_segment(points, segment, n, x3=None):
    delta = segment[1] - segment[0]
    for i in range(int(n) + 1):
        if i < int(n) or torch.rand(1) <= (n - int(n)):
            if x3 is not None:
                points.append(torch.cat((segment[0] + delta * torch.rand(1),
                                         x3.reshape(1))))
            else:
                points.append(segment[0] + delta * torch.rand(1))


def sample_boundary_points(segments, m_all, x3=None):
    dist_all = 0
    for i in segments:
        tmp = torch.stack(i, axis=0)
        dist_all += torch.sum(torch.sum((tmp[:, 0] - tmp[:, 1]) ** 2, axis=1) ** 0.5)

    walls_points = []

    for i in range(len(segments)):
        tmp = torch.stack(segments[i], axis=0)
        dist = torch.sum((tmp[:, 0] - tmp[:, 1]) ** 2, axis=1) ** 0.5
        for j in range(len(segments[i])):
            m = dist[j] / (dist_all / m_all)
            get_point_from_segment(walls_points, segments[i][j], m, x3[i].reshape(1) if x3 is not None else x3)

    x = torch.stack(walls_points)
    return x


def is_inside(triangles, X, buffer=False):
    """Copyright 2018 Alexandre Devert

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software."""
    
    # Вычисление определителя 3x3 вдоль оси 1
    def adet(X, Y, Z):
        ret  = (X[:,0] * Y[:,1] * Z[:,2] + Y[:,0] * Z[:,1] * X[:,2] + Z[:,0] * X[:,1] * Y[:,2] - 
                Z[:,0] * Y[:,1] * X[:,2] - Y[:,0] * X[:,1] * Z[:,2] - X[:,0] * Z[:,1] * Y[:,2])
        return ret

    # Инициализация обобщенного порядка точки
    ret = torch.zeros(X.shape[0], dtype=X.dtype).to(X.device)
    
    # Накопление обобщенного порядок точки для каждого треугольника
    for U, V, W in triangles:
        A, B, C = U - X, V - X, W - X
        omega = adet(A, B, C)

        a, b, c = torch.norm(A, dim=1), torch.norm(B, dim=1), torch.norm(C, dim=1)
        k  = a * b * c + c * torch.sum(A * B, dim=1) + a * torch.sum(B * C, dim=1) + b * torch.sum(C * A,dim=1)
        
        ret += torch.arctan2(omega, k)

    return ret >= 2 * np.pi - (buffer if buffer else 0.)


def points_on_triangle(triangle, m):
    p = m % 1
    m = int(np.floor(m)) + (1 if np.random.random() < p else 0)
    x, y = torch.rand(m), torch.rand(m)
    q = abs(x - y)
    s, t, u = q, 0.5 * (x + y - q), 1 - 0.5 * (q + x + y)
    return torch.stack((s * triangle[0] + t * triangle[3] + u * triangle[6],
                        s * triangle[1] + t * triangle[4] + u * triangle[7],
                        s * triangle[2] + t * triangle[5] + u * triangle[8]), 1)


def sample_boundary_points_from_stl(path, centering, max_coord, m_all, return_norm=False):
    mesh_ = mesh.Mesh.from_file(path)

    points = torch.tensor(np.array(mesh_.points))

    points[:, :3] -= centering.cpu().numpy()
    points[:, 3:6] -= centering.cpu().numpy()
    points[:, 6:9] -= centering.cpu().numpy()

    points = points / max_coord.cpu().numpy() / 2

    areas = torch.tensor(np.array(mesh_.areas))

    areas_all = areas.sum()

    boundary_points = torch.zeros(0, 3)

    for i in range(len(points)):
        m = areas[i] / (areas_all / m_all)

        boundary_points = torch.concatenate((boundary_points,
                                             points_on_triangle(points[i], m)))

    x = boundary_points
    if return_norm:
        norm = torch.tensor(mesh_.normals[0])
        norm = norm / torch.norm(norm)
        return x, norm, areas_all / 1e6
    return x


def load_stl(path, n=50, n_interior=2000000, n_interior_phi=100000, n_outerior=100000, n_walls=100000, n_inlet=20000, n_outlet=20000, odd=False, length=[1., 1., 1.], device='cpu', inside_buffer=0.001, gen_p=True):
    x_dict = {}
    phi_w_dict = {}
    phi_in_dict = {}
    phi_out_dict = {}
    n_dict = {}
    
    closed_mesh = mesh.Mesh.from_file(path)
    
    centering = torch.zeros(3).to(device)

    closed_points = torch.tensor(np.array(closed_mesh.points)).to(device)

    centering[0] = closed_points[:, ::3].min() + (closed_points[:, ::3].max() - closed_points[:, ::3].min()) / 2
    centering[1] = closed_points[:, 1::3].min() + (closed_points[:, 1::3].max() - closed_points[:, 1::3].min()) / 2
    centering[2] = closed_points[:, 2::3].min() + (closed_points[:, 2::3].max() - closed_points[:, 2::3].min()) / 2

    closed_points[:, :3] -= centering
    closed_points[:, 3:6] -= centering
    closed_points[:, 6:9] -= centering

    max_coord = closed_points.__abs__().max()

    closed_points = closed_points / max_coord / 2

    print(f'Outerior points generation with path: {path}')

    if gen_p:
        x1 = torch.linspace(-length[0] / 2, length[0] / 2, n)
        x2 = torch.linspace(-length[1] / 2, length[1] / 2, n)
        x3 = torch.linspace(-length[2] / 2, length[2] / 2, n)

        x1, x2, x3 = torch.meshgrid(x1, x2, x3, indexing='ij')

        x = torch.stack([x1, x2, x3])
        x = x.reshape(3, -1).T.to(device)

        mask = is_inside(zip(closed_points[:, :3], 
                            closed_points[:, 3:6],
                            closed_points[:, 6:9]), x, inside_buffer)
        x_dict['outerior'] = x.cpu()[~mask.cpu()]
        x_dict['outerior'] = x_dict['outerior'][torch.randperm(len(x_dict['outerior']))[:n_outerior]]
        
        
        print('done\n\nInterior points generation')

        x = x[mask]
        x = x.repeat(int(n_interior * 1.3 / len(x)), 1)
        x = x + ((torch.rand(*x.shape).to(device) - 0.5) / n)

        mask_ = is_inside(zip(closed_points[:, :3], 
                            closed_points[:, 3:6],
                            closed_points[:, 6:9]), x, inside_buffer)
        
        x_dict['interior'] = x[mask_].cpu()
        x_dict['interior'] = x_dict['interior'][torch.randperm(len(x_dict['interior']))[:n_interior]]

        closed_mesh = None
        closed_points = None
        x1 = None 
        x2 = None
        x3 = None
        dx = None
        x = None
        mask_ = None
        torch.save(x_dict['interior'], path.replace('.stl', '_interior.pt'))
        torch.save(x_dict['outerior'], path.replace('.stl', '_outerior.pt'))
    else:
        print('done\n\nInterior points generation')
        x_dict['interior'] = torch.load(path.replace('.stl', '_interior.pt'))
        x_dict['outerior'] = torch.load(path.replace('.stl', '_outerior.pt'))

    tr_walls = mesh.Mesh.from_file(path.replace('.stl', '_3.stl'))
    
    tr_walls = torch.tensor(np.array(tr_walls.points)).to(device)

    tr_walls[:, :3] -= centering
    tr_walls[:, 3:6] -= centering
    tr_walls[:, 6:9] -= centering

    tr_walls = tr_walls / max_coord / 2
    tr_walls = tr_walls.reshape(-1, 3, 3)

    tr_in = mesh.Mesh.from_file(path.replace('.stl', '_1.stl' if odd else '_2.stl'))
    
    tr_in = torch.tensor(np.array(tr_in.points)).to(device)

    tr_in[:, :3] -= centering
    tr_in[:, 3:6] -= centering
    tr_in[:, 6:9] -= centering

    tr_in = tr_in / max_coord / 2
    tr_in = tr_in.reshape(-1, 3, 3)

    tr_out = mesh.Mesh.from_file(path.replace('.stl', '_1.stl' if odd else '_2.stl'))
    
    tr_out = torch.tensor(np.array(tr_out.points)).to(device)

    tr_out[:, :3] -= centering
    tr_out[:, 3:6] -= centering
    tr_out[:, 6:9] -= centering

    tr_out = tr_out / max_coord / 2
    tr_out = tr_out.reshape(-1, 3, 3)

    print('done\n\nInlet points generation')
    x_dict['inlet'], n_dict['inlet'], s_in = sample_boundary_points_from_stl(path.replace('.stl', '_1.stl' if odd else '_2.stl'), centering, max_coord, int(n_inlet * 1.1), return_norm=True)
    print('done\n\nOutlet points generation')
    x_dict['outlet'], n_dict['outlet'], s_out = sample_boundary_points_from_stl(path.replace('.stl', '_2.stl' if odd else '_1.stl'), centering, max_coord, int(n_outlet * 1.1), return_norm=True)
    n_dict['inlet_center'] = x_dict['inlet'].mean(0)
    n_dict['outlet_center'] = x_dict['outlet'].mean(0)
    
    print('done\n\nWalls points generation')
    x_dict['walls'] = sample_boundary_points_from_stl(path.replace('.stl', '_3.stl'), centering, max_coord, int(n_walls * 1.1))
    print('done\n\n')

    x_dict['walls'] = x_dict['walls'][torch.randperm(len(x_dict['walls']))[:n_walls]]
    x_dict['inlet'] = x_dict['inlet'][torch.randperm(len(x_dict['inlet']))[:n_inlet]]
    x_dict['outlet'] = x_dict['outlet'][torch.randperm(len(x_dict['outlet']))[:n_outlet]]

    phi_w_dict['interior'] = calc_phi(x_dict['interior'][:n_interior_phi].to(device), tr_walls)
    max_phi_w = phi_w_dict['interior'].max()
    phi_w_dict['interior'] = phi_w_dict['interior'] / max_phi_w
    phi_w_dict['outerior'] = - calc_phi(x_dict['outerior'].to(device), tr_walls) / max_phi_w
    phi_w_dict['inlet'] = calc_phi(x_dict['inlet'].to(device), tr_walls) / max_phi_w
    phi_w_dict['outlet'] = calc_phi(x_dict['outlet'].to(device), tr_walls) / max_phi_w
    phi_w_dict['walls'] = torch.zeros(len(x_dict['walls']))

    phi_in_dict['interior'] = calc_phi(x_dict['interior'][:n_interior_phi].to(device), torch.cat((tr_walls, tr_in), 0))
    max_phi_in = phi_in_dict['interior'].max()
    phi_in_dict['interior'] = phi_in_dict['interior'] / max_phi_in
    phi_in_dict['outerior'] = - calc_phi(x_dict['outerior'].to(device), torch.cat((tr_walls, tr_in), 0)) / max_phi_in
    phi_in_dict['inlet'] = torch.zeros(len(x_dict['inlet']))
    phi_in_dict['outlet'] = calc_phi(x_dict['outlet'].to(device), torch.cat((tr_walls, tr_in), 0))  / max_phi_in
    phi_in_dict['walls'] = torch.zeros(len(x_dict['walls']))

    phi_out_dict['interior'] = calc_phi(x_dict['interior'][:n_interior_phi].to(device), torch.cat((tr_walls, tr_out), 0))
    max_phi_in = phi_out_dict['interior'].max()
    phi_out_dict['interior'] = phi_out_dict['interior'] / max_phi_in
    phi_out_dict['outerior'] = - calc_phi(x_dict['outerior'].to(device), torch.cat((tr_walls, tr_out), 0)) / max_phi_in
    phi_out_dict['inlet'] = torch.zeros(len(x_dict['outlet']))
    phi_out_dict['outlet'] = calc_phi(x_dict['inlet'].to(device), torch.cat((tr_walls, tr_out), 0))  / max_phi_in
    phi_out_dict['walls'] = torch.zeros(len(x_dict['walls']))

    agg_dict = {'x_dict': x_dict, 'phi_w_dict': phi_w_dict, 'phi_in_dict': phi_in_dict, 'phi_out_dict': phi_out_dict, 'n_dict': n_dict, 'l': max_coord / 1000, 's_in': s_in, 's_out': s_out, 'v_mean_in': torch.norm(phi_w_dict['inlet'].mean() * n_dict['inlet']), 'v_mean_out': torch.norm(phi_w_dict['outlet'].mean() * n_dict['outlet'])}

    tr_walls = None
    tr_in = None

    torch.cuda.empty_cache()

    return agg_dict


class SaveBest():
    """Callback for save model if there is an improvement.
    
    Args:
        monitor (str): value for monitoring.
        model_path (str): Path for saving model.
        mode (str): One of {"min", "max"}. In min mode, training will stop when the quantity monitored has stopped decreasing.
            In "max" mode it will stop when the quantity monitored has stopped increasing.
    
    Attributes:
        history (dict): Dict of lists with train history. Key "monitor" contains list of monitoring values. 
        steps (int): Number of passed epoches. 
        best_step (int): Number of best epoch. 
        best_monitor (float): Best of monitoring value.
        model (Model): Training model
    """
    
    def __init__(self, monitor, model_path, mode):
        self.monitor = monitor
        self.model_path = model_path
        self.mode = mode
        self.history = None
        self.best_monitor = None
    
    def start(self, history, model):
        """Start and init callback. Save first version of model.
        
        Args:
            history (dict): Dict of lists with train history. Key "monitor" contains list of monitoring values. 
            model (Model): Training model
        """
        
        self.history = history
        self.model = model
        torch.save(self.model.state_dict(), self.model_path)
    
    def step(self):
        """Make a step of callback.
        
        Returns:
            tuple: (event, stop):
                event (str): Decription of event. If event not did not happen then event = ''.
                stop (bool): Flag of stopping train process.
        """
        
        if self.mode == 'max':
            if self.best_monitor is None or self.history[self.monitor][-1] > self.best_monitor:
                self.best_monitor = self.history[self.monitor][-1]
                torch.save(self.model.state_dict(), self.model_path)
        elif self.mode == 'min':
            if self.best_monitor is None or self.history[self.monitor][-1] < self.best_monitor:
                self.best_monitor = self.history[self.monitor][-1]
                torch.save(self.model.state_dict(), self.model_path)
    
    def stop(self):
        """Delete model from callback."""
        
        self.model = None
        torch.cuda.empty_cache()