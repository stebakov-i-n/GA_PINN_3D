import numpy as np
import torch
from stl import mesh
from torch import autograd

def calc_grad(v1, v2, v3, p, x):
    dv1 = autograd.grad(v1.sum(), x, create_graph=True)[0]
    dv2 = autograd.grad(v2.sum(), x, create_graph=True)[0]
    dv3 = autograd.grad(v3.sum(), x, create_graph=True)[0]
    dp = autograd.grad(p.sum(), x, create_graph=True)[0]
    d2v1 = autograd.grad(dv1.sum(), x, create_graph=True)[0]
    d2v2 = autograd.grad(dv2.sum(), x, create_graph=True)[0]
    d2v3 = autograd.grad(dv3.sum(), x, create_graph=True)[0]
    return dv1, dv2, dv3, d2v1, d2v2, d2v3, dp

def calc_res(v1, v2, v3, p, dv1, dv2, dv3, d2v1, d2v2, d2v3, dp):
    mu = 3e-3
    rho = 1050
    res1 = (v1 * dv1[..., 0:1] + v2 * dv1[..., 1:2] + v3 * dv1[..., 2:3]) - mu * (d2v1[..., 0:1] + d2v1[..., 1:2] + d2v1[..., 2:3]) / rho + dp[..., 0:1] / rho
    res2 = (v1 * dv2[..., 0:1] + v2 * dv2[..., 1:2] + v3 * dv2[..., 2:3]) - mu * (d2v2[..., 0:1] + d2v2[..., 1:2] + d2v2[..., 2:3]) / rho + dp[..., 1:2] / rho
    res3 = (v1 * dv3[..., 0:1] + v2 * dv3[..., 1:2] + v3 * dv3[..., 2:3]) - mu * (d2v3[..., 0:1] + d2v3[..., 1:2] + d2v3[..., 2:3]) / rho + dp[..., 2:3] / rho
    res4 = dv1[..., 0:1] + dv2[..., 1:2] + dv3[..., 2:3]
    return [res1, res2, res3, res4]

def mse_zero_loss(f):
    return (f ** 2).mean()

def zero_loss(outputs, n=4):
    loss = 0
    for i in range(n):
        loss += mse_zero_loss(outputs[i])
    loss = loss / n
    return loss

def dist(a, b):
    return ((a - b) ** 2).sum(axis=1) ** 0.5


def lin_seg(x, x_c):
    return dist(x, x_c)

def phi(x, segments, m=3.):
    tmp = 1 / (lin_seg(x, segments[0]) ** m)
    for i in range(1, len(segments)):
        phi_ = lin_seg(x, segments[i])
        tmp = tmp + 1 / (phi_ ** m)
    x = None
    segments = None
    return 1 / (tmp ** (1 / m))

def lin_seg_(x, x_seg):
    # d = ((x_seg[0] - x_seg[1]) ** 2).sum() ** 0.5
    v1 = x_seg[1] - x_seg[0]
    v2 = x_seg[2] - x_seg[0]
    
    normal = np.cross(v1, v2)
    d = np.linalg.norm(normal) / 2
    x_c = x_seg.mean(axis=0)
    f = torch.tensor(np.dot(x - x_seg[0], normal) / d)
    t = (1 / d) * ((d / 2.) ** 2 - dist(x, x_c) ** 2)
    varphi = (t ** 2 + f ** 4) ** 0.5
    tmp = (f ** 2 + (1 / 4.) * (varphi - t) ** 2) ** 0.5
    return tmp

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
    # [~np.isclose(mesh_.normals[:, 0], 0., rtol=1e-07, atol=1e-10) & (np.isclose(mesh_.normals[:, 1], 0., rtol=1e-07, atol=1e-10))])

    points[:, :3] -= centering.cpu().numpy()
    points[:, 3:6] -= centering.cpu().numpy()
    points[:, 6:9] -= centering.cpu().numpy()

    points = points / max_coord.cpu().numpy() / 2

    areas = torch.tensor(np.array(mesh_.areas))
    # [~np.isclose(mesh_.normals[:, 0], 0., rtol=1e-07, atol=1e-10) & (np.isclose(mesh_.normals[:, 1], 0., rtol=1e-07, atol=1e-10))])

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


def load_stl(path, n=64, n_interior=5000000, n_walls=10000, n_inlet=10000, n_outlet=10000, odd=False, length=[1., 1., 1.], device='cpu', use_3d=True, inside_buffer=0.001, gen_int_p=True):
    x_dict = {}
    phi_w_dict = {}
    phi_in_dict = {}
    n_dict = {}
    
    print(f'Mask generation with path: {path}')
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

    x1 = torch.linspace(-length[0] / 2, length[0] / 2, n)
    x2 = torch.linspace(-length[1] / 2, length[1] / 2, n) if use_3d else torch.tensor(0.001 * length[1])
    x3 = torch.linspace(-length[2] / 2, length[2] / 2, n)

    x1, x2, x3 = torch.meshgrid(x1, x2, x3, indexing='ij')

    dx = torch.tensor([closed_points[:, ::3].max() - closed_points[:, ::3].min(),
                       closed_points[:, 1::3].max() - closed_points[:, 1::3].min(),
                       closed_points[:, 2::3].max() - closed_points[:, 2::3].min()]).to(device)
    
    x = torch.stack([x1, x2, x3])
    x = x.reshape(3, -1).T.to(device)

    mask = is_inside(zip(closed_points[:, :3], 
                         closed_points[:, 3:6],
                         closed_points[:, 6:9]), x, inside_buffer)
    x_dict['outerior'] = x.cpu()[~mask.cpu()]
    mask = mask.reshape(n, n, n).cpu() if use_3d else mask.reshape(n, n).cpu()
    mask = {'num': mask.float(), 'bool': mask}
    
    print('done\n\nInterior points generation')
    if gen_int_p:
        x = (torch.rand(int(0.2 * n_interior), 3) * dx.cpu() * 1.1 - (dx.cpu() * 1.1 / 2)).to(device)
        mask_ = is_inside(zip(closed_points[:, :3], 
                            closed_points[:, 3:6],
                            closed_points[:, 6:9]), x, inside_buffer)
        
        x = x[mask_]
        x = x.repeat(int(n_interior * 1.3 / len(x)), 1)
        x = x + ((torch.rand(*x.shape).to(device) - 0.5) * dx * 0.05)

        mask_ = is_inside(zip(closed_points[:, :3], 
                            closed_points[:, 3:6],
                            closed_points[:, 6:9]), x, inside_buffer)
        
        x_dict['interior'] = x[mask_].cpu()
        x_dict['interior'] = x_dict['interior'][torch.randperm(len(x_dict['interior']))[:n_interior]]

        closed_mesh = None
        closed_points = closed_points.cpu()
        x1 = None 
        x2 = None
        x3 = None
        dx = None
        x = None
        mask_ = None
        torch.save(x_dict['interior'], path.replace('.stl', '.pt'))
    else:
        x_dict['interior'] = torch.load(path.replace('.stl', '.pt'))

    print('done\n\nInlet points generation')
    x_dict['inlet'], n_dict['inlet'], s = sample_boundary_points_from_stl(path.replace('.stl', '_1.stl'), centering, max_coord, int(n_inlet * 1.1), return_norm=True)
    print('done\n\nOutlet points generation')
    x_dict['outlet'], n_dict['outlet'], _ = sample_boundary_points_from_stl(path.replace('.stl', '_2.stl'), centering, max_coord, int(n_outlet * 1.1), return_norm=True)
    n_dict['outlet_center'] = x_dict['outlet'].mean(0)
    if odd:
        x_dict['inlet'], x_dict['outlet'] = x_dict['outlet'], x_dict['inlet']
    print('done\n\nWalls points generation')
    x_dict['walls'] = sample_boundary_points_from_stl(path.replace('.stl', '_3.stl'), centering, max_coord, int(n_walls * 1.1))
    print('done\n\n')
    if not use_3d:
        x_dict['interior'] = x_dict['interior'][:, ::2]
        x_dict['walls'] = x_dict['walls'][:, ::2]
        x_dict['inlet'] = x_dict['inlet'][:, ::2]
        x_dict['outlet'] = x_dict['outlet'][:, ::2]
        x_q_fix = x_q_fix[:, ::2]

    x_dict['walls'] = x_dict['walls'][torch.randperm(len(x_dict['walls']))[:n_walls]]
    x_dict['inlet'] = x_dict['inlet'][torch.randperm(len(x_dict['inlet']))[:n_inlet]]
    x_dict['outlet'] = x_dict['outlet'][torch.randperm(len(x_dict['outlet']))[:n_outlet]]

    phi_w_dict['interior'] = calc_phi(x_dict['interior'].to(device), x_dict['walls'].to(device))
    max_phi_w = phi_w_dict['interior'].max()
    phi_w_dict['interior'] = phi_w_dict['interior'] / max_phi_w
    phi_w_dict['outerior'] = - calc_phi(x_dict['outerior'].to(device), x_dict['walls'].to(device)) / max_phi_w
    phi_w_dict['inlet'] = calc_phi(x_dict['inlet'].to(device), x_dict['walls'].to(device)) / max_phi_w
    phi_w_dict['outlet'] = calc_phi(x_dict['outlet'].to(device), x_dict['walls'].to(device)) / max_phi_w
    phi_w_dict['walls'] = torch.zeros(len(x_dict['walls']))

    phi_in_dict['interior'] = calc_phi(x_dict['interior'].to(device), torch.cat((x_dict['walls'], x_dict['inlet'])).to(device))
    max_phi_in = phi_in_dict['interior'].max()
    phi_in_dict['interior'] = phi_in_dict['interior'] / max_phi_in
    phi_in_dict['outerior'] = - calc_phi(x_dict['outerior'].to(device), torch.cat((x_dict['walls'], x_dict['inlet'])).to(device)) / max_phi_in
    phi_in_dict['inlet'] = torch.zeros(len(x_dict['inlet']))
    phi_in_dict['outlet'] = calc_phi(x_dict['outlet'].to(device), torch.cat((x_dict['walls'], x_dict['inlet'])).to(device))  / max_phi_in
    phi_in_dict['walls'] = torch.zeros(len(x_dict['walls']))

    # closed_points_tmp = []

    # for point in closed_points:
    #     if point[2].cpu() < 0.1 and point[2].cpu() > -0.01:
    #         closed_points_tmp.append(torch.stack([point[:3].cpu(), point[3:6].cpu(), point[6:9].cpu()]))
    # # closed_points.shape

    agg_dict = {'mask': mask, 'x_dict': x_dict, 'phi_w_dict': phi_w_dict, 'phi_in_dict': phi_in_dict, 'n_dict': n_dict, 'l': max_coord / 1000, 's': s, 'v_mean': torch.norm(phi_w_dict['inlet'].mean() * n_dict['inlet'])}

    return agg_dict
