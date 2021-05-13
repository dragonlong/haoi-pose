import math
import random
import torch
import torch.nn as nn
import numpy as np
from math import ceil, cos, pi, sin, sqrt
from collections import defaultdict
from os import makedirs, remove
from os.path import exists, join
import matplotlib.pylab as plt
# import pointnet2_cuda as pointnet2

from typing import Tuple
from torch.autograd import Function, Variable
import joblib
from mpl_toolkits.mplot3d import Axes3D
def bp():
    import pdb;pdb.set_trace()
#
# class GetConcatPoints(object):
#     def __init__(self):
#         with open("./Data/sphere/sphere_8196_fps.pk", "rb") as f:
#             self.sphere_pc = joblib.load(f)
#
#     def get_concat_points(self, batch_size, point_num, shape, shape_num=1, overlap=True, device=None):
#         if shape == "sheet":
#             x = torch.rand(batch_size, point_num, 1) - 0.5
#             y = torch.rand(batch_size, point_num, 1) - 0.5
#             concat_points = torch.cat((x, y), 2)
#             if shape_num > 1:
#                 if overlap:
#                     z = torch.randint(0, shape_num, (batch_size, point_num, 1)).float()
#                     concat_points = torch.cat((concat_points, z), 2)
#                 else:
#                     z = torch.floor((x + 0.5) * shape_num)
#                     concat_points = torch.cat((concat_points, z), 2)
#         elif shape == "sphere":
#             if point_num < 8192:
#                 rand_perm = np.random.permutation(self.sphere_pc.shape[0])
#                 sphere_pcs = self.sphere_pc[rand_perm[0:batch_size]]
#                 concat_points = sphere_pcs[:, 0:point_num]
#                 x = torch.unsqueeze(torch.Tensor(concat_points[:, :, 0]), 2)
#                 y = torch.unsqueeze(torch.Tensor(concat_points[:, :, 1]), 2)
#                 concat_points = torch.Tensor(concat_points)
#             else:
#                 # if point_num == 10242:
#                 #     concat_points = torch.Tensor(self.sphere_mesh).unsqueeze(0).repeat((batch_size, 1, 1))
#                 # else:
#                 u = (torch.rand(batch_size, point_num, 1) - 0.5) * 2
#                 theta = torch.rand(batch_size, point_num, 1) * 2 * math.pi
#                 x = torch.sqrt(1 - u * u) * torch.cos(theta) / 2
#                 y = torch.sqrt(1 - u * u) * torch.sin(theta) / 2
#                 z = u / 2
#                 concat_points = torch.cat((x, y, z), 2)
#             if shape_num > 1:
#                 if overlap:
#                     t = torch.randint(0, shape_num, (batch_size, point_num, 1)).float()
#                     concat_points = torch.cat((concat_points, t), 2)
#                 else:
#                     if shape_num == 2:
#                         t = torch.floor((x + 0.5) * shape_num)
#                         concat_points = torch.cat((concat_points, t), 2)
#                     else:  # shape_num == 4
#                         t_1 = torch.floor((x + 0.5) * 2)
#                         t_2 = torch.floor((y + 0.5) * 2)
#                         t = t_1 * 2 + t_2
#                         concat_points = torch.cat((concat_points, t), 2)
#         else:
#             concat_points = None
#         if device is not None:
#             concat_points = concat_points.to(device)
#         return concat_points


def get_lattice_points(batch_size, point_num, group=1):
    n = int(np.sqrt(point_num))
    m = int(np.sqrt(group))

    x = torch.arange(n).unsqueeze(1).repeat((1, m)).view((-1, m ** 2)).repeat((1, n // m)).view((n * n, 1))
    y = torch.arange(n).view((-1, m)).repeat((1, m)).view((-1, m ** 2)).repeat((n // m, 1)).view((n * n, 1))
    patch = torch.cat((x, y), 1).unsqueeze(0).repeat((batch_size, 1, 1)).float() * (1 / n) + (0.5 / n)

    return patch


def get_lattice_grid(batch_size, point_num_x, point_num_y, delta_x=0, delta_y=0):
    a = (torch.arange(0.0, point_num_x)+delta_x)/point_num_x
    x = a.view(1, 1, point_num_x).repeat(1, point_num_y, 1)
    b = (torch.arange(0.0, point_num_y)+delta_y)/point_num_y
    y = b.view(1, point_num_y, 1).repeat(1, 1, point_num_x)

    grid = torch.cat((x, y), dim=0).unsqueeze(0)
    grid = grid.repeat(batch_size, 1, 1, 1)

    return grid

def sphere_mesh(n=512, radius=1.0, device="cpu"):
    """Return a unit sphere mesh discretization with *at least* n vertices.

    Initially this returns a standard uv sphere but in the future we could
    change it to return a more sophisticated discretization.

    Code adapted from github.com/caosdoar/spheres/blob/master/src/spheres.cpp .
    """
    # Select the subdivisions such that we get at least n vertices
    parallels = ceil(sqrt(n/2))+2
    meridians = 2*parallels
    if exists(f'./{n}_spheres.npy'):
        mesh_dict = np.load(f'./{n}_spheres.npy', allow_pickle=True).item()
        vertices = mesh_dict['vertices']
        faces    = mesh_dict['faces']
    else:
        # Create all the vertices
        vertices = [[0, 1., 0]]
        for i in range(1, parallels):
            polar = pi * i / parallels
            sp = sin(polar)
            cp = cos(polar)
            for j in range(meridians):
                azimuth = 2 * pi * j / meridians
                sa = sin(azimuth)
                ca = cos(azimuth)
                vertices.append([sp * ca, cp, sp * sa])
        vertices.append([0, -1., 0])

        # Create the triangles
        faces = []
        for j in range(meridians):
            faces.append([0, (j+1) % meridians + 1, j+1])
        for i in range(parallels-2):
            a_start = i*meridians + 1
            b_start = (i+1)*meridians + 1
            for j in range(meridians):
                a1 = a_start + j
                a2 = a_start + (j+1) % meridians
                b1 = b_start + j
                b2 = b_start + (j+1) % meridians
                faces.append([a1, a2, b1])
                faces.append([b1, a2, b2])
        for j in range(meridians):
            a = j + meridians * (parallels-2) + 1
            b = (j+1) % meridians + meridians * (parallels-2) + 1
            faces.append([len(vertices)-1, a, b])
        print('--saving shape into ', f'./{n}_spheres.npy')
        np.save(f'./{n}_spheres.npy', arr={'vertices': np.array(vertices), 'faces': np.array(faces)})
    vertices = torch.tensor(vertices, dtype=torch.float, device=device)
    faces = torch.tensor(faces, dtype=torch.long, device=device)

    return vertices * radius, faces

def get_spherical_lattice_grid(batch_size, point_num_x, point_num_y, delta_x=0, delta_y=0.5):
    azimuth = 2*np.pi*(torch.arange(0.0, point_num_x)+delta_x)/point_num_x
    azimuth = azimuth.view(1, 1, point_num_x).repeat(1, point_num_y, 1)

    inclination = np.pi*(torch.arange(0.0, point_num_y)+delta_y)/point_num_y
    inclination = inclination.view(1, point_num_y, 1).repeat(1, 1, point_num_x)

    x = torch.sin(inclination)*torch.cos(azimuth)
    y = torch.sin(inclination)*torch.sin(azimuth)
    z = torch.cos(inclination)

    grid = torch.cat((x, y, z), dim=0).unsqueeze(0)# (1, 3, point_num_y, point_num_x)
    grid = grid.repeat(batch_size, 1, 1, 1)

    return grid


def get_random_spherical_points(batch_size, point_num):
    pcs = torch.randn((batch_size, 3, point_num), dtype=torch.float32)
    pcs = pcs/(torch.norm(pcs, dim=1, keepdim=True) + 1e-7)
    return pcs


def get_lattice_grid_old(batch_size, point_num_x, point_num_y, delta_x=0, delta_y=0):
    a = torch.linspace(0.0, 1.0, point_num_x)
    x = a.repeat(point_num_y).view(1, point_num_y, point_num_x) + delta_x
    b = torch.linspace(0.0, 1.0, point_num_y)
    y = b.repeat(point_num_x).view(point_num_x, point_num_y)
    y = y.transpose(0, 1).view(1, point_num_y, point_num_x) + delta_y

    grid = torch.cat((x, y), dim=0).view(1, 2, point_num_y, point_num_x)
    grid = grid.repeat((batch_size, 1, 1, 1))

    return grid

def get_spherical_lattice_grid_old(batch_size, point_num_x, point_num_y, delta_x=0, delta_y=0):
    point_num_y = point_num_y + 2
    azimuth = 2*np.pi*(torch.linspace(0.0, 1.0, point_num_x) + delta_x/(point_num_x-1))
    azimuth = azimuth.repeat(point_num_y).view(1, point_num_y, point_num_x)
    inclination = np.pi*torch.linspace(0, 1.0, point_num_y)
    inclination = inclination.repeat(point_num_x).view(point_num_x, point_num_y)
    inclination = inclination.transpose(0, 1).view(1, point_num_y, point_num_x)

    x = torch.sin(inclination)*torch.cos(azimuth)
    y = torch.sin(inclination)*torch.sin(azimuth)
    z = torch.cos(inclination)

    grid = torch.cat((x, y, z), dim=0).view(1, 3, point_num_y, point_num_x)
    grid_left = grid[:, :, :, 0].view(1, 3, point_num_y, 1)
    grid_right = grid[:, :, :, -1].view(1, 3, point_num_y, 1)
    grid = torch.cat((grid_right, grid, grid_left), dim=3)
    grid = grid.repeat((batch_size, 1, 1, 1))

    return grid


def get_random_lattice_points(batch_size, point_num):
    x = torch.rand(batch_size, point_num, 1)
    y = torch.rand(batch_size, point_num, 1)
    patch = torch.cat((x, y), 2)
    return patch


def get_cube_points(batch_size, edge_num, x_min, y_min, z_min, x_max, y_max, z_max, device):
    b = batch_size
    n = edge_num
    x_min = x_min.view(b, 1, 1).repeat((1, n * n * n, 1))
    y_min = y_min.view(b, 1, 1).repeat((1, n * n * n, 1))
    z_min = z_min.view(b, 1, 1).repeat((1, n * n * n, 1))
    x_max = x_max.view(b, 1, 1).repeat((1, n * n * n, 1))
    y_max = y_max.view(b, 1, 1).repeat((1, n * n * n, 1))
    z_max = z_max.view(b, 1, 1).repeat((1, n * n * n, 1))
    x = torch.arange(n).unsqueeze(1).repeat((1, n * n)).view(-1, 1).unsqueeze(0).repeat((b, 1, 1)).to(device)
    x = (x.float() + 1.0) / (n + 1) * (x_max - x_min) + x_min
    y = torch.arange(n).unsqueeze(1).repeat((1, n)).repeat((n, 1)).view(-1, 1).unsqueeze(0).repeat((b, 1, 1)).to(device)
    y = (y.float() + 1.0) / (n + 1) * (y_max - y_min) + y_min
    z = torch.arange(n).unsqueeze(1).repeat((n * n, 1)).view(-1, 1).unsqueeze(0).repeat((b, 1, 1)).to(device)
    z = (z.float() + 1.0) / (n + 1) * (z_max - z_min) + z_min
    cube = torch.cat((x, y, z), 2)
    return cube


def get_mesh_points(path, point_num):
    with open(path, "rb") as f:
        pc = joblib.load(f)
    return torch.Tensor(pc[0:point_num, 0:2])


def plot_multi_3d_point_clouds(pcs, figsize, show=True, show_axis=True, in_u_sphere=False, marker='.',
                               s=8, alpha=.8, elev=10, azim=240, axis=None, title=None, *args,
                               **kwargs):
    num = pcs.shape[0]
    fig = plt.figure(figsize=figsize)
    for i in range(num):
        pc = pcs[i]
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        ax = plt.subplot(1, num, i + 1, projection='3d')

        if title is not None:
            plt.title(title)

        sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
        ax.view_init(elev=elev, azim=azim)

        if in_u_sphere:
            ax.set_xlim3d(-0.5, 0.5)
            ax.set_ylim3d(-0.5, 0.5)
            ax.set_zlim3d(-0.5, 0.5)
        else:
            miv = np.min([np.min(x), np.min(y), np.min(z)])  # Multiply with 0.7 to squeeze free-space.
            mav = np.max([np.max(x), np.max(y), np.max(z)])
            ax.set_xlim(miv, mav)
            ax.set_ylim(miv, mav)
            ax.set_zlim(miv, mav)
            plt.tight_layout()

        if not show_axis:
            plt.axis('off')

        if 'c' in kwargs:
            plt.colorbar(sc)

    if show:
        plt.show()

    return fig


def plot_batch_pcs(pc_batch, col, in_u_sphere=True, show=True):
    # print(pc_batch.shape)
    batch_size = pc_batch.shape[0]
    row = math.floor(batch_size / col)
    fig_list = []
    for i in range(row):
        pcs = pc_batch[i * col: (i + 1) * col]
        fig = plot_multi_3d_point_clouds(pcs, figsize=(col * 4, 4), in_u_sphere=in_u_sphere, show=show)
        fig_list.append(fig)
    if row * col < batch_size:
        pcs = pc_batch[row * col:]
        fig = plot_multi_3d_point_clouds(pcs, figsize=((batch_size - row * col) * 4, 4), in_u_sphere=in_u_sphere,
                                         show=show)
        fig_list.append(fig)
    return fig_list


def chamfer_distance_with_batch(p1, p2):
    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)
    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)
    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3)
    dist = torch.min(dist, dim=2)[0]
    dist = torch.sum(dist, dim=1, keepdim=False)

    return dist, None


def get_alpha_linear(epoch, params):
    if epoch <= params["blending_begin"] or epoch >= params["blending_end"]:
        return 0.0
    else:
        return float(epoch - params["blending_begin"]) / (params["blending_end"] - params["blending_begin"])


def get_alpha_twolinear(epoch, params):
    if epoch <= params["blending_begin"]:
        return 0.0
    elif epoch < params["blending_end"]:
        return float(epoch - params["blending_begin"]) / (params["blending_end"] - params["blending_begin"])
    elif epoch < 3000:
        return float(epoch - 2000) / 1000
    else:
        return 0.0


def get_alpha_exp(epoch, params):
    if epoch <= params["blending_begin"] or epoch >= params["blending_end"]:
        return 0.0
    else:
        return math.exp((epoch - params["blending_end"]) / 10)

def get_alpha_step(epoch, params):
    if epoch < params["blending_begin"] or epoch > params["blending_end"]:
        return 0.0
    else:
        step = [1700.0, 1800.0, 2000.0, 2200.0, 2500.0, 3000.0]
        # step = [1700.0, 1900.0, 2100.0, 2300.0, 2600.0, 3000.0]
        # step = [1700.0, 1900.0, 2200.0, 2500.0, 3000.0, 3500.0]
        alpha = [0.0, 0.03, 0.05, 0.1, 0.3, 1.0]
        for i in range(7):
            if step[i] > epoch:
                a = alpha[i - 1]
                b = alpha[i]
                c = step[i - 1]
                d = step[i]
                return (epoch - c) / (d - c) * b + (epoch - d) / (c - d) * a


def uniform_smp1(pts, nsmp, ndiavox = 30):
    npoint = pts.shape[0]
    assert(nsmp < npoint)
    diameter = np.linalg.norm(np.max(pts, 0) - np.min(pts, 0))
    voxres = diameter/ndiavox
    vidx = np.floor((pts - np.min(pts, 0))/voxres).astype(np.uint8)
    voxdict = defaultdict(list)
    npervoxpointrest = defaultdict(int)
    for i, idx in enumerate(vidx):
        voxdict[tuple(idx)].append(i)
        npervoxpointrest[tuple(idx)] += 1
    nvox = len(voxdict.keys())
    voxnsmp = defaultdict(int)
    nsmprest = nsmp
    nvoxrest = nvox
    keylist = voxdict.keys()
    keyflag = np.ones(len(keylist)).astype(np.bool)
    while nsmprest>0:
        npergrid = int(np.ceil(np.float(nsmprest)/nvoxrest))
        for i, k in enumerate(keylist):
            if npervoxpointrest[k] <= npergrid:
                keyflag[i] = False
                voxnsmp[k] += npervoxpointrest[k]
                nsmprest -= npervoxpointrest[k]
                npervoxpointrest[k] = 0
                nvoxrest -= 1
            else:
                voxnsmp[k] += npergrid
                nsmprest -= npergrid
                npervoxpointrest[k] -= npergrid

        keylist = [tuple(x) for x in np.array(list(keylist))[keyflag]]
        keyflag = np.ones(len(keylist)).astype(np.bool)
        nvoxres = len(keylist)
    pts_out = []
    prob = []
    for k in voxnsmp.keys():
        pts_out.append(pts[np.random.choice(voxdict[k], voxnsmp[k], replace=False)])
        prob.append(1/float(voxnsmp[k])*np.ones(voxnsmp[k]))
    pts_out = np.concatenate(pts_out, 0)
    prob = np.concatenate(prob, 0)
    prob = prob/np.sum(prob)
    final_idx = np.random.choice(pts_out.shape[0], nsmp, replace=False, p=prob)
    pts_out = pts_out[final_idx, :]
    return pts_out

def uniform_smp2(pts, nsmp, ndiavox = 30):
    npoint = pts.shape[0]
    assert(nsmp < npoint)
    diameter = np.linalg.norm(np.max(pts, 0) - np.min(pts, 0))
    voxres = diameter/ndiavox
    vidx = np.floor((pts - np.min(pts, 0))/voxres).astype(np.uint8)
    voxdict = defaultdict(list)
    npervoxpointrest = defaultdict(int)
    for i, idx in enumerate(vidx):
        voxdict[tuple(idx)].append(i)
        npervoxpointrest[tuple(idx)] += 1
    pts_out = []
    prob = []
    for k in voxdict.keys():
        pts_out.append(pts[voxdict[k], :])
        prob.append(1/float(npervoxpointrest[k])*np.ones(npervoxpointrest[k]))
    pts_out = np.concatenate(pts_out, 0)
    prob = np.concatenate(prob, 0)
    prob = prob/np.sum(prob)
    final_idx = np.random.choice(pts_out.shape[0], nsmp, replace=False, p=prob)
    pts_out = pts_out[final_idx, :]
    return pts_out

def set_random_seed(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id):
    """The function is designed for pytorch multi-process dataloader.
    Note that we use the pytorch random generator to generate a base_seed.
    Please try to be consistent.

    References:
        https://pytorch.org/docs/stable/notes/faq.html#dataloader-workers-random-seed

    """
    base_seed = torch.IntTensor(1).random_().item()
    # print(worker_id, base_seed)
    np.random.seed(base_seed + worker_id)
