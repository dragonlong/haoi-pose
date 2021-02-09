import sys
import os
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple, List

import dgl
from dgl.nn.pytorch import GraphConv, NNConv
from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from equivariant_attention.fibers import Fiber
#
from kaolin.models.PointNet2 import furthest_point_sampling
from kaolin.models.PointNet2 import fps_gather_by_index
from kaolin.models.PointNet2 import ball_query

def bp():
    import pdb;pdb.set_trace()
def square_distance(src, dst):
    '''
    Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    '''
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    '''
    Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    '''
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

class FixedRadiusNearNeighbors(nn.Module):
    '''
    Ball Query - Find the neighbors with-in a fixed radius
    '''
    def __init__(self, radius, n_neighbor):
        super(FixedRadiusNearNeighbors, self).__init__()
        self.radius = radius
        self.n_neighbor = n_neighbor

    def forward(self, pos, centroids):
        '''
        Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
        '''
        device = pos.device
        B, N, _ = pos.shape
        center_pos = index_points(pos, centroids)
        _, S, _ = center_pos.shape
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        sqrdists = square_distance(center_pos, pos)
        group_idx[sqrdists > self.radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :self.n_neighbor]
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, self.n_neighbor])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]
        return group_idx


class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers: int, input_feature_size: int,
                 num_channels: int, num_nlayers: int=1, num_degrees: int=4,
                 edge_dim: int=4, div: float=4, pooling: str='avg', n_heads: int=1, **kwargs):
        """
        num_layers: control stacked layers;
        num_channels: channel number for each type;
        input_feature_size: the input feat dim;
        num_degrees: control different types;
        edge_dim: embeddin space for edge features?

        """
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads

        self.fibers = {'in': Fiber(1, input_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, num_degrees*self.num_channels)}

        blocks = self._build_gcn(self.fibers, 1)
        self.Gblock, self.FCblock = blocks
        print(self.Gblock)
        print(self.FCblock)

    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim,
                                  div=self.div, n_heads=self.n_heads))
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        # Pooling
        if self.pooling == 'avg':
            Gblock.append(GAvgPooling())
        elif self.pooling == 'max':
            Gblock.append(GMaxPooling())

        # FC layers
        FCblock = []
        FCblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
        FCblock.append(nn.ReLU(inplace=True))
        FCblock.append(nn.Linear(self.fibers['out'].n_features, out_dim))

        return nn.ModuleList(Gblock), nn.ModuleList(FCblock)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)
        bp()
        for layer in self.FCblock:
            h = layer(h)

        return h

if __name__ == '__main__':
    # create model
    model = SE3Transformer(num_layers=2, input_feature_size=1, num_channels=32, edge_dim=0)
    #
    gt_points = np.random.rand(8,11,3).astype(np.float32) # * 10
    inputs    = torch.from_numpy(gt_points)
    frnn      = FixedRadiusNearNeighbors(0.2, 10)

    # data
    pos       = inputs
    centroids = np.tile(np.arange(gt_points.shape[1]).reshape(1, -1),(8, 1))
    centroids = torch.from_numpy(centroids)
    print(pos.shape, centroids.shape)
    feat      = np.ones((pos.shape[0], pos.shape[1], 1))
    dev       = pos.device
    group_idx = frnn(pos, centroids) # cpu function, process per batch, we could put in
    B, N, _ = pos.shape
    glist = []
    n_neighbor = 10
    for i in range(B):
        center = torch.zeros((N)).to(dev)
        center[centroids[i]] = 1 # find the chosen query
        src = group_idx[i].contiguous().view(-1) # real pair
        dst = centroids[i].view(-1, 1).repeat(1, n_neighbor).view(-1) # real pair

        unified = torch.cat([src, dst])
        uniq, inv_idx = torch.unique(unified, return_inverse=True)
        src_idx = inv_idx[:src.shape[0]]
        dst_idx = inv_idx[src.shape[0]:]
        print('src_idx.shape', '\n', src_idx[0:100], '\n', 'dst_idx.shape', '\n', dst_idx[0:100])
        g = dgl.DGLGraph((src_idx, dst_idx))
        g.ndata['x'] = pos[i][uniq]
        g.ndata['f'] = torch.from_numpy(feat[i][uniq].astype(np.float32)[:, :, np.newaxis])
        g.edata['d'] = pos[i][dst_idx] - pos[i][src_idx] #[num_atoms,3]
        glist.append(g)
    batched_graph = dgl.batch(glist)
    # batched_graph = dgl.batch_hetero(glist)
    model.forward(batched_graph)
