import sys
import os
import numpy as np
import random
import torch
from math import pi ,sin, cos, sqrt
from copy import copy
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple, List
from copy import deepcopy
from omegaconf import DictConfig, ListConfig
torch.manual_seed(0)

import dgl
from dgl.nn.pytorch import GraphConv, NNConv
from equivariant_attention.from_se3cnn.SO3 import rot
from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from equivariant_attention.fibers import Fiber
#
from kaolin.models.PointNet2 import furthest_point_sampling
from kaolin.models.PointNet2 import fps_gather_by_index, group_gather_by_index
from kaolin.models.PointNet2 import ball_query
from kaolin.models.PointNet2 import three_nn

from common.d3_utils import compute_rotation_matrix_from_ortho6d
eps = 1e-10
def bp():
    import pdb;pdb.set_trace()
def is_list(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig)

def is_iterable(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig) or isinstance(entity, tuple)

def is_dict(entity):
    return isinstance(entity, dict) or isinstance(entity, DictConfig)

SPECIAL_NAMES = ["radius", "max_num_neighbors", "block_names", "num_degrees"]
def pdist2squared(x, y):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = (y**2).sum(dim=1).unsqueeze(1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), y)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist


# class for fps sampling of points
class Sample(nn.Module):
    def __init__(self, num_points):
        super(Sample, self).__init__()
        self.num_points = num_points

    def forward(self, points):
        """
        points: [B, N, 3]
        return: [B, N1, 3]
        """
        xyz1_ind = furthest_point_sampling(points, self.num_points) # --> [B, N]
        # print(self.num_points, ':', xyz1_ind) # sampling
        xyz1     = fps_gather_by_index(points.permute(0, 2, 1).contiguous(), xyz1_ind)                # batch_size, channel2, nsample
        return xyz1_ind, xyz1.permute(0, 2, 1).contiguous()

# class for neighborhoods sampling of points
class SampleNeighbors(nn.Module):
    def __init__(self, radius, num_samples, knn=False):
        super(SampleNeighbors, self).__init__()

        self.radius = radius
        self.num_samples = num_samples
        self.knn    = knn

    def forward(self, xyz2, xyz1):
        """
        [BS, N, 3],
        find nearest points in xyz2 for every points in xyz1
        return [B, N, K]
        """
        if self.knn:
            dist= pdist2squared(xyz2.permute(0, 2, 1).contiguous(), xyz1.permute(0, 2, 1).contiguous())
            ind = dist.topk(self.num_samples+1, dim=1, largest=False)[1].int().permute(0, 2, 1).contiguous()[:, :, 1:]
            # print('knn neighbors, ', self.num_samples, ':')
            # print(ind[0])
        else:
            # TODO: need to remove self from neighborhood index
            ind = ball_query(self.radius, self.num_samples+1, xyz2, xyz1, False)

        return ind

class SampleK(nn.Module):
    def __init__(self, radius, num_samples, knn=False):
        super(SampleK, self).__init__()

        self.radius = radius
        self.num_samples = num_samples
        self.knn    = knn

    def forward(self, xyz2, xyz1): # [BS, 3, N]
        # find nearest points in xyz2 for every points in xyz1
        if self.knn:
            dist= pdist2squared(xyz2, xyz1)
            ind = dist.topk(self.num_samples, dim=1, largest=False)[1].int().permute(0, 2, 1).contiguous()
        else:
            ind = ball_query(self.radius, self.num_samples, xyz2.permute(0, 2, 1).contiguous(),
                             xyz1.permute(0, 2, 1).contiguous(), False)

        return ind

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


class BuildGraph(nn.Module):
    """
    GPU-based graph builder, given input points, output/update a graph
    """
    def __init__(self, num_samples=20, r=0.1, npoint=256, downsample=False):
        super().__init__()
        self.downsample=downsample
        if downsample:
            self.npoint      = npoint
            self.n_sampler   = Sample(npoint)
        self.num_samples = num_samples
        self.e_sampler   = SampleNeighbors(r, self.num_samples, knn=True)

    def forward(self, xyz=None, G=None, h=None, BS=2):
        """
        xyz: B, N, 3
        G and h are not necessary here(only add them when we want to downsample a graph)
        BS: batch size
        """
        glist = []
        if xyz is None:
            xyz = G.ndata['x'].view(BS, -1, 3).contiguous()
        if self.downsample:
            xyz_ind, pos = self.n_sampler(xyz)
            if h is not None:
                for key in h.keys():
                    h[key] = torch.gather(h[key].view(BS, -1, h[key].shape[-2], h[key].shape[-1]).contiguous(), 1, xyz_ind.long().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h[key].shape[-2], h[key].shape[-1]).contiguous())
                    h[key] = h[key].view(-1, h[key].shape[-2], h[key].shape[-1]).contiguous()
        else:
            pos = xyz

        B, N, _ = pos.shape
        neighbors_ind = self.e_sampler(pos, pos)
        glist = []
        for i in range(B):
            src = neighbors_ind[i].contiguous().view(-1).cpu().long()
            dst = torch.arange(pos[i].shape[0]).view(-1, 1).repeat(1, self.num_samples).view(-1)
            g = dgl.DGLGraph((src.cpu(), dst.cpu()))
            g.ndata['x'] = pos[i]
            g.ndata['f'] = torch.ones(pos[i].shape[0], 1, 1, device=pos.device).float()
            g.edata['d'] = pos[i][dst.long()] - pos[i][src.long()] #[num_atoms,3] but we only supervise the half
            #
            glist.append(g)
        G = dgl.batch(glist)
        return G, h

def build_model():
    num_degrees = 2
    num_features = 1  # todo
    fiber_in = Fiber(1, num_features)
    fiber_mid = Fiber(num_degrees, 16)
    fiber_out = Fiber(num_degrees, 128)

    # We build a module from:
    # 1) a multihead attention block
    # 2) a nonlinearity
    # 3) a TFN layer (no attention)
    # 4) graph max pooling
    # 5) a fully connected layer -> 1 output

    Gblock = []
    fibers = {'in': Fiber(1, num_features),
                   'mid': Fiber(num_degrees, 32),
                   'out': Fiber(num_degrees, 1)}
    fin = fibers['in']
    for i in range(4):
        Gblock.append(GSE3Res(fin, fibers['mid'], n_heads=1, edge_dim=0, div=4)) # edge_dim=4,
        Gblock.append(GNormSE3(fibers['mid']))
        fin = fibers['mid']
    Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True)) # , edge_dim=4
    # Gblock.append(GAvgVecPooling())
    model = nn.ModuleList(Gblock)
    fc_layer = nn.Linear(128, 1)
    return model


def collate(samples):
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(y)

def summary(features):
    if isinstance(features, dict):
        for k, v in features.items():
            print(f'type: {k}; Size: {v.size()}')
    else:
        print(f'Size: {features.size()}')
        print(features[0])
    print('')

def set_feat(G, R, num_features=1):
    G.edata['d'] = G.edata['d'] @ R
    if 'w' in G.edata:
        G.edata['w'] = torch.rand((G.edata['w'].size(0), 0))

    G.ndata['x'] = G.ndata['x'] @ R
    G.ndata['f'] = torch.ones((G.ndata['f'].size(0), num_features, 1))
    # # print(G)

    features = {'0': G.ndata['f']}
    return G, features

def apply_model(model, G, features, num_degrees=2):
    basis, r = get_basis_and_r(G, num_degrees - 1)
    for i, layer in enumerate(model):
        features = layer(features, G=G, r=r, basis=basis)

    return features['0'], features['1']

class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|

    """
    def __init__(self, cfg, latent_dim: int=128, div: float=4, pooling: str='avg', n_heads: int=1, vector_attention=False, **kwargs):
        super().__init__()
        # Build the network
        self.num_nlayers  = cfg.MODEL.num_nlayers
        self.num_in_channels  = cfg.MODEL.num_in_channels
        self.num_mid_channels = cfg.MODEL.num_mid_channels
        self.num_out_channels = cfg.MODEL.num_out_channels
        self.num_channels_R   = cfg.MODEL.num_channels_R
        self.num_degrees     = cfg.MODEL.num_degrees
        self.edge_dim        = cfg.MODEL.edge_dim
        self.encoder_only    = cfg.MODEL.encoder_only
        self.div        = div
        self.pooling    = pooling
        self.n_heads    = n_heads
        self.vector_attention = vector_attention
        self.latent_dim = latent_dim
        self.batch_size = 2 # TODO

        self.fibers = {'in': Fiber(1, self.num_in_channels),
                       'mid': Fiber(self.num_degrees, self.num_mid_channels),         # should match with first downsample layer input
                       'out': Fiber(self.num_degrees, self.num_out_channels),         # should matche last upsampling layer ouput
                       'out_type0': Fiber(1, self.latent_dim),                        # latent_dim matches with Decoder
                       'out_type1_R': Fiber(self.num_degrees, self.num_channels_R),
                       'out_type1_T': Fiber(self.num_degrees, 1)}                     # additional type 1 for center voting;

        self._build_gcn(cfg.MODEL)

    def _build_gcn(self, opt, verbose=False):
        fibers = self.fibers
        self.pre_modules    = nn.ModuleList()
        self.down_modules   = nn.ModuleList()
        self.pre_modules.append( GSE3Res(fibers['in'], fibers['mid'], edge_dim=self.edge_dim,
                                    div=self.div, n_heads=self.n_heads) ) # , vector_attention=self.vector_attention
        self.pre_modules.append( GNormSE3(fibers['mid']) )

        # Down modules
        for i in range(len(opt.down_conv.down_conv_nn)):
            args = self._fetch_arguments(opt.down_conv, i, "DOWN")
            down_module = SE3TBlock(**args)
            self.down_modules.append(down_module)

        # if Up modules
        if not self.encoder_only:
            self.up_modules     = nn.ModuleList()
            for i in range(len(opt.up_conv.up_conv_nn)):
                args = self._fetch_arguments(opt.up_conv, i, "UP")
                if opt.up_conv.module_type == 'GraphFPModule':
                    up_module = GraphFPModule(**args)
                elif opt.up_conv.module_type == 'GraphFPResNoSkipLinkModule':
                    up_module = GraphFPResNoSkipLinkModule(**args)
                else:
                    up_module = GraphFPSumModule(**args)
                self.up_modules.append(up_module)
        else:
            self.up_modules = None

        Oblock = [GConvSE3(fibers['out'], fibers['out_type0'], self_interaction=True, edge_dim=self.edge_dim),
                  GConvSE3(fibers['out'], fibers['out_type1_R'], self_interaction=True, edge_dim=self.edge_dim),
                  GConvSE3(fibers['out'], fibers['out_type1_T'], self_interaction=True, edge_dim=self.edge_dim)]

        self.Oblock = nn.ModuleList(Oblock)

        # Pooling
        if self.pooling == 'avg':
            self.Pblock = GAvgPooling(type='1')
        elif self.pooling == 'max':
            self.Pblock = GMaxPooling()

        return
    # len(tr_agent.net.encoder.Gblock)
    def forward(self, G, verbose=False):
        """
        input graph
        """
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)
        h0 = {'0': G.ndata['f']}
        G0 = G

        for i in range(len(self.pre_modules)):
            h0 = self.pre_modules[i](h0, G=G, r=r, basis=basis)
        pred_dict = {'h0': {'0': h0['0'].detach().clone(), '1': h0['1'].detach().clone()}}
        # encoding
        h1, G1, r1, basis1 = self.down_modules[0](h0, Gin=G0, BS=self.batch_size) # 512-256
        pred_dict['h1'] =  {'0': h1['0'].detach().clone(), '1': h1['1'].detach().clone()}

        h2, G2, r2, basis2 = self.down_modules[1](h1, Gin=G1, BS=self.batch_size) # 256-128
        pred_dict['h2'] =  {'0': h2['0'].detach().clone(), '1': h2['1'].detach().clone()}

        h3, G3, r3, basis3 = self.down_modules[2](h2, Gin=G2, BS=self.batch_size) # 128-64
        pred_dict['h3'] =  {'0': h3['0'].detach().clone(), '1': h3['1'].detach().clone()}

        h4, G4, r4, basis4 = self.down_modules[3](h3, Gin=G3, BS=self.batch_size) # 64-32
        pred_dict['h4'] =  {'0': h4['0'].detach().clone(), '1': h4['1'].detach().clone()}
        # h, G, r, basis = h4, G4, r4, basis4
        # return h['0'], h['1']

        # decoding
        if not self.encoder_only:
            h3 = self.up_modules[0](h3, G=G3, r=r3, basis=basis3, uph=h4, upG=G4, BS=self.batch_size) # 64
            h2 = self.up_modules[1](h2, G=G2, r=r2, basis=basis2, uph=h3, upG=G3, BS=self.batch_size) # 128
            h1 = self.up_modules[2](h1, G=G1, r=r1, basis=basis1, uph=h2, upG=G2, BS=self.batch_size) # 256
            h  = self.up_modules[3](h0, G=G0, r=r, basis=basis, uph=h1, upG=G1, BS=self.batch_size)    # 512
            if verbose:
                i = 1 # choose your interested layer
                print(f'--up{i} GraphFP: ', self.up_modules[i].Tblock[0].f_in.structure, self.up_modules[i].Tblock[0].f_out.structure)
                print('--input ', h0['0'].shape, 'to upsample', h1['0'].shape)
        else:
            h = h4
            G, r, basis = G4, r4, basis4
        pred_dict.update({'h0_u': h, 'h1_u': h1, 'h2_u': h2, 'h3_u': h3})

        pred_S   = self.Oblock[0](h, G=G, r=r, basis=basis) # only one mode
        pred_R   = self.Oblock[1](h, G=G, r=r, basis=basis) #
        pred_T   = self.Oblock[2](h, G=G, r=r, basis=basis) # 1. dense type 1 feature for T

        output_R = pred_R['1']/(torch.norm(pred_R['1'], dim=-1, keepdim=True))
        output_R = compute_rotation_matrix_from_ortho6d(output_R.view(-1, 6)).permute(0, 2, 1).contiguous()
        pred_dict.update({'R': output_R, 'R0': pred_R['0'], 'T': pred_T['1'], 'N': pred_S['0']})

        out      = {'0': pred_S['0'], '1': pred_R['1']}
        out      = self.Pblock(out, G=G, r=r, basis=basis) # pooling
        pred_dict['0'] = out['0']                          # for shape embedding
        pred_dict['1'] = out['1']                          # for rotation average
        pred_dict['G'] = G
        if verbose:
            print(pred_dict['N'].shape, pred_dict['R'].shape)

        return pred_dict

    def _fetch_arguments(self, conv_opt, index, flow):
        """ Fetches arguments for building a convolution (up or down)

        Arguments:
            conv_opt
            index in sequential order (as they come in the config)
            flow "UP" or "DOWN"
        """
        args = self._fetch_arguments_from_list(conv_opt, index)
        args["index"] = index
        return args

    def _fetch_arguments_from_list(self, opt, index):
        """Fetch the arguments for a single convolution from multiple lists
        of arguments - for models specified in the compact format.
        """
        args = {}
        for o, v in opt.items():
            name = str(o)
            if is_list(v) and len(getattr(opt, o)) > 0:
                if name[-1] == "s" and name not in SPECIAL_NAMES:
                    name = name[:-1]
                v_index = v[index]
                if is_list(v_index):
                    v_index = list(v_index)
                args[name] = v_index
            else:
                if is_list(v):
                    v = list(v)
                args[name] = v
        return args

class InterDownGraph(nn.Module): #
    """
    func: given input graph G with N points, downsample to N1 points, output two graphs;
          Gmid:  neighborhoods come from N pts for all N1 pts;
          Gout:  neighborhoods come from N1 pts for all N1 pts;
          xyz_ind: original index of N1 points in N points;
    """
    def __init__(self, npoint=256, num_samples=[20], r=0.1, knn=True):
        super().__init__()
        self.num_samples = num_samples[0]
        self.npoint      = npoint
        self.n_sampler   = Sample(npoint)
        self.e_sampler   = SampleNeighbors(r, self.num_samples, knn=knn)

    def forward(self, G, BS=2):
        """
        G: input Graph
        BS: batch size
        """
        glist = []
        pos = G.ndata['x'].view(BS, -1, 3).contiguous()
        xyz_ind, xyz_query = self.n_sampler(pos)
        neighbors_ind      = self.e_sampler(pos, xyz_query)
        glist              = []
        for i in range(BS):
            src = neighbors_ind[i].contiguous().view(-1)
            dst = xyz_ind[i].view(-1, 1).repeat(1, self.num_samples).view(-1)
            g = dgl.DGLGraph((src.cpu().long(), dst.cpu().long()))
            # try:
            g.ndata['x'] = pos[i] # dgl._ffi.base.DGLError: Expect number of features to match number of nodes (len(u)). Got 256 and 249 instead.
            g.ndata['f'] = torch.ones(pos[i].shape[0], 1, 1, device=pos.device).float()
            g.edata['d'] = pos[i][dst.long()] - pos[i][src.long()] #[num_atoms,3] but we only supervise the half
            # except:
            #     print('---something wrong!!!')
            #     g = dgl.unbatch(G)[i]
            #     g.remove_edges( np.arange( len(g.all_edges()[0]) ).tolist()) # this line comes with bug
            #     g.add_edges(src.cpu().long(), dst.cpu().long())
            #     g.edata['d'] = pos[i][dst.long()] - pos[i][src.long()]
            glist.append(g)

        Gmid = dgl.batch(glist)

        # updated graph
        glist = []
        pos   = xyz_query
        neighbors_ind = self.e_sampler(pos, pos)
        for i in range(BS):
            src = neighbors_ind[i].contiguous().view(-1).cpu()
            dst = torch.arange(pos[i].shape[0]).view(-1, 1).repeat(1, self.num_samples).view(-1)
            g = dgl.DGLGraph((src.cpu(), dst.cpu()))
            g.ndata['x'] = pos[i]
            g.ndata['f'] = torch.ones(pos[i].shape[0], 1, 1, device=pos.device).float()
            g.edata['d'] = pos[i][dst.long()] - pos[i][src.long()] #[num_atoms,3] but we only supervise the half
            glist.append(g)
        Gout = dgl.batch(glist)

        return Gmid, Gout, xyz_ind


class SE3TBlock(nn.Module):
    def __init__(self, npoint, nsample, radius, down_conv_nn, num_degrees=2, edge_dim=0, div=4, n_heads=1, knn=True,
                 use_xyz=True, module_type='mid_layer', index=0, vector_attention=False):
        super(SE3TBlock, self).__init__()
        self.nsample = nsample
        self.npoint  = npoint
        self.module_type = module_type
        self.use_xyz = use_xyz
        self.edge_dim= edge_dim
        self.div=div
        self.n_heads=n_heads
        self.num_degrees = num_degrees
        in_channels=down_conv_nn[0][0]
        out_channels=down_conv_nn[0][1:]
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.down_g = InterDownGraph(npoint, nsample, knn=knn)

        # 2. add SE3-layer over intermediate graph, and abstract Graph
        Tblock = []
        fibers  = [Fiber(num_degrees, in_channels)]
        #
        for i in range(len(out_channels)):
            fibers.append( Fiber(num_degrees, out_channels[i]) )
            Tblock.append(GSE3Res(fibers[-2], fibers[-1], edge_dim=self.edge_dim,
                                  div=self.div, n_heads=self.n_heads)) #, vector_attention=vector_attention
            Tblock.append(GNormSE3(fibers[-1]))

        self.stage1 = nn.ModuleList(Tblock[:2]) # for inter
        self.stage2 = nn.ModuleList(Tblock[2:])

    def forward(self, hin, Gin, BS=2):
        """
        hin: input feature, with type 0: [BS*N, C, 1], type 0: [BS*N, C, 1]
        Gin: input graph
        """
        # Compute equivariant weight basis from relative positions
        Gmid, Gout, xyz_ind = self.down_g(Gin, BS=BS)
        basis, r = get_basis_and_r(Gmid, self.num_degrees-1)

        #  intermediate graph
        h = {}
        for key in hin.keys():
            h[key] = hin[key].detach()

        for layer in self.stage1:
            h = layer(h, G=Gmid, r=r, basis=basis)

        # update h
        for key in h.keys():
            h[key] = torch.gather(h[key].view(BS, -1, h[key].shape[-2], h[key].shape[-1]).contiguous(), 1, xyz_ind.long().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h[key].shape[-2], h[key].shape[-1]).contiguous())
            h[key] = h[key].view(-1, h[key].shape[-2], h[key].shape[-1]).contiguous()

        basis, r = get_basis_and_r(Gout, self.num_degrees-1)
        for layer in self.stage2:
            h = layer(h, G=Gout, r=r, basis=basis)

        return h, Gout, r, basis

class GraphFPModule(nn.Module):
    def __init__(self, up_conv_nn, num_degrees=2, edge_dim=0, div=4, n_heads=1, knn=False, use_xyz=True, module_type='mid_layer', index=0, vector_attention=False):
        super(GraphFPModule, self).__init__()
        self.module_type = module_type
        self.use_xyz = use_xyz
        self.edge_dim= edge_dim
        self.div=div
        self.n_heads=n_heads
        self.num_degrees = num_degrees
        self.index = index

        # 2. add SE3-layer over intermediate graph, and abstract Graph
        Tblock = []
        in_channels  = eval(up_conv_nn[0]) # concatenated channels
        out_channels = up_conv_nn[1:]
        fibers  = [Fiber(num_degrees, in_channels)]
        for i in range(len(out_channels)):
            fibers.append( Fiber(num_degrees, out_channels[i]) )

        for i in range(len(out_channels)):
            Tblock.append(GConvSE3(fibers[i], fibers[i+1], self_interaction=True, edge_dim=self.edge_dim))
            Tblock.append(GNormSE3(fibers[i+1]))
        self.Tblock = nn.ModuleList(Tblock)

    def forward(self, h, G, r, basis, uph=None, upG=None, BS=2):
        """
        h: input skip feature, with type 0: [BS*N, C, 1], type 0: [BS*N, C, 1]
        G, input skip graph,
        r: relative distance
        basis: basis function in SE3 layer
        upG: previous layer Graph, need upsampling;
        uph: previous layer, need upsampling;
        """
        xyz_prev = upG.ndata['x'].view(BS, -1, 3).contiguous()
        xyz = G.ndata['x'].view(BS, -1, 3).contiguous()

        # upsampling + concatenation
        dist, ind = three_nn(xyz, xyz_prev)
        dist = dist * dist
        dist[dist < 1e-10] = 1e-10
        inverse_dist = 1.0 / (dist + 1e-8)
        norm = torch.sum(inverse_dist, dim=2, keepdim=True)
        weights = inverse_dist / norm
        keys = h.keys()
        for key in keys:
            nC, nF= uph[key].shape[-2], uph[key].shape[-1]
            fp    = uph[key].view(uph[key].shape[0], -1).contiguous() # BS*N, C, 1/3 -> BS*N, C
            fp    = fp.view(BS, -1, fp.shape[-1]).contiguous().permute(0, 2, 1).contiguous() # BS, C, N
            new_features = torch.sum(group_gather_by_index(fp, ind) * weights.unsqueeze(1), dim=3) # BS, C, N
            nC1   = new_features.shape[1]
            assert nC1 == nC * nF
            new_features = new_features.permute(0, 2, 1).contiguous().view(-1, nC1).contiguous()
            new_features = new_features.view(new_features.shape[0], nC, -1)
            h[key] = torch.cat([new_features, h[key]], dim=1)

        for i, layer in enumerate(self.Tblock):
            h = layer(h, G=G, r=r, basis=basis)

        return h

def equivariance_test(model1):
    B, N, M        = 2, 512, 10
    base_points = np.random.rand(B, N, 3).astype(np.float32) * 1.0
    frnn      = FixedRadiusNearNeighbors(0.2, 10)
    gt_points = np.copy(base_points)
    inputs    = torch.from_numpy(gt_points)
    # data
    pos       = inputs
    centroids = np.tile(np.arange(pos.shape[1]).reshape(1, -1),(B, 1))
    centroids = torch.from_numpy(centroids)
    print(pos.shape, centroids.shape)
    feat      = np.ones((pos.shape[0], pos.shape[1], 1))
    dev       = pos.device
    neighbors_sample = SampleK(0.1, M+1, knn=True)
    # require cuda
    neighbors_ind = neighbors_sample(pos.transpose(2, 1).cuda(), pos.transpose(2, 1).cuda())
    group_idx = neighbors_ind.cpu()
    group_idx = group_idx[:, :, 1:]

    glist   = []
    n_neighbor = group_idx.shape[-1]

    # # test orth prediction
    # tx, ty, tz = np.random.rand(1, 3)[0] * 180
    # out1 = rot(tx, ty, tz) # 3 * 3
    # tx, ty, tz = np.random.rand(1, 3)[0] * 180
    # R2 = rot(tx, ty, tz)
    # out2 = out1 @ R2
    # p_out1 = compute_rotation_matrix_from_ortho6d(out1[:, :2].contiguous().permute(1, 0).contiguous().view(-1, 6))
    # p_out2 = compute_rotation_matrix_from_ortho6d(out2[:, :2].contiguous().permute(1, 0).contiguous().view(-1, 6))
    # diff = torch.max(torch.abs(p_out2 - p_out1 @ R2)).item()
    # print('case 1: type 1 diff max: ', diff)
    #
    # # test random prediction
    # out1 = torch.from_numpy(np.random.rand(3, 3)*5).float()
    # out1 = out1 / torch.norm(out1, dim=1, keepdim=True) # must do the normalization
    # tx, ty, tz = np.random.rand(1, 3)[0] * 180
    # R2 = rot(tx, ty, tz)
    # out2   = out1 @ R2
    # p_out1 = compute_rotation_matrix_from_ortho6d(out1[:2].contiguous().view(-1, 6)).permute(0, 2, 1).contiguous()
    # p_out2 = compute_rotation_matrix_from_ortho6d(out2[:2].contiguous().view(-1, 6)).permute(0, 2, 1).contiguous()
    # diff = torch.max(torch.abs(p_out2 - p_out1 @ R2)).item()
    # print('case 2: type 1 diff max: ', diff)

    builder = BuildGraph(num_samples=10)
    G, _     = builder(pos)

    dev = torch.device('cuda:0')
    model1.eval()
    for i in range(2):
        print(f'---{i}th data---')
        torch.cuda.empty_cache()
        G1 = deepcopy(G)
        G2 = deepcopy(G)

        R1 = rot(0, 0, 0)
        G1, features = set_feat(G1, R1)
        out_dict1 = model1.forward(G1.to(dev))
        f1, out1  = out_dict1['N'], out_dict1['R']
        # print('...out1: ')
        # summary(out1)
        tx, ty, tz = np.random.rand(1, 3)[0] * 180
        R2 = rot(tx, ty, tz)
        G2, features = set_feat(G2, R2)
        # f2, out2 = apply_model(model, G2, features)
        out_dict2 = model1.forward(G2.to(dev))
        f2, out2  = out_dict2['N'], out_dict2['R']
        # print('...out2: ')
        # summary(out2)
        # print('...out1 * R: ')
        # summary(out1 @ R2.cuda())
        diff = torch.max(torch.abs(f2 - f1)).item()
        print('type 0 diff max: ', diff)
        diff = torch.max(torch.abs(out2 - out1 @ R2.cuda())).item()
        print('type 1 diff max: ', diff)
        print('')

if __name__ == '__main__':
    from copy import copy
    from hydra.experimental import compose, initialize
    initialize("./config/", strict=True)
    cfg = compose("completion.yaml")
    npoints        = cfg.num_points
    # model
    model1 = SE3Transformer(cfg=cfg, edge_dim=0, pooling='avg', n_heads=cfg.MODEL.n_heads).cuda()
    print(model1)
    # model  = build_model()
    np.random.seed(0)
    equivariance_test(model1)
