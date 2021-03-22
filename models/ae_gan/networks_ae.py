"""
Log: Monday, 3.1
1. change concatenation into sum;
2. change module into GraphFPSumModule;
3. flexible neighbors & points;
4. confidence tested;

Tuesday: 3.2
1. add multiple-mode R prediction;
2. add head classifier_mode;
3. remove decoder part(?)

"""
import torch
import torch.nn as nn
import sys

import numpy as np
from math import pi ,sin, cos, sqrt
from dgl.nn.pytorch import GraphConv, NNConv
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple, List

from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling, G1x1SE3, GSum
from equivariant_attention.fibers import Fiber
eps=1e-10
# only for pointnet++ baseline
try:
    from common.debugger import *
    from models.model_factory import ModelBuilder
    from models.pointnet_lib.networks import PointTransformer
    # from models.decoders.pointnet_2 import PointNet2Segmenter
    # from models.decoders.equivariant_model import EquivariantDGCNN
except:
    print('~need env paths~')
from kaolin.models.PointNet2 import furthest_point_sampling
from kaolin.models.PointNet2 import fps_gather_by_index
from kaolin.models.PointNet2 import ball_query
from kaolin.models.PointNet2 import three_nn
from kaolin.models.PointNet2 import group_gather_by_index
from omegaconf import DictConfig, ListConfig
import dgl

def bp():
    import pdb;pdb.set_trace()
def is_list(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig)

def is_iterable(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig) or isinstance(entity, tuple)

def is_dict(entity):
    return isinstance(entity, dict) or isinstance(entity, DictConfig)

SPECIAL_NAMES = ["radius", "max_num_neighbors", "block_names", "num_degrees"]

"""
class InterDownGraph()
class SE3TBlock(): downsampling block
class GraphFPModule(): upsampling block
"""
def eval_torch_func(key):
    if key == 'sigmoid':
        return nn.Sigmoid()
    elif key == 'tanh':
        return nn.Tanh()
    elif key == 'softmax':
        return nn.Softmax(1)
    else:
        return NotImplementedError

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

    # def forward(self, G, BS=2):
    #     """
    #     G: input Graph
    #     BS: batch size
    #     """
    #     glist = []
    #     pos = G.ndata['x'].view(BS, -1, 3).contiguous() # it should be 256, but only got 249, then the input doesn't have enough points
    #     B, N, _ = pos.shape
    #     xyz_ind, xyz_query = self.n_sampler(pos)        # downsample, might be that I actually sampled 256 > 249, so that
    #     neighbors_ind      = self.e_sampler(pos, xyz_query) #
    #     glist              = []                          # works for all complete shapes
    #     for i in range(BS):
    #         src = neighbors_ind[i].contiguous().view(-1)
    #         dst = xyz_ind[i].view(-1, 1).repeat(1, self.num_samples).view(-1)
    #         g = dgl.graph((src.long(), dst.long()), num_nodes=len(pos[i])).to(pos.device)
    #         g.ndata['x'] = pos[i]
    #         g.ndata['f'] = torch.ones(pos[i].shape[0], 1, 1, device=pos.device).float()
    #         g.edata['d'] = pos[i][dst.long()] - pos[i][src.long()]
    #         glist.append(g)
    #
    #     Gmid = dgl.batch(glist)
    #     # updated graph
    #     glist = []
    #     pos   = xyz_query
    #     neighbors_ind = self.e_sampler(pos, pos)
    #     for i in range(B):
    #         src = neighbors_ind[i].contiguous().view(-1).cpu()
    #         dst = torch.arange(pos[i].shape[0]).view(-1, 1).repeat(1, self.num_samples).view(-1)
    #         g = dgl.graph((src.long(), dst.long()), num_nodes=len(pos[i])).to(pos.device)
    #         g.ndata['x'] = pos[i]
    #         g.ndata['f'] = torch.ones(pos[i].shape[0], 1, 1, device=pos.device).float()
    #         g.edata['d'] = pos[i][dst.long()] - pos[i][src.long()] #[num_atoms,3] but we only supervise the half
    #         glist.append(g)
    #     Gout = dgl.batch(glist)
    #
    #     return Gmid, Gout, xyz_ind

    def forward(self, G, BS=2):
        """
        G: input Graph
        BS: batch size
        """
        glist = []
        pos = G.ndata['x'].view(BS, -1, 3).contiguous() # it should be 256, but only got 249, then the input doesn't have enough points
        B, N, _ = pos.shape
        xyz_ind, xyz_query = self.n_sampler(pos)        # downsample, might be that I actually sampled 256 > 249, so that
        neighbors_ind      = self.e_sampler(pos, xyz_query) #
        glist              = []                          # works for all complete shapes
        for i in range(BS):
            src = neighbors_ind[i].contiguous().view(-1)
            dst = xyz_ind[i].view(-1, 1).repeat(1, self.num_samples).view(-1)
            g = dgl.DGLGraph((src.cpu().long(), dst.cpu().long()))
            try:
                g.ndata['x'] = pos[i] # dgl._ffi.base.DGLError: Expect number of features to match number of nodes (len(u)). Got 256 and 249 instead.
                g.ndata['f'] = torch.ones(pos[i].shape[0], 1, 1, device=pos.device).float()
                g.edata['d'] = pos[i][dst.long()] - pos[i][src.long()] #[num_atoms,3] but we only supervise the half
            except:
                print('---something wrong!!!')
                g = dgl.unbatch(G)[i]
                g.remove_edges( np.arange( len(g.all_edges()[0]) ).tolist()) # this line comes with bug
                g.add_edges(src.cpu().long(), dst.cpu().long())
                g.edata['d'] = pos[i][dst.long()] - pos[i][src.long()]
            glist.append(g)

        Gmid = dgl.batch(glist)

        # updated graph
        glist = []
        pos   = xyz_query
        neighbors_ind = self.e_sampler(pos, pos)
        for i in range(B):
            src = neighbors_ind[i].contiguous().view(-1).cpu()
            dst = torch.arange(pos[i].shape[0]).view(-1, 1).repeat(1, self.num_samples).view(-1)
            g = dgl.DGLGraph((src.cpu(), dst.cpu()))
            g.ndata['x'] = pos[i]
            g.ndata['f'] = torch.ones(pos[i].shape[0], 1, 1, device=pos.device).float()
            g.edata['d'] = pos[i][dst.long()] - pos[i][src.long()] #[num_atoms,3] but we only supervise the half
            glist.append(g)
        Gout = dgl.batch(glist)

        return Gmid, Gout, xyz_ind

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
            src = neighbors_ind[i].contiguous().view(-1).cpu()
            dst = torch.arange(pos[i].shape[0]).view(-1, 1).repeat(1, self.num_samples).view(-1)
            g = dgl.DGLGraph((src.cpu(), dst.cpu()))
            g.ndata['x'] = pos[i]
            g.ndata['f'] = torch.ones(pos[i].shape[0], 1, 1, device=pos.device).float()
            g.edata['d'] = pos[i][dst.long()] - pos[i][src.long()] #[num_atoms,3] but we only supervise the half
            #
            glist.append(g)
        G = dgl.batch(glist)
        return G, h

def pdist2squared(x, y):
    """
    x: [B, 3, N]
    y: [B, 3, N]
    """
    xx = (x**2).sum(dim=1).unsqueeze(2) # B, N, 1
    yy = (y**2).sum(dim=1).unsqueeze(1) # B, 1, N
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

# PointNet++ as Encoder
class PointNetplusplus(nn.Module):
    """PointNet++ with multiple heads"""
    def __init__(self, cfg):
        super().__init__()
        self.backbone   = PointNet2Segmenter(num_classes=3, use_random_ball_query=True)
        net_header, head_names  = ModelBuilder.build_header(layer_specs=cfg.HEAD)
        self.head       = net_header
        self.head_names = head_names

    def forward(self, xyz):
        net, bottle_neck = self.backbone(xyz)
        pred_dict = {}
        for i, sub_head in enumerate(self.head):
            if 'regression' in self.head_names[i]:
                if bottle_neck.size(-1) !=1:
                    bottle_neck = bottle_neck.max(-1)[0]
                pred_dict[self.head_names[i]] = self.head[i](bottle_neck.view(-1, 1024))
            else:
                pred_dict[self.head_names[i]] = self.head[i](net)

        return pred_dict

class RegressorC1D(nn.Module):
    def __init__(self, out_channels=[256, 256, 3], latent_dim=128):
        super(RegressorC1D, self).__init__()
        layers = []#
        out_channels = [latent_dim] + out_channels
        for i in range(1, len(out_channels)-1):
            layers += [nn.Conv1d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.LeakyReLU(inplace=True)] # nn.BatchNorm1d(out_channels[i], eps=0.001)
        if out_channels[-1] in ['sigmoid', 'tanh', 'relu', 'softmax']:
            layers +=[eval_torch_func(out_channels[-1])]
        else:
            layers += [nn.Conv1d(out_channels[-2], out_channels[-1], 1, bias=True)]
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)

class RegressorFC(nn.Module):
    def __init__(self, latent_dim=128, bn=False):
        super(RegressorFC, self).__init__()
        self.latent_dim = latent_dim

    def forward(self, x):
        return x # B, 3, 3

class DecoderFC(nn.Module):
    def __init__(self, n_features=(256, 256), latent_dim=128, output_pts=2048, bn=False):
        super(DecoderFC, self).__init__()
        self.n_features = list(n_features)
        self.output_pts = output_pts
        self.latent_dim = latent_dim

        model = []
        prev_nf = self.latent_dim
        for idx, nf in enumerate(self.n_features):
            fc_layer = nn.Linear(prev_nf, nf)
            model.append(fc_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        fc_layer = nn.Linear(self.n_features[-1], output_pts*3)
        model.append(fc_layer)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        x = x.view((-1, 3, self.output_pts))
        return x


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
        pred_dict = {}
        for i in range(len(self.pre_modules)):
            h0 = self.pre_modules[i](h0, G=G, r=r, basis=basis)
        # pred_dict = {'h0': {'0': h0['0'].detach().clone(), '1': h0['1'].detach().clone()}}
        # encoding
        h1, G1, r1, basis1 = self.down_modules[0](h0, Gin=G0, BS=self.batch_size) # 512-256
        # pred_dict['h1'] =  {'0': h1['0'].detach().clone(), '1': h1['1'].detach().clone()}

        h2, G2, r2, basis2 = self.down_modules[1](h1, Gin=G1, BS=self.batch_size) # 256-128
        # pred_dict['h2'] =  {'0': h2['0'].detach().clone(), '1': h2['1'].detach().clone()}

        h3, G3, r3, basis3 = self.down_modules[2](h2, Gin=G2, BS=self.batch_size) # 128-64
        # pred_dict['h3'] =  {'0': h3['0'].detach().clone(), '1': h3['1'].detach().clone()}

        h4, G4, r4, basis4 = self.down_modules[3](h3, Gin=G3, BS=self.batch_size) # 64-32
        # pred_dict['h4'] =  {'0': h4['0'].detach().clone(), '1': h4['1'].detach().clone()}
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
        # pred_dict.update({'h0_u': h, 'h1_u': h1, 'h2_u': h2, 'h3_u': h3})

        pred_S   = self.Oblock[0](h, G=G, r=r, basis=basis) # only one mode
        pred_R   = self.Oblock[1](h, G=G, r=r, basis=basis) #
        pred_T   = self.Oblock[2](h, G=G, r=r, basis=basis) # 1. dense type 1 feature for T

        output_R = pred_R['1']/(torch.norm(pred_R['1'], dim=-1, keepdim=True) + eps)
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

class GraphFPSumModule(nn.Module):
    def __init__(self, up_conv_nn, num_degrees=2, edge_dim=0, div=4, n_heads=1, knn=False, use_xyz=True,
                 module_type='mid_layer', index=0, vector_attention=False):
        super(GraphFPSumModule, self).__init__()
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
            Tblock.append(GSE3Res(fibers[i], fibers[i+1], edge_dim=self.edge_dim, n_heads=self.n_heads)) # vector_attention=vector_attention
            Tblock.append(GNormSE3(fibers[i+1]))
        self.Tblock = nn.ModuleList(Tblock)
        self.add = GSum(fibers[0], fibers[-1])

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
        h_interpolated = {}
        for key in keys:
            nC, nF= uph[key].shape[-2], uph[key].shape[-1]
            fp    = uph[key].view(uph[key].shape[0], -1).contiguous() # BS*N, C, 1/3 -> BS*N, C
            fp    = fp.view(BS, -1, fp.shape[-1]).contiguous().permute(0, 2, 1).contiguous() # BS, C, N
            new_features = torch.sum(group_gather_by_index(fp, ind) * weights.unsqueeze(1), dim=3) # BS, C, N
            nC1   = new_features.shape[1]
            assert nC1 == nC * nF
            new_features = new_features.permute(0, 2, 1).contiguous().view(-1, nC1).contiguous()
            new_features = new_features.view(new_features.shape[0], nC, -1)
            h_interpolated[key] = new_features

        h = self.add(h, h_interpolated) # TODO
        # h = h_interpolated
        for i, layer in enumerate(self.Tblock):
            h = layer(h, G=G, r=r, basis=basis)

        return h


class GraphFPResNoSkipLinkModule(nn.Module):
    def __init__(self, up_conv_nn, num_degrees=2, edge_dim=0, div=4, n_heads=1, knn=False, use_xyz=True, module_type='mid_layer', index=0):
        super(GraphFPResNoSkipLinkModule, self).__init__()
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
            Tblock.append(GSE3Res(fibers[i], fibers[i+1], edge_dim=self.edge_dim, n_heads=self.n_heads))
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
        h_interpolated = {}
        for key in keys:
            nC, nF= uph[key].shape[-2], uph[key].shape[-1]
            fp    = uph[key].view(uph[key].shape[0], -1).contiguous() # BS*N, C, 1/3 -> BS*N, C
            fp    = fp.view(BS, -1, fp.shape[-1]).contiguous().permute(0, 2, 1).contiguous() # BS, C, N
            new_features = torch.sum(group_gather_by_index(fp, ind) * weights.unsqueeze(1), dim=3) # BS, C, N
            nC1   = new_features.shape[1]
            assert nC1 == nC * nF
            new_features = new_features.permute(0, 2, 1).contiguous().view(-1, nC1).contiguous()
            new_features = new_features.view(new_features.shape[0], nC, -1)
            h_interpolated[key] = new_features

        h = h_interpolated  # no skip link at all
        for i, layer in enumerate(self.Tblock):
            h = layer(h, G=G, r=r, basis=basis)

        return h

class en3_transformer(nn.Module):
    """en3_transformer with multiple heads"""
    def __init__(self, cfg):
        super().__init__()
        k = 16
        C               = cfg.MODEL.num_mid_channels
        C_in            = cfg.MODEL.num_in_channels
        self.num_R      = cfg.MODEL.num_channels_R
        C_out           = self.num_R                # for 6D rotation, use 2; for 3D rotation, use 1.
        self.backbone   = EquivariantDGCNN(k, C, C_in, C_out) # k, C, C_in=1, C_out=2):
        net_header, head_names  = ModelBuilder.build_header(layer_specs=cfg.HEAD)
        self.head       = net_header
        self.head_names = head_names


    def forward(self, xyz):
        BS, _, N = xyz.shape
        x = torch.cat([xyz.double(), torch.ones((BS, 1, N), device=xyz.device).double()], dim=1)
        x_out, f_out = self.backbone(x) # x: # [batch_size, C_out, 3, num_points];  f: [batch_size, 64, num_points]
        pred_dict = {}
        pred_dict['R'] = x_out[:, :self.num_R, :, :]# N, C, 3, N
        pred_dict['T'] = x_out[:, -1, :, :]
        for i, sub_head in enumerate(self.head):
            pred_dict[self.head_names[i]] = self.head[i](f_out)

        return pred_dict

# PointNet as Encoder
##############################################################################
class EncoderPointNet(nn.Module):
    def __init__(self, n_filters=(64, 128, 128, 256), latent_dim=128, bn=True):
        super(EncoderPointNet, self).__init__()
        self.n_filters = list(n_filters) + [latent_dim]
        self.latent_dim = latent_dim

        model = []
        prev_nf = 3
        for idx, nf in enumerate(self.n_filters):
            conv_layer = nn.Conv1d(prev_nf, nf, kernel_size=1, stride=1)
            model.append(conv_layer)

            if bn:
                bn_layer = nn.BatchNorm1d(nf)
                model.append(bn_layer)

            act_layer = nn.LeakyReLU(inplace=True)
            model.append(act_layer)
            prev_nf = nf

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        x = torch.max(x, dim=2)[0]
        return x


class PointAE(nn.Module):
    def __init__(self, cfg):
        super(PointAE, self).__init__()
        self.encoder_type = cfg.encoder_type
        if 'se3' in self.encoder_type:
             self.encoder = SE3Transformer(cfg=cfg, edge_dim=0, pooling='avg', n_heads=cfg.MODEL.n_heads,
                                           vector_attention=cfg.MODEL.vector_attention)
        elif 'plus' in self.encoder_type:
            self.encoder = PointNetplusplus(cfg)
        elif 'en3' in self.encoder_type:
            default_type = torch.DoubleTensor
            torch.set_default_tensor_type(default_type)
            self.encoder = en3_transformer(cfg)
        elif 'point_transformer' in self.encoder_type:
            self.encoder = PointTransformer(num_channels_R=cfg.MODEL.num_channels_R,
                                            R_dim=6 if cfg.pred_6d else 3)
        else:
            self.encoder = EncoderPointNet(eval(cfg.enc_filters), cfg.latent_dim, cfg.enc_bn)

        if 'pose' in cfg.task:
            self.regressor = RegressorFC(cfg.MODEL.num_channels, bn=False)
            if cfg.pred_nocs:
                self.regressor_nocs = RegressorC1D(list(cfg.nocs_features), cfg.latent_dim)
            if cfg.pred_seg:
                self.classifier_seg = RegressorC1D(list(cfg.seg_features), cfg.latent_dim)
            if cfg.pred_conf:
                self.regressor_confi= RegressorC1D(list(cfg.confi_features), cfg.latent_dim)
            if cfg.pred_mode:
                self.classifier_mode= RegressorC1D(list(cfg.mode_features), cfg.MODEL.num_channels_R)
        self.decoder = DecoderFC(eval(cfg.dec_features), cfg.latent_dim, cfg.n_pts, cfg.dec_bn)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def regress(self, x):
        return self.regressor(x)

    def forward(self, x):
        # print('---PointAE forwarding---')
        z = self.encoder(x)
        if 'se3' in self.encoder_type:
            x = self.decoder(z['0'])   # shape recontruction
            p = self.regressor(z['1']) # R
            pred_dict = {'S':x, 'R': p, 'T': z['T']}
        else:
            pred_dict = z

        return pred_dict

def rotate_about_axis(theta, axis='x'):
    if axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, cos(theta), -sin(theta)],
                      [0, sin(theta), cos(theta)]])

    elif axis == 'y':
        R = np.array([[cos(theta), 0, sin(theta)],
                      [0, 1, 0],
                      [-sin(theta), 0, cos(theta)]])

    elif axis == 'z':
        R = np.array([[cos(theta), -sin(theta), 0],
                      [sin(theta), cos(theta),  0],
                      [0, 0, 1]])
    return R

def square_distance(src, dst):
    '''
    Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
    '''
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1).contiguous())
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist #

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
    def __init__(self, radius, n_neighbor, knn=False):
        super(FixedRadiusNearNeighbors, self).__init__()
        self.radius = radius
        self.n_neighbor = n_neighbor
        self.knn = knn

    def forward(self, pos, centroids):
        '''
        Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
        '''
        device = pos.device
        B, N, _ = pos.shape
        # center_pos = index_points(pos, centroids)
        center_pos = pos
        _, S, _ = center_pos.shape
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        # print('before square_distance ', center_pos.shape, pos.shape)
        sqrdists = square_distance(center_pos, pos)
        if self.knn:
            _, group_idx = torch.topk(sqrdists, self.n_neighbor+1, dim=-1, largest=False, sorted=True)
            group_idx = group_idx[:, :, 1:self.n_neighbor+1]
        else:
            group_idx[sqrdists > self.radius ** 2] = N
            group_idx = group_idx.sort(dim=-1)[0][:, :, :self.n_neighbor]
            group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, self.n_neighbor])
            mask = group_idx == N
            group_idx[mask] = group_first[mask]
        return group_idx

# python networks_ae.py models=se3_transformer_default num_points=512
if __name__ == '__main__':
    from copy import copy
    from hydra.experimental import compose, initialize
    initialize("../../config/", strict=True)
    cfg = compose("completion.yaml")
    npoints        = cfg.num_points

    # model
    model = SE3Transformer(cfg=cfg, edge_dim=0, pooling='avg', n_heads=cfg.MODEL.n_heads).cuda()
    frnn      = FixedRadiusNearNeighbors(0.2, 10)
    device = torch.device("cuda")
    batch_size = 2
    num_points = 256
    k = 10
    torch.manual_seed(0)

    pts = torch.randn(batch_size, num_points, 3)
    # create input
    pos       = pts.detach().clone()
    centroids = np.tile(np.arange(pts.shape[1]).reshape(1, -1),(batch_size, 1))
    centroids = torch.from_numpy(centroids)
    feat      = np.ones((pos.shape[0], pos.shape[1], 1))
    dev       = pos.device
    group_idx = frnn(pos, centroids) # cpu function, process per batch, we could put in
    glist = []
    n_neighbor = 10
    for i in range(batch_size):
        src = group_idx[i].contiguous().view(-1) # real pair
        dst = centroids[i].view(-1, 1).repeat(1, n_neighbor).view(-1) # real pair

        unified = torch.cat([src, dst])
        uniq, inv_idx = torch.unique(unified, return_inverse=True)
        src_idx = inv_idx[:src.shape[0]]
        dst_idx = inv_idx[src.shape[0]:]

        # print('src_idx.shape', '\n', src_idx[0:100], '\n', 'dst_idx.shape', '\n', dst_idx[0:100])
        g = dgl.DGLGraph((src_idx, dst_idx))
        g.ndata['x'] = pos[i][uniq]
        g.ndata['f'] = torch.from_numpy(feat[i][uniq].astype(np.float32)[:, :, np.newaxis])
        g.edata['d'] = pos[i][dst_idx] - pos[i][src_idx] #[num_atoms,3]
        glist.append(g)
    batched_graph = dgl.batch(copy(glist)).to(device)
    out1 = model(batched_graph)
    bp()

    # print('\n')
    # print('Test rotation equivariance')
    # #rot = R.random(random_state=1234).as_matrix()
    # rot = np.array([[0.5, np.sqrt(3)/2, 0], [-np.sqrt(3)/2, 0.5, 0], [0, 0, 1]])
    # rot = torch.from_numpy(rot).type(default_type).to(device)
    # rot = rot.unsqueeze(0).repeat(batch_size, 1, 1)
    # x_rotated = torch.matmul(rot, x)
    #
    #
    # rotated_pts = torch.cat((x_rotated, f), dim=1)
    #     batched_graph.ndata['x'] = torch.matmul(batched_graph.ndata['x'], torch.from_numpy(R.astype(np.float32)))
    #     batched_graph.edata['d'] = torch.matmul(batched_graph.edata['d'], torch.from_numpy(R.astype(np.float32)))
    # x3, f3 = model(rotated_pts)
    #
    # rot = rot.unsqueeze(1).repeat(1, C_out, 1, 1)
    # x1_rotated = torch.matmul(rot, x1)
    # x_diff_rotation = x3 - x1_rotated
    # f_diff_rotation = f3 - f1
    #
    # print('x diff mean:', torch.mean(torch.abs(x_diff_rotation)))
    # print('x diff max:', torch.max(torch.abs(x_diff_rotation)))
    # print('f diff mean:', torch.mean(torch.abs(f_diff_rotation)))
    # print('f diff max:', torch.max(torch.abs(f_diff_rotation)))
    #

    #
    # np.random.seed(0)
    #
    # base_points = np.random.rand(2,256,3).astype(np.float32) # * 10
    # gt_points = np.copy(base_points)
    # inputs    = torch.from_numpy(gt_points)
    # # data

    #     # batched_graph = dgl.batch_hetero(glist)
    # Rs = []
    # outputs = []
    # outputs_w = []
    # theta_xs = [0, 45]
    # theta_zs = [0, 45]
    # for k in range(2):
    #     batched_graph = dgl.batch(copy(glist))
    #     # print(glist[0].ndata['x'])
    #     theta_x = theta_xs[k]
    #     theta_z = theta_zs[k]
    #     Rx = rotate_about_axis(theta_x / 180 * np.pi, axis='x')
    #     Rz = rotate_about_axis(theta_z / 180 * np.pi, axis='z')
    #     R = np.matmul(Rx, Rz)
    #     Rs.append(R)
    #     print(R)

    #     feat_type1 = model.forward(batched_graph)
    #     outputs.append(feat_type1)
    #     f1 = feat_type1[:, :, [2, 0, 1]]
    #     outputs_w.append(f1)
    #
    #
    # # print(outputs[0][1], '\n\n', outputs[1][1])
    # print('using R: ', '\n', outputs[1][0][0], '\n', torch.matmul(outputs[0], torch.from_numpy(Rs[1].astype(np.float32)))[0][0], '\n')
    # print('using R.T: ', '\n', outputs[1][0][0], '\n', torch.matmul(outputs[0], torch.from_numpy(Rs[1].T.astype(np.float32)))[0][0], '\n')
    # print('using R: ', '\n', outputs_w[1][0][0], '\n', torch.matmul(outputs_w[0], torch.from_numpy(Rs[1].astype(np.float32)))[0][0], '\n')
    # print('using R.T: ', '\n', outputs_w[1][0][0], '\n', torch.matmul(outputs_w[0], torch.from_numpy(Rs[1].T.astype(np.float32)))[0][0], '\n')
    # # inputs = []
    # # for i in range(BS):
    # #     full_pts = xyzs[i]
    # #     if fixed_sampling:
    # #         pos = torch.from_numpy(np.copy(full_pts)[:npoints, :]).unsqueeze(0)
    # #     else:
    # #         pos = torch.from_numpy(np.random.permutation(np.copy(full_pts))[:npoints, :]).unsqueeze(0)
    # #     inputs.append(pos)
    # #
    # # N    = 256
    # # xyz1  = torch.rand(1, N, 3, requires_grad=False, device=deploy_device)
    # # builder = BuildGraph()
    # # G, _    = builder(xyz1)
    # # #
    # # encoder = SE3Transformer(cfg=cfg, edge_dim=0, pooling='avg').cuda()
    # # torch.cuda.empty_cache()
    # # out = encoder(G)
    # # print(out)
    # # #
    # # create model
    # #
