import torch
import torch.nn as nn
import sys

import numpy as np
from dgl.nn.pytorch import GraphConv, NNConv
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple, List

from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from equivariant_attention.fibers import Fiber

from common.debugger import *
from models.model_factory import ModelBuilder

# class PointNetplusplus(nn.Module):
#     """PointNet++ with multiple heads"""
#     def __init__(self, cfg):
#         net_header, head_names  = ModelBuilder.build_header(layer_specs=cfg.HEAD, weights=cfg.header_weights)
#         self.decoder   = ModelBuilder.build_decoder(params=cfg.MODEL, options=cfg.models[cfg.name_model], weights=cfg[key].decoder_weights)
#         self.head      = net_header
#         self.head_names= head_names
#
#     def forward(xyz):
#         net, bottle_neck = self.decoder(xyz)
#
#         for i, sub_head in enumerate(self.head):
#             if 'regression' in self.head_names[i]:
#                 if bottle_neck.size(-1) !=1:
#                     bottle_neck = bottle_neck.max(-1)[0]
#                 pred_dict[self.head_names[i]] = self.head[i](bottle_neck.view(-1, 1024))
#             else:
#                 pred_dict[self.head_names[i]] = self.head[i](net)
#
#         return pred_dict

class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers: int, input_feature_size: int,
                 num_channels: int, num_channels_R: int=3, num_nlayers: int=1, num_degrees: int=4, latent_dim: int=128,
                 edge_dim: int=4, div: float=4, pooling: str='avg', n_heads: int=1, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div     = div
        self.pooling = pooling
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        self.num_channels_R = num_channels_R

        self.fibers = {'in': Fiber(1, input_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(2, self.num_channels),
                       'out_type0': Fiber(1, self.latent_dim),
                       'out_type1': Fiber(2, self.num_channels_R),
                       'out_type1_T': Fiber(2, 1)} # control output channels, TODO

        self._build_gcn(self.fibers, 1)
        print(self.Gblock)

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
        Oblock = [GConvSE3(fibers['out'], fibers['out_type0'], self_interaction=True, edge_dim=self.edge_dim),
                  GConvSE3(fibers['out'], fibers['out_type1'], self_interaction=True, edge_dim=self.edge_dim),
                  GConvSE3(fibers['out'], fibers['out_type1_T'], self_interaction=True, edge_dim=self.edge_dim)]
        # Pooling
        if self.pooling == 'avg':
            self.Pblock = GAvgPooling(type='1')
        elif self.pooling == 'max':
            self.Pblock = GMaxPooling()

        self.Gblock = nn.ModuleList(Gblock)
        self.Oblock = nn.ModuleList(Oblock)

        return

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)
        out = {}
        out['0'] = self.Oblock[0](h, G=G, r=r, basis=basis)['0']
        out['1'] = self.Oblock[1](h, G=G, r=r, basis=basis)['1']
        pred_dict = {'R': out['1']}
        out = self.Pblock(out, G=G, r=r, basis=basis)
        pred_dict['T'] = self.Oblock[2](h, G=G, r=r, basis=basis)['1'] # use type 1 feature
        pred_dict['0'] = out['0'] # already has full R output
        pred_dict['1'] = out['1']

        return pred_dict

# PointNet AutoEncoder
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

class PointAE(nn.Module):
    def __init__(self, config):
        super(PointAE, self).__init__()
        self.use_se3 = 'Graph' in config.dataset_class
        if self.use_se3:
            self.encoder = SE3Transformer(num_layers=config.MODEL.num_layers, \
                                            input_feature_size=config.MODEL.input_feature_size, \
                                            num_channels=config.MODEL.num_channels, \
                                            num_channels_R=config.MODEL.num_channels_R, \
                                            num_nlayers=config.MODEL.num_nlayers, \
                                            num_degrees=config.MODEL.num_degrees,
                                            edge_dim=0,
                                            latent_dim=config.latent_dim,
                                            pooling='avg')
        else:
            self.encoder = EncoderPointNet(eval(config.enc_filters), config.latent_dim, config.enc_bn)

        if 'pose' in config.task:
            self.regressor = RegressorFC(config.MODEL.num_channels, bn=config.dec_bn)
        self.decoder = DecoderFC(eval(config.dec_features), config.latent_dim, config.n_pts, config.dec_bn)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def regress(self, x):
        return self.regressor(x)

    def forward(self, x):
        print('---PointAE forwarding---')
        z = self.encoder(x)
        x = self.decoder(z['0'])   # shape recontruction
        p = self.regressor(z['1']) # R
        pred_dict = {'S':x, 'R': p, 'T': z['T']}
        return pred_dict


if __name__ == '__main__':
    pass
