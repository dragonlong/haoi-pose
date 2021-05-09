import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from collections import OrderedDict
import json
import sys


def bp():
    import pdb;pdb.set_trace()
# used for ModelNet40 direct regression of
# [nb, nc, np, na] --> [nb, nr, na], relative rotation;
# [nb, nc, np, na] --> [nb, na] confidence
import __init__
import vgtk
from models import spconv as M
import vgtk.so3conv.functional as L


class InvSO3ConvModel(nn.Module):
    def __init__(self, params, config=None):
        super(InvSO3ConvModel, self).__init__()

        self.backbone = nn.ModuleList()
        for block_param in params['backbone']:
            self.backbone.append(M.BasicSO3ConvBlock(block_param))

        # self.outblock = M.SO3OutBlockR(params['outblock']) # for 0.2
        if config.num_modes_R > 1:
            self.outblock = nn.ModuleList()
            for i in range( config.num_modes_R):
                self.outblock.append(M.SO3OutBlockR(params['outblock'], norm=1)) # for 0.3
        else:
            self.outblock = M.SO3OutBlockR(params['outblock'], norm=1, pooling_method=config.model.pooling_method) # for 0.3
        self.na_in = params['na']
        self.invariance = True
        self.config = config
        self.n_heads= config.num_modes_R

    def forward(self, x):
        # nb, np, 3 -> [nb, 3, np] x [nb, 1, np, na]
        x = M.preprocess_input(x, self.na_in, False)
        # x = M.preprocess_input(x, 1)

        for block_i, block in enumerate(self.backbone):
            x = block(x)

        # x = self.outblock(x)
        if self.n_heads > 1:
            confidence_list, quats_list = [], []
            for head in self.outblock:
                confidence, quats = head(x)
                confidence_list.append(confidence)
                quats_list.append(quats)
            return confidence_list, quats_list
        else:
            confidence, quats = self.outblock(x)
            return confidence, quats
        # (Pdb) x.xyz.shape
        # torch.Size([2, 3, 64]) #
        # (Pdb) x.feats.shape
        # torch.Size([2, 128, 64, 60]) [nb, nc, np, na]
        # (Pdb) x.anchors.shape
        # torch.Size([60, 3, 3])
        # (Pdb) n
        # > /home/lxiaol9/3DGenNet2019/EPN_PointCloud/SPConvNets/models/inv_so3net_pn.py(44)forward()
        # -> return x
        # (Pdb) x[0].shape
        # torch.Size([2, 64]) [nb, np]
        # (Pdb) x[1].shape
        # torch.Size([2, 128, 64, 60]) [nb, nc, np, na]

    def get_anchor(self):
        return self.backbone[-1].get_anchor()

def build_model(opt,
                mlps=[[32,32], [64,64], [128,128], [128,128]],
                out_mlps=[128, 64],
                strides=[2, 2, 2, 2],
                initial_radius_ratio = 0.2,
                sampling_ratio = 0.8, #0.4, 0.36
                sampling_density = 0.5,
                kernel_density = 1,
                kernel_multiplier = 2,
                sigma_ratio= 0.5, # 1e-3, 0.68
                xyz_pooling = None, # None, 'no-stride'
                to_file=None):

    device = torch.device('cuda:{}'.format(0))
    input_num= opt.model.input_num
    dropout_rate= opt.model.dropout_rate
    temperature= opt.train_loss.temperature
    so3_pooling =  opt.model.flag
    input_radius = opt.model.search_radius
    kpconv = opt.model.kpconv

    na = 1 if opt.model.kpconv else opt.model.kanchor

    # to accomodate different input_num
    if input_num > 1024:
        sampling_ratio /= (input_num / 1024)
        strides[0] = int(2 * (input_num / 1024))
        print("Using sampling_ratio:", sampling_ratio)
        print("Using strides:", strides)

    print("[MODEL] USING RADIUS AT %f"%input_radius)
    params = {'name': 'Invariant SPConv Model',
              'backbone': [],
              'na': na
              }
    dim_in = 1

    # process args
    n_layer = len(mlps)
    stride_current = 1
    stride_multipliers = [stride_current]
    for i in range(n_layer):
        stride_current *= 2
        stride_multipliers += [stride_current]

    num_centers = [int(input_num / multiplier) for multiplier in stride_multipliers]

    radius_ratio = [initial_radius_ratio * multiplier**sampling_density for multiplier in stride_multipliers]

    # radius_ratio = [0.25, 0.5]
    radii = [r * input_radius for r in radius_ratio]

    weighted_sigma = [sigma_ratio * radii[0]**2]
    for idx, s in enumerate(strides):
        weighted_sigma.append(weighted_sigma[idx] * s)

    for i, block in enumerate(mlps):
        block_param = []
        for j, dim_out in enumerate(block):
            lazy_sample = i != 0 or j != 0

            stride_conv = i == 0 or xyz_pooling != 'stride'

            # TODO: WARNING: Neighbor here did not consider the actual nn for pooling. Hardcoded in vgtk for now.
            neighbor = int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))

            if i == 0 and j == 0:
                neighbor *= int(input_num / 1024)

            kernel_size = 1
            if j == 0:
                # stride at first (if applicable), enforced at first layer
                inter_stride = strides[i]
                nidx = i if i == 0 else i+1
                # nidx = i if (i == 0 or xyz_pooling != 'stride') else i+1
                if stride_conv:
                    neighbor *= 2 # * int(sampling_ratio * num_centers[i] * radius_ratio[i]**(1/sampling_density))
                    kernel_size = 1 # if inter_stride < 4 else 3
            else:
                inter_stride = 1
                nidx = i+1

            # one-inter one-intra policy
            block_type = 'inter_block' if na != 60  else 'separable_block'
            inter_param = {
                'type': block_type,
                'args': {
                    'dim_in': dim_in,
                    'dim_out': dim_out,
                    'kernel_size': kernel_size,
                    'stride': inter_stride,
                    'radius': radii[nidx],
                    'sigma': weighted_sigma[nidx],
                    'n_neighbor': neighbor,
                    'lazy_sample': lazy_sample,
                    'dropout_rate': dropout_rate,
                    'multiplier': kernel_multiplier,
                    'activation': 'leaky_relu',
                    'pooling': xyz_pooling,
                    'kanchor': na,
                }
            }
            block_param.append(inter_param)

            dim_in = dim_out

        params['backbone'].append(block_param)

    representation = opt.model.representation
    params['outblock'] = {
            'dim_in': dim_in,
            'mlp': out_mlps,
            'fc': [64],
            'k': 40,
            'kanchor': na,
            'pooling': so3_pooling,
            'representation': representation,
            'temperature': temperature,
    }

    if to_file is not None:
        with open(to_file, 'w') as outfile:
            json.dump(params, outfile)

    model = InvSO3ConvModel(params, config=opt).to(device)
    return model

def build_model_from(opt, outfile_path=None):
    return build_model(opt, to_file=outfile_path)

if __name__ == '__main__':
    from models.spconv.options import opt
    BS = 2
    N  = 1024
    C  = 3
    device = torch.device("cuda:0")
    x = torch.randn(BS, N, 3).to(device)
    opt.mode = 'train'
    opt.model.flag = 'rotation'
    print("Performing a regression task...")
    opt.model.model = 'inv_so3net'
    model = build_model_from(opt, outfile_path=None)
    out = model(x)
    print(out[0].shape, out[1].shape)
    print('Con!')
