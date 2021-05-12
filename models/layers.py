import numpy as np
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_

import __init__
import vgtk.so3conv as sptk

def bp():
    import pdb;pdb.set_trace()

def MLP(channels, activation=nn.LeakyReLU(0.2), bn_momentum=0.1, bias=True):
    return nn.Sequential(
        *[
            nn.Sequential(
                nn.Linear(channels[i - 1], channels[i], bias=bias),
                FastBatchNorm1d(channels[i], momentum=bn_momentum),
                activation,
            )
            for i in range(1, len(channels))
        ]
    )

class SimpleMLP(nn.Module):
    def __init__(self, k=2):
        super(SimpleMLP, self).__init__()
        self.k = k
        self.conv1 = torch.nn.Conv1d(64*2+128, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = BatchNorm1d(128)

    def forward(self, feat_list, v2p_ind, r2p_ind):
        p_feat, v_feat, r_feat = feat_list
        # feat from different backbones
        v_feat_per_pt = c2p_map(v_feat, v2p_ind)
        r_feat_per_pt = c2p_map(r_feat, r2p_ind)
        x = torch.cat([p_feat, v_feat_per_pt, r_feat_per_pt], 1)
        # MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x

class SimpleConv(nn.Module):
    def __init__(self, k=2):
        super(SimpleConv, self).__init__()
        self.k = k
        self.conv1 = torch.nn.Conv1d(64*2+128, 128, 1)
        self.conv2 = torch.nn.Conv2d(128, self.k, 3)
        self.bn1 = BatchNorm1d(128)

    def forward(self, feat_list, v2p_ind, r2p_ind):
        p_feat, v_feat, r_feat = feat_list
        # feat from different backbones
        v_feat_per_pt = c2p_map(v_feat, v2p_ind)
        r_feat_per_pt = c2p_map(r_feat, r2p_ind)
        x = torch.cat([p_feat, v_feat_per_pt, r_feat_per_pt], 1)
        # MLP
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.conv2(x)

        return x

class FC(nn.Sequential):
    def __init__(
        self,
        in_size,
        out_size,
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=None,
        preact=False,
        name="",
    ):
        # type: (FC, int, int, Any, bool, Any, bool, AnyStr) -> None
        super(FC, self).__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant_(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + "bn", BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + "activation", activation)

        self.add_module(name + "fc", fc)

        if not preact:
            if bn:
                self.add_module(name + "bn", BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + "activation", activation)



def devoxelization(c_feat, p2c_ind):
    """
    c_feat: [bs, C, h, w]
    p2c_ind:[bs, N, 2]
    """
    c_feat_per_pt = c2p_map(c_feat, p2c_ind)

    return c_feat_per_pt

def fetch_and_replace(r_feat, v_feat, p2v_ind, p2r_ind, unique_indices=None):
    """
    r_feat: [B*seq, C=20, H, W] --> [B, seq, C=20*16, H1, W1]
    v_feat: [B*seq, C=16*3, H, W]
    p2v_ind:[B*seq, N, 3]
    p2r_ind:[B*seq, N, 2]
    """
    bs, seq, c, h, w= v_feat.size()
    bs, seq, n1, d1 = p2v_ind.size()
    bs, seq, n2, d2 = p2r_ind.size()

    v_feat = v_feat.view(-1, c, h, w).contiguous()
    p2v_ind= p2v_ind.view(-1, n1, d1).contiguous()
    p2r_ind= p2r_ind.view(-1, n2, d2).contiguous()
    if unique_indices is not None: # use unique voxel indices
      _, _, n3 = unique_indices.size()
      unique_indices = unique_indices.view(-1, n3).contiguous()
      with torch.no_grad():
        p2r_ind = torch.gather(p2r_ind[:, :, :2], 1, unique_indices.unsqueeze(2).repeat(1, 1, 2)) # b, n, 2
        p2v_ind = torch.gather(p2v_ind[:, :, :3], 1, unique_indices.unsqueeze(2).repeat(1, 1, 3))

    # better have a mask
    r_feat_per_pt = c2p_map(r_feat, p2r_ind) # TODO
    # update new feature
    xv_mapped   = p2c_map3d(r_feat_per_pt, p2v_ind, v_feat)
    _, cn, _, _ = xv_mapped.size()
    xv_mapped = xv_mapped.view(bs, seq, cn, h, w).contiguous()

    return xv_mapped

def feat_fusion(p_feat, v_feat, v2p_ind, r_feat, r2p_ind):
    """
    inputs:
		p_feat:
		v_feat:
		v2p_ind: [B,N]
		r_feat:
		r2p_ind: [B, N]
    return:
    	feat_concat: [B, N, 64*3]
    """
    bs, cv, wv, hv = v_feat.size()
    v_feat_flat = v_feat.view(bs, cv, -1)
    bs, cr, wr, hr = r_feat.size()
    r_feat_flat = r_feat.view(bs, cr, -1)
    #
    choose_v = v2p_ind.unsqueeze(1).repeat(1, cv, 1) # B, C, N
    v_feat_per_pt = torch.gather(v_feat_flat, 2, choose_v).contiguous()
    choose_r = r2p_ind.unsqueeze(1).repeat(1, cr, 1) # B, C, N
    r_feat_per_pt = torch.gather(r_feat_flat, 2, choose_r).contiguous()
    feat_concat = torch.cat((p_feat, v_feat_per_pt, r_feat_per_pt), 1)

    return feat_concat

def c2p_map(c_feat, c2p_ind, unique_indices=None, gpu=None):
    """
    inputs:
		c_feat: B, C, H, W  --> B, C, H*W
		v2p_ind: [B, N, 3]  --> B, N --> B, C, N
		r_feat:
		r2p_ind: [B, N, 2]
    return:
    	feat_concat: [B, N, 64*3]
    """
    bs, cv, hv, wv = c_feat.size()
    c_feat_flat    = c_feat.view(bs, cv, -1)
    #
    # breakpoint()
    c2p_ind_flat  = c2p_ind[:, :, 0] * wv + c2p_ind[:, :, 1] # x* 480 + y
    if unique_indices:
        with torch.no_grad():
            c2p_ind_flat = torch.gather(c2p_ind_flat, 1, unique_indices) # B, N --> B, N'
    choose_c = c2p_ind_flat.unsqueeze(1).repeat(1, cv, 1) # B, C, N
    c_feat_per_pt = torch.gather(c_feat_flat, 2, choose_c).contiguous()

    return c_feat_per_pt

# assume all these variables on gpu already
def p2c_map(p_feat, p2c_ind, c_feat, unique_indices=None, gpu=None):
    """
    A general function for mapping point features to camera pixel,
    p_feat:  [B, C, N]
    p2c_ind: [B, N, 2]
    c_feat:  [B, C, W, L]
    c_feat[b, :, x, y] = p_feat[b, :, n]

    1 - (30, 40, 5)
    	...
    2 - (30, 40, 5)
    3 - (30, 45, 8)
        .
        .
        .
    4 - (100, 100, 1)
    ..	...
    N - (300, 100, 4)
    	...
    """
    if unique_indices:
        with torch.no_grad():
            p2c_ind = torch.gather(p2c_ind, 1, unique_indices[:, :, :2].unsqueeze(2).repeat(1, 1, 2))
    bs, n, _    = p2c_ind.size()
    bs, c, _    = p_feat.size()
    bs, _, w, h = c_feat.size()
    ind_b       = torch.arange(bs).unsqueeze(1).unsqueeze(2).repeat(1, n, 1)

    flat_p2c    = torch.cat((ind_b, p2c_ind.cpu()[:, :, 0:2]), 2).view(-1, 3)
    c_feat_n    = torch.zeros(bs, c, w, h, device=c_feat.device)
    # c_feat_n    = c_feat
    c_feat_n[flat_p2c[:, 0], :, flat_p2c[:, 1], flat_p2c[:, 2]] = p_feat.transpose(2, 1).contiguous().view(-1, c) # densely assign

    return c_feat_n

# assume all these variables on gpu already
def p2c_map3d(p_feat, p2c_ind, c_feat, unique_indices=None, gpu=None):
    """
    A general function for mapping point features to camera pixel,
    p_feat:  [B, C, N]
    p2c_ind: [B, N, 2]
    c_feat:  [B, C, W, L, H]
    c_feat[b, :, x, y] = p_feat[b, :, n]

    1 - (30, 40, 5)
        ...
    2 - (30, 40, 5)
    3 - (30, 45, 8)
        .
        .
        .
    4 - (100, 100, 1)
    ..  ...
    N - (300, 100, 4)
        ...
    """
    if unique_indices:
        with torch.no_grad():
            p2c_ind = torch.gather(p2c_ind, 1, unique_indices[:, :, :3].unsqueeze(2).repeat(1, 1, 3))
    bs, n, _    = p2c_ind.size()
    bs, c, _    = p_feat.size()
    bs, c_total, w, h = c_feat.size()
    ind_b       = torch.arange(bs).unsqueeze(1).unsqueeze(2).repeat(1, n, 1)

    flat_p2c    = torch.cat((ind_b, p2c_ind.cpu()[:, :, 0:3]), 2).view(-1, 4)
    c_feat_n    = torch.zeros(bs, w, h, int(c_total), c, device=c_feat.device)
    # c_feat_n    = c_feat
    c_feat_n[flat_p2c[:, 0], flat_p2c[:, 1], flat_p2c[:, 2], flat_p2c[:, 3], :] = p_feat.transpose(2, 1).contiguous().view(-1, c).contiguous() # densely assign
    c_feat_n = c_feat_n.view(bs, w, h, -1).contiguous().permute(0, 3, 1, 2).contiguous()

    return c_feat_n

class rv_model(nn.Module):
    def __init__(self):
        super(rv_model, self).__init__()
        self.conv1 = nn.Conv2d(64, 128, 5)
        self.conv2 = nn.Conv2d(128, 64, 5)

    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))

class bev_model(nn.Module):
    def __init__(self):
        super(bev_model, self).__init__()
        self.conv1 = nn.Conv2d(64, 128, 5)
        self.conv2 = nn.Conv2d(128, 64, 5)

    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))

class pt_model(nn.Module):
    def __init__(self):
        super(pt_model, self).__init__()
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 64, 1)

    def forward(self, x):
       x = F.relu(self.conv1(x))
       return F.relu(self.conv2(x))

class AttentGraph_Sum(nn.Module):
    def __init__(self, n_layer=1, n_head=2, c_node=64, c_emb=64, c_ins=[128, 64, 20], k=20, sum_type='concat'):
        super(AttentGraph_Sum, self).__init__()
        self.c_node      = c_node
        self.c_emb       = c_emb
        self.num_class   = k
        self.n_head      = n_head
        self.n_layer     = n_layer
        self.kernel_size = 3
        self.c_feat_list = [c_ins] + [c_ins] * n_layer
        self.c_emb_list  = [self.c_emb] * n_layer
        self.sum_type    = sum_type

        self.att_block_list = nn.ModuleList()
        for i in range(n_layer):
            self.att_block_list.append( AttentGraph_Block( self.c_feat_list[i+1], self.c_feat_list[i], self.c_emb_list[i], self.n_head) )

        if sum_type == 'concat':
            self.last_p = nn.Conv1d(self.c_feat_list[-1][0]*2, self.num_class, 1)
            self.last_r = nn.Conv2d(self.c_feat_list[-1][2]*2, self.num_class, self.kernel_size, padding=int((self.kernel_size-1)/2))
        else:
            self.last_p = nn.Conv1d(self.c_feat_list[-1][0], self.num_class, 1)
            self.last_r = nn.Conv2d(self.c_feat_list[-1][2], self.num_class, self.kernel_size, padding=int((self.kernel_size-1)/2))
        # self.last_v = nn.Conv2d(self.c_feat_list[-1][1], self.num_class, self.kernel_size, padding=int((self.kernel_size-1)/2))

        self.r_dp   = nn.Dropout2d(0.1)
        self.p_dp   = nn.Dropout(p=0.1)

    def forward(self, feat_list, v2p_ind, r2p_ind, verbose=False):
        feature_in  = feat_list.copy()
        feature_out = feat_list
        for i in range(self.n_layer):
            feature_out = self.att_block_list[i](feature_out, v2p_ind, r2p_ind)
        p_feat, v_feat, r_feat = feature_out
        p_feat_in, v_feat_in, r_feat_in = feature_in
        if self.sum_type == 'concat':
            p_feat_new = torch.cat([p_feat, p_feat_in], dim=1)
            r_feat_new = torch.cat([r_feat, r_feat_in], dim=1)
        elif self.sum_type == 'sum':
            p_feat_new = p_feat + p_feat_in
            r_feat_new = r_feat + r_feat_in

        pred_p = self.last_p( self.p_dp( p_feat_new))
        # pred_v = self.last_v(v_feat)
        pred_r = self.last_r( self.r_dp( r_feat_new))

        if verbose:
            print(pred_p.size(), pred_v.size(), pred_r.size())

        # return pred_p, pred_v, pred_r
        return pred_p, pred_r

class AttentGraph_MultiScale(nn.Module):
    def __init__(self, n_layer=1, n_head=2, c_node=64, c_emb=64, k=20):
        super(AttentGraph_MultiScale, self).__init__()
        self.c_node      = c_node
        self.c_emb       = c_emb
        self.num_class   = k
        self.n_head      = n_head
        self.n_layer     = n_layer
        self.kernel_size = 3
        self.c_feat_list = [[128, 64, 20]] + [[self.c_node, self.c_node, self.c_node]] * n_layer
        self.c_emb_list  = [self.c_emb] * n_layer

        self.att_block_list = nn.ModuleList()
        for i in range(n_layer):
            self.att_block_list.append( AttentGraph_Block( self.c_feat_list[i+1], self.c_feat_list[i], self.c_emb_list[i], self.n_head) )
        self.last_p = nn.Conv1d(self.c_feat_list[-1][0], self.num_class, 1)
        self.last_v = nn.Conv2d(self.c_feat_list[-1][1], self.num_class, self.kernel_size, padding=int((self.kernel_size-1)/2))
        self.last_r = nn.Conv2d(self.c_feat_list[-1][2], self.num_class, self.kernel_size, padding=int((self.kernel_size-1)/2))

    def forward(self, feat_list, v2p_ind, r2p_ind):
        feature_out = feat_list
        for i in range(self.n_layer):
            feature_out = self.att_block_list[i](feature_out, v2p_ind, r2p_ind)
        p_feat, v_feat, r_feat = feature_out

        pred_p = self.last_p(p_feat)
        pred_v = self.last_v(v_feat)
        pred_r = self.last_r(r_feat)

        return [pred_p, pred_v, pred_r]


# Resnet Blocks
class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class CResnetBlockConv1d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks
    '''

    def __init__(self, c_dim, size_in, size_h=None, size_out=None,
                 norm_method='batch_norm', legacy=False):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if not legacy:
            self.bn_0 = CBatchNorm1d(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d(
                c_dim, size_h, norm_method=norm_method)
        else:
            self.bn_0 = CBatchNorm1d_legacy(
                c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d_legacy(
                c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetBlockConv1d(nn.Module):
    ''' 1D-Convolutional ResNet block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_h=None, size_out=None):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.bn_0 = nn.BatchNorm1d(size_in)
        self.bn_1 = nn.BatchNorm1d(size_h)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)

        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(self.bn_0(x)))
        dx = self.fc_1(self.actvn(self.bn_1(net)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


# Utility modules
class AffineLayer(nn.Module):
    ''' Affine layer class.

    Args:
        c_dim (tensor): dimension of latent conditioned code c
        dim (int): input dimension
    '''

    def __init__(self, c_dim, dim=3):
        super().__init__()
        self.c_dim = c_dim
        self.dim = dim
        # Submodules
        self.fc_A = nn.Linear(c_dim, dim * dim)
        self.fc_b = nn.Linear(c_dim, dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fc_A.weight)
        nn.init.zeros_(self.fc_b.weight)
        with torch.no_grad():
            self.fc_A.bias.copy_(torch.eye(3).view(-1))
            self.fc_b.bias.copy_(torch.tensor([0., 0., 2.]))

    def forward(self, x, p):
        assert(x.size(0) == p.size(0))
        assert(p.size(2) == self.dim)
        batch_size = x.size(0)
        A = self.fc_A(x).view(batch_size, 3, 3)
        b = self.fc_b(x).view(batch_size, 1, 3)
        out = p @ A + b
        return out


class CBatchNorm1d(nn.Module):
    ''' Conditional batch normalization layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert(x.size(0) == c.size(0))
        assert(c.size(1) == self.c_dim)

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out

class CBatchNorm1d_legacy(nn.Module):
    ''' Conditional batch normalization legacy layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim, norm_method='batch_norm'):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.fc_gamma = nn.Linear(c_dim, f_dim)
        self.fc_beta = nn.Linear(c_dim, f_dim)
        if norm_method == 'batch_norm':
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == 'instance_norm':
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == 'group_norm':
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError('Invalid normalization method!')
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.ones_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x, c):
        batch_size = x.size(0)
        # Affine mapping
        gamma = self.fc_gamma(c)
        beta = self.fc_beta(c)
        gamma = gamma.view(batch_size, self.f_dim, 1)
        beta = beta.view(batch_size, self.f_dim, 1)
        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out

def add_ones(query_points, x, add_one):
    if add_one:
        ones = torch.ones(query_points.shape[0], dtype=torch.float).unsqueeze(-1).to(query_points.device)
        if x is not None:
            x = torch.cat([ones.to(x.dtype), x], dim=-1)
        else:
            x = ones
    return x

class BaseModule(nn.Module):
    """ Base module class with some basic additions to the pytorch Module class
    """
    @property
    def nb_params(self):
        """This property is used to return the number of trainable parameters for a given layer
        It is useful for debugging and reproducibility.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self._nb_params = sum([np.prod(p.size()) for p in model_parameters])
        return self._nb_params

class Identity(BaseModule):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, data):
        return data

def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i+1)
            new_s = list(x.size())
            new_s[i+1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i+n)
            new_s = list(idx.size())
            new_s[i+n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')

def max_pool(x, inds):
    """
    Pools features with the maximum values.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] pooling indices
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get all features for each pooling location [n2, max_num, d]
    pool_features = gather(x, inds)

    # Pool the maximum [n2, d]
    max_features, _ = torch.max(pool_features, 1)
    return max_features


def global_average(x, batch_lengths):
    """
    Block performing a global average over batch pooling
    :param x: [N, D] input features
    :param batch_lengths: [B] list of batch lengths
    :return: [B, D] averaged features
    """

    # Loop over the clouds of the batch
    averaged_features = []
    i0 = 0
    for b_i, length in enumerate(batch_lengths):

        # Average features for each batch cloud
        averaged_features.append(torch.mean(x[i0:i0 + length], dim=0))

        # Increment for next cloud
        i0 += length

    # Average features in each batch
    return torch.stack(averaged_features)

# outblock for rotation regression model
class GeneralOutBlockRT(nn.Module):
    def __init__(self, latent_dim=128, norm=None, pooling_method='max', global_scalar=False):
        super(GeneralOutBlockRT, self).__init__()

        c_in = 128
        mlp = [128, 128]
        self.temperature = 3
        self.representation = 'quat'

        self.global_scalar = global_scalar
        self.pooling_method = pooling_method
        self.linear = nn.ModuleList()
        self.norm   = nn.ModuleList()
        self.attention_layer = nn.Conv1d(mlp[-1], 1, (1))
        self.regressor_layer = nn.Conv1d(mlp[-1],4*60,(1))
        self.regressor_scalar_layer = nn.Conv1d(mlp[-1],1,(1)) # [B, C, A] --> [B, 1, A] scalar, local
        self.pointnet = nn.Sequential(nn.Conv1d(mlp[-1], mlp[-1], 1),
                                      nn.BatchNorm1d(mlp[-1]),
                                      nn.LeakyReLU(inplace=True)) # for Z
        # ------------------ uniary conv ----------------
        for c in mlp: #
            self.linear.append(nn.Conv1d(c_in, c, 1))
            self.norm.append(nn.BatchNorm1d(c))
            c_in = c

        # ----------------- dense regression ------------------
        self.regressor_dense_layer = nn.Sequential(nn.Conv1d(2* mlp[-1], mlp[-1], 1),
                                                 nn.BatchNorm1d(mlp[-1]),
                                                 nn.LeakyReLU(inplace=True),
                                                 nn.Conv1d(c, 3, 1)) # randomly

    def forward(self, feats, xyz=None):
        x_out = feats         # nb, nc, np
        end = len(self.linear)
        for lid, linear in enumerate(self.linear):
            x_out = linear(x_out)
            x_out = self.norm[lid](x_out)
            x_out = F.relu(x_out)

        shared_feat = x_out
        # 1. R prediction
        # mean pool at xyz ->  BxC
        if self.pooling_method == 'mean':
            x_out = x_out.mean(2)
        elif self.pooling_method == 'max':
            x_out = x_out.max(2)[0]
        x_out = x_out.unsqueeze(-1)
        attention_wts = self.attention_layer(x_out)  # Bx1XA
        confidence    = F.softmax(attention_wts * self.temperature, dim=2).view(x_out.shape[0], x_out.shape[2])
        y_r           = self.regressor_layer(x_out).view(xyz.shape[0], 4, -1) # Bx4xA

        # 2. t prediction
        t_out = self.regressor_dense_layer(torch.cat([x_out.repeat(1, 1, shared_feat.shape[2]).contiguous(), shared_feat], dim=1)) # dense branch
        y_t = self.regressor_scalar_layer(shared_feat) # [nb, 1, p]
        y_t = F.normalize(t_out, p=2, dim=1) * y_t     # scalar from invariant T
        y_t = y_t.mean(dim=2).unsqueeze(-1)


        # 3. Z prediction
        y_z = self.pointnet(shared_feat).max(dim=-1)[0]        # pooling over all points, [nb, nc]

        # regressor
        output = {}
        output['1'] = confidence #
        output['0'] = y_z #
        output['R'] = y_r
        output['T'] = y_t

        return output

if __name__ == '__main__':
    gpu = 0
    deploy_device   = torch.device('cuda:{}'.format(gpu))

    N    = 512
    T    = 3
    xyz2  = torch.rand(1, T, 3, N, requires_grad=False, device=deploy_device).float() # [BS, T, C, N] --> later [BS, C, T, N]
    xyz1  = xyz2[:, 0, :, :]
    feat2 = torch.rand(1, T, 1, N, requires_grad=False, device=deploy_device).float()

    # print('output is ', new_feat.size())
    print('Con!!!')
