import numpy as np
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_

import __init__

def breakpoint():
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