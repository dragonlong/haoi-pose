import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

# simply define a silu function
def silu(input):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    '''
    return input * torch.sigmoid(input) # use torch.sigmoid to make sure that we created the most efficient implemetation based on builtin PyTorch functions

def bp():
    import pdb;pdb.set_trace()

class MLP(nn.Module):
    def __init__(self, dim, in_channel, mlp, use_bn=True, skip_last=True, last_acti=None):
        super(MLP, self).__init__()
        layers = []
        conv = nn.Conv1d if dim == 1 else nn.Conv2d
        bn = nn.BatchNorm1d if dim == 1 else nn.BatchNorm2d
        last_channel = in_channel
        for i, out_channel in enumerate(mlp):
            layers.append(conv(last_channel, out_channel, 1))
            if use_bn and (not skip_last or i != len(mlp) - 1):
                layers.append(bn(out_channel))
            if (not skip_last or i != len(mlp) - 1):
                layers.append(nn.ReLU())
            last_channel = out_channel
        if last_acti is not None:
            if last_acti == 'softmax':
                layers.append(nn.Softmax(dim=1))
            elif last_acti == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                assert 0, f'Unsupported activation type {last_acti}'
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class SiLU(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf
    Examples:
        >>> m = silu()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return silu(input) # simply apply already implemented SiLU


def knn(x, k):
    x_transpose = x.transpose(1, 2).contiguous()
    pairwise_distance = torch.cdist(x_transpose, x_transpose)
    #inner = -2*torch.matmul(x.transpose(2, 1), x)
    #xx = torch.sum(x**2, dim=1, keepdim=True)
    #pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k+1, dim=-1)[1][:, :, 1:]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = (feature-x).permute(0, 3, 1, 2).contiguous()

    return feature

def get_equivariant_graph_feature(x, f, k=20, idx=None):
    """
        x: [batch_size, 3, num_points]
        f: [batch_size, num_dims, num_points]
        k: integer
        idx: [batch_size, num_points, k]

    """
    batch_size = x.size(0)
    num_points = x.size(2)
    num_dims = f.size(1)

    x = x.view(batch_size, -1, num_points)
    num_channels = x.size(1)

    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous() #(batch_size, num_points, num_channels)
    x_neighbours = x.view(batch_size*num_points, num_channels)[idx, :]
    x_neighbours = x_neighbours.view(batch_size, num_points, k, num_channels)
    x = x.view(batch_size, num_points, 1, num_channels).repeat(1, 1, k, 1)
    x_diff = x_neighbours - x
    x_diff_square = torch.sum(x_diff.pow(2),dim=-1, keepdim=True)
    x_diff = x_diff.permute(0, 3, 1, 2).contiguous()

    f = f.transpose(2, 1).contiguous() #(batch_size, num_points, num_dims)
    f_neighbours = f.view(batch_size*num_points, num_dims)[idx, :]
    f_neighbours = f_neighbours.view(batch_size, num_points, k, num_dims)
    f = f.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # (batch_size, num_points, k, 2*num_dims + 1) -> (batch_size, 2*num_dims + 1, num_points, k)
    feature = torch.cat((f_neighbours-f, f, x_diff_square), dim=3).permute(0, 3, 1, 2).contiguous()
    # (batch_size, num_points, k, 2*num_dims + 1) -> (batch_size, 3, num_points, k)

    return feature, x_diff


class EquivariantDGCNN(nn.Module):
    def __init__(self, k=16, C=1, C_in=1, C_out=2, C_mid=64, num_mode=1, depth=4):
        super(EquivariantDGCNN, self).__init__()
        self.k = k # nsamples
        self.C = C # output x
        self.C_in  = C_in
        self.C_out = C_out

        self.e_conv1 = nn.Sequential(nn.Conv2d(self.C_in*2+1, 64, kernel_size=1, bias=True),
                                   SiLU())
        self.e_conv2 = nn.Sequential(nn.Conv2d(64*2+1, 64, kernel_size=1, bias=True),
                                   SiLU())
        self.e_conv3 = nn.Sequential(nn.Conv2d(64*2+1, 64, kernel_size=1, bias=True),
                                   SiLU())
        self.e_conv4 = nn.Sequential(nn.Conv2d(64*2+1, 64, kernel_size=1, bias=True),
                                   SiLU())

        self.x_conv1 = nn.Sequential(nn.Conv2d(64, self.C, kernel_size=1, bias=True),
                                   SiLU())
        self.x_conv2 = nn.Sequential(nn.Conv2d(64, self.C, kernel_size=1, bias=True),
                                   SiLU())
        self.x_conv3 = nn.Sequential(nn.Conv2d(64, self.C, kernel_size=1, bias=True),
                                   SiLU())
        self.x_conv4 = nn.Sequential(nn.Conv2d(64, self.C_out, kernel_size=1, bias=True),
                                   SiLU())

        self.f_conv1 = nn.Sequential(nn.Conv1d(self.C_in+64, 64, kernel_size=1, bias=True),
                                   SiLU())
        self.f_conv2 = nn.Sequential(nn.Conv1d(64*2, 64, kernel_size=1, bias=True),
                                   SiLU())
        self.f_conv3 = nn.Sequential(nn.Conv1d(64*2, 64, kernel_size=1, bias=True),
                                   SiLU())
        self.f_conv4 = nn.Sequential(nn.Conv1d(64*2, 64, kernel_size=1, bias=True),
                                   SiLU())
        head_cfg = {
            "N": [128, 3, 'sigmoid'],
            "M": [128, num_mode, 'softmax'],
        }
        self.heads  = nn.ModuleDict()
        for key, mlp in head_cfg.items():
            self.heads[key] = MLP(dim=1, in_channel=C_mid, mlp=mlp[:-1], use_bn=True, skip_last=True, last_acti=mlp[-1])
        # self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=args.dropout)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=args.dropout)
        # self.linear3 = nn.Linear(256, C_outhannels)

    def forward(self, pts):
        batch_size = pts.size(0)
        num_points = pts.size(2)

        x = pts[:, :3, :]
        if pts.shape[1] > 3:
            f = pts[:, 3:, :]
        else:
            f = torch.ones(pts.shape[0], 1, pts.shape[-1], device=pts.device)

        feature_1, x_diff_1 = get_equivariant_graph_feature(x, f, k=self.k)
        x_c  = x.unsqueeze(1).repeat(1, self.C, 1, 1).contiguous() # batch_size, C, 3, num_points
        x_diff_1_equi =  x_diff_1.unsqueeze(1).repeat(1, self.C, 1, 1, 1) # batch_size, C, 3, num_points, k

        m1 = self.e_conv1(feature_1)
        phi_x_1 = self.x_conv1(m1).unsqueeze(2).repeat(1, 1, 3, 1, 1)# batch_size, C, 3, num_points, k
        x1 = x_c + torch.mean(x_diff_1_equi * phi_x_1, dim=-1)# batch_size, C, 3, num_points
        f1 = self.f_conv1(torch.cat((f,torch.sum(m1, dim=-1)), dim=1))# batch_size, 64, num_points
        x1_out = x1.view(batch_size, -1, num_points).contiguous()

        feature_2, x_diff_2 = get_equivariant_graph_feature(x1_out, f1, k=self.k)
        x_diff_2_equi = x_diff_2.view(batch_size, self.C, 3, num_points, self.k)
        m2 = self.e_conv2(feature_2)
        phi_x_2 = self.x_conv2(m2).unsqueeze(2).repeat(1, 1, 3, 1, 1)# batch_size, C, 3, num_points, k
        x2 = x1 + torch.mean(x_diff_2_equi * phi_x_2, dim=-1)# batch_size, C, 3, num_points
        f2 = self.f_conv2(torch.cat((f1, torch.sum(m2, dim=-1)), dim=1))# batch_size, 64, num_points
        x2_out = x2.view(batch_size, -1, num_points).contiguous()

        feature_3, x_diff_3 = get_equivariant_graph_feature(x2_out, f2, k=self.k)
        x_diff_3_equi = x_diff_3.view(batch_size, self.C, 3, num_points, self.k)
        m3 = self.e_conv3(feature_3)
        phi_x_3 = self.x_conv3(m3).unsqueeze(2).repeat(1, 1, 3, 1, 1)# batch_size, C, 3, num_points, k
        x3 = x2 + torch.mean(x_diff_3_equi * phi_x_3, dim=-1)# batch_size, C, 3, num_points
        f3 = self.f_conv3(torch.cat((f2, torch.sum(m3, dim=-1)), dim=1))# batch_size, 64, num_points
        x3_out = x3.view(batch_size, -1, num_points).contiguous()

        feature_4, x_diff_4 = get_equivariant_graph_feature(x3_out, f3, k=self.k)
        x_diff_4_equi = x_diff_4.view(batch_size, self.C, 3, num_points, self.k)
        x_diff_4_equi = torch.mean(x_diff_4.view(batch_size, self.C_out, -1, 3, num_points, self.k), dim=2)
        x_3_equi = torch.mean(x3.view(batch_size, self.C_out, -1, 3, num_points), dim=2)
        m4 = self.e_conv4(feature_4)
        phi_x_4 = self.x_conv4(m4).unsqueeze(2).repeat(1, 1, 3, 1, 1)# batch_size, C_out, 3, num_points, k
        x4 = x_3_equi + torch.mean(x_diff_4_equi * phi_x_4, dim=-1)# batch_size, C_out, 3, num_points
        f4 = self.f_conv4(torch.cat((f3, torch.sum(m4, dim=-1)), dim=1))# batch_size, 64, num_points
        x4_out = x4.view(batch_size, -1, num_points).contiguous()

        #return x1, f1, x_diff_1, phi_x_1
        #return x2, f2, x_diff_2, phi_x_2
        #return x3, f3, x_diff_3, phi_x_3
        bp()
        pred_dict = {}
        pred_dict.update({'R': x4_out, 'T': x4_out})
        for key, head in self.heads.items():
            pred_dict[key] = head(f.permute(0, 2, 1).contiguous())

        return pred_dict #, x_diff_4, phi_x_4

if __name__ == '__main__':
    device = torch.device("cuda")
    batch_size = 15
    num_points = 1024
    k = 16
    C = 4
    C_in  = 1
    C_out = 1 ## for 6D rotation, use 2; for 3D rotation, use 1.

    torch.manual_seed(0)
    default_type = torch.DoubleTensor
    # default_type = torch.FloatTensor
    torch.set_default_tensor_type(default_type)
    j = 0
    # g_raw, n_arr, instance_name, up_axis, center_offset, idx, category_name = dset.__getitem__(j)
    # gt_points = n_arr
    # input = g_raw.ndata['x'].unsqueeze(0).to(device)

    pts = torch.randn(batch_size, 4, num_points).to(device)
    x_old = pts[:, :3, :]
    x = x_old/torch.norm(x_old, dim=1, keepdim=True)
    f = pts[:, 3:, :]
    pts = torch.cat((x, f), dim=1)

    # idx = knn(x, k=k)
    # idx_new = knn(x+3, k=k)
    # idx_diff = (idx-idx_new).type(torch.FloatTensor)
    # print(torch.max(idx_diff))


    model = EquivariantDGCNN(k, C, C_in, C_out).to(device)
    x1, f1 = model(pts)
    #x1, f1, x1_diff, phi_x_1 = model(pts)
    #x1 = x1.view(batch_size, C, 3, num_points)
    #x1 = torch.mean(x1, dim=1)

    print('Test translation equivariance')
    translation = 3
    translated_pts = torch.cat((x+translation, f), dim=1)
    x2, f2= model(translated_pts)
    #x2, f2, x2_diff, phi_x_2= model(translated_pts)
    # print('x_diff max:', torch.max(torch.abs(x2_diff - x1_diff)))
    # print('x_diff mean', torch.mean(torch.abs(x2_diff - x1_diff)))
    # print('phi_x max:', torch.max(torch.abs(phi_x_1 - phi_x_2)))
    # print('phi_x mean:', torch.mean(torch.abs(phi_x_1 - phi_x_2)))
    x_diff = x2 - (x1+translation)
    f_diff = f2 - f1
    max_x_diff_translation = torch.max(torch.abs(x_diff))
    max_f_diff_translation = torch.max(torch.abs(f_diff))
    print('x_diff max: ', max_x_diff_translation)
    print('f_diff max: ', max_f_diff_translation)

    print('\n')
    print('---Test rotation equivariance')
    #rot = R.random(random_state=1234).as_matrix()
    rot = np.array([[0.5, np.sqrt(3)/2, 0], [-np.sqrt(3)/2, 0.5, 0], [0, 0, 1]])
    rot = torch.from_numpy(rot).type(default_type).to(device)
    rot = rot.unsqueeze(0).repeat(batch_size, 1, 1)
    x_rotated = torch.matmul(rot, x)

    rotated_pts = torch.cat((x_rotated, f), dim=1)
    x3, f3 = model(rotated_pts)

    rot = rot.unsqueeze(1).repeat(1, C_out, 1, 1)
    x1_rotated = torch.matmul(rot, x1)
    x_diff_rotation = x3 - x1_rotated
    f_diff_rotation = f3 - f1

    print('x diff mean:', torch.mean(torch.abs(x_diff_rotation)))
    print('x diff max:', torch.max(torch.abs(x_diff_rotation)))
    print('f diff mean:', torch.mean(torch.abs(f_diff_rotation)))
    print('f diff max:', torch.max(torch.abs(f_diff_rotation)))
