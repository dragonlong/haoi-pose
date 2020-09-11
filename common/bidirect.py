"""
Author: Xiaolong Li;
email: lxiaol9@vt.edu
func: fetch and concatenaet feature vectors according to index
      in pytorch, index matrix
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import numpy as np
import __init__
from common.debugger import breakpoint
import global_info

setting = global_info.setting()
if setting.USE_MULTI_GPU:
    # BatchNorm2d = SynchronizedBatchNorm2d
    BatchNorm2d = nn.SyncBatchNorm
    BatchNorm1d = nn.SyncBatchNorm
else:
    BatchNorm2d = nn.BatchNorm2d
    BatchNorm1d = nn.BatchNorm1d

input_remap = {
          0: 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
          1: 1,    # "car"
          2: 1,    # "bicycle"
          3: 1,    # "motorcycle"
          4: 1,    # "truck"
          5: 1,    # "on-rails" mapped to "other-vehicle" ---------------------mapped
          6: 1,    # "person"
          7: 1,    # "bicyclist"
          8: 1,    # "motorcyclist"
          9: 0,    # "road"
          10: 0,    # "parking"
          11: 0,  # "sidewalk"
          12: 0,  # "other-ground"
          13: 0,  # "building"
          14: 0,  # "fence"
          15: 0,  # "vegetation"
          16: 0,  # "trunk"
          17: 0,  # "terrain"
          18: 0,   # "pole"
          19: 0,   # "traffic-sign"}
                }

def tensor_replace(x, lut_dict):
    with torch.no_grad():
        for key, value in lut_dict.items():
            x[x==key] = value

    return x

class shape_transformer(nn.Module):
    def __init__(self, c_in, c_out):
        super(shape_transformer, self).__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 1)
        self.bn1 = BatchNorm2d(c_out)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return x

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


	# # x,y index flatten
	# v2p_ind = torch.empty(32, 480, 480, dtype=torch.long, device=deploy_device).random_(0, 480*480)
	# b = 2
	# x = 10
	# y = 300
	# print('Target value is ', v2p_ind[b, x, y].item())
	# print('Target value is ', v2p_ind.view(32, -1)[b, x*480+y].item())

if __name__ == '__main__':
    # gpu = 0
    # # test.cuda(gpu)
    # deploy_device   = torch.device('cuda:{}'.format(gpu))
    # N = 1500
    # BS = 2
    # k  = 20
    # p_in= torch.rand(BS, 3, N, requires_grad=True, device=deploy_device)
    # v_in= torch.rand(BS, 42, 480, 480, requires_grad=True, device=deploy_device)
    # v_in1= torch.rand(BS, 3, 480, 480, 14, requires_grad=True, device=deploy_device)
    # r_in= torch.rand(BS, 5, 64, 2048, requires_grad=True, device=deploy_device)

    # p_feat= torch.rand(BS, 64  , N, requires_grad=True, device=deploy_device)
    # v_feat= torch.rand(BS, 64, 480, 480, requires_grad=True, device=deploy_device)
    # r_feat= torch.rand(BS, 64, 64, 2048, requires_grad=True, device=deploy_device)
    # y_c     = torch.empty(BS, 472, 472, dtype=torch.long, device=deploy_device).random_(0, k)
    # y_p     = torch.empty(BS, N, dtype=torch.long, device=deploy_device).random_(-1, k)
    # v2p_ind = torch.empty(BS, N, 3, dtype=torch.long).random_(0, 14)
    # r2p_ind = torch.empty(BS, N, 2, dtype=torch.long).random_(0, 64)

    # # s_t = time.time()
    # # feat_concat = feat_fusion(p_feat, v_feat, v2p_ind, r_feat, r2p_ind)
    # # print(f'feat_concat: {feat_concat.size()}, costing {time.time()-s_t} seconds for {N} points with batch size {32}')

    # p_model = pt_model().cuda(gpu)
    # r_model = rv_model().cuda(gpu)
    # v_model = bev_model().cuda(gpu)

    # p_feat_1 = p_model.forward(p_feat)
    # r_feat_1 = r_model.forward(r_feat)
    # v_feat_1 = v_model.forward(v_feat)
    # p2c_ind  = v2p_ind
    # v_feat_2 = p2c_map(p_feat_1, p2c_ind, v_feat_1, gpu=gpu)
    # print(v_feat_2.size())
    # v_feat_3 = p2c_map3d(p_feat_1, p2c_ind, v_in, gpu=gpu)    
    # print(v_feat_3.size())

    # pred  = v_feat_2
    # # print(y_p)
    # # fuse_model = SimpleMLP(k=k).cuda(gpu)
    # # pred = fuse_model([p_feat, v_feat, r_feat], v2p_ind, r2p_ind)
    # crit = nn.NLLLoss(ignore_index=-1).cuda(gpu)
    # loss = crit(pred, y_c)
    # print(loss)
    # loss.backward()
    # # print(p_model.conv1.weight.grad)
    # # print(r_model.conv1.weight.grad)
    # # print(v_model.conv1.weight.grad)

    N = 1500
    BS = 2
    tt = torch.empty(BS, N, 2, dtype=torch.long).random_(0, 20)
    tt = tensor_replace(tt, input_remap)
    print(tt)
	#
