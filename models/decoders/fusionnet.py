# This file was originally modified from https://github.com/BobLiu20/YOLOv3_PyTorch
# And it is recently adapted from https://github.com/PRBonn/lidar-bonnetal/blob/master/train/tasks/semantic/decoders/darknet.py
# We add feature merging from Point and VoxelNet branch;
#>>>>>>>>>>>>>>>>>>>>>log<<<<<<<<<<<<<<<<<<<<<<<<#
"""
1. we add more efficient feature injection operation in Pytorch, which create sparse pesudo-image tensor from 1d;
2. we fuse features from PointNet, and BeV branch; Yes!
3. we need feature propagation layer during decoding stage;  Yes!
4. we also try a tiny different attention mechanism(:learnable attention weights, 2F * (F+F'') ) Yes!
Note: as for aggregating point features into RV, we could actually use max pooling together with sampling to get voxel-wise or pixel-wise
corresponding features from point features, but now we just use a unique index, to help gather corresponding features between BeV & RV;
"""
#>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<#
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import imp
import yaml

import __init__
import global_info
from common.bidirect import c2p_map, p2c_map
from models.fusor import attent1d, maxpl1d, attent1d_spatial

setting = global_info.setting()
if setting.USE_MULTI_GPU:
  BatchNorm2d = nn.SyncBatchNorm
else:
  BatchNorm2d = nn.BatchNorm2d

def breakpoint():
    import pdb; pdb.set_trace()


class BasicBlock(nn.Module):
  def __init__(self, inplanes, planes, bn_d=0.1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                           stride=1, padding=0, bias=False)
    self.bn1 = BatchNorm2d(planes[0], momentum=bn_d)
    self.relu1 = nn.LeakyReLU(0.1)
    self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn2   = BatchNorm2d(planes[1], momentum=bn_d)
    self.relu2 = nn.LeakyReLU(0.1)

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu2(out)

    out += residual
    return out

class AttentionBlock(nn.Module):
  def __init__(self, c_outs=[64, 64, 64], c_ins=[128, 64, 20], c_emb=64, n_head=1):
    super(AttentionBlock, self).__init__()
    self.c_outs = c_outs
    self.c_ins = c_ins
    self.c_emb = c_emb
    self.n_head= n_head
    self.kernel_size = 3
    self.att1  = attent1d(self.c_ins, self.c_emb, n_head=self.n_head) # 128-align dim->-128-linear trans->64-attention->64

  def forward(self, feat_list, v2p_ind, r2p_ind, unique_indices=None, masks=None): # a little bit different here is that we adopt unique indices to select points
    p_feat, v_feat, r_feat = feat_list
    # extract all necessary features together with indexs, thus reducing computation head;
    if unique_indices is not None:
      bs, cp, _ = p_feat.size()
      p_feat = torch.gather(p_feat, 2, unique_indices.unsqueeze(1).repeat(1, cp, 1))
      with torch.no_grad():
        r2p_ind = torch.gather(r2p_ind, 1, unique_indices.unsqueeze(2).repeat(1, 1, 2))
        v2p_ind = torch.gather(v2p_ind[:, :, :2], 1, unique_indices.unsqueeze(2).repeat(1, 1, 2))

    v_feat_per_pt = c2p_map(v_feat.detach(), v2p_ind)
    r_feat_per_pt = c2p_map(r_feat.detach(), r2p_ind)
    feat_att  = self.att1([p_feat, v_feat_per_pt, r_feat_per_pt])
    xp, xv, xr= feat_att
    # refer to https://github.com/traveller59/second.pytorch/blob/3aba19c9688274f75ebb5e576f65cfe54773c021/second/pytorch/models/pointpillars.py
    # we could also reshape it to 1d, then assign it and reshape
    xv_mapped = p2c_map(xv, v2p_ind, v_feat)
    xr_mapped = p2c_map(xr, r2p_ind, r_feat) #
    # # feature transformation
    # xp = F.relu(self.bn1p(self.conv1p(xp)))
    # xv_mapped = F.relu(self.bn1v(self.conv1v(xv_mapped)))
    # xr_mapped = F.relu(self.bn1r(self.conv1r(xr_mapped)))

    return [xp, xv_mapped, xr_mapped]

class AttentionBlock2(nn.Module):
  def __init__(self, c_outs=[64, 64], c_ins=[64, 20], c_emb=64, n_head=1):
    super(AttentionBlock2, self).__init__()
    self.c_outs = c_outs
    self.c_ins = c_ins
    self.c_emb = c_emb
    self.n_head= n_head
    self.kernel_size = 3
    self.att1  = attent1d(self.c_ins, self.c_emb, n_head=self.n_head) # 128-align dim->-128-linear trans->64-attention->64

  def forward(self, feat_list, v2p_ind, r2p_ind, os=1, unique_indices=None, masks=None, verbose=False): # a little bit different here is that we adopt unique indices to select points
    v_feat, r_feat = feat_list
    bundle_size = max(1, int(os/2))
    if verbose:
      print('bundle size is ', bundle_size)
    # extract all necessary features together with indexs, thus reducing computation head;
    if unique_indices is not None:
      with torch.no_grad():
        r2p_ind = torch.gather(r2p_ind, 1, unique_indices[:, ::bundle_size].unsqueeze(2).repeat(1, 1, 2)) # b, n, 2
        v2p_ind = torch.gather(v2p_ind[:, :, :2], 1, unique_indices.unsqueeze(2).repeat(1, 1, 2))

    v_feat_per_pt = c2p_map(v_feat.detach(), v2p_ind)
    bs, cs, n = v_feat_per_pt.size()
    v_feat_nodes  = torch.split(v_feat_per_pt.view(bs, cs, int(n/bundle_size), bundle_size), 1, dim=-1)
    v_feat_nodes  = [feat_split.squeeze() for feat_split in v_feat_nodes]
    r_feat_per_pt = c2p_map(r_feat.detach(), r2p_ind)
    # breakpoint()
    feat_att  = self.att1(list(v_feat_nodes) + [r_feat_per_pt])
    xr= feat_att[-1]
    # refer to https://github.com/traveller59/second.pytorch/blob/3aba19c9688274f75ebb5e576f65cfe54773c021/second/pytorch/models/pointpillars.py
    # we could also reshape it to 1d, then assign it and reshape
    # xv_mapped = p2c_map(xv, v2p_ind, v_feat)
    xr_mapped = p2c_map(xr, r2p_ind, r_feat) #

    return [None, xr_mapped]

class AttentionBlock3(nn.Module): # add coord for attention
  def __init__(self, c_outs=[128, 128], c_ins=[512, 512], c_emb=128, n_head=2):
    super(AttentionBlock3, self).__init__()
    self.c_outs = c_outs
    self.c_ins = c_ins
    self.c_emb = c_emb
    self.n_head= n_head
    self.kernel_size = 3
    self.att1  = attent1d_spatial(self.c_ins, self.c_emb, n_head=self.n_head) # 128-align dim->-128-linear trans->64-attention->64

  def forward(self, feat_list, v2p_ind, r2p_ind, os=1, unique_indices=None, masks=None, verbose=False): # a little bit different here is that we adopt unique indices to select points
    v_feat, r_feat = feat_list
    bundle_size = max(1, int(os/2))
    if verbose:
      print('bundle size is ', bundle_size)
    # extract all necessary features together with indexs, thus reducing computation head;
    if unique_indices is not None:
      with torch.no_grad():
        r2p_ind = torch.gather(r2p_ind, 1, unique_indices[:, ::bundle_size].unsqueeze(2).repeat(1, 1, 2)) # b, n, 2
        v2p_ind = torch.gather(v2p_ind[:, :, :2], 1, unique_indices.unsqueeze(2).repeat(1, 1, 2)) # b, N, 2

    v_feat_per_pt = c2p_map(v_feat.detach(), v2p_ind)
    bs, cs, n = v_feat_per_pt.size()
    v_feat_nodes  = torch.split(v_feat_per_pt.view(bs, cs, int(n/bundle_size), bundle_size), 1, dim=-1)
    v_feat_nodes  = [feat_split.squeeze() for feat_split in v_feat_nodes]
    r_feat_per_pt = c2p_map(r_feat.detach(), r2p_ind)
    # breakpoint()
    K_v2p_ind = v2p_ind.view(bs, int(n/bundle_size), bundle_size, 2)
    feat_att  = self.att1(list(v_feat_nodes) + [r_feat_per_pt], K_v2p_ind)
    xr= feat_att[-1]
    # refer to https://github.com/traveller59/second.pytorch/blob/3aba19c9688274f75ebb5e576f65cfe54773c021/second/pytorch/models/pointpillars.py
    # we could also reshape it to 1d, then assign it and reshape
    # xv_mapped = p2c_map(xv, v2p_ind, v_feat)
    xr_mapped = p2c_map(xr, r2p_ind, r_feat) #

    return [None, xr_mapped]

class MaxpoolBlock(nn.Module):
  def __init__(self, c_outs=[128, 128], c_ins=[512, 512], c_emb=128, n_head=2):
    super(MaxpoolBlock, self).__init__()
    self.c_outs = c_outs
    self.c_ins = c_ins
    self.c_emb = c_emb
    self.n_head= n_head
    self.kernel_size = 3
    self.maxpl1  = maxpl1d(self.c_ins, self.c_emb, n_head=self.n_head) # 128-align dim->-128-linear trans->64-attention->64

  def forward(self, feat_list, v2p_ind, r2p_ind, os=1, unique_indices=None, masks=None, verbose=False): # a little bit different here is that we adopt unique indices to select points
    v_feat, r_feat = feat_list
    bundle_size = max(1, int(os/2))
    if verbose:
      print('bundle size is ', bundle_size)
    # extract all necessary features together with indexs, thus reducing computation head;
    if unique_indices is not None:
      with torch.no_grad():
        r2p_ind = torch.gather(r2p_ind, 1, unique_indices[:, ::bundle_size].unsqueeze(2).repeat(1, 1, 2)) # b, n, 2
        v2p_ind = torch.gather(v2p_ind[:, :, :2], 1, unique_indices.unsqueeze(2).repeat(1, 1, 2)) # 

    v_feat_per_pt = c2p_map(v_feat.detach(), v2p_ind)
    bs, cs, n = v_feat_per_pt.size()
    v_feat_nodes  = torch.split(v_feat_per_pt.view(bs, cs, int(n/bundle_size), bundle_size), 1, dim=-1)
    v_feat_nodes  = [feat_split.squeeze() for feat_split in v_feat_nodes]
    r_feat_per_pt = c2p_map(r_feat.detach(), r2p_ind)
    # breakpoint()
    feat_att  = self.maxpl1(list(v_feat_nodes) + [r_feat_per_pt])
    xr= feat_att[-1]
    # refer to https://github.com/traveller59/second.pytorch/blob/3aba19c9688274f75ebb5e576f65cfe54773c021/second/pytorch/models/pointpillars.py
    # we could also reshape it to 1d, then assign it and reshape
    # xv_mapped = p2c_map(xv, v2p_ind, v_feat)
    xr_mapped = p2c_map(xr, r2p_ind, r_feat) #

    return [None, xr_mapped]

# ******************************************************************************
class Decoder(nn.Module):
  """
     Class for DarknetSeg. Subclasses PyTorch's own "nn" module
  """
  def __init__(self, params, stub_skips, OS=32, feature_depth=1024, num_att=3, att_type='max_pool', sum_type='concat'):
    super(Decoder, self).__init__()
    self.backbone_OS = OS
    self.backbone_feature_depth = feature_depth
    self.drop_prob = params["dropout"]
    self.bn_d = params["bn_d"]
    self.num_att = num_att
    self.sum_type = sum_type
    self.att_type = att_type

    # stride play
    self.strides = [2, 2, 2, 2, 2]
    self.num_layer = len(self.strides)
    # check current stride
    current_os = 1
    for s in self.strides:
      current_os *= s
    print("Decoder original OS: ", int(current_os))
    # redo strides according to needed stride
    for i, stride in enumerate(self.strides):
      if int(current_os) != self.backbone_OS:
        if stride == 2:
          current_os /= 2
          self.strides[i] = 1
        if int(current_os) == self.backbone_OS:
          break
    print("Decoder new OS: ", int(current_os))
    print("Decoder strides: ", self.strides)

    # self.layers = [self.dec5, self.dec4, self.dec3, self.dec2, self.dec1]
    self.dec=nn.ModuleList()
    self.att=nn.ModuleList()
    self.mlp=nn.ModuleList()
    c_out = [self.backbone_feature_depth, 512, 256, 128, 64, 32]
    for i in range(len(self.strides)):
      decode_layer = self._make_dec_layer(BasicBlock, [c_out[i], c_out[i+1]], bn_d=self.bn_d,
                                          stride=self.strides[i])
      self.dec.append(decode_layer)

    c_in  = [512, 256, 128, 64, 32]
    if self.sum_type == 'concat':
      c_eb  = [128, 128, 128, 128, 128]
    else:
      c_eb  = [512, 256, 128, 64, 32]
    for i in range(self.num_att):
      c_ins = [c_in[i]] * 2 # TODO
      if self.att_type == 'inner_prod':
        print('adopting attention mechanism!!!')
        att_layer = AttentionBlock3(c_ins=c_ins, c_emb=c_eb[i], n_head=2) # TODO
      elif self.att_type == 'max_pool':
        print('adopting simple max_pool operation!!!')
        att_layer = MaxpoolBlock(c_ins=c_ins, c_emb=c_eb[i], n_head=2) # TODO
      mlp_layer = []
      mlp_layer.append(("conv", nn.Conv2d(c_in[i]+c_eb[i], c_in[i], kernel_size=1)))
      mlp_layer.append(("bn", BatchNorm2d(c_in[i], momentum=self.bn_d)))
      mlp_layer.append(("relu", nn.LeakyReLU(0.1)))
      self.att.append(att_layer)
      self.mlp.append(nn.Sequential(OrderedDict(mlp_layer)))

    # for a bit of fun
    self.dropout = nn.Dropout2d(self.drop_prob)

    # last channels
    self.last_channels = 32

  def _make_dec_layer(self, block, planes, bn_d=0.1, stride=2):
    layers = []
    #  downsample
    if stride == 2:
      layers.append(("upconv", nn.ConvTranspose2d(planes[0], planes[1],
                                                  kernel_size=[1, 4], stride=[1, 2],
                                                  padding=[0, 1])))
    else:
      layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                       kernel_size=3, padding=1)))
    layers.append(("bn", BatchNorm2d(planes[1], momentum=bn_d)))
    layers.append(("relu", nn.LeakyReLU(0.1)))

    #  blocks
    layers.append(("residual", block(planes[1], planes, bn_d)))

    return nn.Sequential(OrderedDict(layers))

  def run_layer(self, x, layer, skips, os):
    feats = layer(x)  # up
    if feats.shape[-1] > x.shape[-1]:
      os //= 2  # match skip
      feats = feats + skips[os].detach()  # add skip
    x = feats
    return x, skips, os

  def run_layer_att(self, x, layer, att_layer, mlp_layer, skips, neighbors,  v2p_ind, r2p_ind, layer_index, os, channels, verbose=False):
    feats = layer(x)  # up
    if feats.shape[-1] > x.shape[-1]:
      os //= 2  # match skip
      feats = feats + skips[os].detach()  # add skip
      if verbose: 
        print('feature has os {} for output channels {}'.format(os, channels))
    # add attention module
    x = feats
    feat_in = neighbors + [x]
    #>>>>>>>>>>>> attention
    _, r_feat = att_layer(feat_in, v2p_ind, r2p_ind, os, layer_index)
    _, r_feat_in = feat_in
    #>>>>>>>>>>>> prepare for output
    if self.sum_type == 'concat':
        # print('we are concating the feature for ', channels*2)
        r_feat_new = torch.cat([r_feat, r_feat_in], dim=1)
        x = mlp_layer(r_feat_new)
    elif self.sum_type == 'sum':
        if verbose: 
          print('add attention features back')
        x= r_feat + r_feat_in
    channels //=2

    return x, skips, os, channels

  def forward(self, x, skips, nodes, v2p_ind, r2p_ind, index_map, return_feature_maps=False, verbose=False):
    os = self.backbone_OS
    channels = 512
    os_v = [8, 4, 2, 1]
    os_r = [16, 8, 4, 2, 1]
    for i in range(self.num_layer):
      if i < self.num_att:
        layer_nodes = []
        for k in range(len(nodes)):
          layer_nodes.append(nodes[0][channels]) # point, bev
        if verbose:
          print('now RV feat has os: ', os)
        layer_index = index_map[os_r[i]] # os
        v2p_ind_scale = v2p_ind/os_v[i]
        r2p_ind_scale = r2p_ind/os_r[i]
        x, skips, os, channels = self.run_layer_att(x, self.dec[i], self.att[i], self.mlp[i], skips, layer_nodes, v2p_ind_scale.long(), r2p_ind_scale.long(), layer_index, os, channels)
      else:
        if i==3:
          last_feat = x
        if verbose:
          print('now RV feat has os: ', os)
        x, skips, os = self.run_layer(x, self.dec[i], skips, os)

    x = self.dropout(x)
    if return_feature_maps:
        return x, last_feat
    else:
        return x

  def get_last_depth(self):
    return self.last_channels


if __name__ == '__main__':
  ARCH = yaml.safe_load(open('../config/arch/darknet21.yaml', 'r'))
  bboneModule = imp.load_source("bboneModule", '../backbones/' + ARCH["backbone"]["name"] + '.py')
  backbone = bboneModule.Backbone(params=ARCH["backbone"])

  # do a pass of the backbone to initialize the skip connections
  stub = torch.zeros((1,
                      backbone.get_input_depth(),
                      ARCH["dataset"]["sensor"]["img_prop"]["height"],
                      ARCH["dataset"]["sensor"]["img_prop"]["width"]))

  if torch.cuda.is_available():
    stub = stub.cuda()
    backbone.cuda()
  _, stub_skips = backbone(stub)
  print(list(stub_skips.keys()))

  decoder = Decoder(params=ARCH["decoder"], stub_skips=stub_skips,
                    OS=ARCH["backbone"]["OS"],
                    feature_depth=backbone.get_last_depth())
  # print('decoder params: \n', decoder)
