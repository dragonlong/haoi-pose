import sys
import time
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_

from omegaconf import DictConfig, ListConfig
from kaolin.models.PointNet2 import group_gather_by_index
from kaolin.models.PointNet2 import fps_gather_by_index
from kaolin.models.PointNet2 import furthest_point_sampling
from kaolin.models.PointNet2 import three_nn
from kaolin.models.PointNet2 import three_interpolate

# from torch_points3d.core.spatial_ops import RadiusNeighbourFinder
# from torch_points3d.core.data_transform import GridSampling
import __init__
from modules.sampling import SampleK, Group
from modules.layers import Identity, BaseModule, MLP
from common.pytorch_utils import SharedMLP

def breakpoint():
    import pdb;pdb.set_trace()

def is_list(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig)

def is_iterable(entity):
    return isinstance(entity, list) or isinstance(entity, ListConfig) or isinstance(entity, tuple)

def is_dict(entity):
    return isinstance(entity, dict) or isinstance(entity, DictConfig)

def get_activation(act_opt, create_cls=True):
    if is_dict(act_opt):
        act_opt = dict(act_opt)
        act = getattr(torch.nn, act_opt["name"])
        del act_opt["name"]
        args = dict(act_opt)
    else:
        act = getattr(torch.nn, act_opt)
        args = {}

    if create_cls:
        return act(**args)
    else:
        return act

class PointNetMSGDown3d(nn.Module):
    def __init__(
        self,
        npoint,
        radii,
        nsample,
        module_name='qunima',
        down_conv_nn=None,
        window_size=3,
        layer_size=1, # number of MLP in spatial kernel
        bn=True,
        activation="LeakyReLU",
        use_xyz=True,
        use_time=False,
        knn=True,
        share=True,
        scale_idx=0,
        padding=0,
        kernel_type='pointnet++',
        **kwargs
    ):
        super(PointNetMSGDown3d, self).__init__()
        self.scale_idx = scale_idx # as name
        self.npoint = npoint
        self.nsample= nsample
        self.window_size= window_size
        self.layer_size = layer_size
        self.kernel_type= kernel_type
        self.padding = padding

        self.radii  = radii
        self.knn    = knn
        self.share  = share

        self.sample = Sample(npoint, fps=True)
        self.blocks = nn.ModuleList()
        self.conv3d = nn.ModuleList()
        self.unarys = nn.ModuleList()
        self.neighbors_sample_list = nn.ModuleList()
        if self.kernel_type == 'attention':
            self.pos_nn = nn.ModuleList()
            self.att_nn = nn.ModuleList()

        self.use_xyz = use_xyz
        self.use_time= use_time
        print(f'scale block {self.scale_idx}, knn: {knn}, sample: {nsample}, padding: {self.padding}')
        for i in range(len(radii)):
            in_channels=eval(down_conv_nn[i][0])
            if self.use_time:
                in_channels+=1
            out_channels=down_conv_nn[i][1:]
            neighbors_sample = SampleK(radii[i], nsample[i], knn=knn)
            self.neighbors_sample_list.append(neighbors_sample)
            out_channels = [in_channels, *out_channels]
            if self.kernel_type == 'pointnet++':

                layers = []
                j = 1
                if self.share:
                    # >>>>>>>>>> same MLP <<<<<<<<< #
                    while j < len(out_channels):
                        inc = 1
                        layers+=[nn.Conv2d(out_channels[j - 1], out_channels[j], 1, bias=True), nn.BatchNorm2d(out_channels[j], eps=0.001), nn.LeakyReLU()]
                        j+=inc
                    layers = nn.Sequential(*layers)
                    self.blocks.append(layers)
                    # >>>>>>>>>> end here <<<<<<<<<<< #
                else:
                    # >>>>>>> different MLP <<<<<<< #
                    while j < len(out_channels) - 1:
                        inc = 1
                        layers+=[nn.Conv2d(out_channels[j - 1], out_channels[j], 1, bias=True), nn.BatchNorm2d(out_channels[j], eps=0.001), nn.LeakyReLU()]
                        j+=inc
                    layers = nn.Sequential(*layers)
                    self.blocks.append(layers)
                    for j in range(self.window_size):
                        self.conv3d.append(nn.Sequential(nn.Conv2d(out_channels[-2], out_channels[-1], 1, bias=True), nn.BatchNorm2d(out_channels[-1], eps=0.001), nn.LeakyReLU()))
                    # >>>>>>> end here <<<<<<<<<<<< #

            elif self.kernel_type == 'attentative':
                # position
                layers = []
                mlp_pos = [10, 16, 32]
                for j in range(1, len(mlp_pos)):
                    layers+=[nn.Conv2d(mlp_pos[j - 1], mlp_pos[j], 1, bias=True), nn.BatchNorm2d(mlp_pos[j], eps=0.001), nn.LeakyReLU(0.2)]
                self.pos_nn.append(nn.Sequential(*layers))

                # attention
                layers  = []
                mlp_att = [out_channels[0]+32-3, 2*(out_channels[0]+32-3), out_channels[0]+32-3]
                for j in range(1, len(mlp_att)):
                    layers+=[nn.Conv2d(mlp_att[j - 1], mlp_att[j], 1, bias=True), nn.BatchNorm2d(mlp_att[j], eps=0.001), nn.LeakyReLU(0.2)]
                self.att_nn.append(nn.Sequential(*layers))

                # spatial pooling
                layers  = []
                out_channels[0] = out_channels[0]+32-3
                for j in range(1, len(out_channels)):
                    layers+=[nn.Conv1d(out_channels[j - 1], out_channels[j], 1, bias=True), nn.BatchNorm1d(out_channels[j], eps=0.001), nn.LeakyReLU(0.2)]
                self.blocks.append(nn.Sequential(*layers))

            elif self.kernel_type == 'attention':
                # attention
                layers  = []
                mlp_att = [out_channels[0], 64, 32]
                for j in range(1, len(mlp_att)):
                    layers+=[nn.Conv2d(mlp_att[j - 1], mlp_att[j], 1, bias=True), nn.BatchNorm2d(mlp_att[j], eps=0.001), nn.LeakyReLU(0.2)]
                self.att_nn.append(nn.Sequential(*layers))

                # spatial pooling
                layers          = []
                out_channels[0] = out_channels[0]
                for j in range(1, len(out_channels)):
                    layers+=[nn.Conv1d(out_channels[j - 1], out_channels[j], 1, bias=True), nn.BatchNorm1d(out_channels[j], eps=0.001), nn.LeakyReLU(0.2)]
                self.blocks.append(nn.Sequential(*layers))

            # self.unarys.append(nn.Sequential(
            #         nn.Conv1d(out_channels[-1]*self.window_size, out_channels[-1], 1, bias=True),
            #         nn.BatchNorm1d(out_channels[-1], eps=0.001),
            #         nn.LeakyReLU()
            #     ))
    def forward(self, xyz, times, feat, verbose=False): # x, pos, new_pos, radius_idx, scale_idx
        """
           xyz:   [B, T, 3, N]
           times: [B, T, 1, N]
           feat: original input for feature: [B, 1, T, N]
        """
        # x = torch.cat([xyz.permute(0, 2, 1, 3).contiguous(), times.permute(0, 2, 1, 3).contiguous(), feat], dim=1)
        B = xyz.size(0)
        T = xyz.size(1)
        N = xyz.size(3)
        x = feat
        xyz_ind = self.sample( xyz.view(-1, 3, N).contiguous() ) # FPS to get exact number of target points
        xyz1    = fps_gather_by_index(xyz.view(-1, 3, N).contiguous(), xyz_ind)
        t_flag  = fps_gather_by_index(times.view(-1, 1, N).contiguous(), xyz_ind) # gather by index
        feat1   = fps_gather_by_index(x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), N).contiguous(), xyz_ind)

        xyz1    = xyz1.view(-1, T, 3, xyz1.size(-1)).contiguous()
        t_flag  = t_flag.view(-1, T, 1, t_flag.size(-1)).contiguous()
        feat1   = feat1.view(-1, T, x.size(1), feat1.size(-1)).contiguous().view(B, x.size(1), -1).contiguous()
        points  = xyz1.permute(0, 2, 1, 3).contiguous().view(B, 3, -1).contiguous()

        feat_branchs = []
        for i in range(len(self.radii)):
            # #>>>>>>>>>>>> 3D PointNet++, with seperate FPS sampling<<<<< #
            # feat_frames = []
            # for j in range(T):
            #     support      = xyz[:, j, :, :].contiguous()
            #     neighbors_ind= self.neighbors_sample_list[i](support, points)

            #     xyz_grouped      = group_gather_by_index(xyz[:, j, :, :].contiguous(), neighbors_ind)
            #     features_grouped = group_gather_by_index(x[:, :, j, :].contiguous(), neighbors_ind) # batch_size, channel2, npoint1, nsample
            #     times_grouped    = group_gather_by_index(times[:, j, :, :].contiguous(), neighbors_ind)

            #     xyz_diff     = xyz_grouped - points.unsqueeze(3) # distance as part of feature
            #     if self.use_xyz:
            #         feat_grouped = torch.cat([xyz_diff, features_grouped], dim=1) # batch_size, channel2+3, npoint1, nsample
            #         if self.use_time:
            #             feat_grouped = torch.cat([xyz_diff, times_grouped, features_grouped], dim=1) # batch_size, channel2+3, npoint1, nsample
            #     else:
            #         feat_grouped = features_grouped

            #     new_feat = self.blocks[i](feat_grouped)
            #     # new_feat = self.conv3d[i*self.window_size+j](new_feat) # different for different t
            #     if verbose:
            #         print(f'frame {j} out neighbors feature {new_feat.size()}')
            #     feat_frames.append(torch.max(new_feat, dim=-1)[0])

            # feat_temporal = torch.cat(feat_frames, dim=1)
            # feat = self.unarys[i](feat_temporal)
            # # feat = feat_frames[0] - feat_frames[1]
            # #>>>>>>>>>>>>>>>>>>>>>>>>>> end here <<<<<<<<<<<<<<<<<<<<<<< #

            # #>>>>>>>>>>>> MeteorNet, with seperate FPS sampling<<<<< #
            # feat_frames = []
            # for j in range(T):
            #     support      = xyz[:, j, :, :].contiguous()
            #     neighbors_ind= self.neighbors_sample_list[i](support, points)

            #     xyz_grouped      = group_gather_by_index(xyz[:, j, :, :].contiguous(), neighbors_ind)
            #     features_grouped = group_gather_by_index(x[:, :, j, :].contiguous(), neighbors_ind) # batch_size, channel2, npoint1, nsample
            #     times_grouped    = group_gather_by_index(times[:, j, :, :].contiguous(), neighbors_ind)

            #     xyz_diff     = xyz_grouped - points.unsqueeze(3) # distance as part of feature
            #     if self.use_xyz:
            #         feat_grouped = torch.cat([xyz_diff, features_grouped], dim=1) # batch_size, channel2+3, npoint1, nsample
            #         if self.use_time:
            #             feat_grouped = torch.cat([xyz_diff, times_grouped, features_grouped], dim=1) # batch_size, channel2+3, npoint1, nsample
            #     else:
            #         feat_grouped = features_grouped

            #     new_feat = self.blocks[i](feat_grouped)
            #     # new_feat = self.conv3d[i*self.window_size+j](new_feat) # different for different t
            #     if verbose:
            #         print(f'frame {j} out neighbors feature {new_feat.size()}')
            #     feat_frames.append(new_feat)

            # feat = torch.max(torch.cat(feat_frames, dim=-1), dim=-1)[0]
            # # feat_temporal = torch.cat(feat_frames, dim=1)
            # # feat = self.unarys[i](feat_temporal)
            # # feat = feat_frames[0] - feat_frames[1]
            # #>>>>>>>>>>>>>>>>>>>>>>>>>> end here <<<<<<<<<<<<<<<<<<<<<<< #

            #>>>>>>>>>>>> 3D PointNet++, with seperate FPS sampling, different MLP<<<<< #
            feat_frames = []
            for j in range(T):
                support      = xyz[:, j, :, :].contiguous()
                neighbors_ind= self.neighbors_sample_list[i](support, points) # [B, N, M]
                N1, M1 = neighbors_ind.size(1), neighbors_ind.size(2)
                xyz_grouped      = group_gather_by_index(xyz[:, j, :, :].contiguous(), neighbors_ind)
                features_grouped = group_gather_by_index(x[:, :, j, :].contiguous(), neighbors_ind) # batch_size, channel2, npoint1, nsample
                times_grouped    = group_gather_by_index(times[:, j, :, :].contiguous(), neighbors_ind)

                xyz_diff     = xyz_grouped - points.unsqueeze(3) # distance as part of feature
                xyz_zero     = points - points
                if self.kernel_type == 'pointnet++':
                    if self.use_xyz:
                        feat_grouped = torch.cat([xyz_diff, features_grouped], dim=1) # batch_size, channel2+3, npoint1, nsample
                        if self.use_time:
                            feat_grouped = torch.cat([xyz_diff, times_grouped, features_grouped], dim=1) # batch_size, channel2+3, npoint1, nsample
                    else:
                        feat_grouped = features_grouped

                    new_feat = self.blocks[i](feat_grouped)
                    if not self.share:
                        new_feat = self.conv3d[i*self.window_size+j](new_feat) # different for different t
                    if verbose:
                        print(f'frame {j} out neighbors feature {new_feat.size()}')
                    feat_frames.append(torch.max(new_feat, dim=-1)[0])

                elif self.kernel_type == 'attentative':
                    vij = xyz_diff
                    dij = torch.norm(vij, dim=1).unsqueeze(1)

                    relPointPos = torch.cat([points.unsqueeze(3).repeat(1, 1, 1, xyz_grouped.size(-1)), xyz_grouped, vij, dij], dim=1)
                    rij = self.pos_nn[i](relPointPos)

                    # concatenate position encoding with feature vector
                    fij_hat = torch.cat([features_grouped, rij], dim=1)

                    # attentative pooling
                    g_fij = self.att_nn[i](fij_hat)
                    s_ij = F.softmax(g_fij, -1)
                    msg = torch.sum(s_ij * fij_hat, dim=-1)

                    feat_frames.append(self.blocks[i](msg))

                elif self.kernel_type == 'attention':
                    vij = xyz_diff
                    # concatenate position encoding with feature vector
                    fij_hat = torch.cat([features_grouped, vij], dim=1) # [B, C, N, M]
                    fi_hat  = torch.cat([feat1, xyz_zero], dim=1)

                    # attentation
                    g_fij = self.att_nn[i](fij_hat).permute(0, 2, 3, 1).contiguous()
                    g_fi  = self.att_nn[i](fi_hat.unsqueeze(3)).permute(0, 2, 1, 3).contiguous()

                    g_fij_hat = g_fij.view(-1, g_fij.size(2), g_fij.size(3)).contiguous()
                    g_fi_hat  = g_fi.view(-1, g_fi.size(2), g_fi.size(3)).contiguous()

                    s_ij = F.softmax(torch.bmm(g_fij_hat, g_fi_hat), 1) # [B*N, M, 1]
                    s_ij = s_ij.view(B, N1, M1, 1).contiguous().permute(0, 3, 1, 2).contiguous() # [B, N, M, 1]-> [B, 1, N, M]
                    msg  = torch.sum(s_ij * fij_hat, dim=-1)
                    feat_frames.append(self.blocks[i](msg))

            # feat_temporal = torch.cat(feat_frames, dim=1)
            # feat = self.unarys[i](feat_temporal)
            feat = feat_frames[0] - feat_frames[1]
            #>>>>>>>>>>>>>>>>>>>>>>>>>> end here <<<<<<<<<<<<<<<<<<<<<<< #
            feat_branchs.append(feat)
            if verbose:
                print(f'branch {i} out feature {feat.size()}')
        features = torch.cat(feat_branchs, dim=1)
        features = features.view(B, features.size(1), T, -1)
        return xyz1, t_flag, features #

# simple pointnet++ down module
class PointNetMSGDown(nn.Module):
    def __init__(
        self,
        npoint,
        radii,
        nsample,
        down_conv_nn=None,
        bn=True,
        activation="LeakyReLU",
        use_xyz=True,
        scale_idx=0,
        kernel_type='pointnet++',
        verbose=False,
        **kwargs
    ):
        super(PointNetMSGDown, self).__init__()
        self.npoint = npoint
        self.blocks = nn.ModuleList()
        self.radii  = radii
        self.nsample=nsample
        self.scale_idx = scale_idx

        self.neighbors_sample_list = nn.ModuleList()
        for i in range(len(radii)):
            in_channels=eval(down_conv_nn[i][0])
            out_channels=down_conv_nn[i][1:]
            neighbors_sample = SampleK(radii[i], nsample[i], knn=False)
            self.neighbors_sample_list.append(neighbors_sample)

            layers = []
            out_channels = [in_channels, *out_channels]
            print(f'Output channel in block {i} is {out_channels}')
            for j in range(1, len(out_channels)):
                    layers+=[nn.Conv2d(out_channels[j - 1], out_channels[j], 1, bias=True), nn.BatchNorm2d(out_channels[j], eps=0.001), nn.LeakyReLU()]
            layers = nn.Sequential(*layers)
            if verbose:
                print('Current block has: \n', layers)
            self.blocks.append(layers)

    def forward(self, query_points, support_points, x): # x, pos, new_pos, radius_idx, scale_idx
        """
        Module: PointNetMSGDown
        - query_points(torch Tensor):   [Batch_size, 3, N]
        - support_points(torch Tensor): [Batch_size, 3, N0]
        - x : feature of size [Batch_size, d, N0] with xyz included)
        """
        # by default, we use fps sampling
        xyz_ind = furthest_point_sampling(query_points.permute(0, 2, 1).contiguous().float(), self.npoint) # FPS
        xyz1    = fps_gather_by_index(query_points, xyz_ind)
        new_features = []
        for i in range(len(self.radii)):
            neighbor_indices = self.neighbors_sample_list[i](support_points.view(-1, 3, support_points.size(-1)).contiguous(), xyz1) #
            neighbors_feat = group_gather_by_index(x, neighbor_indices) # batch_size, I, N, K -->
            new_feat = self.blocks[i](neighbors_feat)
            new_features.append(torch.max(new_feat, dim=-1)[0]) # pointnet++-like operation
        features = torch.cat(new_features, dim=1)
        return xyz1, features

class Sample(nn.Module):
    def __init__(self, npoint, fps=True):
        super(Sample, self).__init__()
        self.npoint = npoint
        self.fps=fps

    def forward(self, xyz):
        """

        """
        if self.fps:
            xyz_ind = furthest_point_sampling(xyz.permute(0, 2, 1).contiguous().float(), self.npoint)
        else:
            xyz_ind = None

        return xyz_ind

class GlobalDenseBaseModule(nn.Module):
    def __init__(self, nn, aggr="max", bn=True, activation="LeakyReLU", **kwargs):
        super(GlobalDenseBaseModule, self).__init__()
        nn = [eval(nn[0]), *nn[1:]]
        self.nn = SharedMLP(nn, bn=bn, activation=get_activation(activation))
        if aggr.lower() not in ["mean", "max"]:
            raise Exception("The aggregation provided is unrecognized {}".format(aggr))
        self._aggr = aggr.lower()

    @property
    def nb_params(self):
        """[This property is used to return the number of trainable parameters for a given layer]
        It is useful for debugging and reproducibility.
        Returns:
            [type] -- [description]
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self._nb_params = sum([np.prod(p.size()) for p in model_parameters])
        return self._nb_params

    def forward(self, pos, x, **kwargs):

        x = self.nn(torch.cat([x, pos], dim=1).unsqueeze(-1))

        if self._aggr == "max": # global
            x = x.squeeze(-1).max(-1)[0]
        elif self._aggr == "mean":
            x = x.squeeze(-1).mean(-1)
        else:
            raise NotImplementedError("The following aggregation {} is not recognized".format(self._aggr))

        pos = None  # pos.mean(1).unsqueeze(1)
        x = x.unsqueeze(-1)
        return pos, x

    def __repr__(self):
        return "{}: {} (aggr={}, {})".format(self.__class__.__name__, self.nb_params, self._aggr, self.nn)

class DenseFPModule(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|

    """
    def __init__(self, up_conv_nn, bn=True, bias=False, activation=nn.LeakyReLU(negative_slope=0.01), **kwargs):
        super(DenseFPModule, self).__init__()
        in_channels  = eval(up_conv_nn[0])
        out_channels = up_conv_nn[1:]
        out_channels = [in_channels, *out_channels]
        layers = []
        for i in range(1, len(out_channels)):
            layers += [nn.Conv1d(out_channels[i - 1], out_channels[i], 1, bias=False), nn.BatchNorm1d(out_channels[i], eps=0.001), activation]
        self.conv = nn.Sequential(*layers)

    def forward(self, xyz, skip, xyz_prev=None, feat_prev=None):
        """
        xyz:       current full points, [BS, 3, N0]
        skip:      features for current current full points, [BS, C, N0]
        xyz_prev:  points to interpolate; [BS, C, N]
        feat_prev: features to interpolate; [BS, C, N]
        """
        if xyz_prev is not None:
            dist, ind = three_nn(xyz.permute(0, 2, 1).contiguous(), xyz_prev.permute(0, 2, 1).contiguous())
            dist = dist * dist
            dist[dist < 1e-10] = 1e-10
            inverse_dist = 1.0 / (dist + 1e-8)
            norm = torch.sum(inverse_dist, dim=2, keepdim=True)
            weights = inverse_dist / norm
            new_features = torch.sum(group_gather_by_index(feat_prev, ind) * weights.unsqueeze(1), dim = 3)
            if skip is not None:
                new_features = torch.cat([new_features, skip], dim=1)
        else:
            new_features = torch.cat([skip, feat_prev.repeat(1, 1, skip.size(-1))], dim=1)

        new_features = self.conv(new_features) # no temporal channel just simple features for current points
        return xyz, new_features
