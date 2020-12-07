import torch 
import time
import torch.nn as nn
import numpy as np
from kaolin.models.PointNet2 import furthest_point_sampling
from kaolin.models.PointNet2 import fps_gather_by_index
from kaolin.models.PointNet2 import ball_query
# import torch
from kaolin.models.PointNet2 import group_gather_by_index
from kaolin.models.PointNet2 import three_nn
from kaolin.models.PointNet2 import three_interpolate
from torch_scatter import scatter

import __init__
from modules.sampling import SampleK

def breakpoint():
    import pdb;pdb.set_trace()

def pdist2squared(x, y):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = (y**2).sum(dim=1).unsqueeze(1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), y)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist

# Simply sample npoint xyz, return index
class Sample(nn.Module):
    def __init__(self, npoint, fps=False):
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

# Group points according to index
class Group(nn.Module):
    def __init__(self, radius, nsample, knn=False):
        super(Group, self).__init__()

        self.radius = radius
        self.nsample = nsample
        self.knn    = knn

    def forward(self, xyz1, xyz2, flag=None, nframe=2):
        # for every xyz in xyz1, find nearest xyz in xyz2, or
        if self.knn:
            # breakpoint()
            dist = pdist2squared(xyz2, xyz1)
            ind = dist.topk(self.nsample, dim=1, largest=False)[1].int().permute(0, 2, 1).contiguous()
        else:
            ind = ball_query(self.radius, self.nsample, xyz2.permute(0, 2, 1).contiguous(),
                             xyz1.permute(0, 2, 1).contiguous(), False)

        return ind


# Group xyz according to index
class GroupRR(nn.Module):
    def __init__(self, radius_list, nsample, knn=False, use_xyz=True):
        super(GroupRR, self).__init__()

        self.radius_list = radius_list
        self.nsample= nsample
        self.knn    = knn

        self.groups = nn.ModuleList()
        for radius in radius_list:
            self.groups.append( Group(radius, nsample, knn=knn)) # number of samples per frame

    def forward(self, xyz1, xyz2, flag, nframe):
        # for every xyz in xyz1, find nearest xyz in xyz2, or
        bs = len(flag.size(0))

        # for i in range(bs):
        #     xyz_batch = []
        #     for j in range(nframe): # j is frame index
        #         ind = torch.nonzero(flag[i]==j) # ind: 2, N
        #         xyz_patch = xyz1[i:i+1, :, ]

        ind = None
        return ind


class meteor_direct_module(nn.Module):
    def __init__(self, npoint, nsample, radius_list, in_channels, out_channels, pooling='max', knn=False, use_xyz=True, module_type='ind'):
        super(meteor_direct_module, self).__init__()

        self.nsample = nsample
        self.window_size  = 2
        self.npoint  = npoint # npoint in total
        self.module_type = module_type
        self.pooling = pooling
        self.use_xyz = use_xyz

        self.sampling = Sample(npoint, fps=True) # sample npoint, but how to distribute time
        if radius_list[0] != radius_list[1]:
            self.grouping = GroupRR(radius_list, nsample, knn=knn)
        else:
            self.grouping = Group(radius_list[0], nsample, knn=knn)

        layers = []
        out_channels = [in_channels, *out_channels] #
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        self.unary= nn.Sequential(
                    nn.Conv1d(out_channels[-1]*self.window_size, out_channels[-1], 1, bias=True),
                    nn.BatchNorm1d(out_channels[-1], eps=0.001),
                    nn.LeakyReLU()
                    )

    def forward(self, xyz, times, features, verbose=False):
        """
        Input:
        - xyz  : [Batch_size, 3, T * N0]
        - times  : [Batch_size, 1, T * N0]
        - features(torch Tensor): [Batch_size, C, T * N0]
        Output:
            xyz : [Batch_size, 3, T*N1]
            time: updated frame labels;
            new_feature: [Batch_size, 3, T, N1]
        """
        # 1. sample, random sample number of xyz
        xyz_ind = self.sampling(xyz)                 # [B, N]
        points  = fps_gather_by_index(xyz, xyz_ind)
        t_flag  = fps_gather_by_index(times, xyz_ind) # gather by index

        # 2. group by radius, and feature concats
        neighbors_ind    = self.grouping(points, xyz)

        xyz_grouped      = group_gather_by_index(xyz, neighbors_ind)
        features_grouped = group_gather_by_index(features, neighbors_ind) # batch_size, channel2, npoint1, nsample
        times_grouped    = group_gather_by_index(times, neighbors_ind)

        xyz_diff     = xyz_grouped - points.unsqueeze(3) # distance as part of feature
        if self.use_xyz:
            feat_grouped = torch.cat([xyz_diff, times_grouped, features_grouped], dim=1) # batch_size, channel2+3, npoint1, nsample
        else:
            feat_grouped = features_grouped

        # 3. compute features
        new_features = self.conv(feat_grouped)

        bs, c, n, m = new_features.size()
        x_max = torch.zeros(bs, c, n, 2, device=new_features.device) # TODO
        x_max = scatter(new_features, times_grouped.repeat(1, c, 1, 1).long(), dim=-1, out=x_max, reduce="max")

        # new_features = new_features.max(dim=3)[0]
        new_features = self.unary(x_max.permute(0, 3, 1, 2).contiguous().view(bs, -1, n).contiguous())

        return points, t_flag, new_features

class meteor_direct_module_original(nn.Module):
    def __init__(self, npoint, nsample, radius_list, in_channels, out_channels, nframe=2, pooling='max', knn=False, use_xyz=True, module_type='ind'):
        super(meteor_direct_module_original, self).__init__()

        self.nsample = nsample
        self.nframe  = nframe
        self.npoint  = npoint # npoint in total
        self.module_type = module_type
        self.pooling = pooling
        self.use_xyz = use_xyz

        self.sampling = Sample(npoint, fps=True) # sample npoint, but how to distribute time
        if radius_list[0] != radius_list[1]:
            self.grouping = GroupRR(radius_list, nsample, knn=knn)
        else:
            self.grouping = Group(radius_list[0], nsample, knn=knn)

        layers = []
        out_channels = [in_channels, *out_channels] #
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)

    def forward(self, xyz, times, features, verbose=False):
        """
        Input:
        - xyz  : [Batch_size, 3, T * N0]
        - times  : [Batch_size, 1, T * N0]
        - features(torch Tensor): [Batch_size, C, T * N0]
        Output:
            xyz : [Batch_size, 3, T*N1]
            time: updated frame labels;
            new_feature: [Batch_size, 3, T, N1]
        """
        # 1. sample, random sample number of xyz
        xyz_ind = self.sampling(xyz)                 # [B, 3, N] --> [B, N]
        points  = fps_gather_by_index(xyz, xyz_ind)
        t_flag  = fps_gather_by_index(times, xyz_ind) # gather by index

        # 2. group by radius, and feature concats
        # get xyz1, xyz2
        #
        xyzs = []
        for i in range(self.nframe):
            ind = torch.nonzero(times==i)
            breakpoint()
            xyzs.append(xyz[ind[:, 0], :, ind[:, 2]])

            neighbors_ind    = self.grouping(points, xyzs[i])

            xyz_grouped      = group_gather_by_index(xyz, neighbors_ind)
            features_grouped = group_gather_by_index(features, neighbors_ind) # batch_size, channel2, npoint1, nsample
            times_grouped    = group_gather_by_index(times, neighbors_ind)

            xyz_diff     = xyz_grouped - points.unsqueeze(3) # distance as part of feature
            if self.use_xyz:
                feat_grouped = torch.cat([xyz_diff, times_grouped, features_grouped], dim=1) # batch_size, channel2+3, npoint1, nsample
            else:
                feat_grouped = features_grouped

            # 3. compute features
            new_features = self.conv(feat_grouped)
            new_features = new_features.max(dim=3)[0]

        return points, t_flag, new_features

class pointnet_fp_module(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(pointnet_fp_module, self).__init__()

        layers = []
        out_channels = [in_channels1+in_channels2, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)

    def forward(self, xyz2, xyz1, feat2, feat1):
        """
        xyz1 : [B, 3, N], the sparse points to intepolate
        feat2: previous layer to concat, may be None for first
        """
        dist, ind = three_nn(xyz2.permute(0, 2, 1).contiguous(), xyz1.permute(0, 2, 1).contiguous())
        dist = dist * dist
        dist[dist < 1e-10] = 1e-10
        inverse_dist = 1.0 / dist
        norm = torch.sum(inverse_dist, dim=2, keepdim=True)
        weights = inverse_dist / norm
        #new_features = three_interpolate(feat1, ind, weights) # wrong gradients
        new_features = torch.sum(group_gather_by_index(feat1, ind) * weights.unsqueeze(1), dim = 3)
        new_features = torch.cat([new_features, feat2], dim=1)
        new_features = self.conv(new_features.unsqueeze(3)).squeeze(3)

        return new_features

class MeteorNet(nn.Module):
    def __init__(self, nframe=2, radius_list=[0.05, 0.05], num_class=20, nsample=32, knn=False):
        super(MeteorNet, self).__init__()
        self.nframe = nframe
        RADIUS1 = np.array(radius_list, dtype='float32')
        RADIUS2 = RADIUS1 * 2
        RADIUS3 = RADIUS1 * 4
        RADIUS4 = RADIUS1 * 8
        print(f'We have {nsample} neighbors')
        self.meteor_conv1 = meteor_direct_module(npoint=2048, nsample=nsample, radius_list=RADIUS1, in_channels=2+4, out_channels=[32,32,128], knn=knn)
        self.meteor_conv2 = meteor_direct_module(npoint=512, nsample=nsample, radius_list=RADIUS2, in_channels=128+4, out_channels=[64,64,256], knn=knn)
        self.meteor_conv3 = meteor_direct_module(npoint=128, nsample=nsample, radius_list=RADIUS3, in_channels=256+4, out_channels=[128,128,512], knn=knn)
        self.meteor_conv4 = meteor_direct_module(npoint=64, nsample=nsample, radius_list=RADIUS4, in_channels=512+4, out_channels=[256,256,1024], knn=knn)

        # No innermost for global pooling
        self.fp1 = pointnet_fp_module(1024, 512, [256, 256]) # input 1, 2
        self.fp2 = pointnet_fp_module(256, 256, [256, 256])
        self.fp3 = pointnet_fp_module(256, 128, [256, 128])
        self.fp4 = pointnet_fp_module(128, 2, [128, 128]) # use reflectence + time dimension
        print(f'-----------we have {num_class} classes')
        self.classifier = nn.Sequential(
            nn.Conv1d(128, num_class, 1, bias=True)
        ) # will need softmax
        # self.classifier_motion = nn.Sequential(
        #     nn.Conv1d(128, 3, 1, bias=True)
        # ) # will need softmax

    def forward(self, xyzs, feat, times=None):
        """
        xyzs: []
        times:
        """
        l0_xyz = xyzs
        l0_time= times
        l0_points = torch.cat([feat, l0_time], axis=1)
        # breakpoint()
        l1_xyz, l1_time, l1_points = self.meteor_conv1(l0_xyz, l0_time, l0_points)
        l2_xyz, l2_time, l2_points = self.meteor_conv2(l1_xyz, l1_time, l1_points)
        l3_xyz, l3_time, l3_points = self.meteor_conv3(l2_xyz, l2_time, l2_points)
        l4_xyz, l4_time, l4_points = self.meteor_conv4(l3_xyz, l3_time, l3_points)

        l3_points = self.fp1(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp3(l1_xyz, l2_xyz, l1_points, l2_points)
        new_feat  = self.fp4(l0_xyz, l1_xyz, l0_points, l1_points)
        # print(l0_xyz.size(), l1_xyz.size(), l0_points.size(), l1_points.size())

        pred = self.classifier(new_feat) # B, 3, N*T
        pred_curr = pred.view(pred.size(0), pred.size(1), self.nframe, -1).contiguous()[:, :, 0, :]

        # only return preds in current frame
        return pred_curr

        # c_pred = self.classifier(new_feat) # B, 3, N*T
        # m_pred = self.classifier_motion(new_feat) # B, 3, N*T

        # c_pred_curr = c_pred.view(pred.size(0), pred.size(1), self.nframe, -1).contiguous()[:, :, 0, :]

        # # only return preds in current frame
        # return c_pred, m_pred

def criterion(pred_flow, flow, mask=None):
    """
    pred_flow: B, 3, N*T;
    flow: B, 3, N*T
    mask: B, 3, B*T
    """
    if mask is not None:
        loss = torch.mean(torch.sum((pred_flow - flow) * (pred_flow - flow) * mask, dim=1) / 2.0)
    else:
        loss = torch.mean(torch.sum((pred_flow - flow) * (pred_flow - flow), dim=1) / 2.0) # why divided by 2
    return loss

if __name__ == '__main__':
    gpu = 0
    deploy_device   = torch.device('cuda:{}'.format(gpu))
    N    = 2048
    B    = 2
    xyz1 = np.random.rand(B, 3, N)
    xyz2 = xyz1 + np.array([0.2, 0.2, 0]).reshape(1, 3, 1)
    xyz1 = torch.from_numpy(xyz1.copy()).cuda(gpu)
    xyz2 = torch.from_numpy(xyz2.copy()).cuda(gpu)
    feat1 = torch.rand(B, 1, N, requires_grad=False, device=deploy_device).float()
    feat2 = feat1
    time1 = torch.zeros(B, 1, N, device=deploy_device).float()
    time2 = torch.ones(B, 1, N, device=deploy_device).float()

    xyzs  = torch.cat([xyz1, xyz2], dim=2).float()
    times = torch.cat([time1, time2], dim=2).float()
    feat  = torch.cat([feat1, feat2], dim=2).float()
    model = MeteorNet(2).cuda()
    new_feat = model(xyzs, times, feat)
    print('output is ', new_feat.size())

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=0.00001,
    #     momentum=0.9,
    #     weight_decay=1e-4)

    # zeros_t = time.time()
    # for i in range(1000):

    #     start_t = time.time()
    #     pred_flow  = model(xyz1, xyz2, feat1, feat2)
    #     loss = criterion(pred_flow, flow, mask=None)
    #     loss.backward()
    #     optimizer.step()
    #     end_t = time.time()
    #     print(f'Step {i} has loss {loss.cpu():0.04f}, compute time {end_t - start_t} seconds')

    # print(f'flow estimation training takes {end_t - zeros_t} seconds')
    # print('Con!!!')
