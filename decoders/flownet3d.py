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

def pdist2squared(x, y):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = (y**2).sum(dim=1).unsqueeze(1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), y)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0, np.inf)
    return dist

def breakpoint():
    import pdb;pdb.set_trace()

# Simply sample num_points points
class Sample(nn.Module):
    def __init__(self, num_points):
        super(Sample, self).__init__()
        
        self.num_points = num_points
        
    def forward(self, points):
        # breakpoint()
        xyz1_ind = furthest_point_sampling(points.permute(0, 2, 1).contiguous(), self.num_points)
        xyz1 = fps_gather_by_index(points, xyz1_ind)    # batch_size, channel2, nsample
        return xyz1

# Group points according to index
class Group(nn.Module):
    def __init__(self, radius, num_samples, knn=False):
        super(Group, self).__init__()
        
        self.radius = radius
        self.num_samples = num_samples
        self.knn    = knn
        
    def forward(self, xyz2, xyz1, features):
        # find nearest points in xyz2 for every points in xyz1
        if self.knn:
            dist = pdist2squared(xyz2, xyz1)
            ind = dist.topk(self.num_samples, dim=1, largest=False)[1].int().permute(0, 2, 1).contiguous()
        else:
            ind = ball_query(self.radius, self.num_samples, xyz2.permute(0, 2, 1).contiguous(),
                             xyz1.permute(0, 2, 1).contiguous(), False) 
        xyz2_grouped = group_gather_by_index(xyz2, ind)
        xyz_diff     = xyz2_grouped - xyz1.unsqueeze(3) 
        features_grouped = group_gather_by_index(features, ind) # batch_size, channel2, npoint1, nsample
        new_features = torch.cat([xyz_diff, features_grouped], dim=1) # batch_size, channel2+3, npoint1, nsample
        return new_features

# point feature learning
# sampling and grouping, followed by convolution over kernel points
class SetConv(nn.Module):
    def __init__(self, num_points, radius, num_samples, in_channels, out_channels):
        super(SetConv, self).__init__()
        
        self.sample = Sample(num_points)
        self.group = Group(radius, num_samples)
        
        layers = []
        out_channels = [in_channels+3, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, points, features):
        xyz1 = self.sample(points)
        new_features = self.group(points, xyz1, features)
        new_features = self.conv(new_features)
        new_features = new_features.max(dim=3)[0]
        return xyz1, new_features

# 
class FlowEmbedding(nn.Module):
    def __init__(self, num_samples, in_channels, out_channels, pooling='max', knn=True, corr_func='elementwise_product'):
        super(FlowEmbedding, self).__init__()
        
        self.num_samples = num_samples
        
        self.group = Group(None, self.num_samples, knn=True) # TODO
        
        layers = []
        out_channels = [2*in_channels+3, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, xyz1, xyz2, feat1, feat2):
        """
        Input:
            xyz1: (batch_size, npoint, 3)
            xyz2: (batch_size, npoint, 3)
            feat1: (batch_size, npoint, channel)
            feat2: (batch_size, npoint, channel)
        Output:
            xyz1: (batch_size, npoint, 3)
            feat1_new: (batch_size, npoint, mlp[-1])
        """
        # breakpoint()
        feat2_grouped = self.group(xyz2, xyz1, feat2) # here we group feat2
        feat1_expanded= feat1.unsqueeze(3) 
        # corr_function
        
        new_features = feat2_grouped
        new_features = torch.cat([new_features, feat1.unsqueeze(3).expand(-1, -1, -1, self.num_samples)], dim=1)
        new_features = self.conv(new_features)
        new_features = new_features.max(dim=3)[0]
        return new_features

class SetUpConv(nn.Module):
    def __init__(self, num_samples, in_channels1, in_channels2, out_channels1, out_channels2):
        super(SetUpConv, self).__init__()
        
        self.group = Group(None, num_samples, knn=True)
        
        layers = []
        out_channels1 = [in_channels1+3, *out_channels1]
        for i in range(1, len(out_channels1)):
            layers += [nn.Conv2d(out_channels1[i - 1], out_channels1[i], 1, bias=True), nn.BatchNorm2d(out_channels1[i], eps=0.001), nn.ReLU()]
        self.conv1 = nn.Sequential(*layers)
        
        layers = []
        if len(out_channels1) == 1:
            out_channels2 = [in_channels1+in_channels2+3, *out_channels2]
        else:
            out_channels2 = [out_channels1[-1]+in_channels2, *out_channels2]
        for i in range(1, len(out_channels2)):
            layers += [nn.Conv2d(out_channels2[i - 1], out_channels2[i], 1, bias=True), nn.BatchNorm2d(out_channels2[i], eps=0.001), nn.ReLU()]
        self.conv2 = nn.Sequential(*layers)
        
    def forward(self, xyz1, xyz2, feat1, feat2):
        """
        Inputs:
            xyz1: (batch_size, npoint1, 3)
            xyz2: (batch_size, npoint2, 3)
            feat1: (batch_size, npoint1, channel1) features for xyz1 points (earlier layers)
            feat2: (batch_size, npoint2, channel2) features for xyz2 points
        Output:
            feat1_new: (batch_size, npoint2, mlp[-1] or mlp2[-1] or channel1+3)
        """
        new_features = self.group(xyz1, xyz2, feat1)
        new_features = self.conv1(new_features)
        new_features = new_features.max(dim=3)[0]
        new_features = torch.cat([new_features, feat2], dim=1)
        new_features = new_features.unsqueeze(3)
        new_features = self.conv2(new_features)
        new_features = new_features.squeeze(3)
        return new_features


class FeaturePropagation(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(FeaturePropagation, self).__init__()
        
        layers = []
        out_channels = [in_channels1+in_channels2, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        
    def forward(self, xyz1, xyz2, feat1, feat2):
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

class FlowNet3D(nn.Module):
    def __init__(self):
        super(FlowNet3D, self).__init__()
        # self.set_conv1 = SetConv(8, 3.0, 4, 1, [32, 32, 64]) #num_points, radius, num_samples, in_channels, out_channels
        # self.set_conv2 = SetConv(8, 3.0, 4, 64, [64, 64, 128])
        # self.flow_embedding = FlowEmbedding(4, 128, [128, 128, 128]) # num_samples, in_channels, out_channels, pooling='max',
        # self.set_conv3 = SetConv(4, 2.0, 8, 128, [128, 128, 256])
        # self.set_conv4 = SetConv(4, 4.0, 8, 256, [256, 256, 512])
        # self.set_upconv1 = SetUpConv(4, 512, 256, [], [256, 256]) # num_samples, in_channels1, in_channels2, out_channels1, out_channels2
        # self.set_upconv2 = SetUpConv(4, 256, 256, [128, 128, 256], [256])
        # self.set_upconv3 = SetUpConv(4, 256, 64, [128, 128, 256], [256])
        # self.fp = FeaturePropagation(256, 3, [256, 256]) # in_channels1, in_channels2, out_channels
        # self.classifier = nn.Sequential(
        #     nn.Conv1d(256, 128, 1, bias=True),
        #     nn.BatchNorm1d(128, eps=0.001),
        #     nn.ReLU(),
        #     nn.Conv1d(128, 3, 1, bias=True)
        # ) 
        self.set_conv1 = SetConv(1024, 0.5, 16, 1, [32, 32, 64]) # TODO
        self.set_conv2 = SetConv(256, 1.0, 16, 64, [64, 64, 128])
        self.flow_embedding = FlowEmbedding(64, 128, [128, 128, 128])
        self.set_conv3 = SetConv(64, 2.0, 8, 128, [128, 128, 256])
        self.set_conv4 = SetConv(16, 4.0, 8, 256, [256, 256, 512])
        self.set_upconv1 = SetUpConv(8, 512, 256, [], [256, 256])
        self.set_upconv2 = SetUpConv(8, 256, 256, [128, 128, 256], [256])
        self.set_upconv3 = SetUpConv(8, 256, 64, [128, 128, 256], [256])
        self.fp = FeaturePropagation(256, 1, [256, 256])
        self.classifier = nn.Sequential(
            nn.Conv1d(256, 128, 1, bias=True),
            nn.BatchNorm1d(128, eps=0.001),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1, bias=True)
        )
         
   # pointnet2_charlesmsg:
   #      class: pointnet2.PointNet2_D
   #      conv_type: "DENSE"
   #      use_category: ${data.use_category}
   #      down_conv:
   #          module_name: PointNetMSGDown
   #          npoint: [512, 128]
   #          radii: [[0.1, 0.2, 0.4], [0.4, 0.8]]
   #          nsamples: [[32, 64, 128], [64, 128]]
   #          down_conv_nn:
   #              [
   #                  [
   #                      [FEAT+3, 32, 32, 64],
   #                      [FEAT+3, 64, 64, 128],
   #                      [FEAT+3, 64, 96, 128],
   #                  ],
   #                  [
   #                      [64 + 128 + 128+3, 128, 128, 256],
   #                      [64 + 128 + 128+3, 128, 196, 256],
   #                  ],
   #              ]
   #      innermost:
   #          module_name: GlobalDenseBaseModule
   #          nn: [256 * 2 + 3, 256, 512, 1024]
   #      up_conv:
   #          module_name: DenseFPModule
   #          up_conv_nn:
   #              [
   #                  [1024 + 256*2, 256, 256],
   #                  [256 + 128 * 2 + 64, 256, 128],
   #                  [128 + FEAT, 128, 128],
   #              ]
   #          skip: True
   #      mlp_cls:
   #          nn: [128, 128]
   #          dropout: 0.5
   
    def forward(self, xyz1, xyz2, feat1, feat2):
        xyz1_1, feat1_1 = self.set_conv1(xyz1, feat1)
        xyz1_2, feat1_2 = self.set_conv2(xyz1_1, feat1_1)

        xyz2_1, feat2_1 = self.set_conv1(xyz2, feat2)
        xyz2_2, feat2_2 = self.set_conv2(xyz2_1, feat2_1)

        embedding = self.flow_embedding(xyz1_2, xyz2_2, feat1_2, feat2_2)
        
        xyz1_3, feat1_3 = self.set_conv3(xyz1_2, embedding)
        xyz1_4, feat1_4 = self.set_conv4(xyz1_3, feat1_3)
        
        new_feat1_3 = self.set_upconv1(xyz1_4, xyz1_3, feat1_4, feat1_3)
        new_feat1_2 = self.set_upconv2(xyz1_3, xyz1_2, new_feat1_3, torch.cat([feat1_2, embedding], dim=1))
        new_feat1_1 = self.set_upconv3(xyz1_2, xyz1_1, new_feat1_2, feat1_1)
        new_feat1 = self.fp(xyz1_1, xyz1, new_feat1_1, feat1)

        flow = self.classifier(new_feat1)
        
        return flow

def criterion(pred_flow, flow, mask=None):
    loss = torch.mean(torch.sum((pred_flow - flow) * (pred_flow - flow), dim=1) / 2.0)
    return loss

if __name__ == '__main__':
    gpu = 0
    deploy_device   = torch.device('cuda:{}'.format(gpu))
    # xyz1  = torch.empty(BS, 3, N, dtype=torch.long, device=deploy_device).random_(2, 420)
    N    = 8
    xyz1 = np.array([[1, 4, 0], [1, 3, 0], [1, 2, 0], [1, 1, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [3, 1, 0]], dtype=np.float32)
    xyz2 = xyz1 + np.array([[2, 1, 0]], dtype=np.float32)
    xyz1 = torch.from_numpy(xyz1.copy().transpose(1, 0)[np.newaxis, :, :]).cuda(gpu).float()
    xyz2 = torch.from_numpy(xyz2.copy().transpose(1, 0)[np.newaxis, :, :]).cuda(gpu) + 0.1 * torch.rand(1, 3, N, requires_grad=False, device=deploy_device) 
    xyz2 = xyz2.float()
    print(xyz1, xyz2)
    flow = xyz2 - xyz1
    feat1 = torch.rand(1, 1, N, requires_grad=False, device=deploy_device) 
    feat2 = feat1
    model = FlowNet3D().cuda(gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.00001,
        momentum=0.9,
        weight_decay=1e-4)

    zeros_t = time.time()
    for i in range(1000):

        start_t = time.time()
        pred_flow  = model(xyz1, xyz2, feat1, feat2)
        loss = criterion(pred_flow, flow, mask=None)
        loss.backward()
        optimizer.step()
        end_t = time.time()
        print(f'Step {i} has loss {loss.cpu():0.04f}, compute time {end_t - start_t} seconds')

    print(f'flow estimation training takes {end_t - zeros_t} seconds')
    print('Con!!!')
