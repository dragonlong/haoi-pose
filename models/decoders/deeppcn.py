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
            # breakpoint()
        else:
            ind = ball_query(self.radius, self.num_samples, xyz2.permute(0, 2, 1).contiguous(),
                             xyz1.permute(0, 2, 1).contiguous(), False)
        xyz2_grouped = group_gather_by_index(xyz2, ind)
        xyz_diff     = xyz2_grouped - xyz1.unsqueeze(3)
        features_grouped = group_gather_by_index(features, ind) # batch_size, channel2, npoint1, nsample
        new_features = torch.cat([xyz_diff, features_grouped], dim=1) # batch_size, channel2+3, npoint1, nsample
        return new_features

class SampleK(nn.Module):
    def __init__(self, radius, num_samples, knn=False):
        super(SampleK, self).__init__()

        self.radius = radius
        self.num_samples = num_samples
        self.knn    = knn

    def forward(self, xyz2, xyz1):
        # find nearest points in xyz2 for every points in xyz1
        if self.knn:
            dist= pdist2squared(xyz2, xyz1)
            ind = dist.topk(self.num_samples, dim=1, largest=False)[1].int().permute(0, 2, 1).contiguous()
        else:
            ind = ball_query(self.radius, self.num_samples, xyz2.permute(0, 2, 1).contiguous(),
                             xyz1.permute(0, 2, 1).contiguous(), False)

        return ind

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
        # dist[dist < 1e-10] = 1e-10
        inverse_dist = 1.0 / (dist + 1e-10)
        norm = torch.sum(inverse_dist, dim=2, keepdim=True)
        weights = inverse_dist / norm
        #new_features = three_interpolate(feat1, ind, weights) # wrong gradients
        new_features = torch.sum(group_gather_by_index(feat1, ind) * weights.unsqueeze(1), dim = 3)
        new_features = torch.cat([new_features, feat2], dim=1)
        #
        # breakpoint()
        new_features = self.conv(new_features.unsqueeze(3)).squeeze(3)
        return new_features

# point feature learning
# sampling and grouping, followed by convolution over kernel points
class PcConv(nn.Module):
    def __init__(self, in_channels=3, out_channels=[16, 32]):
        super(PcConv, self).__init__()
        # self.sample = SampleK(radius, num_samples, knn=True)

        layers = []
        out_channels = [in_channels, *out_channels]
        for i in range(1, len(out_channels)):
            layers += [nn.Conv2d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm2d(out_channels[i], eps=0.001), nn.ReLU()]

        self.conv = nn.Sequential(*layers)

    def forward(self, xyz, feat, ind=None):
        # xyz1 = self.sample(points)
        # new_features = self.group(points, xyz1, features)
        # ind = self.sample(xyz, xyz)

        # xyz_grouped  = group_gather_by_index(xyz, ind) # batch_size, 3, N, K
        xyz_grouped  = xyz
        feat_grouped = group_gather_by_index(feat, ind) # batch_size, I, N, K --> batch_size, 1, N, K, I

        # learn weights
        weights = self.conv(xyz_grouped) # batch_size, 32, N, K --> batch_size,32, N, K, I

        # weighted sum
        new_features = torch.sum(feat_grouped.permute(0, 2, 3, 1).contiguous().unsqueeze(1) * weights.unsqueeze(-1), dim=[3, 4])/weights.size(3)# for point features, we also learn local dominant features, like pointnet
        return new_features

class DPCN(nn.Module):
    def __init__(self, num_points=1024):
        super(DPCN, self).__init__()
        self.radius      = 10 # use real distance
        self.num_samples = 16
        self.num_layer   = 8
        self.init_sample = Sample(num_points)
        self.sample = SampleK(self.radius, self.num_samples, knn=True)
        layers = []
        for i in range(self.num_layer):
            if i < self.num_layer -1:
                cconv_layer = PcConv()
            else:
                cconv_layer = PcConv(out_channels=[64, 128])
            cconv_layer.apply(DPCN.weights_init)
            layers.append(cconv_layer)

        self.conv = nn.Sequential(*layers)
        self.linear = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, eps=0.001),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, eps=0.001),
            nn.ReLU(), # we use activation function as this is not the final layer
            )
        self.fp = FeaturePropagation(512, 1, [256, 256])
        self.classifier = nn.Sequential(
            nn.Conv1d(256, 256, 1, bias=True),
            nn.BatchNorm1d(256, eps=0.001),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1, bias=True),
            nn.BatchNorm1d(128, eps=0.001),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1, bias=True)
        )

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and classname.find('Set')== -1 and classname.find('Pc')== -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        elif classname.find('Linear') != -1:
           m.weight.data.normal_(0.0, 0.0001)

    def forward(self, xyz1, xyz2, feat1, feat2):
        xyz1_1 = self.init_sample(xyz1.float())
        ind1 = self.sample(xyz1.float(), xyz1_1) #
        ind2 = self.sample(xyz2.float(), xyz1_1)

        # breakpoint()
        xyz1_grouped  = group_gather_by_index(xyz1.float(), ind1) - xyz1_1.float().unsqueeze(3)
        xyz2_grouped  = group_gather_by_index(xyz2.float(), ind2) - xyz1_1.float().unsqueeze(3)
        #
        feat1_updated = self.conv[0](xyz1_grouped, feat1, ind1)
        feat2_updated = self.conv[0](xyz2_grouped, feat2, ind2)
        for i in range(1, self.num_layer-1):
            feat1_updated = feat1_updated + self.conv[i](xyz1_grouped, feat1_updated, ind1)
            feat2_updated = feat2_updated + self.conv[i](xyz2_grouped, feat2_updated, ind2)

        feat1_updated = self.conv[-1](xyz1_grouped, feat1_updated, ind1)
        feat2_updated = self.conv[-1](xyz2_grouped, feat2_updated, ind2)
        print('last feature has size ', feat1_updated.size())
        # global maxpooling, transformation
        feat1_pooled = self.linear(torch.max(feat1_updated, 2)[0]).unsqueeze(-1)
        feat2_pooled = self.linear(torch.max(feat2_updated, 2)[0]).unsqueeze(-1)

        # concat
        feat_final = torch.cat([feat1_pooled.repeat(1, 1, feat1_updated.size(-1)), feat1_updated, feat2_pooled.repeat(1, 1, feat2_updated.size(-1)), feat2_updated], dim=1)
        feat_final = self.fp(xyz1_1, xyz1.float(), feat_final, feat1)
        flow = self.classifier(feat_final)

        return flow

def criterion(pred_flow, flow, mask=None):
    loss = torch.mean(torch.sum((pred_flow - flow) * (pred_flow - flow), dim=1) / 2.0)
    return loss

if __name__ == '__main__':
    gpu = 0
    deploy_device   = torch.device('cuda:{}'.format(gpu))
    # xyz1  = torch.empty(BS, 3, N, dtype=torch.long, device=deploy_device).random_(2, 420)
    N    = 2048
    # xyz1 = np.array([[1, 4, 0], [1, 3, 0], [1, 2, 0], [1, 1, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [3, 1, 0]], dtype=np.float32)
    xyz1 = np.random.rand(N, 3)
    xyz2 = xyz1 + np.array([[2, 1, 0]], dtype=np.float32)
    xyz1 = torch.from_numpy(xyz1.copy().transpose(1, 0)[np.newaxis, :, :]).cuda(gpu)
    xyz2 = torch.from_numpy(xyz2.copy().transpose(1, 0)[np.newaxis, :, :]).cuda(gpu) + 0.1 * torch.rand(1, 3, N, requires_grad=False, device=deploy_device)
    flow = xyz2 - xyz1
    feat1 = torch.rand(1, 4, N, requires_grad=False, device=deploy_device).float()
    feat2 = feat1
    # model = PcConv(1024, 1, 32, 3).cuda(gpu) #
    # new_feat = model(xyz1, feat1)
    model = DPCN().cuda()
    new_feat = model(xyz1, xyz2, feat1, feat2)
    print('output is ', new_feat.size())

    # model = FlowNet3D().cuda(gpu)
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
