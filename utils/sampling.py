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

class SampleK(nn.Module):
    def __init__(self, radius, num_samples, knn=False):
        super(SampleK, self).__init__()

        self.radius = radius
        self.num_samples = num_samples
        self.knn    = knn

    def forward(self, xyz2, xyz1): # [BS, 3, N]
        # find nearest points in xyz2 for every points in xyz1
        if self.knn:
            dist= pdist2squared(xyz2, xyz1)
            ind = dist.topk(self.num_samples, dim=1, largest=False)[1].int().permute(0, 2, 1).contiguous()
        else:
            ind = ball_query(self.radius, self.num_samples, xyz2.permute(0, 2, 1).contiguous(),
                             xyz1.permute(0, 2, 1).contiguous(), False)

        return ind


def criterion(pred_flow, flow, mask=None):
    loss = torch.mean(torch.sum((pred_flow - flow) * (pred_flow - flow), dim=1) / 2.0)
    return loss

if __name__ == '__main__':
    gpu = 0
    deploy_device   = torch.device('cuda:{}'.format(gpu))
    # xyz1  = torch.empty(BS, 3, N, dtype=torch.long, device=deploy_device).random_(2, 420)
    N    = 1024
    # xyz1 = np.array([[1, 4, 0], [1, 3, 0], [1, 2, 0], [1, 1, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [3, 1, 0]], dtype=np.float32)
    # xyz2 = xyz1 + np.array([[2, 1, 0]], dtype=np.float32)
    # xyz1 = torch.from_numpy(xyz1.copy().transpose(1, 0)[np.newaxis, :, :]).cuda(gpu)
    # xyz2 = torch.from_numpy(xyz2.copy().transpose(1, 0)[np.newaxis, :, :]).cuda(gpu) + 0.1 * torch.rand(1, 3, N, requires_grad=False, device=deploy_device)
    # flow = xyz2 - xyz1
    xyz1  = torch.rand(1, 3, N, requires_grad=False, device=deploy_device)
    feat1 = torch.rand(1, 3, N, requires_grad=False, device=deploy_device)
    feat2 = feat1
    model = PcConv(1024, 1, 32, 3).cuda(gpu) #
    new_feat = model(xyz1, feat1)
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
