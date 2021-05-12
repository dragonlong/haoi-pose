from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F
import sys
TRAIN_PATH = "../../"
sys.path.insert(0, TRAIN_PATH)

import global_info

setting = global_info.setting()
# if setting.USE_MULTI_GPU:
#     BatchNorm1d = nn.SyncBatchNorm
# else:
BatchNorm1d = nn.BatchNorm1d

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x, gpu=0):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32)).requires_grad_().view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x, gpu=0):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).requires_grad_().view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda(gpu)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = BatchNorm1d(64)
        self.bn2 = BatchNorm1d(128)
        self.bn3 = BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x, gpu=0):
        n_pts = x.size()[2]
        # print('input size: ', x.size())
        trans = self.stn(x, gpu=gpu)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = x

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = BatchNorm1d(512)
        self.bn2 = BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x, gpu=0):
        x, trans, trans_feat = self.feat(x, gpu=gpu)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat

class PointNetEncoder(nn.Module):
    def __init__(self, feature_transform=False):
        super(PointNetEncoder, self).__init__()
        self.feature_transform=feature_transform
        self.feat  = PointNetfeat(global_feat=False, feature_transform=feature_transform)

    def forward(self, x, gpu=0, return_feature_maps=True):
        feat_concat, trans, trans_feat = self.feat(x, gpu=gpu)

        return feat_concat

class PointNetDecoder(nn.Module):
    def __init__(self, k = 2):
        super(PointNetDecoder, self).__init__()
        self.k = k
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = BatchNorm1d(512)
        self.bn2 = BatchNorm1d(256)
        self.bn3 = BatchNorm1d(128)

    def forward(self, x, return_feature_maps=False):
        feat_dict = {}
        x = F.relu(self.bn1(self.conv1(x)))
        feat_dict[512] = x
        x = F.relu(self.bn2(self.conv2(x)))
        feat_dict[256] = x
        x = F.relu(self.bn3(self.conv3(x)))
        feat_dict[128] = x
        x = self.conv4(x)
        if setting.USE_PT_PRED:
            feat_dict[self.k] = x
        # x = x.transpose(2,1).contiguous() # ! transpose here
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        # x = x.view(batchsize, n_pts, self.k)
        if return_feature_maps:
            return feat_dict
        else:
            return x

class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat  = PointNetfeat(global_feat=False, feature_transform=feature_transform)

        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = BatchNorm1d(512)
        self.bn2 = BatchNorm1d(256)
        self.bn3 = BatchNorm1d(128)

    def forward(self, x, gpu=0):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x, gpu=gpu)
        feat_concat = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        # x = x.transpose(2,1).contiguous() # ! transpose here
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        # x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

def feature_transform_regularizer(trans, gpu=0):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda(gpu)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    deploy_device   = torch.device('cuda:0')
    sim_data = torch.rand(32,3,1024, requires_grad=True, device=deploy_device)
    y = torch.empty(32, 1024, dtype=torch.long, device=deploy_device).random_(0, 20)
    trans = STN3d().cuda()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    # sim_data_64d = Variable(torch.rand(32, 64, 2500))
    # trans = STNkd(k=64)
    # out = trans(sim_data_64d)
    # print('stn64d', out.size())
    # print('loss', feature_transform_regularizer(out))

    # pointfeat = PointNetfeat(global_feat=True)
    # out, _, _ = pointfeat(sim_data)
    # print('global feat', out.size())

    # pointfeat = PointNetfeat(global_feat=False)
    # out, _, _ = pointfeat(sim_data)
    # print('point feat', out.size())

    # cls = PointNetCls(k = 5)
    # out, _, _ = cls(sim_data)
    # print('class', out.size())

    # seg = PointNetDenseCls(k = 20).cuda(0) # gpu name, with no feature transform (STN3D + featnet + global + mapping back)
    # out, trans, trans_feat = seg(sim_data, gpu=0)            # also specify the gpu name
    # print('seg', out.size())    # [B, K, N]
    # print('feat', trans.size()) # [B, 3, 3]
    # print('trans_feat', trans_feat.size()) # [B, C, N] C is the feature channels

    enc = PointNetEncoder().cuda(0)
    dec = PointNetDecoder(k=20).cuda(0)
    pred = dec(enc(sim_data, gpu=0))
    crit = nn.NLLLoss(ignore_index=-1).cuda(0)

    print(pred.shape)
    loss = crit(nn.functional.log_softmax(pred, dim=1), y) # add log softmax
    loss.backward()
    print('seg', pred.size()) # [B, K, N]
    print('loss', loss.size())
