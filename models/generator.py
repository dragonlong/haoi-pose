import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
# from src.dnn_lib import *
# from src.utils import *
import __init__
from common.generator_utils import get_spherical_lattice_grid, get_lattice_grid, sphere_mesh
import models.pointnet_lib.pointnet2_utils as pointnet2_utils

def bp():
    import pdb;pdb.set_trace()

class FcGenerator(nn.Module):
    def __init__(self, device=torch.device("cpu"), num_points=2048, equal_lr=False):
        super(FcGenerator, self).__init__()

        linear = EqualLinear if equal_lr else nn.Linear
        self.mlp = nn.Sequential(
            linear(512, 512),
            nn.LeakyReLU(),
            linear(512, 512),
            nn.LeakyReLU(),
            linear(512, 512),
            nn.LeakyReLU(),
            linear(512, num_points),
            nn.LeakyReLU(),
            linear(num_points, num_points * 3),
            nn.Sigmoid()
        ).to(device)

    def forward(self, z, _):
        """ input   z:      (B, 128)        B:batch size
            output  pcs:    (B, 512, 3)     B:batch size """
        batch_size = z.shape[0]
        out_pc = (self.mlp(z) - 0.5).reshape(batch_size, -1, 3)
        return out_pc, None

class DeconvGenerator(nn.Module):
    def __init__(self, device=torch.device("cpu"), num_points=1024, equal_lr=False):
        super(DeconvGenerator, self).__init__()
        self.num_points = num_points
        self.mlp = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=[2,2], stride=[1,1]),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 512, kernel_size=[3, 3], stride=[1, 1]),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 512, kernel_size=[4, 4], stride=[2, 2]),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 512, kernel_size=[5, 10], stride=[3, 6]),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 3, kernel_size=[1, 1], stride=[1, 1]),
            nn.Sigmoid()
        ).to(device)

    def forward(self, z, point_num):
        """ input   z:      (B, 512)        B:batch size
            output  pcs:    (B, num_points, 3)     B:batch size """
        batch_size = z.shape[0]
        latent_dim = z.shape[1]
        z = z.view(batch_size, -1, 1, 1)

        out_pc = (self.mlp(z) - 0.5).reshape(batch_size, 3, -1).contiguous()
        assert out_pc.shape[-1] == point_num
        out_pc = out_pc.transpose(1, 2).contiguous()

        return out_pc, None

class AtlasNetGenerator(nn.Module):
    def __init__(self, shape="sphere", shape_num=1, device=None, code_dim=512, point_num=2048, equal_lr=False, prob=False):
        super(AtlasNetGenerator, self).__init__()
        self.device = device
        self.shape = shape
        self.point_num= point_num
        if 'sphere' in shape:
            self.concat_dim = 3
        elif shape == "sheet":
            self.concat_dim = 2
        else:
            self.concat_dim = 0
        if shape_num > 1:
            self.concat_dim = self.concat_dim + 1

        # conv_1d = EqualConv1d if equal_lr else nn.Conv1d
        # linear = EqualLinear if equal_lr else nn.Linear
        conv_1d = nn.Conv1d
        linear = nn.Linear

        self.mlp = nn.Sequential(
            conv_1d(code_dim + self.concat_dim, 512, kernel_size=1),
            nn.LeakyReLU(),
            conv_1d(512, 512, kernel_size=1),
            nn.LeakyReLU(),
            conv_1d(512, 512, kernel_size=1),
            nn.LeakyReLU(),
            conv_1d(512, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU()
        ).to(device)
        if prob:
            self.toXYZ = nn.Sequential(
            linear(512, 64),
            nn.LeakyReLU(),
            linear(64, 4),
            nn.Sigmoid()).to(device)
        else:
            self.toXYZ = nn.Sequential(
                linear(512, 64),
                nn.LeakyReLU(),
                linear(64, 3),
                nn.Sigmoid()
            ).to(device)
        self.shape_num = shape_num
        self.vis = False


    def forward(self, z, use_fps=False):
        """ input   z:      (B, 512)        B:batch size
            output  pcs:    (B, N, 3)       B:batch size,   N:points number
                    cat:    (B, N_o, 3)     B:batch size,   N_o:origin points number """
        # import time
        # st = time.time()
        batch_size = z.shape[0]
        if use_fps:
            origin_point_num = max(8192, self.point_num)
        else:
            origin_point_num = self.point_num

        z_duplicate = torch.unsqueeze(z, 2).repeat(1, 1, origin_point_num)

        if self.shape == "sphere":
            concat_shape = get_spherical_lattice_grid(batch_size, 64, 32, delta_x=0, delta_y=0.5).to(self.device)
        elif self.shape == "uniform_sphere":
            concat_shape, faces = sphere_mesh(n=1024, radius=1.0, device=self.device)
            concat_shape = concat_shape.transpose(0, 1).contiguous().unsqueeze(0).contiguous().repeat(batch_size, 1, 1).contiguous()
        elif self.shape == "sheet":
            concat_shape = get_lattice_grid(batch_size, 64, 32, delta_x=0.5, delta_y=0.5).to(self.device)

        concat_shape = concat_shape.view(batch_size, self.concat_dim, -1)
        mlp_input = torch.cat((z_duplicate, concat_shape), 1)
        feature = self.mlp(mlp_input).transpose(1, 2).contiguous()
        pcs = self.toXYZ(feature)
        if not use_fps:
            return pcs.transpose(1, 2).contiguous()#, concat_shape
        else:
            pcs = pointnet2_utils.fps(pcs, origin_point_num)
            return pcs.transpose(1, 2).contiguous()#, concat_shape

    def set_vis(self, vis):
        self.vis = vis


class MultiAtlasNetConvGenerator(nn.Module):
    def __init__(self, shape="sphere", device=None, code_dim=512, utils=None, num=4):
        super(MultiAtlasNetConvGenerator, self).__init__()
        self.generators = nn.ModuleList([AtlasNetConvGeneratorV3(shape, device, code_dim, utils) for i in
                                         range(num)])
        self.generator_num = num
        self.device = device
        self.fps_begin = False
        self.resolution = False
        self.vis = False

    def forward(self, z, point_num):
        pcs_list = []
        concat_shape_list = []
        if self.fps_begin:
            point_num_per_generator = point_num // self.generator_num * 16
        else:
            point_num_per_generator = point_num // self.generator_num
        for i in range(self.generator_num):
            generator = self.generators[i]
            pcs_i, concat_shape = generator(z, point_num_per_generator)
            pcs_list.append(pcs_i)
            concat_shape_list.append(concat_shape)
        pcs = torch.cat(tuple(pcs_list), 1)
        point_num_pcs = pcs.shape[1]
        if point_num_pcs == point_num:
            return pcs, concat_shape_list
        else:
            pcs = pointnet2_utils.fps(pcs, point_num)
            return pcs, concat_shape_list


class AtlasNetConvGeneratorV3(nn.Module):
    def __init__(self, shape="sphere", device=None, code_dim=512, fps=False):
        super(AtlasNetConvGeneratorV3, self).__init__()

        self.shape = shape
        if self.shape == "sphere":
            self.padding_mode = "lr_circular"
            self.concat_dim = 3
        elif self.shape == "sheet":
            self.padding_mode = "zeros"
            self.concat_dim = 2
        else:
            raise NotImplementedError

        self.mlp = nn.Sequential(
            CircularConv2d(code_dim + self.concat_dim, 512, kernel_size=3, padding_mode=self.padding_mode),
            nn.LeakyReLU(0.2),
            CircularConv2d(512, 512, kernel_size=3, padding_mode=self.padding_mode),
            nn.LeakyReLU(0.2),
            CircularConv2d(512, 512, kernel_size=3, padding_mode=self.padding_mode),
            nn.LeakyReLU(0.2),
            CircularConv2d(512, 512, kernel_size=3, padding_mode=self.padding_mode),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        ).to(device)

        self.toXYZ = nn.Sequential(
            nn.Conv1d(512, 64, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 3, kernel_size=1),
            nn.Sigmoid()
        ).to(device)

        self.toXYZ = nn.Sequential(
            nn.Conv1d(512, 64, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 3, kernel_size=1),
            nn.Sigmoid()
        ).to(device)

        self.device = device
        self.if_upsample = False
        self.sampling_method = None
        self.resolution = False
        self.vis = False

    def forward(self, z, point_num):
        """ input   z:      (B, 512)        B:batch size
            output  pcs:    (B, N, 3)       B:batch size,   N:points number
                    cat:    (B, N_o, 3)     B:batch size,   N_o:origin points number """
        # import time
        # st = time.time()
        batch_size = z.shape[0]

        if not np.sqrt(point_num) == np.round(np.sqrt(point_num)):
            width = int(np.sqrt(point_num * 2))
            height = int(np.sqrt(point_num // 2))
        else:
            width = np.sqrt(point_num)
            assert width == np.round(width)
            width = int(width)
            height = width

        if not self.if_upsample:
            z_duplicate = z.unsqueeze(-1).unsqueeze(-1)
            z_duplicate = z_duplicate.repeat(1, 1, height, width)

            if self.shape == "sphere":
                concat_grid = get_spherical_lattice_grid(batch_size, width, height, delta_x=0, delta_y=0.5).to(self.device)
            else:
                concat_grid = get_lattice_grid(batch_size, width, height, delta_x=0.5, delta_y=0.5).to(self.device)

            mlp_input = torch.cat((z_duplicate, concat_grid), dim=1)#B, 512+C, H, W

            feature = self.mlp(mlp_input).view(batch_size, -1, point_num).contiguous()
            pcs = (self.toXYZ(feature) - 0.5).transpose(1, 2).contiguous()
        else:
            num_repeat_x = 3
            num_repeat_y = 3
            num_repeat = num_repeat_x * num_repeat_y

            if self.shape == "sphere":
                concat_grid = torch.zeros((0, 3, height, width)).to(self.device)
                for i in range(num_repeat):
                    delta_x = np.float(i)/num_repeat
                    new_concat_grid = get_spherical_lattice_grid(1, width, height, delta_x=delta_x, delta_y=0.5).to(self.device)
                    concat_grid = torch.cat((concat_grid, new_concat_grid), dim=0)
            else:##sheet
                concat_grid = torch.zeros((0, 2, height, width)).to(self.device)
                for i in range(num_repeat_x):
                    for j in range(num_repeat_y):
                        delta_x = np.float(i)/num_repeat_x
                        delta_y = np.float(j)/num_repeat_y
                        new_concat_grid = get_lattice_grid(1, width, height, delta_x=delta_x, delta_y=delta_y).to(self.device)
                        concat_grid = torch.cat((concat_grid, new_concat_grid), dim=0)

            z_duplicate = z.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)#B, 1, D, 1, 1
            z_duplicate = z_duplicate.repeat(1, num_repeat, 1, height, width)#B, N, D, height, width
            z_duplicate = z_duplicate.view(-1, z.shape[1], height, width) #B*N, D, height, width

            concat_grid = concat_grid.repeat(batch_size, 1, 1, 1) # (B*N, C, height, width)
            mlp_input = torch.cat((z_duplicate, concat_grid), dim=1) # (B*N, D+C, height, width)

            feature = self.mlp(mlp_input)
            feat_out_dim = feature.shape[1]
            feature = feature.view(batch_size, num_repeat, feat_out_dim, height, width).transpose(1, 2).contiguous()
            feature = feature.view(batch_size, feat_out_dim, -1)

            pcs = (self.toXYZ(feature) - 0.5).transpose(1, 2).contiguous()

            if self.sampling_method == "fps":
                pcs = pointnet2_utils.fps(pcs, point_num)
            elif self.sampling_method == "random":
                idx = torch.randperm(num_repeat*point_num)[:point_num]
                pcs = pcs[:, idx, :]
            elif self.sampling_method == "uniform":
                pcs_cpu = pcs.cpu().detach().numpy()
                new_pcs = np.zeros((0, point_num, 3), dtype=np.float32)
                for i in range(len(pcs_cpu)):
                    pcs_batch = uniform_smp2(pcs_cpu[i], point_num, ndiavox=30)
                    pcs_batch = pcs_batch[np.newaxis, :]
                    new_pcs = np.concatenate((new_pcs, pcs_batch), axis=0)
                pcs = torch.from_numpy(new_pcs).to(self.device)
            else:
                raise NotImplementedError

        return pcs, None


    def set_upsample(self):
        self.if_upsample = True

    def set_sampling_method(self, sampling_method):
        self.sampling_method = sampling_method

    def set_vis(self, vis):
        self.vis = vis

    def open_resolution(self):
        self.resolution = True

class AtlasNetSphereEdgeConvGenerator(nn.Module):
    def __init__(self, device=None, k=8, code_dim=512, num_points=2048):
        super(AtlasNetSphereEdgeConvGenerator, self).__init__()

        self.concat_dim = 3
        self.num_points = num_points
        self.k = k
        #sphere_1024 = np.loadtxt("Data/sphere/sphere_1024.txt")

        self.sphere_parameterization = get_spherical_lattice_grid(1, 64, 32, delta_x=0, delta_y=0.5).to(device)
        #self.sphere_parameterization = get_random_spherical_points(1, num_points).to(device)
        self.sphere_parameterization = self.sphere_parameterization.view(1, 3, -1).transpose(1, 2)

        self.conv_1  = nn.Sequential(
            nn.Conv2d(code_dim + 2*self.concat_dim, 512, kernel_size=1),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)).to(device)

        self.conv_2  = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)).to(device)

        self.conv_3  = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)).to(device)

        self.conv_4  = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)).to(device)

        self.toXYZ = nn.Sequential(
            nn.Conv1d(512, 64, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 3, kernel_size=1),
            nn.Sigmoid()
        ).to(device)

        self.device = device
        self.fps_begin = False
        self.resolution = False
        self.vis = False

    def forward(self, z, point_num):
        """ input   z:      (B, 512)        B:batch size
            output  pcs:    (B, N, 3)       B:batch size,   N:points number
                    cat:    (B, N_o, 3)     B:batch size,   N_o:origin points number """
        # import time
        # st = time.time()
        batch_size = z.shape[0]

        sphere_parameterization = self.sphere_parameterization.repeat(batch_size, 1, 1) #(B, N, 3)
        sphere_xyz = sphere_parameterization.transpose(1, 2).contiguous()# (B, 3, N)

        with torch.no_grad():
            sphere_knn_idx = pointnet2_utils.knn_unary(self.sphere_parameterization.transpose(1, 2), k=self.k)
            sphere_knn_idx = sphere_knn_idx.repeat(batch_size, 1, 1).detach() # (B, N, K)
        #z_duplicate = z.unsqueeze(dim=1) # (B, 1, D)
        # feature = torch.cat((sphere_parameterization, z_duplicate), dim=-1) # (B, N, 3+D)
        # feature = feature.transpose(1, 2).contiguous()#B, 3+D, N

        grouped_xyz = pointnet2_utils.grouping_operation(sphere_xyz, sphere_knn_idx) # (B, 3, N, K)
        sphere_xyz = sphere_xyz.unsqueeze(-1).repeat(1, 1, 1, self.k)# (B, 3, N, K)
        grouped_xyz -= sphere_xyz #(B, 3, N, K)
        z_duplicate = z.reshape(batch_size, -1, 1, 1).repeat(1, 1, self.num_points, self.k)# B, 512, N, K

        #print(sphere_xyz.shape, grouped_xyz.shape, z_duplicate.shape)
        grouped_feature_1 = torch.cat((sphere_xyz, grouped_xyz, z_duplicate), dim=1)# (B, 3+3+512, N, K)
        out = self.conv_1(grouped_feature_1) # B, 512, N, K
        out = out.mean(dim=-1) # B, 512, N

        grouped_feature_2 = pointnet2_utils.grouping_operation(out, sphere_knn_idx) # B, 512, N, K
        out = self.conv_2(grouped_feature_2) # B, 512, N, K
        out = out.mean(dim=-1) # B, 512, N

        grouped_feature_3 = pointnet2_utils.grouping_operation(out, sphere_knn_idx) # B, 512, N, K
        out = self.conv_3(grouped_feature_3) # B, 512, N, K
        out = out.mean(dim=-1) # B, 512, N

        grouped_feature_4 = pointnet2_utils.grouping_operation(out, sphere_knn_idx) # B, 512, N, K
        out = self.conv_4(grouped_feature_4) # B, 512, N, K
        feature = out.mean(dim=-1) # B, 512, N
        # z_duplicate = z.unsqueeze(dim=-1).repeat(1, 1, self.num_points) # (B, 1, D)
        # feature = torch.cat((sphere_xyz, z_duplicate), dim=1) # (B, 3+D, N)
        # feature = self.mlp(feature)

        pcs = (self.toXYZ(feature) - 0.5).transpose(1, 2).contiguous()

        return pcs, feature


    def set_fps(self, fps):
        self.fps_begin = fps

    def set_vis(self, vis):
        self.vis = vis

    def open_resolution(self):
        self.resolution = True


class AtlasNetSphereDynamicEdgeConvGenerator(nn.Module):
    def __init__(self, device=None, k=8, code_dim=512, num_points=2048):
        super(AtlasNetSphereDynamicEdgeConvGenerator, self).__init__()

        self.concat_dim = 3
        self.num_points = num_points
        self.k = k
        #sphere_1024 = np.loadtxt("Data/sphere/sphere_1024.txt")

        self.sphere_parameterization = get_random_spherical_points(1, num_points).to(device)
        self.sphere_parameterization = self.sphere_parameterization.transpose(1, 2)

        self.conv_1  = nn.Sequential(
            nn.Conv2d(code_dim + 2*self.concat_dim, 512, kernel_size=1),
            nn.LeakyReLU(0.2)).to(device)

        self.conv_2  = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(0.2)).to(device)

        self.conv_3  = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(0.2)).to(device)

        self.conv_4  = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)).to(device)

        self.toXYZ = nn.Sequential(
            nn.Conv1d(512, 64, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 3, kernel_size=1),
            nn.Sigmoid()
        ).to(device)

        self.device = device
        self.fps_begin = False
        self.resolution = False
        self.vis = False

    def forward(self, z, point_num):
        """ input   z:      (B, 512)        B:batch size
            output  pcs:    (B, N, 3)       B:batch size,   N:points number
                    cat:    (B, N_o, 3)     B:batch size,   N_o:origin points number """
        # import time
        # st = time.time()
        batch_size = z.shape[0]

        sphere_parameterization = self.sphere_parameterization.repeat(batch_size, 1, 1) #(B, N, 3)
        sphere_xyz = sphere_parameterization.transpose(1, 2).contiguous()# (B, 3, N)


        with torch.no_grad():
            sphere_knn_idx = pointnet2_utils.knn_unary(self.sphere_parameterization.transpose(1, 2), k=self.k)
            sphere_knn_idx = sphere_knn_idx.repeat(batch_size, 1, 1).detach() # (B, N, K)
        #z_duplicate = z.unsqueeze(dim=1) # (B, 1, D)
        # feature = torch.cat((sphere_parameterization, z_duplicate), dim=-1) # (B, N, 3+D)
        # feature = feature.transpose(1, 2).contiguous()#B, 3+D, N

        grouped_xyz = pointnet2_utils.grouping_operation(sphere_xyz, sphere_knn_idx) # (B, 3, N, K)
        sphere_xyz = sphere_xyz.unsqueeze(-1).repeat(1, 1, 1, self.k)# (B, 3, N, K)
        grouped_xyz -= sphere_xyz #(B, 3, N, K)
        z_duplicate = z.reshape(batch_size, -1, 1, 1).repeat(1, 1, self.num_points, self.k)# B, 512, N, K

        #print(sphere_xyz.shape, grouped_xyz.shape, z_duplicate.shape)
        grouped_feature_1 = torch.cat((sphere_xyz, grouped_xyz, z_duplicate), dim=1)# (B, 3+3+512, N, K)
        out = self.conv_1(grouped_feature_1) # B, 512, N, K
        out = out.mean(dim=-1) # B, 512, N

        with torch.no_grad():
            knn_idx_2 = pointnet2_utils.knn_unary(out, k=self.k).detach()

        grouped_feature_2 = pointnet2_utils.grouping_operation(out, knn_idx_2) # B, 512, N, K
        out = self.conv_2(grouped_feature_2) # B, 512, N, K
        out = out.mean(dim=-1) # B, 512, N

        with torch.no_grad():
            knn_idx_3 = pointnet2_utils.knn_unary(out, k=self.k).detach()

        grouped_feature_3 = pointnet2_utils.grouping_operation(out, knn_idx_3) # B, 512, N, K
        out = self.conv_3(grouped_feature_3) # B, 512, N, K
        out = out.mean(dim=-1) # B, 512, N

        with torch.no_grad():
            knn_idx_4 = pointnet2_utils.knn_unary(out, k=self.k).detach()

        grouped_feature_4 = pointnet2_utils.grouping_operation(out, knn_idx_4) # B, 512, N, K
        out = self.conv_4(grouped_feature_4) # B, 512, N, K
        feature = out.mean(dim=-1) # B, 512, N
        # z_duplicate = z.unsqueeze(dim=-1).repeat(1, 1, self.num_points) # (B, 1, D)
        # feature = torch.cat((sphere_xyz, z_duplicate), dim=1) # (B, 3+D, N)
        # feature = self.mlp(feature)

        pcs = (self.toXYZ(feature) - 0.5).transpose(1, 2).contiguous()

        return pcs, feature


    def set_fps(self, fps):
        self.fps_begin = fps

    def set_vis(self, vis):
        self.vis = vis

    def open_resolution(self):
        self.resolution = True


class TreeGCN(nn.Module):
    def __init__(self, depth, features, degrees, support=10, node=1, upsample=False, activation=True, loop_non_linear=False):
        self.depth = depth
        self.in_feature = features[depth]
        self.out_feature = features[depth+1]
        self.node = node
        self.degree = degrees[depth]
        self.upsample = upsample
        self.activation = activation
        self.loop_non_linear = loop_non_linear
        super(TreeGCN, self).__init__()

        # ancestor term
        self.W_root = nn.ModuleList([nn.Linear(features[inx], self.out_feature, bias=False) for inx in range(self.depth+1)])

        if self.upsample:
            # shape (node, in_feature, out_feature)
            self.W_branch = nn.Parameter(torch.FloatTensor(self.node, self.in_feature, self.degree*self.in_feature))

        if self.loop_non_linear:
            self.W_loop = nn.Sequential(nn.Linear(self.in_feature, self.in_feature*support, bias=False),
                                        nn.LeakyReLU(negative_slope=0.2),
                                        nn.Linear(self.in_feature*support, self.out_feature, bias=False))
            print('loop non linear',self.in_feature, self.in_feature*support)
        else:
            self.W_loop = nn.Sequential(nn.Linear(self.in_feature, self.in_feature*support, bias=False),
                                        nn.Linear(self.in_feature*support, self.out_feature, bias=False))

        if activation:
            self.bias = nn.Parameter(torch.FloatTensor(1, self.degree, self.out_feature))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.init_param()

    def init_param(self):
        if self.upsample:
            init.xavier_uniform_(self.W_branch.data, gain=init.calculate_gain('relu'))

        stdv = 1. / math.sqrt(self.out_feature)
        if self.activation:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, tree):
        batch_size = tree[0].shape[0]
        root = 0
        # ancestor term
        for inx in range(self.depth+1):
            root_num = tree[inx].size(1)
            repeat_num = int(self.node / root_num)
            root_node = self.W_root[inx](tree[inx])
            root = root + root_node.repeat(1,1,repeat_num).view(batch_size,-1,self.out_feature)
            # after reshape, for node = 2,
        branch = 0
        if self.upsample:
            branch = tree[-1].unsqueeze(2) @ self.W_branch
            branch = self.leaky_relu(branch)
            branch = branch.view(batch_size,self.node*self.degree,self.in_feature)
            # loop term
            branch = self.W_loop(branch)
            # add ancestor term
            branch = root.repeat(1,1,self.degree).view(batch_size,-1,self.out_feature) + branch
        else:
            branch = self.W_loop(tree[-1])

            branch = root + branch

        if self.activation:
            branch = self.leaky_relu(branch + self.bias.repeat(1,self.node,1))
        tree.append(branch)
        # tree.append(nn.Sigmoid())
        # at depth = 2, node = 2, W-branch (2, 256, 512), in feature,
        # root = (b, 2, 256)
        return tree


class TreeGANGenerator(nn.Module):
    def __init__(self,features,degrees,support,loop_non_linear=False):
        self.layer_num = len(features)-1
        assert self.layer_num == len(degrees), "Number of features should be one more than number of degrees."
        self.pointcloud = None
        super(TreeGANGenerator, self).__init__()

        vertex_num = 1
        self.gcn = nn.Sequential()
        # self.sigmoid    = nn.Sigmoid()
        for inx in range(self.layer_num):
            # NOTE last layer activation False
            if inx == self.layer_num-1:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(inx, features, degrees,
                                            support=support, node=vertex_num, upsample=True, activation=False, loop_non_linear=loop_non_linear))
            else:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(inx, features, degrees,
                                            support=support, node=vertex_num, upsample=True, activation=False, loop_non_linear=loop_non_linear))
            vertex_num = int(vertex_num * degrees[inx])

    def forward(self, z):
        tree = [z.unsqueeze(1)]
        feat = self.gcn(tree)

        self.pointcloud = feat[-1] # 0.5

        return self.pointcloud.permute(0, 2, 1).contiguous()

    def getPointcloud(self):
        return self.pointcloud[-1] + 0.5 # return a single point cloud (2048,3)

    def get_params(self,index):

        if index < 7:
            for param in self.gcn[index].parameters():
                yield param
        else:
            raise ValueError('Index out of range')

class SimpleAtlasNetGenerator(nn.Module):
    def __init__(self, shape="sphere", shape_num=1, overlap=True, device=None, utils=None):
        super(SimpleAtlasNetGenerator, self).__init__()
        if shape == "sphere":
            concat_dim = 3
        elif shape == "sheet":
            concat_dim = 2
        else:
            concat_dim = 0
        if shape_num > 1:
            concat_dim = concat_dim + 1
        self.mlp = nn.Sequential(
            nn.Conv1d(128 + concat_dim, 512, kernel_size=1),
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Conv1d(512, 1024, kernel_size=1),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Conv1d(1024, 256, kernel_size=1),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU()
        ).to(device)
        self.toXYZ = nn.Sequential(
            nn.Linear(256, 3),
            nn.Sigmoid()
        ).to(device)
        self.shape = shape
        self.shape_num = shape_num
        self.overlap = overlap
        self.device = device
        self.fps_begin = False
        self.resolution = False
        self.vis = False
        if utils is None:
            self.utils = GetConcatPoints()
        else:
            self.utils = utils

    def forward(self, z, point_num):
        """ input   z:      (B, 128)        B:batch size
            output  pcs:    (B, N, 3)       B:batch size,   N:points number
                    cat:    (B, N_o, 3)     B:batch size,   N_o:origin points number """
        # import time
        # st = time.time()
        batch_size = z.shape[0]
        if self.resolution:
            point_num = 8196
        if self.fps_begin:
            origin_point_num = max(8192, point_num)
        else:
            origin_point_num = point_num

        z_duplicate = torch.unsqueeze(z, 1).repeat(1, origin_point_num, 1)
        if self.vis is False:
            concat_shape = self.utils.get_concat_points(batch_size, origin_point_num, self.shape, self.shape_num,
                                                        self.overlap, self.device)
        else:
            concat_shape = get_mesh_points("mesh/mesh_points.pk", origin_point_num).to(self.device).unsqueeze(0).repeat(
                batch_size, 1, 1)
        mlp_input = torch.cat((z_duplicate, concat_shape), 2)
        feature = self.mlp(mlp_input.transpose(1, 2)).transpose(1, 2)
        pcs = self.toXYZ(feature) - 0.5
        if origin_point_num == point_num:
            return pcs, concat_shape
        else:
            pcs_flipped = pcs.transpose(1, 2).contiguous()
            pcs = gather_operation(pcs_flipped, furthest_point_sample(pcs, point_num)).transpose(1, 2).contiguous()
            return pcs, concat_shape

    def set_fps(self, fps):
        self.fps_begin = fps

    def set_vis(self, vis):
        self.vis = vis

    def open_resolution(self):
        self.resolution = True


class MultiAtlasNetGenerator(nn.Module):
    def __init__(self, generator_num=4, shape="sphere", shape_num=1, overlap=True, device=None):
        super(MultiAtlasNetGenerator, self).__init__()
        self.generator_num = generator_num
        self.utils = GetConcatPoints()
        self.generators = nn.ModuleList([AtlasNetGenerator(shape, shape_num, overlap, device, self.utils) for i in
                                         range(generator_num)])
        self.fps_begin = False

    def forward(self, z, point_num):
        """ input   z:      (B, 128)        B:batch size
            output  pcs:    (B, N, 3)       B:batch size,   N:points number
                    cat:    [(B, N_o, 3),]  B:batch size,   N_o:origin points number """
        pcs_list = []
        concat_shape_list = []
        if self.fps_begin:
            point_num_per_generator = 256
        else:
            point_num_per_generator = point_num // self.generator_num
        for i in range(self.generator_num):
            generator = self.generators[i]
            pcs_i, concat_shape = generator(z, point_num_per_generator)
            pcs_list.append(pcs_i)
            concat_shape_list.append(concat_shape)
        pcs = torch.cat(tuple(pcs_list), 1)
        point_num_pcs = pcs.shape[1]
        if point_num_pcs == point_num:
            return pcs, concat_shape_list
        else:
            pcs_flipped = pcs.transpose(1, 2).contiguous()
            pcs = gather_operation(pcs_flipped, furthest_point_sample(pcs, point_num)).transpose(1, 2).contiguous()
            return pcs, concat_shape_list

    def set_fps(self, fps):
        self.fps_begin = fps
        # for i in range(self.generator_num):
        #     self.generators[i].open_fps()

    def set_vis(self, vis):
        for i in range(self.generator_num):
            self.generators[i].set_vis(vis)


class TwoStageGenerator(nn.Module):
    def __init__(self, device=torch.device("cpu"), random=False, feature=1):
        super(TwoStageGenerator, self).__init__()
        self.show = False
        self.random = random
        self.feature = feature
        self.mlp1 = FirstAtlasNet(device, self.show)
        self.toXYZ1 = ToXYZ(device, self.show)
        self.grouper = QueryAndGroup(radius=0.15, nsample=4, use_xyz=False)
        feature_size = 256 * (self.feature + 1)
        # self.mlp2 = TriangleAtlasNet(device, self.show, feature_size)
        self.mlp2 = OldSecondAtlasNet(device, self.show, feature_size)
        self.toXYZ2 = ToXYZ(device, self.show)
        self.fps = False
        self.vis = False

    def forward(self, z, num1, num2, use_fps=False):
        """
        :param z: latent code
        :param num1: number of first stage point cloud
        :param num2: number of second stage point cloud
        :return pc1: first stage point cloud
        :return pc2: second stage point cloud
        """
        if self.show:
            print("Two Stage Generator Module")
            print("TwoStageG_z", z.size())
            print("TwoStageG_num1", num1)
            print("TwoStageG_num2", num2)
            print("")

        feature1 = self.mlp1(z, num1, group=1, random=self.random)
        pc1 = self.toXYZ1(feature1)
        # if self.fps:
        #     pc2 = self.mlp2(z, feature1, pc1, num2 * 16, random=self.random, vis=self.vis)
        # else:
        #     pc2 = self.mlp2(z, feature1, pc1, num2, random=self.random, vis=self.vis)

        feature1_flip = feature1.transpose(1, 2)

        if self.feature >= 1:
            local_feature_group = self.grouper(pc1, pc1, feature1_flip)
            local_feature = torch.nn.functional.max_pool2d(local_feature_group,
                                                           kernel_size=[1, local_feature_group.size(3)])
            local_feature = local_feature.squeeze(-1).transpose(1, 2)
            feature1 = torch.cat((feature1, local_feature), dim=2)
            if self.show:
                print("TwoStageG_local_feature_group", local_feature_group.size())
                print("TwoStageG_local_feature", local_feature.size())
                print("TwoStageG_feature1", feature1.size())
                print("")

        if self.feature >= 2:
            global_feature = torch.nn.functional.max_pool1d(feature1_flip, kernel_size=num1)
            global_feature = global_feature.repeat(1, 1, num1).transpose(1, 2)
            feature1 = torch.cat((feature1, global_feature), dim=2)
            if self.show:
                print("TwoStageG_global_feature", global_feature.size())
                print("TwoStageG_feature1", feature1.size())
                print("")

        # feature = feature1.detach()
        feature = feature1
        if use_fps:
            feature2 = self.mlp2(feature, num2 * 16, group=num2 * 16 // num1, random=self.random, vis=self.vis)
        else:
            feature2 = self.mlp2(feature, num2, group=num2 // num1, random=self.random, vis=self.vis)
        pc2_delta = self.toXYZ2(feature2)  # B * N2 * 3
        pc2 = pc2_delta + pc1.repeat(1, 1, pc2_delta.size(1) // num1).view(-1, pc2_delta.size(1), 3)
        if use_fps:
            pc2_flipped = pc2.transpose(1, 2).contiguous()
            pc2 = gather_operation(pc2_flipped, furthest_point_sample(pc2, num2)).transpose(1, 2).contiguous()

        if self.show:
            # print("TwoStageG_global_feature2", feature2.size())
            # print("TwoStageG_pc2_delta", pc2_delta.size())
            print("TwoStageG_pc2", pc2.size())
            print("")
        return pc1, pc2, pc2

    def set_fps(self, fps):
        self.fps = fps

    def set_vis(self, vis):
        self.vis = vis


class ToXYZ(nn.Module):
    def __init__(self, device=None, show=False):
        super(ToXYZ, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        ).to(device)
        self.device = device if device is not None else torch.device("cpu")
        self.show = show

    def forward(self, feature):
        """
        :param feature: (B, N, F)
        :return pcs: (B, N, D)
        """
        pcs = self.model(feature) - 0.5

        if self.show:
            print("ToXYZ Module:")
            print("G_ToXYZ_input_feature:", feature.shape)
            print("G_ToXYZ_output_pcs:", pcs.shape)
            print("")
        return pcs


class FirstAtlasNet(nn.Module):
    # from latent code z to 256 feature
    # input latent code z : B * 128
    # output feature : B * N * F, N=256, F=256
    def __init__(self, device=None, show=False):
        super(FirstAtlasNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(128 + 2, 256, kernel_size=1),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Conv1d(256, 512, kernel_size=1),
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Conv1d(512, 1024, kernel_size=1),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Conv1d(1024, 256, kernel_size=1),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        ).to(device)
        self.device = device if device is not None else torch.device("cpu")
        self.show = show
        self.get_concat_points = GetConcatPoints()

    def forward(self, z, point_num, group=1, random=True):
        """
        :param z: (B, 128) latent code
        :param point_num: point_num = N int
        :param group: int
        :param random: boolean
        :return: feature: (B, N, F)
        """
        B, _ = z.shape
        N = point_num

        z_duplicate = torch.unsqueeze(z, 1).repeat((1, N, 1))
        if random:
            concat_points = get_random_lattice_points(B, N).to(self.device)
            # concat_points = self.get_concat_points.get_concat_points(B, N, "sheet", 4, False, self.device)
        else:
            concat_points = get_lattice_points(B, N, group).to(self.device)
        z_concat = torch.cat((z_duplicate, concat_points), 2)
        feature = self.model(z_concat.transpose(1, 2)).transpose(1, 2)

        if self.show:
            print("Generator_Block1 Module:")
            print("G_Block1_input_point_num", point_num)
            print("G_Block1_input_group", group)
            print("G_Block1_input_random", random)
            print("G_Block1_input_z", z.shape)
            print("G_Block1_z_duplicate", z_duplicate.shape)
            print("G_Block1_concat_points", concat_points.shape)
            print("G_Block1_z_concat", z_concat.shape)
            print("G_Block1_output_feature", feature.shape)
            print("")

        return feature


class OldSecondAtlasNet(nn.Module):
    # form the 256 feature of FirstBlock to 4096 feature
    # input feature: B * N1 * F, N1=256 F=256
    # output feature: B * N2 * F, N2=4096 (256 * 16) F=256
    def __init__(self, device=None, show=False, feature_size=256):
        super(OldSecondAtlasNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(feature_size + 2, 512, kernel_size=1),
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Conv1d(512, 1024, kernel_size=1),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Conv1d(1024, 256, kernel_size=1),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        ).to(device)
        self.device = device if device is not None else torch.device("cpu")
        self.show = show

    def forward(self, feature_N1, point_num, group=1, random=False, vis=False):
        """
        :param feature_N1: (B, N1, F)
        :param point_num: int
        :param group: int
        :param random: boolean
        :return feature_N2: (B, N2, F)
        """
        B, N1, F = feature_N1.shape
        N2 = point_num
        up_sample = N2 // N1

        feature_duplicate = feature_N1.repeat((1, 1, up_sample)).view(B, N2, F)
        if vis is False:
            if random:
                concat_points = get_random_lattice_points(B, up_sample).to(self.device).repeat((1, N1, 1))
            else:
                concat_points = get_lattice_points(B, up_sample, group).to(self.device).repeat((1, N1, 1))
        else:
            concat_points = get_mesh_points("mesh/mesh_points.pk", N2).to(self.device).unsqueeze(0).repeat(B, 1, 1)
        feature_concat = torch.cat((feature_duplicate, concat_points), 2)
        feature_N2 = self.model(feature_concat.transpose(1, 2)).transpose(1, 2)

        if self.show:
            print("Generator_Block2 Module:")
            print("G_Block2_input_feature_N1", feature_N1.shape)
            print("G_Block2_input_point_num", point_num)
            print("G_Block2_input_group", group)
            print("G_Block2_input_random", random)
            print("G_Block2_feature_duplicate", feature_duplicate.shape)
            print("G_Block2_concat_points", concat_points.shape)
            print("G_Block2_feature_concat", feature_concat.shape)
            print("G_Block2_output_feature_N2", feature_N2.shape)
            print("")

        return feature_N2


class SecondAtlasNet(nn.Module):
    def __init__(self, device=None, show=False, feature_size=256):
        super(SecondAtlasNet, self).__init__()
        self.concat_z = False
        if self.concat_z:
            self.feature_size = feature_size + 128
        else:
            self.feature_size = feature_size
        self.model = nn.Sequential(
            nn.Conv1d(self.feature_size + 2, 512, kernel_size=1),
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Conv1d(512, 1024, kernel_size=1),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Conv1d(1024, 256, kernel_size=1),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        ).to(device)
        self.device = device if device is not None else torch.device("cpu")
        self.grouper = QueryAndGroup(0.15, 4, use_xyz=False)
        self.show = show

    def forward(self, z, coarse_feature, coarse_pc, point_num, random=False, vis=False):
        """
        :param z: latent code B * ndim
        :param coarse_feature: B * N1 * F
        :param coarse_pc: B * N1 * 3
        :param point_num: N2
        :param random: False
        :param vis: False
        :return:
        """
        if self.show:
            print("Generator_Block2 Module:")
        B, N, F = coarse_feature.shape
        M = point_num // N
        coarse_feature_flip = coarse_feature.transpose(1, 2)
        if self.concat_z:
            patch_feature = torch.cat((z.repeat(1, N).view(B, N, -1), coarse_feature), 2)
        else:
            patch_feature = coarse_feature
        if self.feature_size > 256 + 128:
            local_feature_group = self.grouper(coarse_pc, coarse_pc, coarse_feature_flip)
            local_feature = torch.nn.functional.avg_pool2d(local_feature_group,
                                                           kernel_size=[1, local_feature_group.size(3)])
            local_feature = local_feature.squeeze(-1).transpose(1, 2)
            patch_feature = torch.cat((patch_feature, local_feature), dim=2)
            if self.show:
                print("G_Block2_local_feature_group", local_feature_group.size())
                print("G_Block2_local_feature", local_feature.size())
                print("G_Block2_patch_feature", patch_feature.size())
        if self.feature_size > 512 + 128:
            global_feature = torch.nn.functional.avg_pool1d(coarse_feature_flip, kernel_size=N)
            global_feature = global_feature.repeat(1, 1, N).transpose(1, 2)
            patch_feature = torch.cat((patch_feature, global_feature), dim=2)
            if self.show:
                print("G_Block2_global_feature", global_feature.size())
                print("G_Block2_patch_feature", patch_feature.size())

        patch_feature = patch_feature.repeat((1, 1, M)).view((B, point_num, -1))
        if vis is False:
            if random:
                concat_points = get_random_lattice_points(B, M).to(self.device).repeat((1, N, 1))
            else:
                concat_points = get_lattice_points(B, M, M).to(self.device).repeat((1, N, 1))
        else:
            concat_points = get_mesh_points("mesh/mesh_points.pk", point_num).to(self.device).unsqueeze(0).repeat(B, 1,
                                                                                                                  1)

        feature_concat = torch.cat((patch_feature, concat_points), 2)
        fine_feature = self.model(feature_concat.transpose(1, 2)).transpose(1, 2)
        if self.show:
            print("G_Block2_patch_feature", patch_feature.shape)
            print("G_Block2_concat_points", concat_points.shape)
            print("G_Block2_feature_concat", feature_concat.shape)
            print("G_Block2_fine_feature", fine_feature.shape)
            print("")
        return fine_feature


# class TriangleAtlasNet(nn.Module):

class TwoStageGenerator_V2(nn.Module):
    def __init__(self, device=torch.device("cpu"), feature="global"):
        super(TwoStageGenerator_V2, self).__init__()
        self.first_stage_feature = nn.Sequential(
            nn.Conv1d(512 + 2, 512, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(512, 512, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(512, 1024, kernel_size=1),
            nn.LeakyReLU(),
        ).to(device)
        self.first_stage_points = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 3),
            nn.Sigmoid()
        ).to(device)
        self.feature = feature
        if feature == "self":
            self.feature_size = 1024
        elif feature == "local":
            self.feature_size = 1024 * 2
        elif feature == "global":
            self.feature_size = 1024 * 3
        else:
            self.feature_size = 0
            print("wrong feature type!!!")
        self.second_stage_feature = nn.Sequential(
            nn.Conv1d(self.feature_size, 1024, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(1024, 512, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv1d(512, 1024, kernel_size=1),
            nn.LeakyReLU()
        ).to(device)
        self.second_stage_points = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 3),
        ).to(device)
        self.grouper = QueryAndGroup(0.1, 64, use_xyz=False)
        self.device = device

    def forward(self, z, n1, n2):
        b, dim = z.shape

        z_duplicate = z.view(b, 1, dim).repeat(1, n1, 1)
        concat_points = get_lattice_points(b, n1).to(self.device)
        z_concat = torch.cat((z_duplicate, concat_points), 2)
        first_stage_feature = self.first_stage_feature(z_concat.transpose(1, 2)).transpose(1, 2)
        first_stage_points = self.first_stage_points(first_stage_feature) - 0.5

        z_duplicate = z.view(b, 1, dim).repeat(1, n2, 1)
        concat_points = get_lattice_points(b, n2).to(self.device)
        z_concat = torch.cat((z_duplicate, concat_points), 2)
        self_feature = self.first_stage_feature(z_concat.transpose(1, 2))
        coarse_points = self.first_stage_points(self_feature.transpose(1, 2)) - 0.5
        second_stage_feature_list = [self_feature]
        # print(self_feature.shape)
        if self.feature == "local" or self.feature == "global":
            local_feature_group = self.grouper(coarse_points, coarse_points, self_feature)
            # print(local_feature_group.shape)
            local_feature = F.max_pool2d(local_feature_group, kernel_size=[1, 64]).squeeze(-1)
            second_stage_feature_list.append(local_feature)
            # print(local_feature.shape)
        if self.feature == "global":
            global_feature = F.max_pool1d(self_feature, kernel_size=n2).view(b, 1024, 1).repeat(1, 1, n2)
            second_stage_feature_list.append(global_feature)
            # print(global_feature.shape)
        second_stage_feature = torch.cat(second_stage_feature_list, 1)
        # print(second_stage_feature.shape)
        second_stage_feature = self.second_stage_feature(second_stage_feature)
        second_stage_points = self.second_stage_points(second_stage_feature) + coarse_points

        return first_stage_points, second_stage_points, coarse_points

class TwoStageAtlasNetGenerator(nn.Module):
    def __init__(self, device=torch.device("cpu"), feature="global"):
        super(TwoStageAtlasNetGenerator, self).__init__()
        self.first_stage_feature = nn.Sequential(
            nn.Conv1d(512 + 2, 512, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, kernel_size=1),
            nn.LeakyReLU(0.2),
        ).to(device)
        self.to_xyz_1 = nn.Sequential(
            nn.Conv1d(512, 3, kernel_size=1),
            nn.Sigmoid()
        ).to(device)
        self.feature = feature
        if feature == "self":
            self.feature_size = 512
        elif feature == "local":
            self.feature_size = 512 * 2
        elif feature == "global":
            self.feature_size = 512 * 3
        else:
            self.feature_size = 0
            print("wrong feature type!!!")
        self.second_stage_feature = nn.Sequential(
            nn.Conv1d(self.feature_size, 512, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, kernel_size=1),
            nn.LeakyReLU(0.2)
        ).to(device)
        self.to_xyz_2 = nn.Sequential(
            nn.Conv1d(512, 3, kernel_size=1),
            nn.Sigmoid()
        ).to(device)
        self.grouper = QueryAndGroup(0.1, 64, use_xyz=False)
        self.device = device

    def forward(self, z, n1, n2):
        b, dim = z.shape

        z_duplicate_1 = z.view(b, 1, dim).repeat(1, n1, 1)
        concat_points_1 = get_lattice_points(b, n1).to(self.device)
        z_concat_1 = torch.cat((z_duplicate_1, concat_points_1), 2).transpose(1, 2)

        first_stage_coarse_feature = self.first_stage_feature(z_concat_1)
        first_stage_points = self.to_xyz_1(first_stage_coarse_feature).transpose(1, 2).contiguous() - 0.5## B, 64, 3


        z_duplicate_2 = z.view(b, 1, dim).repeat(1, n2, 1)
        concat_points_2 = get_lattice_points(b, n2).to(self.device)
        z_concat_2 = torch.cat((z_duplicate_2, concat_points_2), 2).transpose(1, 2)
        first_stage_fine_feature = self.first_stage_feature(z_concat_2)   ## B, 512, 1024
        first_stage_fine_points = self.to_xyz_1(first_stage_fine_feature).transpose(1,2).contiguous() - 0.5## B, 1024, 3

        self_feature = first_stage_fine_feature
        second_stage_feature_list = [self_feature]
        # print(self_feature.shape)
        if self.feature == "local" or self.feature == "global":
            local_feature_group = self.grouper(first_stage_fine_points, first_stage_fine_points, self_feature) ##
            # print(local_feature_group.shape)
            local_feature = torch.max(local_feature_group, dim=-1)[0]
            second_stage_feature_list.append(local_feature)
            # print(local_feature.shape)
        if self.feature == "global":
            global_feature = torch.max(first_stage_fine_feature, dim=2, keepdim=True)[0]
            global_feature = global_feature.repeat(1, 1, n2) ## B, 512, 1024
            second_stage_feature_list.append(global_feature)
            # print(global_feature.shape)
        second_stage_feature = torch.cat(second_stage_feature_list, 1)
        # print(second_stage_feature.shape)
        second_stage_feature = self.second_stage_feature(second_stage_feature)
        delta_points = 0.1*(self.to_xyz_2(second_stage_feature) - 0.5).transpose(1, 2).contiguous()
        second_stage_points = delta_points + first_stage_fine_points

        return first_stage_points, second_stage_points, first_stage_fine_points

if __name__ == '__main__':
    # generator_ins = AtlasNetGenerator(device=torch.device('cuda:0'), code_dim=128, point_num=2048)
    # print(generator_ins)
    # BS = 4
    # gpu = 0
    # device   = torch.device('cuda:{}'.format(gpu))
    # z        = torch.rand(BS, 128, requires_grad=True, device=device)
    # outputs = generator_ins(z)
    # print(outputs[0].shape, outputs[1].shape )
    # print('Con!!!')
    DEGREE=[1,  2,   2,   2,   2,   2,   64]
    G_FEAT=[128, 256, 256, 256, 128, 128, 128, 3]
    D_FEAT=[3, 64,  128, 256, 256, 512]
    support=10
    loop_non_linear= False
    generator_ins = TreeGANGenerator(features=G_FEAT, degrees=DEGREE, support=support, loop_non_linear=loop_non_linear).cuda()
    print(generator_ins)
    z = torch.zeros((4, 1, 128)).normal_().cuda()
    bp()
    out = generator_ins([z])
