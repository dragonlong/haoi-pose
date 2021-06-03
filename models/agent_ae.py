import torch
import torch.nn as nn
import numpy as np
import wandb
import torchvision
import torch.nn.functional as F

import pytorch3d
from pytorch3d.renderer import compositing as compositing
from pytorch3d.renderer.points import rasterize_points
from pytorch3d.structures import Pointclouds

import __init__
from models.ae_gan import get_network
from models.base import BaseAgent
from utils.extensions.chamfer_dist import ChamferDistance
from utils.p2i_utils import look_at
from models.losses import loss_geodesic, loss_vectors, compute_vect_loss, compute_1vN_nocs_loss, compute_miou_loss
from common.d3_utils import compute_rotation_matrix_from_euler, compute_euler_angles_from_rotation_matrices, compute_rotation_matrix_from_ortho6d, mean_angular_error, angle_from_R
from common.yj_pose import compute_pose_diff, rot_diff_degree, rot_diff_rad
from common.quaternion import matrix_to_unit_quaternion, unit_quaternion_to_matrix
from common.rotations import rotate
from models.pointnet_lib.pointnet2_modules import farthest_point_sample, gather_operation
from os import makedirs, remove
from os.path import exists, join
from utils.ycb_eval_utils import Basic_Utils

import vgtk.so3conv.functional as L
from vgtk.functional import compute_rotation_matrix_from_quaternion, compute_rotation_matrix_from_ortho6d, so3_mean
from global_info import global_info

infos           = global_info()
my_dir          = infos.base_path
delta_R         = infos.delta_R
delta_T         = infos.delta_T
def bp():
    import pdb;pdb.set_trace()

epsilon=1e-10
THRESHOLD_GOOD = 5  # in degrees
THRESHOLD_BAD  = 50 # in degrees

def process_rotation(config, raw, batch_size):
    R_dim = 6 if config.pred_6d else 3
    M = config.MODEL.num_channels_R
    B = batch_size
    if 'se3' in config.encoder_type:  # raw: [B * N, M * R_dim / 3, 3]
        raw = raw.view(B, -1, M, R_dim).contiguous().permute(0, 3, 1, 2)  # [B, R_dim, N, M]
    else:  # raw: [B, M * R_dim, N]
        raw = raw.view(B, M, R_dim, -1).contiguous().permute(0, 2, 3, 1)  # [B, R_dim, N, M]

    def normalize_3d(rot):  # rot: [B, 3, *]
        return rot / (torch.norm(rot, dim=1, keepdim=True) + epsilon)

    if config.pred_6d:
        pass
    else:
        rotation = normalize_3d(raw)  # [B, R_dim, N, M]
        avg_rotation = rotation.mean(dim=2)
        avg_rotation = normalize_3d(avg_rotation)

    return rotation, avg_rotation

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PointAEPoseAgent(BaseAgent):
    def __init__(self, config):
        super(PointAEPoseAgent, self).__init__(config)
        if 'ycb' in self.config.task:
            self.bs_utils = Basic_Utils(config=self.config) #, chamfer_dist=self.chamfer_dist)
            self.ycb_last_pose_err = None
        self.flip_axis = config.target_category in ['can']
        print('flip axis', self.flip_axis)

    def build_net(self, config):
        # customize your build_net function
        net = get_network(config, "pointAE")

        print(net)
        if config.parallel:
            net = nn.DataParallel(net)
        net = net.cuda()
        return net

    def synchronize_input(self, data):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        target_keys = ['xyz', 'points', 'C', 'T', 'R_gt', 'R', 'R_label']
        for key in target_keys:
            if key in data:
                data[key] = data[key].to(device)
                if (key == 'xyz' or key == 'points') and self.config.use_fps_points:
                    fps_idx = farthest_point_sample(data[key], npoint=1024)
                    new_xyz = gather_operation(data[key].permute(0, 2, 1).contiguous(), fps_idx.int())  # [B, C, S]
                    data[key] = new_xyz.permute(0, 2, 1).contiguous()

    def forward(self, data, verbose=False):
        """
        1. forwarding;
        2. pred+loss+info for pose;
        3. extra losses;
        """
        self.infos    = {}
        self.synchronize_input(data) #
        if 'se3' in self.config.encoder_type:
            self.predict_se3(data)
        else:
            self.predict_pnet2(data)

        # for general loss
        if self.config.pred_6d:
            self.compute_and_eval_6d(data)
        if self.config.pred_axis:
            self.compute_and_eval_axis(data)
        if 'so3' in self.config.encoder_type or self.config.use_head_assemble:
            self.compute_and_eval_so3(data)
        #
        self.compute_loss(data)

    def predict_se3(self, data):
        # bp()
        BS = data['points'].shape[0]
        N  = data['points'].shape[1] # B, N, 3
        M  = self.config.num_modes_R
        CS = self.config.MODEL.num_channels_R
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.latent_vect = self.net.encoder(G=None, xyz=data['xyz'])
        # self.latent_vect = self.net.encoder(data['G'].to(device))
        if self.config.pred_nocs:
            self.output_N = self.net.regressor_nocs(self.latent_vect['N'].squeeze(-1).view(BS, -1, self.latent_vect['N'].shape[1]).contiguous().permute(0, 2, 1).contiguous()) # assume to be B*N, 128? --> 3 channel
        # self.output_T = self.latent_vect['T'].permute(0, 2, 1).contiguous().squeeze(-1).view(BS, -1, 3).contiguous().permute(0, 2, 1).contiguous()# [B, 3, N]
        # BS * N, M, 3 + [BS, N, 3, 1]--> BS, 3, M
        self.output_T = -self.latent_vect['T'].permute(0, 2, 1).contiguous().view(BS, N, 3, M).contiguous() + (data['xyz'] - data['xyz'].mean(dim=1, keepdim=True)).unsqueeze(-1).contiguous() # [B, 3, N]
        self.output_T = self.output_T.mean(dim=1)
        if torch.isnan(self.output_T).any():
            print('we have nan in self.output_T')
        if self.config.pred_6d:
            self.output_R = compute_rotation_matrix_from_ortho6d(self.latent_vect['R'].view(-1, 6).contiguous())
            self.output_R_pooled = compute_rotation_matrix_from_ortho6d(self.latent_vect['1'].view(-1, 6).contiguous()) # dense by default!!! B, 2*C, 3

        if self.config.pred_seg:
            self.output_C = self.net.classifier_seg(self.latent_vect['N'].squeeze(-1).view(BS, -1, self.latent_vect['N'].shape[1]).contiguous().permute(0, 2, 1).contiguous())

        if self.config.pred_mode: # map 4 to 2
            self.output_M = self.net.classifier_mode(self.latent_vect['R0']).squeeze(-1) # [BS * N, M, 1]-->[BS * N, M] --> [BS, M, N]

        if 'completion' in self.config.task:
            if 'ycb' in self.config.task:
                self.output_pts = data['full'].permute(0, 2, 1).to(
                    self.latent_vect['R'].device).float()  # [B, N, 3] -> [B, 3, N]
            elif isinstance(self.latent_vect, dict):
                self.output_pts = self.net.decoder(self.latent_vect['0'])
            else:
                self.output_pts = self.net.decoder(self.latent_vect)

    def predict_pnet2(self, data):
        BS = data['points'].shape[0]
        N  = data['points'].shape[1] # B, N, 3
        M  = self.config.num_modes_R
        CS = self.config.MODEL.num_channels_R
        # B, 3, N
        if 'xyz' in data:
            input_pts = data['xyz'].permute(0, 2, 1).contiguous()
        else:
            input_pts    = data['G'].ndata['x'].view(BS, -1, 3).contiguous().permute(0, 2, 1).contiguous() #
        input_pts        = input_pts - input_pts.mean(dim=-1, keepdim=True)
        # input_pts = input_pts - 0.5 * (input_pts.min(dim=-1, keepdim=True)[0] + input_pts.max(dim=-1, keepdim=True)[0])
        if self.config.use_rgb:
            rgb_feat = data['G'].ndata['f'].view(BS, -1, 3).contiguous().permute(0, 2, 1).contiguous() #
            feat = torch.cat([input_pts, rgb_feat], dim=1)
        else:
            feat = input_pts
        self.latent_vect = self.net.encoder(feat) # double

        self.output_T    = self.latent_vect['T']  # 3, N, no activation
        if self.config.pred_nocs:
            self.output_N = self.latent_vect['N']
        if self.config.pred_6d:
            if M > 1:
                self.output_R = self.latent_vect['R'].view(BS, -1, 3, N).contiguous().view(BS, M, 2, 3, N).contiguous() # B, 6, N
                self.output_R = self.output_R/(torch.norm(self.output_R, dim=3, keepdim=True) + epsilon)
                self.output_R = self.output_R.permute(0, 1, 4, 2, 3).contiguous().view(-1, 2, 3).contiguous().view(-1, 6).contiguous() # -1, 6
                self.output_R = compute_rotation_matrix_from_ortho6d(self.output_R)
            else:
                self.output_R = self.latent_vect['R'].view(BS, 2, 3, N).contiguous() # B, 6, N
                self.output_R = self.output_R/(torch.norm(self.output_R, dim=2, keepdim=True) + epsilon)
                self.output_R = self.output_R.permute(0, 3, 1, 2).contiguous().view(-1, 2, 3).contiguous().view(-1, 6).contiguous()
                self.output_R = compute_rotation_matrix_from_ortho6d(self.output_R)

        if self.config.pred_seg:
            self.output_C = self.latent_vect['C']

        if self.config.pred_mode: # B, M, N
            self.output_M = self.latent_vect['M'].permute(0, 2, 1).contiguous().view(-1, M).contiguous() # BS*N, M

        if 'completion' in self.config.task:
            if 'ycb' in self.config.task:
                self.output_pts = data['full'].permute(0, 2, 1).to(
                    self.latent_vect['R'].device).float()  # [B, N, 3] -> [B, 3, N]
            elif isinstance(self.latent_vect, dict):
                self.output_pts = self.net.decoder(self.latent_vect['0'])
            else:
                self.output_pts = self.net.decoder(self.latent_vect)

    def compute_and_eval_6d(self, data):
        input_pts  = data['xyz'] # 'points'
        BS = data['points'].shape[0]
        N  = data['points'].shape[1] # B, N, 3
        M  = self.config.num_modes_R
        CS = self.config.MODEL.num_channels_R
        target_R   = data['R']
        if 'C' in data:
            mask = data['C'] # B, N
        else:
            mask = torch.ones(BS, N, device=target_R.device)

        target_R_tiled  = target_R.unsqueeze(1).contiguous().repeat(1, N * M, 1, 1).contiguous()
        geodesic_loss   = loss_geodesic(self.output_R, target_R_tiled.view(-1, 3, 3).contiguous().float())  # BS*N* C/2

        self.degree_err_full = geodesic_loss.view(BS, M, N).contiguous()   # BS, N
        self.output_R_full   = self.output_R.view(BS, M, N, 3, 3).contiguous() # BS, N, 3, 3
        self.output_R   = self.output_R_full

        # BS, N*M
        self.norm_err   = torch.norm(self.output_R.view(BS, N*M, 9).contiguous() - target_R_tiled.view(BS, N*M, 9).contiguous(), dim=-1)

        if self.config.check_consistency: # BS, M, N, 3, 3
            mean_R   = torch.mean(self.output_R_full, dim=2, keepdim=True).view(BS, M, 1, -1).contiguous()
            variance = torch.norm(self.output_R.view(BS, M, N, -1).contiguous() - mean_R, dim=-1).mean()
            self.consistency_loss = variance * self.config.consistency_loss_multiplier
            self.infos['consistency'] = self.consistency_loss
        else:
            self.consistency_loss = 0.0

        if M > 1: # multi-mode dense
            self.norm_err = self.norm_err.view(BS, M, N).contiguous()
            if self.config.use_adaptive_mode:
                min_loss, min_indices = torch.min(self.norm_err, dim=1) # B, M, N
                self.regressionR_loss = min_loss.mean(dim=1)
                self.output_R         = self.output_R[torch.arange(BS).reshape(-1, 1), min_indices, torch.arange(N).reshape(1, -1)].contiguous()
                self.degree_err       = self.degree_err_full[torch.arange(BS).reshape(-1, 1), min_indices, torch.arange(N).reshape(1, -1)].contiguous()
            else:
                min_loss, min_indices = torch.min(self.norm_err.mean(dim=-1), dim=-1) # we only allow one mode to be True
                self.regressionR_loss = min_loss
                self.degree_err = self.degree_err_full[torch.arange(BS), :, min_indices] # degree err is raw GT mode prediction
        else:
            self.regressionR_loss = torch.sum(self.norm_err * mask, dim=1)/(torch.sum(mask, dim=1) + epsilon)
            self.degree_err = self.degree_err_full.squeeze() * mask
            self.output_R = self.output_R.squeeze()

        self.regressionR_loss = self.regressionR_loss.mean()

        if self.config.use_objective_T:
            target_T = data['T'].permute(0, 2, 1).contiguous()  # B, 3, N
            self.regressionT_loss = compute_vect_loss(self.output_T, target_T, confidence=mask).mean() # TYPE_LOSS='SOFT_L1'

        correctness_mask = torch.zeros((self.degree_err.shape[0], self.degree_err.shape[1]), device=self.output_R.device) # B, N
        correctness_mask[self.degree_err<THRESHOLD_GOOD] = 1.0
        correctness_mask[mask<1] = 0.0
        good_pred_ratio   = torch.sum(correctness_mask, dim=1)/(torch.sum(mask, dim=1) + epsilon) # B
        degree_err_masked = torch.sum(self.degree_err, dim=1)/(torch.sum(mask, dim=1) + epsilon)
        self.infos.update({'5deg': good_pred_ratio.mean(), 'rdiff': degree_err_masked.mean()})

    def compute_and_eval_axis(self, data):
        target_R   = data['R']
        target_R   = target_R.permute(0, 2, 1).contiguous() # B, 3, 1
        if self.config.MODEL.num_channels_R > 1:
            BS, CS = target_R.shape[0], self.latent_vect['R'].shape[-2]     # [BS*N, C, 3] -> [Bs, N, C, 3] -> [Bs, 3, N, C]
            self.output_R  = self.latent_vect['R'].view(BS, -1, CS, 3).contiguous().permute(0, 3, 1, 2).contiguous()
            self.output_R = self.output_R/(torch.norm(self.output_R, dim=1, keepdim=True) + epsilon)
            self.output_R_full = self.output_R
            self.degree_err = torch.acos( torch.sum(self.output_R*target_R.unsqueeze(-1), dim=1)) * 180 / np.pi
        else:
            self.output_R = self.latent_vect['R'].view(target_R.shape[0], -1, 3).contiguous().permute(0, 2, 1).contiguous() # B, 3, N
            self.output_R = self.output_R/(torch.norm(self.output_R, dim=1, keepdim=True) + epsilon)
            self.degree_err = torch.acos( torch.sum(self.output_R*target_R, dim=1)) * 180 / np.pi

        self.output_R_pooled = self.output_R.mean(dim=2) #
        self.output_R_pooled = self.output_R_pooled/(torch.norm(self.output_R_pooled, dim=1, keepdim=True) + epsilon)

        confidence = target_C
        if self.config.check_consistency: # on all points, all modes
            mean_R = torch.mean(self.output_R, dim=2, keepdim=True)
            variance = torch.norm(self.output_R - mean_R, dim=1).mean()
            self.consistency_loss = variance * self.config.consistency_loss_multiplier
            self.infos['consistency'] = self.consistency_loss
        else:
            self.consistency_loss = 0.0

        if self.config.MODEL.num_channels_R > 1: # apply to all, add
            # add per-mode minimum calculation, using the symmetry type, when we have symmetry on z, x axis
            if self.config.rotation_use_dense:
                regressionR_loss = compute_vect_loss(self.output_R, target_R.unsqueeze(-1), target_category=self.config.target_category, use_one2many=self.config.use_one2many)
            else:
                regressionR_loss = compute_vect_loss(self.output_R_pooled.unsqueeze(2), target_R.unsqueeze(2), target_category=self.config.target_category, use_one2many=self.config.use_one2many) # [2, 3, 2], [2, 3, 1]
            min_loss, min_indices = torch.min(regressionR_loss, dim=-1)
            self.regressionR_loss = min_loss
            # update the output_R here
            self.output_R   =  self.output_R_full[torch.arange(target_R.shape[0]), :, :, min_indices[:]]
            self.degree_err =  self.degree_err[torch.arange(target_R.shape[0]), :, min_indices[:]] # B, N, C
        else: # we have confidence for hand points
            if self.config.rotation_use_dense:
                self.regressionR_loss = compute_vect_loss(self.output_R, target_R, confidence=confidence, target_category=self.config.target_category, use_one2many=self.config.use_one2many) # B
            else:
                self.regressionR_loss = compute_vect_loss(self.output_R_pooled.unsqueeze(-1), target_R, target_category=self.config.target_category, use_one2many=self.config.use_one2many) # B, 3, N
            self.infos['regressionR'] = self.regressionR_loss.mean()

        self.regressionR_loss = self.regressionR_loss.mean()
        if self.config.use_objective_T:
            self.regressionT_loss = compute_vect_loss(self.output_T, target_T, confidence=mask).mean() # TYPE_LOSS='SOFT_L1'

        correctness_mask = torch.zeros((self.degree_err.shape[0], self.degree_err.shape[1]), device=self.output_R.device) # B, N
        correctness_mask[self.degree_err<THRESHOLD_GOOD] = 1.0
        good_pred_ratio = torch.mean(correctness_mask, dim=1) # TODO
        self.infos['good_pred_ratio'] = good_pred_ratio.mean() # scalar

        if self.config.pred_mode and self.config.MODEL.num_channels_R > 1:
            self.target_M       = min_indices.unsqueeze(1).repeat(1, self.output_R.shape[-1]).contiguous().view(-1).contiguous()
            self.classifyM_loss = compute_miou_loss(self.output_M, min_indices.unsqueeze(1).repeat(1, self.output_R.shape[-1]).contiguous().view(-1).contiguous(), loss_type='xentropy')
            self.output_M_label = torch.argmax(self.output_M, dim=-1)  # [B * N] in [0...M-1]
            self.classifyM_acc = (self.output_M_label == min_indices.unsqueeze(1).repeat(1, self.output_R.shape[-1]).contiguous().view(-1).contiguous()).float().mean()
            # [B, 3, N, M]
            self.output_R_chosen = self.output_R_full[torch.arange(BS).reshape(-1, 1), :,
                                   torch.arange(N).reshape(1, -1), self.output_M_label.reshape(BS, N)].permute(0, 2, 1)  # [B, N, 3]
            self.degree_err_chosen = torch.acos(torch.sum(self.output_R_chosen * target_R, dim=1)) * 180 / np.pi

    def compute_and_eval_so3(self, data):
        # bp()
        input_pts      = data['xyz']
        shift_dis      = input_pts.mean(dim=1, keepdim=True) if self.config.pred_t else 0
        target_pts     = data['points']
        target_T       = data['T'].permute(0, 2, 1).contiguous() # B, 3, N

        BS, N = target_pts.shape[0:2]
        M  = self.config.num_modes_R
        self.threshold = 1.0
        if 'se3' in self.config.encoder_type: # we have [BS, M*2, 3]
            self.latent_vect['R'] = self.latent_vect['R'].view(-1, M, 6).contiguous().permute(0, 2, 1).contiguous()
            if torch.isnan(self.latent_vect['R']).any():
                print('we have nan in .latent_vect R')
        nb, nr, na = self.latent_vect['R'].shape  #
        r_gt        = data['R_gt'].float() # GT R,torch.Size([2, 3, 3])
        rlabel_gt   = data['R_label'].view(-1).contiguous()   # GT R label, torch.Size([2])
        ranchor_gt  = data['R'].float() # GT relative R, torch.Size([2, 60, 3, 3])
        with torch.no_grad():
            ranchor_gt_quat = matrix_to_unit_quaternion(ranchor_gt)

        anchors = self.anchors
        rlabel_pred = self.latent_vect['1']

        rotation_mapping = compute_rotation_matrix_from_quaternion if nr == 4 else compute_rotation_matrix_from_ortho6d
        ranchor_pred = rotation_mapping(self.latent_vect['R'].transpose(1,2).contiguous().view(-1,nr)).view(nb,-1,3,3)
        if 'so3' in self.config.encoder_type and na>1:
            pred_R = torch.matmul(anchors, ranchor_pred) # [60, 3, 3], [nb, 60, 3, 3] --> [nb, 60, 3, 3]
        else:
            pred_R = ranchor_pred

        if self.config.pred_t:
            if self.config.t_method_type == -1:
                pred_T          = self.output_T.permute(0, 2, 1).contiguous().unsqueeze(-1).contiguous()
                pred_T          = torch.matmul(pred_R, pred_T) # nb, na, 3, 1,
            elif self.config.t_method_type == 0:
                pred_T          = self.output_T.permute(0, 2, 1).contiguous().unsqueeze(-1).contiguous()
                if na > 1:
                    pred_T          = torch.matmul(anchors, pred_T) # nb, na, 3, 1,
            else: # type 1, 2, 3, dense voting
                pred_T          = self.output_T.permute(0, 2, 1).contiguous().unsqueeze(-1).contiguous() #  [B, num_heads, 3, A]--> nb, na, 3, 1

        if 'ssl' in self.config.task:
            np_out = self.output_pts.shape[-1]
            if self.config.r_method_type < 1:  # Q to R first
                if self.config.pred_t:
                    transformed_pts = torch.matmul(pred_R, self.output_pts.unsqueeze(1).contiguous() - 0.5) + pred_T
                    transformed_pts = transformed_pts.permute(0, 1, 3, 2).contiguous() # nb, na, np, 3
                    dist1, dist2 = self.chamfer_dist(transformed_pts.view(-1, np_out, 3).contiguous(), (input_pts - shift_dis).unsqueeze(1).repeat(1, na, 1, 1).contiguous().view(-1, N, 3).contiguous(), return_raw=True)
                else:
                    transformed_pts = torch.matmul(pred_R, self.output_pts.unsqueeze(1).contiguous() - 0.5).permute(0, 1, 3, 2).contiguous() #
                    dist1, dist2 = self.chamfer_dist(transformed_pts.view(-1, np_out, 3).contiguous(), input_pts.unsqueeze(1).repeat(1, na, 1, 1).contiguous().view(-1, N, 3).contiguous(), return_raw=True)

            elif self.config.r_method_type == 1: # raw quaternion
                qw, qxyz = torch.split(self.latent_vect['R'].permute(0, 2, 1).contiguous(), [1, 3], dim=-1)
                # theta_max= torch.Tensor([1.2]).cuda()
                theta_max= torch.Tensor([36/180 * np.pi]).cuda()
                qw       = torch.cos(theta_max) + (1- torch.cos(theta_max)) * F.sigmoid(qw)
                constrained_quat = torch.cat([qw, qxyz], dim=-1)
                ranchor_pred = rotation_mapping(constrained_quat.view(-1,nr)).view(nb,-1,3,3)
                pred_R = torch.matmul(anchors, ranchor_pred) # [60, 3, 3], [nb, 60, 3, 3] --> [nb, 60, 3, 3]

                constrained_quat_tiled = constrained_quat.unsqueeze(2).contiguous().repeat(1, 1, np_out, 1).contiguous() # nb, na, np, 4
                canon_pts= self.output_pts.permute(0, 2, 1).contiguous() - 0.5 # nb, np, 3
                canon_pts_tiled= canon_pts.unsqueeze(1).contiguous().repeat(1, na, 1, 1).contiguous() # nb, na, np, 3

                transformed_pts = rotate(constrained_quat_tiled, canon_pts_tiled) # nb, na, np, 3
                if self.config.pred_t:
                    transformed_pts = torch.matmul(anchors, transformed_pts.permute(0, 1, 3, 2).contiguous()) + pred_T
                    transformed_pts = transformed_pts.permute(0, 1, 3, 2).contiguous()
                    dist1, dist2    = self.chamfer_dist(transformed_pts.view(-1, np_out, 3).contiguous(), (input_pts - shift_dis).unsqueeze(1).repeat(1, na, 1, 1).contiguous().view(-1, N, 3).contiguous(), return_raw=True)
                else:
                    transformed_pts = torch.matmul(anchors, transformed_pts.permute(0, 1, 3, 2).contiguous())
                    transformed_pts = transformed_pts.permute(0, 1, 3, 2).contiguous()
                    dist1, dist2    = self.chamfer_dist(transformed_pts.view(-1, np_out, 3).contiguous(), input_pts.unsqueeze(1).repeat(1, na, 1, 1).contiguous().view(-1, N, 3).contiguous(), return_raw=True)

                # self.regu_quat_loss = torch.mean( torch.norm( torch.norm(constrained_quat, dim=-1) - 1))
                self.regu_quat_loss = torch.mean( torch.pow( torch.norm(constrained_quat, dim=-1) - 1, 2))

            if 'partial' in self.config.task:
                all_dist = (dist2).mean(-1).view(nb, -1).contiguous()
            else:
                all_dist = (dist1.mean(-1) + dist2.mean(-1)).view(nb, -1).contiguous()
            if na > 1:
                min_loss, min_indices = torch.min(all_dist, dim=-1) # we only allow one mode to be True
                if torch.isnan(ranchor_pred).any() or torch.isnan(ranchor_gt).any():
                    self.recon_loss  = torch.Tensor([0]).cuda()#
                else:
                    self.recon_loss   = min_loss.mean()
                self.rlabel_pred  = min_indices.detach().clone()
                self.rlabel_pred.requires_grad = False
                self.transformed_pts = transformed_pts[torch.arange(0, BS), min_indices] + shift_dis
                self.r_pred = pred_R[torch.arange(0, BS), min_indices].detach().clone() # correct R by searching
                self.r_pred.requires_grad = False
                if self.config.pred_t:
                    self.t_pred = pred_T.squeeze(-1)[torch.arange(0, BS), min_indices] + shift_dis.squeeze()
                else:
                    self.t_pred = None
            else:
                self.transformed_pts = transformed_pts.squeeze()
                self.r_pred = pred_R.squeeze()
                self.recon_loss   = all_dist.mean()
                if self.config.pred_t:
                    self.t_pred = pred_T[:, 0, :, :].squeeze() + shift_dis.squeeze()
                else:
                    self.t_pred = None
            self.infos["recon"] = self.recon_loss
            if self.config.eval:
                print('chamferL1', torch.sqrt(self.recon_loss))

            if self.config.pred_t:
                self.projection_loss = torch.Tensor([0]).cuda()
                # if self.config.p_method_type == -1: # use naive project loss
                #     gt_view_matrix = look_at(
                #         eyes=torch.tensor([[0, 0, 0]], dtype=torch.float32).repeat(BS, 1).contiguous(), # can multiply 0.8 if the eye is too close?
                #         centers=data['T'].cpu().squeeze(),
                #         ups=torch.tensor([[0, 0, 1]], dtype=torch.float32).repeat(BS, 1).contiguous(),
                #     )
                #     gt_pre_matrix = self.render.projection_matrix @ gt_view_matrix
                #     self.gt_depth_map = self.render(input_pts, view_id=0, radius_list=[15.0, 20.0], pre_matrix=gt_pre_matrix)
                #
                #     # pred
                #     pred_view_matrix =look_at(
                #         eyes=torch.tensor([[0, 0, 0]], dtype=torch.float32).repeat(BS, 1).contiguous(), # can multiply 0.8 if the eye is too close?
                #         centers=self.t_pred.cpu(),
                #         ups=torch.tensor([[0, 0, 1]], dtype=torch.float32).repeat(BS, 1).contiguous(),
                #     )
                #     pred_pre_matrix = self.render.projection_matrix @ pred_view_matrix
                #     self.pred_depth_map = self.render(self.transformed_pts, view_id=0, radius_list=[20.0, 25.0], pre_matrix=pred_pre_matrix)
                #     self.projection_loss   = 0.1 * self.render_loss(self.pred_depth_map, self.gt_depth_map.detach())
                #     self.infos['projection'] = self.projection_loss
                # elif self.config.p_method_type == 0:
                #     # pre-set the camera
                #     # put points into Pytorch3D world space
                #     points_p3d = self.transformed_pts.float() # [B, N, 3]
                #     points_p3d[:, :, -1] = -points_p3d[:, :, -1] # reverse z
                #     points_p3d[:, :, 0]  = -points_p3d[:, :, 0]  # reverse x to fit pytorch3d convention
                #     # convert into NDC
                #     pred_proj = self.cam.transform_points(points_p3d, eps=1e-10)
                #     pred_proj = pred_proj/pred_proj[:, :, -1:]
                #
                #     points_p3d = input_pts # [B, N, 3]
                #     points_p3d[:, :, -1] = -points_p3d[:, :, -1] # reverse z
                #     points_p3d[:, :, 0]  = -points_p3d[:, :, 0]  # reverse x to fit pytorch3d convention
                #     input_proj = self.cam.transform_points(points_p3d)
                #     input_proj = input_proj/input_proj[:, :, -1:]
                #     dist1, dist2 = self.chamfer_dist_2d(input_proj, pred_proj, return_raw=True)
                #     self.projection_loss = torch.clamp(0.01 * (dist1 + dist2).mean(), min=0, max=0.1)
                # elif self.config.p_method_type == 1:
                #     # pre-set the camera
                #     # put points into Pytorch3D world space
                #     points_p3d = self.transformed_pts # [B, N, 3]
                #     points_p3d[:, :, -1] = -points_p3d[:, :, -1] # reverse z
                #     points_p3d[:, :, 0]  = -points_p3d[:, :, 0]  # reverse x to fit pytorch3d convention
                #     # convert into NDC
                #     points_ndc = self.cam.transform_points(points_p3d, eps=1e-10)
                #     points_features = points_ndc.clone().detach()
                #
                #     # Rasterize points -- bins set heurisically
                #     pointcloud = pytorch3d.structures.Pointclouds(points_ndc, features=points_features)
                #     radius = 15 # pixels, set emperically
                #     K_pt   = 16
                #     K_num  = 2
                #     img_h  = 480
                #     img_w  = 640
                #     ndc_r = 2 * radius / float(img_h)
                #     idx, zbuf, dist_xy = rasterize_points(pointcloud, [img_h, img_w], ndc_r, K_pt)
                #
                #     visible_ids = idx[:, 0:K_num, :, :] # assume the 3 closest points are visible, may need tune point radius
                #     visible_ids = visible_ids[visible_ids>0].unique()
                #     if len(visible_ids) > 0:
                #         self.projection_loss = dist1.view(nb, na, np_out).contiguous()[torch.arange(0, BS), min_indices]# [nb*na, np] --> [nb, np]
                #         self.projection_loss = self.projection_loss.view(-1).contiguous()[visible_ids.long()].mean()
                #         self.projection_loss = torch.clamp(self.projection_loss, min=0, max=0.1)
                #     else:
                #         self.projection_loss = torch.Tensor([0]).cuda()

        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>supervised >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        else:
            rlabel_pred = rlabel_pred.view(nb,-1)
            if na == 1:
                r_acc = torch.zeros(1) + 1
                self.classifyM_loss = torch.zeros(1).cuda()
                if self.config.use_axis:
                    gt_up = torch.matmul(r_gt, self.symmetry_axis.view(1, 3, 1).contiguous())
                    pred_up = torch.matmul(ranchor_pred.squeeze(), self.symmetry_axis.view(1, 3, 1).contiguous())
                    self.regressionR_loss = torch.pow(gt_up - pred_up, 2).sum()
                else:
                    self.regressionR_loss = torch.pow(ranchor_pred.squeeze() - r_gt, 2).sum()
                self.r_pred = ranchor_pred.squeeze()
                self.ranchor_pred = ranchor_pred.squeeze()
                self.ranchor_gt = r_gt
                if self.config.pred_t:
                    self.t_pred = self.output_T.squeeze() + shift_dis.squeeze()
                    self.regressionT_loss = torch.norm(self.t_pred - target_T.squeeze(), dim=1).mean()
                else:
                    self.t_pred = None
            else:
                cls_loss, r_acc = self.classifier(rlabel_pred, rlabel_gt.view(-1).contiguous())
                self.classifyM_loss = cls_loss.mean()
                if self.config.use_axis:
                    if self.config.r_method_type <1: # default r with 3*3 matrix
                        gt_bias = angle_from_R(ranchor_gt.view(-1,3,3)).view(nb,-1) # keep the same
                        # gt_bias = rot_diff_rad(ranchor_gt.view(-1,3,3), torch.eye(3).cuda().unsqueeze(0), chosen_axis=self.chosen_axis).view(nb, -1)
                        mask = (gt_bias < self.threshold)[:,:,None].float()
                        self.regressionR_loss = torch.pow(ranchor_gt[:, :, :, -2] * mask - ranchor_pred[:, :, :, -2] * mask,2).sum()
                    else:
                        gt_bias = angle_from_R(ranchor_gt.view(-1,3,3)).view(nb,-1)
                        mask = (gt_bias < self.threshold)[:,:,None].float()
                        qw, qxyz = torch.split(self.latent_vect['R'].permute(0, 2, 1).contiguous(), [1, 3], dim=-1)
                        theta_max= torch.Tensor([36/180 * np.pi]).cuda() #
                        qw       = torch.cos(theta_max) + (1- torch.cos(theta_max)) * F.sigmoid(qw)
                        constrained_quat = torch.cat([qw, qxyz], dim=-1)

                        if torch.isnan(constrained_quat).any() or torch.isnan(ranchor_gt_quat).any():
                            self.regressionR_loss  = torch.Tensor([0]).cuda()#
                            self.regu_quat_loss    = torch.Tensor([0]).cuda()
                        else: # y axis as up
                            gt_up     = rotate(ranchor_gt_quat, self.symmetry_axis.repeat(nb, na, 1).contiguous()) # nb, na, 3
                            pred_up   = rotate(constrained_quat, self.symmetry_axis.repeat(nb, na, 1).contiguous()) # nb, na, 3
                            self.regressionR_loss  = torch.pow(gt_up * mask -pred_up * mask, 2).sum() #
                            self.regu_quat_loss    = torch.mean( torch.pow( torch.norm(constrained_quat, dim=-1) - 1, 2))

                        # recover to R for inference
                        ranchor_pred = rotation_mapping(constrained_quat.view(-1,nr)).view(nb,-1,3,3)
                        pred_R = torch.matmul(anchors, ranchor_pred) # [60, 3, 3], [nb, 60, 3, 3] --> [nb, 60, 3, 3]
                    # [4, 60, 3, 1024]  [nb, na, 3, np] -->  [nb, na, np, 3]
                    self.rlabel_pred  = torch.argmax(rlabel_pred, 1).detach().clone()
                    self.rlabel_pred.requires_grad = False
                    self.r_pred       = L.batched_index_select(pred_R, 1, self.rlabel_pred.long().view(nb,-1)).view(nb,3,3)
                    self.ranchor_pred = L.batched_index_select(ranchor_pred, 1, self.rlabel_pred.long().view(nb,-1)).view(nb,3,3)
                    self.ranchor_gt   = L.batched_index_select(ranchor_gt, 1, rlabel_gt.long().view(nb,-1)).view(nb,3,3)
                else:
                    if self.config.r_method_type <1: # default r with 3*3 matrix
                        gt_bias = angle_from_R(ranchor_gt.view(-1,3,3)).view(nb,-1)
                        mask = (gt_bias < self.threshold)[:,:,None,None].float()
                        if torch.isnan(ranchor_pred).any() or torch.isnan(ranchor_gt).any():
                            self.regressionR_loss  = torch.Tensor([0]).cuda()#
                        self.regressionR_loss = torch.pow(ranchor_gt * mask - ranchor_pred * mask, 2).sum()
                    else:
                        gt_bias = angle_from_R(ranchor_gt.view(-1,3,3)).view(nb,-1)
                        mask = (gt_bias < self.threshold)[:,:,None].float()
                        qw, qxyz = torch.split(self.latent_vect['R'].permute(0, 2, 1).contiguous(), [1, 3], dim=-1)
                        theta_max= torch.Tensor([36/180 * np.pi]).cuda() #
                        qw       = torch.cos(theta_max) + (1- torch.cos(theta_max)) * F.sigmoid(qw)
                        constrained_quat = torch.cat([qw, qxyz], dim=-1)

                        # [nb, na, 4] - [nb, na, 4], assume constrained_quat close to have modus 1
                        if torch.isnan(constrained_quat).any() or torch.isnan(ranchor_gt_quat).any():
                            self.regressionR_loss  = torch.Tensor([0]).cuda()#
                            self.regu_quat_loss    = torch.Tensor([0]).cuda()
                        else:
                            self.regressionR_loss  = torch.pow(ranchor_gt_quat * mask - constrained_quat * mask, 2).sum() #
                            self.regu_quat_loss    = torch.mean( torch.pow( torch.norm(constrained_quat, dim=-1) - 1, 2))

                        # recover to R for inference
                        ranchor_pred = rotation_mapping(constrained_quat.view(-1,nr)).view(nb,-1,3,3)
                        pred_R = torch.matmul(anchors, ranchor_pred) # [60, 3, 3], [nb, 60, 3, 3] --> [nb, 60, 3, 3]

                    # [4, 60, 3, 1024]  [nb, na, 3, np] -->  [nb, na, np, 3]
                    self.rlabel_pred  = torch.argmax(rlabel_pred, 1).detach().clone()
                    self.rlabel_pred.requires_grad = False
                    self.r_pred = L.batched_index_select(pred_R, 1, self.rlabel_pred.long().view(nb,-1)).view(nb,3,3)
                    self.ranchor_pred = L.batched_index_select(ranchor_pred, 1, self.rlabel_pred.long().view(nb,-1)).view(nb,3,3)
                    self.ranchor_gt   = L.batched_index_select(ranchor_gt, 1, rlabel_gt.long().view(nb,-1)).view(nb,3,3)

                if self.config.pred_t:
                    # regressionT_loss = torch.norm(
                    #     pred_T + shift_dis.unsqueeze(-1) - target_T.unsqueeze(1).contiguous(),
                    #     dim=2).squeeze()
                    regressionT_loss = torch.pow(pred_T + shift_dis.unsqueeze(-1) - target_T.unsqueeze(1).contiguous(), 2).mean(dim=2).squeeze()  # TYPE_LOSS='SOFT_L1'
                    regressionT_loss = torch.sum(regressionT_loss * mask.squeeze(), dim=1) / mask.squeeze().sum(dim=1)
                    self.regressionT_loss = regressionT_loss.mean()
                    self.t_pred = L.batched_index_select(pred_T, 1, self.rlabel_pred.long().view(nb,-1)).squeeze() + shift_dis.squeeze()
                    if 'completion' in self.config.task:
                        self.transformed_pts = torch.matmul(self.r_pred, self.output_pts - 0.5) + self.t_pred.unsqueeze(-1).contiguous()
                        self.transformed_pts = self.transformed_pts.permute(0, 2, 1).contiguous()
                else:
                    self.t_pred = None

            self.infos['r_acc'] = r_acc
            if not self.config.use_axis:
                self.infos['rdiff'] = mean_angular_error(self.r_pred, r_gt).mean() * 180 / np.pi  # final r error
                self.infos['rdiff_anchor'] = mean_angular_error(self.ranchor_pred,
                                                                self.ranchor_gt).mean() * 180 / np.pi  # final r error
            else:
                self.infos['rdiff'] = rot_diff_degree(self.r_pred, r_gt, chosen_axis=self.chosen_axis, flip_axis=self.flip_axis).view(nb, -1).mean()
                self.infos['rdiff_anchor'] = rot_diff_degree(self.ranchor_pred, self.ranchor_gt,
                                                             chosen_axis=self.chosen_axis, flip_axis=self.flip_axis).view(nb, -1).mean()
            if self.config.pred_t:
                self.infos['tdiff'] = torch.norm(self.t_pred - target_T.squeeze(), dim=1).mean()
        # bp()
        if 'ycb' in self.config.task:
            pred_RT = torch.cat([self.r_pred, self.t_pred.unsqueeze(-1)], dim=-1)  # [B, 3, 1] -> [B, 3, 4]
            gt_RT = torch.cat([data['R_gt'], data['T'].transpose(-1, -2)], dim=-1).to(pred_RT.device)
            cls = int(self.config.instance)
            self.ycb_last_pose_err = self.bs_utils.cal_full_error(pred_RT, gt_RT, cls)


    def compute_loss(self, data):
        """
        check all other losses
        """
        # bp()
        input_pts      = data['xyz']
        shift_dis      = input_pts.mean(dim=1, keepdim=True)
        target_pts     = data['points']
        target_T       = data['T'].permute(0, 2, 1).contiguous() # B, 3, N
        BS, N = target_pts.shape[0:2]

        if 'C' in data:
            mask = data['C'] # B, N
        else:
            mask = torch.ones(BS, N, device=target_T.device)

        if self.config.pred_seg:
            self.seg_loss = compute_miou_loss(self.output_C, mask).mean()

        if self.config.pred_nocs:
            shift_dis     = input_pts.mean(dim=1, keepdim=True)
            self.nocs_err = torch.norm(self.output_N - target_pts.permute(0, 2, 1).contiguous(), dim=1)
            self.nocs_loss= compute_1vN_nocs_loss(self.output_N, target_pts.permute(0, 2, 1).contiguous(), target_category=self.config.target_category, num_parts=1, confidence=mask)
            self.nocs_loss= self.nocs_loss.mean()

        if 'completion' in self.config.task:
            if 'ycb' in self.config.task:
                self.output_pts = data['full'].permute(0, 2, 1).to(self.latent_vect['R'].device).float()  # [B, N, 3] -> [B, 3, N]
                self.recon_canon_loss = 0
                if self.config.instance is None:  # classify the object
                    logits = self.latent_vect['class']  # [B, num_heads]
                    pred_labels = torch.argmax(logits, dim=-1)
                    gt_labels = data['class'].to(logits.device).long()
                    self.classification_loss = nn.CrossEntropyLoss()(logits, gt_labels)
                    self.classification_acc = (pred_labels == gt_labels).float().mean()
                    self.infos['classification_loss'] = self.classification_loss
                    self.infos['classification_acc'] = self.classification_acc

                    chosen_class = pred_labels if self.is_testing else gt_labels
                    batch_idx = torch.arange(len(chosen_class)).to(chosen_class.device).long()

                    for key in ['1', 'R', 'T']:
                        self.latent_vect[key] = self.latent_vect[key][batch_idx, chosen_class]
                    self.output_T = self.latent_vect['T']
            else:
                dist1_canon, dist2_canon = self.chamfer_dist(self.output_pts.permute(0, 2, 1).contiguous(), target_pts, return_raw=True)
                if 'partial' in self.config.task and 'ssl' in self.config.task:
                    self.recon_canon_loss = (dist2_canon).mean()
                else:
                    self.recon_canon_loss = dist1_canon.mean() + dist2_canon.mean()
            self.infos["recon_canon"] = self.recon_canon_loss

    def eval_func(self, data):
        self.net.eval()
        with torch.no_grad():
            self.forward(data)

        # get pose_err and pose_info
        if self.config.pred_6d:
            self.eval_6d(data)
        elif self.config.pred_nocs:
            self.eval_nocs(data)
        else:
            if self.config.pre_compute_delta:
                self.pose_err = None
                self.pose_info = {'delta_r': self.r_pred, 'delta_t': self.t_pred}
            else:
                self.eval_so3(data)

    def eval_so3(self, data):
        # all you need to get a reasonable evalution during test stage, for unseen and seen data
        BS = data['points'].shape[0]
        N  = data['points'].shape[1]
        if 'ssl' in self.config.task and self.config.eval: # only apply this during eval
            if f'{self.config.exp_num}_{self.config.name_dset}_{self.config.target_category}' in delta_R:
                self.delta_r = torch.from_numpy(delta_R[f'{self.config.exp_num}_{self.config.name_dset}_{self.config.target_category}']).cuda()
                print('use precomputed delta_r')
            elif f'{self.config.name_dset}_{self.config.target_category}' in delta_R:
                self.delta_r = torch.from_numpy(delta_R[f'{self.config.name_dset}_{self.config.target_category}']).cuda()
            else:
                print('not found precomputed delta_r!!!')
                self.delta_r= torch.eye(3).reshape((1, 3, 3)).repeat(BS, 1, 1).cuda()
            # t
            if f'{self.config.exp_num}_{self.config.name_dset}_{self.config.target_category}' in delta_T:
                print('use precomputed delta_t')
                self.delta_t = torch.from_numpy(delta_T[f'{self.config.exp_num}_{self.config.name_dset}_{self.config.target_category}']).cuda()
            elif f'{self.config.name_dset}_{self.config.target_category}' in delta_T:
                print('use precomputed delta_t, type 1')
                self.delta_t = torch.from_numpy(delta_T[f'{self.config.name_dset}_{self.config.target_category}']).cuda()
            else:
                self.delta_t = torch.zeros(1, 3).cuda()
        else:
            self.delta_r = torch.eye(3).reshape((1, 3, 3)).cuda()
            self.delta_t = torch.zeros(1, 3).cuda()

        self.delta_r = self.delta_r.reshape(1, 3, 3)
        self.delta_t = self.delta_t.reshape(1, 3)

        # if self.config.use_axis or chosen_axis is not None:
        pred_rot    = torch.matmul(self.r_pred, self.delta_r.float().permute(0, 2, 1).contiguous())
        gt_rot      = data['R_gt'].cuda()  # [B, 3, 3]
        rot_err     = rot_diff_degree(pred_rot, gt_rot, chosen_axis=self.chosen_axis, flip_axis=self.flip_axis)

        if 'xyz' in data:
            input_pts  = data['xyz'].permute(0, 2, 1).contiguous().cuda()
        else:
            input_pts  = data['G'].ndata['x'].view(BS, -1, 3).contiguous().cuda() # B, N, 3

        if self.config.pred_t:
            # if
            if self.r_pred.shape[-1] < 3:
                pred_center= self.t_pred.unsqueeze(-1)
            else:
                pred_center= self.t_pred.unsqueeze(-1) - torch.matmul(pred_rot, self.delta_t.unsqueeze(-1)) # ?
            gt_center  = data['T'].cuda() # B, 3 # from original to
            trans_err  = torch.norm(pred_center[:, :, 0] - gt_center[:, 0, :], dim=-1)
        else:
            trans_err = torch.zeros(BS).cuda()
            gt_center  = torch.zeros(BS, 1, 3).cuda()
            pred_center= torch.zeros(BS, 3, 1).cuda()

        scale_err = torch.zeros(BS).cuda()
        scale = torch.ones(BS)
        self.pose_err  = {'rdiff': rot_err, 'tdiff': trans_err, 'sdiff': scale_err}
        self.pose_info = {'r_gt': gt_rot, 't_gt': gt_center.mean(dim=1), 's_gt': scale, 'r_pred': pred_rot, 't_pred': pred_center.mean(dim=-1), 's_pred': scale}
        return

    def eval_6d(self, data, thres=0.05):
        """one step of validation"""
        BS = data['points'].shape[0]
        N  = data['points'].shape[1]
        M  = self.config.num_modes_R
        if 'C' in data:
            mask = data['C'].cuda() # B, N
        else:
            mask = torch.ones(BS, N, device=data['points'].device)
        flatten_r    = self.output_R.view(BS, N, -1).contiguous()
        pairwise_dis = torch.norm(flatten_r.unsqueeze(2) - flatten_r.unsqueeze(1), dim=-1) # [2, 512, 512]
        inliers_mask = torch.zeros_like(pairwise_dis)
        inliers_mask[pairwise_dis<thres] = 1.0   # [B, N, N]
        inliers_mask = inliers_mask * mask.unsqueeze(-1)
        score = inliers_mask.mean(dim=-1)
        select_ind  = torch.argmax(score, dim=1) # B
        pred_rot    = self.output_R[torch.arange(BS), select_ind]
        gt_rot = data['R'].cuda()  # [B, 3, 3]
        rot_err = rot_diff_degree(gt_rot, pred_rot, chosen_axis=None)  # [B, M]
        if 'xyz' in data:
            input_pts = data['xyz'].permute(0, 2, 1).contiguous().cuda()
        else:
            input_pts  = data['G'].ndata['x'].view(BS, -1, 3).contiguous().cuda() # B, N, 3
        pred_center= input_pts - self.output_T.permute(0, 2, 1).contiguous()
        gt_center  = input_pts - data['T'].cuda() # B, N, 3
        mean_pred_c = torch.sum(pred_center * mask.unsqueeze(-1), dim=1) / (torch.sum(mask.unsqueeze(-1), dim=1) + epsilon)
        trans_err = torch.norm(mean_pred_c - gt_center[:, 0, :], dim=1) #

        scale_err = torch.Tensor([0, 0]) #
        scale = torch.Tensor([1.0, 1.0]) #
        self.pose_err  = {'rdiff': rot_err, 'tdiff': trans_err, 'sdiff': scale_err}
        self.pose_info = {'r_gt': gt_rot, 't_gt': gt_center.mean(dim=1), 's_gt': scale, 'r_pred': pred_rot, 'r_raw': self.output_R, 't_pred': pred_center.mean(dim=1), 's_pred': scale}
        return

    def eval_nocs(self, data):
        target_T   = data['T'].cuda().permute(0, 2, 1).contiguous() # B, 3, N
        BS = data['points'].shape[0]
        N  = data['points'].shape[1] # B, N, 3
        target_pts = data['points'].cuda()
        if 'xyz' in data:
            input_pts = data['xyz'].permute(0, 2, 1).contiguous().cuda()
        else:
            input_pts = data['G'].ndata['x'].view(target_pts.shape[0], -1, 3).contiguous().permute(0, 2, 1).contiguous().cuda()

        if 'C' in data:
            mask = data['C'].cuda() # B, N
        else:
            mask = None

        pred_nocs = self.output_N # B, 3, N
        if self.output_N.shape[-1] < target_pts.shape[1]:
            input_pts        = data['G'].ndata['x'].view(target_pts.shape[0], -1, 3).contiguous().permute(0, 2, 1).contiguous().cuda()
            shift_dis        = input_pts.mean(dim=-1, keepdim=True)
            reduced_pts      = self.latent_vect['xyz']
            input_pts        = reduced_pts + shift_dis
            target_pts       = torch.matmul(data['R'].cuda(), input_pts  - target_T) / data['S'].cuda().unsqueeze(-1) + 0.5
            target_pts       = target_pts.permute(0, 2, 1).contiguous()
        gt_nocs = target_pts.transpose(-1, -2) # B, 3, N
        if mask is None:
            self.pose_err, self.pose_info = compute_pose_diff(nocs_gt=gt_nocs.transpose(-1, -2),
                    nocs_pred=pred_nocs.transpose(-1, -2), target=input_pts.transpose(-1, -2),
                                              category=self.config.target_category)
        else:
            self.pose_err, self.pose_info = compute_pose_diff(nocs_gt=gt_nocs[0:1].transpose(-1, -2)[mask[0:1]>0],
                    nocs_pred=pred_nocs[0:1].transpose(-1, -2)[mask[0:1]>0], target=input_pts[0:1].transpose(-1, -2)[mask[0:1]>0],
                                              category=self.config.target_category)

    # seems vector is more stable
    def collect_loss(self):
        loss_dict = {}
        if 'pose' in self.config.task:
            if self.config.use_objective_T:
                loss_dict["regressionT"]= self.regressionT_loss
            if self.config.use_objective_R:
                loss_dict["regressionR"]= self.regressionR_loss
            if self.config.use_objective_N:
                loss_dict["nocs"]= self.nocs_loss
            if self.config.use_objective_C:
                loss_dict["seg"]= self.seg_loss
            if self.config.use_confidence_R:
                loss_dict['confidence'] = self.regressionCi_loss
            if self.config.use_objective_M:
                loss_dict['classifyM'] = 0.1 * self.classifyM_loss
            if self.config.use_objective_V:
                loss_dict['consistency'] = self.consistency_loss
            if self.config.use_objective_P:
                loss_dict['projection'] = self.projection_loss
        if 'completion' in self.config.task:
            if self.config.use_objective_canon:
                recon_multiplier = 1
                if 'finetune' in self.config.task:
                    recon_multiplier = 0
                loss_dict['recon'] = recon_multiplier * self.recon_canon_loss
            else:
                loss_dict['recon'] = self.recon_loss
            if 'ycb' in self.config.task and self.config.instance is None:
                loss_dict['classification'] = self.classification_loss * 0.01
            if self.config.use_symmetry_loss:
                loss_dict['chirality'] = self.recon_chirality_loss
            if self.config.r_method_type == 1:
                # loss_dict['regu_quat'] = 0.001 * self.regu_quat_loss
                loss_dict['regu_quat'] = 0.1 * self.regu_quat_loss # only for MSE loss

        return loss_dict

    # seems vector is more stable
    def collect_info(self):
        if 'pose' in self.config.task:
            return self.infos
        else:
            return None

    def visualize_batch(self, data, mode, **kwargs):
        tb = self.train_tb if mode == 'train' else self.val_tb
        num = min(data['points'].shape[0], 12)
        target_pts = data['points'].detach().cpu().numpy() # canonical space
        if 'xyz' in data:
            input_pts = data['xyz'].detach().cpu().numpy()
        else:
            input_pts = data['G'].ndata['x'].view(target_pts.shape[0], -1, 3).contiguous().cpu().numpy()
        ids  = data['id']
        idxs = data['idx']
        BS    = data['points'].shape[0]
        M     = self.config.num_modes_R
        N     = input_pts.shape[1]
        save_path = f'{self.config.log_dir}/generation'
        if not exists(save_path):
            print('making directories', save_path)
            makedirs(save_path)

        if self.config.pred_nocs:
            nocs_err   = self.nocs_err.cpu().detach().numpy() # GT mode degree error, [B, N]
            output_N   = self.output_N.transpose(-1, -2).cpu().detach().numpy() # B, 3, N --> B, N, 3
            if len(nocs_err.shape) < 3:
                nocs_err = nocs_err[:, :, np.newaxis]

            for j in range(nocs_err.shape[-1]):
                if self.output_N.shape[-1] < target_pts.shape[1]:
                    input_pts  = self.latent_vect['xyz']
                save_arr  = np.concatenate([input_pts, nocs_err[:, :, j:j+1], output_N], axis=-1)
                for k in range(num): # batch
                    save_name = f'{save_path}/{mode}_{self.clock.step}_{ids[k]}_{idxs[k]}_{j}.txt' #
                    np.savetxt(save_name, save_arr[k])

        if self.config.pred_6d or self.config.pred_nocs:
            # save degree error
            degree_err = self.degree_err.cpu().detach().numpy() # GT mode degree error, [B, N]
            output_R   = self.output_R.cpu().detach().numpy().reshape(BS, N, -1) #
            if len(degree_err.shape) < 3:
                degree_err = degree_err[:, :, np.newaxis]

            # degree_err
            for j in range(degree_err.shape[-1]):
                save_arr  = np.concatenate([input_pts, degree_err[:, :, j:j+1], output_R], axis=-1)
                for k in range(degree_err.shape[0]): # batch
                    save_name = f'{save_path}/{mode}_{self.clock.step}_{ids[k]}_{idxs[k]}_{j}.txt' #
                    np.savetxt(save_name, save_arr[k])

        # mode prediction
        if self.config.pred_mode:
            output_M = self.output_M.cpu().detach().numpy().reshape(BS, -1, M)
            save_path = f'{self.config.log_dir}/mode'
            if not exists(save_path):
                print('making directories', save_path)
                makedirs(save_path)
            for j in range(M):
                save_arr  = np.concatenate([input_pts, output_M[:, :, j:j+1]], axis=-1)
                for k in range(num):
                    save_name = f'{save_path}/{mode}_{self.clock.step}_{ids[k]}_{idxs[k]}_{j}.txt'
                    np.savetxt(save_name, save_arr[k])

        # shape reconstruction
        if 'completion' in self.config.task:
            outputs_pts     = self.output_pts[:num].transpose(1, 2).detach().cpu().numpy()
            transformed_pts = self.transformed_pts[:num].detach().cpu().numpy()
            tb.add_mesh("gt", vertices=target_pts, global_step=self.clock.step)
            tb.add_mesh("output", vertices=outputs_pts, global_step=self.clock.step)
            if self.use_wandb and self.clock.step % 500 == 0:
                outputs_pts[0] = outputs_pts[0] + np.array([0, 1, 0]).reshape(1, -1)
                pts = np.concatenate([target_pts[0], outputs_pts[0]], axis=0)
                wandb.log({"input+AE_output": [wandb.Object3D(pts)], 'step': self.clock.step})
                # camera space
                pts = np.concatenate([input_pts[0], transformed_pts[0]], axis=0)
                wandb.log({"camera_space": [wandb.Object3D(pts)], 'step': self.clock.step})

            # canon shape, camera shape, input shape
            for k in range(num): # batch
                save_name = f'{save_path}/{mode}_{self.clock.step}_{ids[k]}_input.txt'
                np.savetxt(save_name, np.concatenate( [input_pts[k], 0.1 * np.ones((input_pts[k].shape[0], 1))], axis=1))
                save_name = f'{save_path}/{mode}_{self.clock.step}_{ids[k]}_target.txt'
                np.savetxt(save_name, np.concatenate( [target_pts[k], 0.25 * np.ones((target_pts[k].shape[0], 1))], axis=1))
                save_name = f'{save_path}/{mode}_{self.clock.step}_{ids[k]}_canon.txt'
                np.savetxt(save_name, np.concatenate( [outputs_pts[k], 0.5 * np.ones((outputs_pts[k].shape[0], 1))], axis=1))
                save_name = f'{save_path}/{mode}_{self.clock.step}_{ids[k]}_pred.txt'
                np.savetxt(save_name, np.concatenate( [transformed_pts[k], 0.75 * np.ones((transformed_pts[k].shape[0], 1))], axis=1))

class PointVAEAgent(BaseAgent):
    def __init__(self, config):
        super(PointVAEAgent, self).__init__(config)
        self.z_dim = config.z_dim
        self.weight_kl_vae = config.weight_kl_vae

    def build_net(self, config):
        # customize your build_net function
        net = get_network(config, "pointVAE")
        print('-----pointVAE architecture-----')
        print(net)
        if config.parallel:
            net = nn.DataParallel(net)
        net = net.cuda()
        return net

    def set_loss_function(self):
        pass

    def forward(self, data):
        input_pts = data['points'].cuda()
        target_pts = input_pts.clone().detach()

        self.output_pts, mu, logvar = self.net(input_pts)

        self.emd_loss = earth_mover_distance(self.output_pts, target_pts)
        self.emd_loss = torch.mean(self.emd_loss)

        self.kl_loss = -0.5 * torch.mean(1 + logvar - mu ** 2 - torch.exp(logvar)) * self.weight_kl_vae

    def collect_loss(self):
        loss_dict = {"emd": self.emd_loss, "kl": self.kl_loss}
        return loss_dict

    def random_sample(self, num):
        z = torch.normal(torch.zeros((num, self.z_dim)), torch.ones((num, self.z_dim))).cuda()
        gen_pts = self.net.decoder(z)
        return gen_pts

    def visualize_batch(self, data, mode, **kwargs):
        tb = self.train_tb if mode == 'train' else self.val_tb

        num = 2

        target_pts = data['points'][:num].transpose(1, 2).detach().cpu().numpy()
        outputs_pts = self.output_pts[:num].transpose(1, 2).detach().cpu().numpy()

        tb.add_mesh("gt", vertices=target_pts, global_step=self.clock.step)
        tb.add_mesh("output", vertices=outputs_pts, global_step=self.clock.step)

        self.net.eval()
        with torch.no_grad():
            gen_pts = self.random_sample(num)
        gen_pts = gen_pts.transpose(1, 2).detach().cpu().numpy()
        tb.add_mesh("generated", vertices=gen_pts, global_step=self.clock.step)
