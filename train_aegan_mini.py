from os import makedirs, remove
from os.path import exists, join
from time import time
import hydra
from hydra import utils
import logging
from omegaconf import DictConfig, ListConfig, OmegaConf
import pickle
import wandb
import random
import torch
import dgl

from collections import OrderedDict
from tqdm import tqdm
from sklearn.externals import joblib
from dataset.obman_parser import ObmanParser
from dataset.modelnet40_parser import ModelParser
from common.train_utils import cycle

from models.ae_gan.networks_ae import BuildGraph
from models.base import BaseAgent
from models.ae_gan import get_network
from utils.emd import earth_mover_distance
from models.losses import loss_geodesic, compute_vect_loss, compute_1vN_nocs_loss, compute_miou_loss
from common.yj_pose import compute_pose_diff, rot_diff_degree

from common.debugger import *
from evaluation.pred_check import post_summary, prepare_pose_eval
from common.algorithms import compute_pose_ransac
from common.d3_utils import compute_rotation_matrix_from_ortho6d, axis_diff_degree, rot_diff_rad, rotate_about_axis
from common.vis_utils import plot_distribution
from global_info import global_info

infos           = global_info()
my_dir          = infos.base_path
project_path    = infos.project_path
categories_id   = infos.categories_id
categories      = infos.categories

whole_obj = infos.whole_obj
part_obj  = infos.part_obj
obj_urdf  = infos.obj_urdf
categories_id = infos.categories_id
project_path  = infos.project_path

THRESHOLD_GOOD = 5 # 5 degrees
def check_r_bb(test_data, tr_agent):
    BS = 2
    input_pts     = test_data['G'].ndata['x'].view(BS, -1, 3).contiguous().cpu().numpy() # B, N, 3
    pred_vertices = tr_agent.output_R.permute(0, 2, 3, 1).cpu().numpy() + input_pts[:, :, np.newaxis, :]# Bs, 3, N, M --> B, N, M, 3
    target_R      = test_data['R'].numpy() + input_pts[:, :, np.newaxis, :] # B, N, M, 3
    return np.linalg.norm(pred_vertices - target_R, axis=-1).mean(axis=0).mean(axis=0)

def check_t(data, tr_agent):
    BS = 2
    input_pts     = test_data['G'].ndata['x'].view(BS, -1, 3).contiguous().cpu().numpy() # B, N, 3
    gt_center     = input_pts - test_data['T'].cpu().numpy()
    pred_center   = input_pts - tr_agent.output_T.cpu().detach().numpy().transpose(0, 2, 1)

    return np.linalg.norm(np.mean(pred_center, axis=1) - np.mean(gt_center, axis=1))

def get_single_data():
    category_name = 'airplane'
    instance_name = '002'
    train_pts= []
    targets = []
    for i in range(2):
        fn  = [category_name, f'{my_dir}/data/modelnet40_normal_resampled/airplane/airplane_0002.txt']
        point_normal_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        full_pts  = np.copy(point_normal_set[:, 0:3])
        r = np.eye(3).astype(np.float32)
        train_pts.append(full_pts)
        targets.append(r)
    print('data loading ...  ready')
    return train_pts, targets

def get_chirality_data(augment=False):
    category_name = 'airplane'
    instance_name = '0002'
    num_points    = 256
    train_pts= []
    targets  = []
    for m in range(4):
        fn  = [category_name, f'{my_dir}/data/modelnet40_normal_resampled/{category_name}/airplane_{instance_name}.txt']
        point_normal_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        # xyz  = np.random.permutation(point_normal_set[:, 0:3])[:256]
        xyz  = point_normal_set[:, 0:3][:256] # fixed
        xyz1 = np.concatenate([xyz, xyz * np.array([[1, 1, -1]])], axis=0)
        r = np.eye(3).astype(np.float32)

        if augment:
            print('doing augment')
            i, j, k = random.sample(range(0, 8), 3)
            theta_x = 360/8 * int(i)
            Rx = rotate_about_axis(theta_x / 180 * np.pi, axis='x')
            theta_y = 360/8 * int(j)
            Ry = rotate_about_axis(theta_y / 180 * np.pi, axis='y')
            theta_z = 360/8 * int(k)
            Rz = rotate_about_axis(theta_z / 180 * np.pi, axis='z')
            r = np.matmul(Ry, Rx).astype(np.float32)
            r = np.matmul(Rz, r).astype(np.float32)
            xyz1 = np.matmul(xyz1, r.T)

        train_pts.append(xyz1)
        targets.append(r.T)
    print('data loading ...  ready')
    return train_pts, targets

# only get one instance
def get_test_data(nx=8, ny=8, nz=8):
    category_name = 'airplane'
    instance_name = '002'
    fn  = [category_name, f'{my_dir}/data/airplane/airplane/0_0_0.txt']
    train_pts= []
    targets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                theta_x = 360/8 * int(i)
                Rx = rotate_about_axis(theta_x / 180 * np.pi, axis='x')
                theta_y = 360/8 * int(j)
                Ry = rotate_about_axis(theta_y / 180 * np.pi, axis='y')
                theta_z = 360/8 * int(k)
                Rz = rotate_about_axis(theta_z / 180 * np.pi, axis='z')
                r = np.matmul(Ry, Rx).astype(np.float32)
                r = np.matmul(Rz, r).astype(np.float32)
                point_normal_set = np.loadtxt(fn[1].replace('0_0_0', f'{i}_{j}_{k}'), delimiter=' ').astype(np.float32)
                full_pts  = np.copy(point_normal_set[:, 0:3])
                train_pts.append(full_pts)
                # targets.append(r)
                targets.append(np.eye(3).astype(np.float32))
    print('data loading ...  ready')
    return train_pts, targets

def get_category_data():
    fpath  = f'{my_dir}/data/modelnet40'
    f_train= f'{fpath}/airplane_train_2048.pk'
    f_test = f'{fpath}/airplane_test_2048.pk'

    with open(f_train, "rb") as f:
       train_pts = joblib.load(f)

    with open(f_test, "rb") as obj_f:
        test_pts  = joblib.load(obj_f)

    print(train_pts.shape, test_pts.shape)
    return train_pts, test_pts

class PointAEPoseAgent(BaseAgent):
    def build_net(self, config):
        # customize your build_net function
        net = get_network(config, "pointAE")
        print(net)
        if config.parallel:
            net = nn.DataParallel(net)
        net = net.cuda()
        return net

    def forward(self, data, verbose=False):
        self.infos    = {}
        if 'se3' in self.config.encoder_type:
            self.predict_se3(data)
        else:
            self.predict_pnet2(data)

        self.compute_loss(data)

    def eval_func(self, data):
        if self.config.pred_6d:
            self.eval_6d(data)

        if self.config.pred_nocs:
            self.eval_nocs(data)

    def eval_6d(self, data, thres=0.1):
        """one step of validation"""
        self.net.eval()
        with torch.no_grad():
            self.forward(data)

        BS = self.config.DATASET.train_batch
        N  = self.config.num_points
        M  = self.config.num_modes_R
        flatten_r = self.output_R.view(BS, N, -1).contiguous()
        pairwise_dis = torch.norm(flatten_r.unsqueeze(2) - flatten_r.unsqueeze(1), dim=-1) # [2, 512, 512]
        inliers_mask = torch.zeros_like(pairwise_dis)
        inliers_mask[pairwise_dis<thres] = 1.0 # [B, N, N]
        score = inliers_mask.mean(dim=-1)
        select_ind  = torch.argmax(score, dim=1) # B
        pred_rot1    = self.output_R[torch.arange(BS), select_ind]
        gt_rot = data['R'].cuda()  # [B, 3, 3]
        rot_err = rot_diff_degree(gt_rot, pred_rot1, chosen_axis=None)  # [B, M]

        input_pts  = data['G'].ndata['x'].view(BS, -1, 3).contiguous() # B, N, 3
        pred_center= input_pts - self.output_T.permute(0, 2, 1).contiguous()
        gt_center  = input_pts - data['T'].cuda() # B, N, 3
        trans_err = torch.norm(pred_center.mean(dim=1) - gt_center.mean(dim=1), dim=1)

        scale_err = torch.Tensor([0, 0])
        self.pose_err = {'rdiff': rot_err, 'tdiff': trans_err, 'sdiff': scale_err}

        return

    def predict_se3(self, data):
        input_pts  = data['points'].cuda()
        BS = self.config.DATASET.train_batch
        N  = self.config.num_points
        M  = self.config.num_modes_R
        CS = self.config.MODEL.num_channels_R
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.latent_vect = self.net.encoder(data['G'].to(device))
        if self.config.pred_nocs:
            self.output_N = self.net.regressor_nocs(self.latent_vect['N'].squeeze().view(BS, -1, self.latent_vect['N'].shape[1]).contiguous().permute(0, 2, 1).contiguous()) # assume to be B*N, 128? --> 3 channel
        self.output_T = self.latent_vect['T'].permute(0, 2, 1).contiguous().squeeze().view(BS, -1, 3).contiguous().permute(0, 2, 1).contiguous()# [B, 3, N]

        if self.config.pred_seg:
            self.output_C = self.net.classifier_seg(self.latent_vect['N'].squeeze().view(BS, -1, self.latent_vect['N'].shape[1]).contiguous().permute(0, 2, 1).contiguous())

        if self.config.pred_mode: # map 4 to 2
            self.output_M = self.net.classifier_mode(self.latent_vect['R0']).squeeze() # [BS * N, M, 1]-->[BS * N, M] --> [BS, M, N]

        if self.config.pred_6d:
            self.compute_and_eval_6d(data)
        else:
            self.compute_and_eval_axis(data)

    def predict_pnet2(self, data):
        input_pts  = data['points'].cuda()
        BS = self.config.DATASET.train_batch
        N  = self.config.num_points
        M  = self.config.num_modes_R
        CS = self.config.MODEL.num_channels_R
        # B, 3, N
        input_pts        = data['G'].ndata['x'].view(target_pts.shape[0], -1, 3).contiguous().permute(0, 2, 1).contiguous().cuda() #
        if 'plus' in self.config.encoder_type:
            input_pts        = input_pts - input_pts.mean(dim=-1, keepdim=True)
        self.latent_vect = self.net.encoder(input_pts) # double
        if self.config.pred_nocs:
            self.output_N = self.latent_vect['N'] #
        elif self.config.pred_6d:
            print('Not implemented yet')
        else:
            self.output_T    = self.latent_vect['T']  # 3, N, no activation
            if self.config.MODEL.num_channels_R > 1:
                self.output_R = self.latent_vect['R'].view(BS, self.config.MODEL.num_channels_R, 3, -1).contiguous().permute(0, 2, 3, 1).contiguous()#  [B, M * 3, C] -> [Bs, M, 3, N] -> [Bs, 3, N, M]
                self.output_R = self.output_R/(torch.norm(self.output_R, dim=1, keepdim=True) + epsilon)
                self.output_R_full = self.output_R
                self.degree_err = torch.acos( torch.sum(self.output_R*target_R.unsqueeze(-1), dim=1)) * 180 / np.pi
            else:
                self.output_R = self.latent_vect['R'].squeeze() # B, 3, N
                self.output_R = self.output_R/(torch.norm(self.output_R, dim=1, keepdim=True) + epsilon)
                self.degree_err = torch.acos( torch.sum(self.output_R*target_R, dim=1)) * 180 / np.pi
                self.output_R_full = self.output_R # all use
            # we have two ways to do the pooling
            self.output_R_pooled = self.output_R.mean(dim=2) # TODO
            self.output_R_pooled = self.output_R_pooled/(torch.norm(self.output_R_pooled, dim=1, keepdim=True) + epsilon)

        if self.config.pred_seg:
            self.output_C = self.latent_vect['C']
        if self.config.pred_mode: # B, M, N
            self.output_M = self.latent_vect['M'].permute(0, 2, 1).contiguous().view(-1, M).contiguous() # BS*N, M

    def eval_nocs(self, data):
        target_pts = data['points'].cuda()
        input_pts = data['G'].ndata['x'].view(target_pts.shape[0], -1, 3).contiguous().permute(0, 2,
                                                                                               1).contiguous().cuda()
        pred_nocs = self.output_N
        gt_nocs = target_pts
        self.pose_err = compute_pose_diff(nocs_gt=gt_nocs.transpose(-1, -2),
                nocs_pred=pred_nocs.transpose(-1, -2), target=input_pts.transpose(-1, -2),
                                          category=self.config.target_category)

    def compute_and_eval_6d(self, data):
        BS = self.config.DATASET.train_batch
        N  = self.config.num_points
        M  = self.config.num_modes_R
        CS = self.config.MODEL.num_channels_R
        target_R   = data['R'].cuda()
        # [B*N, C, 3] --> [B*N*M, 6]
        self.output_R =self.latent_vect['R'].view(-1, 6).contiguous()
        self.output_R = compute_rotation_matrix_from_ortho6d(self.output_R)

        # [B*M, 6]
        self.output_R_pooled =self.latent_vect['1'].view(-1, 6).contiguous() # dense by default!!! B, 2*C, 3
        self.output_R_pooled = compute_rotation_matrix_from_ortho6d(self.output_R_pooled) #
        #     target_R_tiled  = target_R.unsqueeze(1).contiguous().unsqueeze(1).contiguous().repeat(1, N, M, 1, 1).contiguous()
        #     geodesic_loss   = loss_geodesic(self.output_R, target_R_tiled.view(-1, 3, 3).contiguous())  # BS*N* C/2
        #     self.degree_err_full = geodesic_loss.view(BS, N, M).contiguous()       # BS, N, M
        #     self.output_R_full   = self.output_R.view(BS, N, M, 3, 3).contiguous() # BS, N, M, 3, 3
        #     self.output_R   = self.output_R_full.squeeze()
        #     self.norm_err   = torch.norm(self.output_R.view(BS, N, M, -1).contiguous() - target_R_tiled.view(BS, N, M, -1).contiguous(), dim=-1)

        target_R_tiled  = target_R.unsqueeze(1).contiguous().repeat(1, N, 1, 1).contiguous()
        geodesic_loss   = loss_geodesic(self.output_R, target_R_tiled.view(-1, 3, 3).contiguous().float())  # BS*N* C/2
        self.degree_err_full = geodesic_loss.view(BS, N).contiguous()       # BS, N
        self.output_R_full   = self.output_R.view(BS, N, 3, 3).contiguous() # BS, N, 3, 3
        self.output_R   = self.output_R_full.squeeze()
        self.norm_err   = torch.norm(self.output_R.view(BS, N, 9).contiguous() - target_R_tiled.view(BS, N, 9).contiguous(), dim=-1)

        if self.config.check_consistency: # BS, N, M, 3, 3
            mean_R   = torch.mean(self.output_R_full, dim=1, keepdim=True).view(BS, 1, M, -1).contiguous()
            variance = torch.norm(self.output_R.view(BS, N, M, -1).contiguous() - mean_R, dim=-1).mean()
            self.consistency_loss = variance * self.config.consistency_loss_multiplier
            self.infos['consistency'] = self.consistency_loss
        else:
            self.consistency_loss = 0.0

        if M > 1: # multi-mode dense
            if self.config.use_adaptive_mode:
                min_loss, min_indices = torch.min(self.norm_err, dim=-1) # B, N
                self.regressionR_loss = min_loss.mean(dim=1)
                self.degree_err       = self.degree_err_full[torch.arange(BS).reshape(-1, 1), torch.arange(N).reshape(1, -1), min_indices].contiguous()
            else:
                min_loss, min_indices = torch.min(self.norm_err.mean(dim=1), dim=-1) # we only allow one mode to be True
                self.regressionR_loss = min_loss
                self.degree_err = self.degree_err_full[torch.arange(BS), :, min_indices] # degree err is raw GT mode prediction
        else:
            self.regressionR_loss = self.norm_err.mean(dim=1)
            self.degree_err = self.degree_err_full.squeeze()

        self.regressionR_loss = self.regressionR_loss.mean()
        correctness_mask = torch.zeros((self.degree_err.shape[0], self.degree_err.shape[1]), device=self.output_R.device) # B, N
        correctness_mask[self.degree_err<THRESHOLD_GOOD] = 1.0
        good_pred_ratio = torch.mean(correctness_mask, dim=1) # TODO
        self.infos.update({'5deg': good_pred_ratio.mean(), 'rdiff': self.degree_err.mean()})
        if self.config.pred_mode and M > 1:
            if self.config.use_adaptive_mode:
                self.target_M = min_indices.view(-1).contiguous().detach().clone()
            else:
                self.target_M = min_indices.unsqueeze(1).repeat(1, N).contiguous().view(-1).contiguous().detach().clone()
            self.classifyM_loss = compute_miou_loss(self.output_M, self.target_M, loss_type='xentropy')
            self.output_M_label = torch.argmax(self.output_M, dim=-1)  # [B * N] in [0...M-1]
            self.classifyM_acc  = (self.output_M_label == self.target_M).float().mean()
            self.degree_err_chosen  = self.degree_err_full[torch.arange(BS).reshape(-1, 1), torch.arange(N).reshape(1, -1),  self.output_M_label.reshape(BS, N)].contiguous()

    def compute_and_eval_axis(self, data):
        target_R   = data['R'].cuda()
        target_R = target_R.permute(0, 2, 1).contiguous() # B, 3, 1
        if self.config.MODEL.num_channels_R > 1:
            BS, CS = target_R.shape[0], self.latent_vect['R'].shape[-2]     # [BS*N, C, 3] -> [Bs, N, C, 3] -> [Bs, 3, N, C]
            self.output_R  = self.latent_vect['R'].view(BS, -1, CS, 3).contiguous().permute(0, 3, 1, 2).contiguous()
            self.output_R = self.output_R/(torch.norm(self.output_R, dim=1, keepdim=True) + epsilon)
            self.output_R_full = self.output_R
            self.degree_err = torch.acos( torch.sum(self.output_R*target_R.unsqueeze(-1), dim=1)) * 180 / np.pi
        else:
            self.output_R = self.latent_vect['R'].squeeze().view(target_R.shape[0], -1, 3).contiguous().permute(0, 2, 1).contiguous() # B, 3, N
            self.output_R = self.output_R/(torch.norm(self.output_R, dim=1, keepdim=True) + epsilon)
            self.degree_err = torch.acos( torch.sum(self.output_R*target_R, dim=1)) * 180 / np.pi

        self.output_R_pooled = self.output_R.mean(dim=2) #
        # self.output_R_pooled = self.latent_vect['1'].permute(0, 2, 1).contiguous() # B, 3, 1
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
                self.regressionR_loss = compute_vect_loss(self.output_R.squeeze(), target_R.double(), confidence=confidence, target_category=self.config.target_category, use_one2many=self.config.use_one2many) # B
            else:
                self.regressionR_loss = compute_vect_loss(self.output_R_pooled.unsqueeze(-1), target_R, target_category=self.config.target_category, use_one2many=self.config.use_one2many) # B, 3, N
            self.infos['regressionR'] = self.regressionR_loss.mean()

        self.regressionR_loss = self.regressionR_loss.mean()

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

    def compute_loss(self, data):
        input_pts  = data['points'].cuda()
        target_pts = input_pts.clone().detach()
        target_T   = data['T'].cuda().permute(0, 2, 1).contiguous() # B, 3, N
        target_C   = None  # target_C   = data['C'].cuda()
        confidence = target_C
        #>>>>>>>>>>>>>>>>>> 2. computing loss <<<<<<<<<<<<<<<<<<<#
        if self.config.pred_seg:
            self.seg_loss = compute_miou_loss(self.output_C, target_C).mean()

        if self.config.pred_nocs:
            self.nocs_loss= compute_1vN_nocs_loss(self.output_N, target_pts, target_category=self.config.target_category, num_parts=1, confidence=target_C)
            self.nocs_loss= self.nocs_loss.mean()

        if self.config.use_objective_T:
            self.regressionT_loss = compute_vect_loss(self.output_T, target_T, confidence=confidence).mean() # TYPE_LOSS='SOFT_L1'

        if 'completion' in self.config.task:
            if isinstance(self.latent_vect, dict):
                self.output_pts = self.net.decoder(self.latent_vect['0'])
            else:
                self.output_pts = self.net.decoder(self.latent_vect)
            self.emd_loss = earth_mover_distance(self.output_pts, target_pts)
            self.emd_loss = torch.mean(self.emd_loss)

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
                loss_dict['classifyM'] = self.classifyM_loss
            if self.config.use_objective_V:
                loss_dict['consistency'] = self.consistency_loss
        if 'completion' in self.config.task:
            loss_dict["emd"]=self.emd_loss

        return loss_dict

    # seems vector is more stable
    def collect_info(self):
        if 'pose' in self.config.task:
            return self.infos
        else:
            return None

    def visualize_batch(self, data, mode, **kwargs):
        tb = self.train_tb if mode == 'train' else self.val_tb
        if self.config.pred_nocs:
            return

        num = 2
        target_pts = data['points'][:num].transpose(1, 2).detach().cpu().numpy() # canonical space
        input_pts = data['G'].ndata['x'].view(target_pts.shape[0], -1, 3).contiguous().cpu().numpy()
        ids  = data['id']
        idxs = data['idx']
        #
        BS    = self.output_R.shape[0]
        M     = self.config.num_modes_R #
        N     = input_pts.shape[1]

        # save degree error
        degree_err = self.degree_err.cpu().detach().numpy() # GT mode degree error, [B, N]
        output_R   = self.output_R.cpu().detach().numpy().reshape(BS, N, -1) #
        if len(degree_err.shape) < 3:
            degree_err = degree_err[:, :, np.newaxis]

        save_path = f'{self.config.log_dir}/generation'
        if not exists(save_path):
            print('making directories', save_path)
            makedirs(save_path)

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
            for j in range(M): # modals
                save_arr  = np.concatenate([input_pts, output_M[:, :, j:j+1]], axis=-1)
                for k in range(BS): # batch
                    save_name = f'{save_path}/{mode}_{self.clock.step}_{ids[k]}_{idxs[k]}_{j}.txt' #
                    np.savetxt(save_name, save_arr[k])

        # shape reconstruction
        if 'completion' in self.config.task:
            outputs_pts = self.output_pts[:num].transpose(1, 2).detach().cpu().numpy()
            tb.add_mesh("gt", vertices=target_pts, global_step=self.clock.step)
            tb.add_mesh("output", vertices=outputs_pts, global_step=self.clock.step)
            if self.use_wandb and self.clock.step % 500 == 0:
                outputs_pts[0] = outputs_pts[0] + np.array([0, 1, 0]).reshape(1, -1)
                pts = np.concatenate([target_pts[0], outputs_pts[0]], axis=0)
                wandb.log({"input+AE_GAN_output": [wandb.Object3D(pts)], 'step': self.clock.step})

@hydra.main(config_path="config/completion.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(hydra.utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    is_cuda = (torch.cuda.is_available() and not cfg.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")
    # Set t0
    t0 = time()
    # category-wise training setup
    infos       = global_info()
    data_infos  = infos.datasets[cfg.item]
    cfg.root_data   = infos.second_path + '/data'
    cfg.log_dir     = infos.second_path + cfg.log_dir
    cfg.model_dir   = cfg.log_dir + '/checkpoints'
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)
        os.makedirs(cfg.log_dir + '/checkpoints'
        )

    if cfg.use_wandb:
        if cfg.eval:
            run_name = f'{cfg.exp_num}_{cfg.target_category}_eval'
        else:
            run_name = f'{cfg.exp_num}_{cfg.target_category}'
        wandb.init(project="haoi-pose", name=run_name)
        wandb.init(config=cfg)
    # copy the project codes into log_dir
    if (not cfg.eval) and (not cfg.debug):
        if not os.path.isdir(f'{cfg.log_dir}/code'):
            os.makedirs(f'{cfg.log_dir}/code')
            os.makedirs(f'{cfg.log_dir}/code/dataset')
        os.system('cp -r ./models {}/code'.format(cfg.log_dir))
        os.system('cp -r ./config {}/code'.format(cfg.log_dir))
        os.system('cp ./dataset/*py {}/code/dataset'.format(cfg.log_dir))

    # Shorthands
    out_dir = cfg.log_dir
    print('Saving to ', out_dir)

    # create network and training agent
    # tr_agent = get_agent(cfg)
    tr_agent = PointAEPoseAgent(cfg)
    if cfg.use_wandb:
        if cfg.module=='gan':
            wandb.watch(tr_agent.netG)
            wandb.watch(tr_agent.netD)
        else:
            wandb.watch(tr_agent.net)

    # load from checkpoint if provided
    if cfg.use_pretrain or cfg.eval:
        tr_agent.load_ckpt(cfg.ckpt)

    # train_pts, targets = get_test_data()
    # train_pts, test_pts = get_category_data()
    # train_pts, train_targets = get_single_data()
    train_pts, train_targets = get_chirality_data()

    # # test data: 1. partial
    # test_pts, test_targets = get_test_data(nx=8, ny=1, nz=1)
    # # test data: 2. what if we rotate
    test_pts, test_targets = get_chirality_data(augment=True)
    builder = BuildGraph(num_samples=10)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    inputs = []
    BS     = 2
    N  = cfg.num_points
    M  = cfg.num_modes_R
    CS = cfg.MODEL.num_channels_R
    npoints= cfg.num_points
    fixed_sampling = cfg.fixed_sampling

    # start training
    clock = tr_agent.clock #
    epoch_size = max(int(len(train_pts)/2), 1000)
    pbar = tqdm(range(0, epoch_size))
    for e in range(clock.epoch, 200):
        for _, b in enumerate(pbar):
            input_pts = np.copy(np.array(train_pts[:2])).astype(np.float32)
            target_R = torch.from_numpy(np.stack(train_targets[:2], axis=0)).cuda()

            # input_pts = np.copy(train_pts[2*b:2*b+2].astype(np.float32))
            if not fixed_sampling:
                input_pts[0] = np.random.permutation(input_pts[0])
                input_pts[1] = np.random.permutation(input_pts[1])
            xyz1 = torch.from_numpy(input_pts[:, :npoints]).cuda()
            g, _     = builder(xyz1)
            data = {}
            data['G'] = g
            data['R'] = target_R
            data['points'] = xyz1
            data['T'] = xyz1
            data['id']= ['0002', '0002']
            data['idx'] = [b, b]

            # loss_dict, info_dict = app_func(tr_agent, g, target_R, cfg)
            torch.cuda.empty_cache()
            # loss_dict, info_dict = tr_agent.train_func(data)
            loss_dict, info_dict = tr_agent.val_func(data)
            x1 = tr_agent.latent_vect['R']
            f1 = tr_agent.latent_vect['N']
            pbar.set_description("EPOCH[{}][{}]".format(0, e)) #
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in  {**loss_dict, **info_dict}.items()}))
            # visualize
            if cfg.vis and clock.step % cfg.vis_frequency == 0:
                tr_agent.visualize_batch(data, "train")

            # if clock.step % cfg.eval_frequency == 0:
            #     track_dict = {'rdiff': [], 'tdiff': [], 'sdiff': [],
            #                   '5deg': [], '5cm': [], '5deg5cm': []}
            #     for num in range(10):
            #         tr_agent.eval_func(data)
            #         pose_diff = tr_agent.pose_err
            #         for key in ['rdiff', 'tdiff', 'sdiff']:
            #             track_dict[key].append(pose_diff[key].cpu().numpy().mean())
            #         deg = pose_diff['rdiff'] <= 5.0
            #         cm = pose_diff['tdiff'] <= 0.05
            #         # degcm = torch.logical_and(deg, cm)
            #         degcm = deg & cm
            #         for key, value in zip(['5deg', '5cm', '5deg5cm'], [deg, cm, degcm]):
            #             track_dict[key].append(value.float().cpu().numpy().mean())
            #     for key, value in track_dict.items():
            #         print(key, ':', np.array(value).mean())
            #     if cfg.use_wandb:
            #         for key, value in track_dict.items():
            #             wandb.log({f'test/{key}': np.array(value).mean(), 'step': clock.step})

            if clock.step % cfg.val_frequency == 0:
                # inds = random.sample(range(0, len(test_pts)-1), BS)
                inds = [0, 1]
                if fixed_sampling:
                    input_pts = np.stack([test_pts[inds[0]][:npoints, :], test_pts[inds[1]][:npoints, :]], axis=0).astype(np.float32)
                else:
                    input_pts = np.stack([np.random.permutation(test_pts[inds[0]])[:npoints, :], np.random.permutation(test_pts[inds[1]])[:npoints, :]], axis=0).astype(np.float32)
                xyz2 = torch.from_numpy(input_pts).cuda()
                target_R = torch.from_numpy(np.stack(test_targets[:2], axis=0).astype(np.float32)).cuda()
                #
                g1, _     = builder(xyz2)
                data = {}
                data['G'] = g1
                data['R'] = target_R
                data['points'] = xyz2
                data['T'] = xyz2
                data['id']= ['0002', '0002']
                data['idx'] = inds
                torch.cuda.empty_cache()
                loss_dict, info_dict = tr_agent.val_func(data)
                x2 = tr_agent.latent_vect['R'] # [B*N, 2, 3]
                f2 = tr_agent.latent_vect['N']
                # check the invariance and equivalence
                if cfg.vis and clock.step % cfg.vis_frequency == 0:
                    tr_agent.visualize_batch(data, "validation")
            print('raw pts diff: ')
            print(np.matmul(train_pts[0], test_targets[0]) - test_pts[0])
            print(np.matmul(xyz1[0].cpu().numpy(), target_R[0].cpu().numpy()) - xyz2[0].cpu().numpy())
            xyz1_rotated    = torch.matmul(xyz1[0], target_R[0]) # 2, 512
            xyz_diff        = xyz2[0] - xyz1_rotated
            x1_rotated      = torch.matmul(x1[0], target_R[0]) # B, 3, 3, [1024, 2, 3]
            x_diff_rotation = x2[0] - x1_rotated
            f_diff_rotation = f2 - f1

            print('xyz diff mean:', torch.mean(torch.abs(xyz_diff)))
            print('xyz diff max:', torch.max(torch.abs(xyz_diff)))
            print('x diff mean:', torch.mean(torch.abs(x_diff_rotation)))
            print('x diff max:', torch.max(torch.abs(x_diff_rotation)))
            print('f diff mean:', torch.mean(torch.abs(f_diff_rotation)))
            print('f diff max:', torch.max(torch.abs(f_diff_rotation)))
            bp()
            clock.tick()
            if clock.step % cfg.save_step_frequency == 0:
                tr_agent.save_ckpt('latest')

        tr_agent.update_learning_rate()
        clock.tock()

        if clock.epoch % 10 == 0:
            tr_agent.save_ckpt()

if __name__ == '__main__':
    main()
