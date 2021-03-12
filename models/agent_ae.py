import torch
import torch.nn as nn
import numpy as np
import wandb
import __init__
from models.ae_gan import get_network
from models.base import BaseAgent
from utils.emd import earth_mover_distance
from models.losses import loss_geodesic, loss_vectors, compute_vect_loss, compute_1vN_nocs_loss, compute_miou_loss
from common.d3_utils import compute_rotation_matrix_from_euler, compute_euler_angles_from_rotation_matrices, compute_rotation_matrix_from_ortho6d, chamfer_distance
from os import makedirs, remove
from os.path import exists, join

def bp():
    import pdb;pdb.set_trace()
# if self.config.pred_conf:
#     pred_c = self.output_Cf.squeeze()
#     gt_c = torch.abs(1 - dis/2)
#     confi_err= torch.abs(gt_c - pred_c)
#     w   = self.config.confidence_loss_multiplier
#     self.regressionCi_loss= w * torch.sum( confi_err * confidence , dim=1) / (torch.sum(confidence, dim=1) + 1)
#     self.regressionCi_loss = self.regressionCi_loss.mean()

epsilon=1e-10
THRESHOLD_GOOD = 5  # in degrees
THRESHOLD_BAD  = 50 # in degrees
class PointAEPoseAgent(BaseAgent):
    def build_net(self, config):
        # customize your build_net function
        net = get_network(config, "pointAE")
        print('-----pointAEPose architecture-----')
        print(net)
        if config.parallel:
            net = nn.DataParallel(net)
        net = net.cuda()
        return net

    def set_loss_function(self):
        pass

    def forward(self, data, verbose=False):
        input_pts  = data['points'].cuda()
        target_pts = input_pts.clone().detach()
        target_R   = data['R'].cuda()
        target_T   = data['T'].cuda()
        if 'C' in data:
            target_C   = data['C'].cuda()
        else:
            target_C   = None
        self.infos = {}

        BS = target_R.shape[0]
        N  = target_T.shape[1]
        M  = self.config.num_modes_R
        CS = self.config.MODEL.num_channels_R
        target_R = target_R.permute(0, 2, 1).contiguous() # B, 3, 1
        target_T      = target_T.permute(0, 2, 1).contiguous() # B, 3, N
        if 'se3' in self.config.encoder_type:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            self.latent_vect = self.net.encoder(data['G'].to(device))

            if self.config.pred_nocs:
                self.output_N = self.net.regressor_nocs(self.latent_vect['N'].squeeze().view(target_R.shape[0], -1, self.latent_vect['N'].shape[1]).contiguous().permute(0, 2, 1).contiguous()) # assume to be B*N, 128? --> 3 channel
            elif self.config.pred_6d: # if we add pred_6d, for some, R^T, R^T;
                # [B*N, C, 3],  --> [B*N, M, 2, 3] --> B*N, M, 6 --> [B*N*M, 6]
                self.output_R = self.latent_vect['R'].view(-1, M, 2, 3).contiguous().view(-1, M, 6).contiguous().view(-1, 6).contiguous()
                self.output_R = compute_rotation_matrix_from_ortho6d(self.output_R)
                # [B*M, 6]
                self.output_R_pooled = self.latent_vect['1'].view(-1, M, 2, 3).contiguous().view(-1, M, 6).contiguous().view(-1, 6).contiguous() # dense by default!!! B, 2*C, 3
                self.output_R_pooled = compute_rotation_matrix_from_ortho6d(self.output_R_pooled) #
            else:
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

            self.output_T = self.latent_vect['T'].permute(0, 2, 1).contiguous().squeeze().view(target_T.shape[0], -1, 3).contiguous().permute(0, 2, 1).contiguous()# [B, 3, N]
            if self.config.pred_seg:
                self.output_C = self.net.classifier_seg(self.latent_vect['N'].squeeze().view(target_R.shape[0], -1, self.latent_vect['N'].shape[1]).contiguous().permute(0, 2, 1).contiguous())

            if self.config.pred_conf:
                self.output_Cf = self.net.regressor_confi(self.latent_vect['N'].squeeze().view(target_R.shape[0], -1, self.latent_vect['N'].shape[1]).contiguous().permute(0, 2, 1).contiguous())

            if self.config.pred_mode: # need to map 4 to 2?
                self.output_M = self.net.classifier_mode(self.latent_vect['R0']).squeeze() # [BS * N, M, 1]-->[BS * N, M] --> [BS, M, N]
        else:
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
                self.output_T    = self.latent_vect['T'] # 3, N, no activation
                if self.config.MODEL.num_channels_R > 1:
                    self.output_R  = self.latent_vect['R'].view(BS, self.config.MODEL.num_channels_R, 3, -1).contiguous().permute(0, 2, 3, 1).contiguous()# # [BS*N, C, 3] -> [Bs, N, C, 3] -> [Bs, 3, N, C]
                    self.output_R  = self.output_R/(torch.norm(self.output_R, dim=1, keepdim=True) + epsilon)
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

        #>>>>>>>>>>>>>>>>>> 2. computing loss <<<<<<<<<<<<<<<<<<<#
        if self.config.pred_seg:
            self.seg_loss = compute_miou_loss(self.output_C, target_C).mean()

        if self.config.pred_nocs:
            self.nocs_loss= compute_1vN_nocs_loss(self.output_N, target_pts, target_category=self.config.target_category, num_parts=1, confidence=target_C)
            self.nocs_loss= self.nocs_loss.mean()

        # self.output_R: [BS*N*M, 3, 3]
        elif self.config.pred_6d:
            target_R_tiled  = target_R.unsqueeze(1).contiguous().unsqueeze(1).contiguous().repeat(1, N, M, 1, 1).contiguous()
            geodesic_loss   = loss_geodesic(self.output_R, target_R_tiled.view(-1, 3, 3).contiguous())  # BS*N* C/2
            self.degree_err_full = geodesic_loss.view(BS, N, M).contiguous()       # BS, N, M
            self.output_R_full   = self.output_R.view(BS, N, M, 3, 3).contiguous() # BS, N, M, 3, 3
            self.output_R   = self.output_R_full.squeeze()
            self.norm_err   = torch.norm(self.output_R.view(BS, N, M, -1).contiguous() - target_R_tiled.view(BS, N, M, -1).contiguous(), dim=-1)
            if self.config.check_consistency: # BS, N, M, 3, 3
                mean_R   = torch.mean(self.output_R_full, dim=1, keepdim=True).view(BS, 1, M, -1).contiguous()
                variance = torch.norm(self.output_R.view(BS, N, M, -1).contiguous() - mean_R, dim=-1).mean()
                self.consistency_loss = variance * self.config.consistency_loss_multiplier
                self.infos['consistency'] = self.consistency_loss
            else:
                self.consistency_loss = 0.0
            # degree err is raw GT mode prediction, degree_err_chosen is using predicted mode
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
                self.degree_err = self.degree_err_full.squeeze()
                if self.config.rotation_use_dense:
                    self.regressionR_loss = self.norm_err.mean(dim=1)
                else: # BS*M, 3, 3
                    self.regressionR_loss = torch.norm(self.output_R_pooled.view(BS, -1).contiguous() - target_R.view(BS, -1).contiguous(), dim=-1)
            self.regressionR_loss = self.regressionR_loss.mean()

            # degree_err_full
            correctness_mask = torch.zeros((self.degree_err.shape[0], self.degree_err.shape[1]), device=self.output_R.device) # B, N
            correctness_mask[self.degree_err<THRESHOLD_GOOD] = 1.0
            good_pred_ratio = torch.mean(correctness_mask, dim=1) # TODO
            self.infos['good_pred_ratio'] = good_pred_ratio.mean() # scalar
            if self.config.pred_mode and M > 1:
                if self.config.use_adaptive_mode:
                    self.target_M = min_indices.view(-1).contiguous().detach().clone()
                else:
                    self.target_M = min_indices.unsqueeze(1).repeat(1, N).contiguous().view(-1).contiguous().detach().clone()
                self.classifyM_loss = compute_miou_loss(self.output_M, self.target_M, loss_type='xentropy')
                self.output_M_label = torch.argmax(self.output_M, dim=-1)  # [B * N] in [0...M-1]
                self.classifyM_acc  = (self.output_M_label == self.target_M).float().mean()
                self.degree_err_chosen  = self.degree_err_full[torch.arange(BS).reshape(-1, 1), torch.arange(N).reshape(1, -1),  self.output_M_label.reshape(BS, N)].contiguous()
        else: # loss for up axis
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
                    bp()
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

                B, N = len(min_indices), self.output_R.shape[-1]
                # [B, 3, N, M]
                self.output_R_chosen = self.output_R_full[torch.arange(B).reshape(-1, 1), :,
                                       torch.arange(N).reshape(1, -1), self.output_M_label.reshape(B, N)].permute(0, 2, 1)  # [B, N, 3]
                self.degree_err_chosen = torch.acos(torch.sum(self.output_R_chosen * target_R, dim=1)) * 180 / np.pi

        if target_T is not None and self.config.use_objective_T:
            self.regressionT_loss = compute_vect_loss(self.output_T, target_T, confidence=confidence).mean() # TYPE_LOSS='SOFT_L1'

        if 'completion' in self.config.task:
            if isinstance(self.latent_vect, dict):
                self.output_pts = self.net.decoder(self.latent_vect['0'])
            else:
                self.output_pts = self.net.decoder(self.latent_vect)
            self.emd_loss = earth_mover_distance(self.output_pts, target_pts)
            self.emd_loss = torch.mean(self.emd_loss)
            self.infos['canonical_recon_loss'] = self.emd_loss
            # if 'unsupervised' in self.config.task:
            #     # B, 3, N
            #     camera_pts = data['G'].ndata['x'].view(target_pts.shape[0], -1, 3).contiguous().permute(0, 2, 1).contiguous()
            #     pred_T = camera_pts - self.output_T
            #     pred_T = pred_T.mean(dim=-1) # B, 3, 1
            #     pred_R = self.output_R_pooled # B, 3, 3
            #     self.output_pts = torch.matmul(self.output_pts.permute(0, 2, 1).contiguous() - 0.5, pred_R) + pred_T.unsqueeze(1).contiguous()  # RR
            #     bp()
            #     self.emd_cam_loss = earth_mover_distance(self.output_pts, camera_pts.permute(0, 2, 1).contiguous())
            #     self.emd_cam_loss = torch.mean(self.emd_cam_loss)
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

        num = 2

        target_pts = data['points'][:num].transpose(1, 2).detach().cpu().numpy()
        if 'completion' in self.config.task:
            outputs_pts = self.output_pts[:num].transpose(1, 2).detach().cpu().numpy()
            tb.add_mesh("gt", vertices=target_pts, global_step=self.clock.step)
            tb.add_mesh("output", vertices=outputs_pts, global_step=self.clock.step)
            if self.use_wandb and self.clock.step % 500 == 0:
                outputs_pts[0] = outputs_pts[0] + np.array([0, 1, 0]).reshape(1, -1)
                pts = np.concatenate([target_pts[0], outputs_pts[0]], axis=0)
                wandb.log({"input+AE_GAN_output": [wandb.Object3D(pts)], 'step': self.clock.step})

        try:
            input_pts = data['G'].ndata['x'].view(target_pts.shape[0], -1, 3).contiguous().cpu().numpy()
        except:
            input_pts  = self.latent_vect['G'].ndata['x'].view(target_pts.shape[0], -1, 3).contiguous().cpu().numpy() # B, N, 3
        ids  = data['id']
        idxs = data['idx']
        if self.config.pred_nocs:
            # save nocs predictions & visualization
            print('To be implemented!!!')
        else:
            BS, M = self.output_R.shape[0], self.config.num_modes_R #
            N     = input_pts.shape[1]
            # save degree error
            degree_err = self.degree_err.cpu().detach().numpy() # GT mode degree error, [B, N]
            if len(degree_err.shape) < 3:
                degree_err = degree_err[:, :, np.newaxis]
            save_path = f'{self.config.log_dir}/generation'
            if not exists(save_path):
                print('making directories', save_path)
                makedirs(save_path)
            for j in range(degree_err.shape[-1]): # modes
                save_arr  = np.concatenate([input_pts, degree_err[:, :, j:j+1]], axis=-1)
                for k in range(degree_err.shape[0]): # batch
                    save_name = f'{save_path}/{mode}_{self.clock.step}_{ids[k]}_{idxs[k]}_{j}.txt' #
                    # print('--saving to ', save_name)
                    np.savetxt(save_name, save_arr[k])
            # save per-point mode prediction
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

class PointAEAgent(BaseAgent):
    def build_net(self, config):
        # customize your build_net function
        net = get_network(config, "pointAE")
        print('-----pointAE architecture-----')
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

        if 'Graph' in self.config.dataset_class:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            pred_dict = self.net(data['G'].to(device))
        else:
            pred_dict = self.net(input_pts)
        self.output_pts  = pred_dict['S']
        # if 'R' in pred_dict:
        #     self.latent_vect = pred_dict['R']
        # if 'T' in pred_dict:
        #     self.output_T      = pred_dict['T']
        self.emd_loss = earth_mover_distance(self.output_pts, target_pts)
        self.emd_loss = torch.mean(self.emd_loss)

    def collect_loss(self):
        loss_dict = {"emd": self.emd_loss}
        return loss_dict

    def visualize_batch(self, data, mode, **kwargs):
        tb = self.train_tb if mode == 'train' else self.val_tb

        num = 2

        target_pts = data['points'][:num].transpose(1, 2).detach().cpu().numpy()
        outputs_pts = self.output_pts[:num].transpose(1, 2).detach().cpu().numpy()

        tb.add_mesh("gt", vertices=target_pts, global_step=self.clock.step)
        tb.add_mesh("output", vertices=outputs_pts, global_step=self.clock.step)
        if self.use_wandb and self.clock.step % 500 == 0:
            outputs_pts[0] = outputs_pts[0] + np.array([0, 1, 0]).reshape(1, -1)
            pts = np.concatenate([target_pts[0], outputs_pts[0]], axis=0)
            wandb.log({"input+AE_GAN_output": [wandb.Object3D(pts)], 'step': self.clock.step})

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
