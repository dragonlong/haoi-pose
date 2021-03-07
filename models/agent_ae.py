import torch
import torch.nn as nn
import numpy as np
import wandb
import __init__
from models.ae_gan import get_network
from models.base import BaseAgent
from utils.emd import earth_mover_distance
from models.losses import loss_geodesic, loss_vectors, compute_vect_loss, compute_1vN_nocs_loss, compute_miou_loss
from common.d3_utils import compute_rotation_matrix_from_euler, compute_euler_angles_from_rotation_matrices, compute_rotation_matrix_from_ortho6d
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
THRESHOLD_GOOD = 0.1 #
THRESHOLD_BAD  = 0.5 #
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
        M  = self.config.MODEL.num_channels_R
        if 'se3' in self.config.encoder_type:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            self.latent_vect = self.net.encoder(data['G'].to(device))

            if self.config.pred_nocs:
                self.output_N = self.net.regressor_nocs(self.latent_vect['N'].squeeze().view(target_R.shape[0], -1, self.latent_vect['N'].shape[1]).contiguous().permute(0, 2, 1).contiguous()) # assume to be B*N, 128? --> 3 channel
            else:
                target_R = target_R.permute(0, 2, 1).contiguous() # B, 3, 1

                if self.config.MODEL.num_channels_R > 1:
                    # [BS*N, C, 3] -> [Bs, N, C, 3] -> [Bs, 3, N, C]
                    BS, CS = target_R.shape[0], self.latent_vect['R'].shape[-2]
                    self.output_R  = self.latent_vect['R'].view(BS, -1, CS, 3).contiguous().permute(0, 3, 1, 2).contiguous()# BS*N, C, 3
                    self.output_R = self.output_R/(torch.norm(self.output_R, dim=1, keepdim=True) + epsilon)
                    self.output_R_raw = self.output_R
                    self.degree_err = torch.acos( torch.sum(self.output_R*target_R.unsqueeze(-1), dim=1)) * 180 / np.pi
                else:
                    self.output_R = self.latent_vect['R'].squeeze().view(target_R.shape[0], -1, 3).contiguous().permute(0, 2, 1).contiguous() # B, 3, N
                    self.output_R = self.output_R/(torch.norm(self.output_R, dim=1, keepdim=True) + epsilon)
                    self.degree_err = torch.acos( torch.sum(self.output_R*target_R, dim=1)) * 180 / np.pi

                # we have two ways to do the pooling
                self.output_R_pooled = self.output_R.mean(dim=2) # TODO
                # self.output_R_pooled = self.latent_vect['1'].permute(0, 2, 1).contiguous() # B, 3, 1
                self.output_R_pooled = self.output_R_pooled/(torch.norm(self.output_R_pooled, dim=1, keepdim=True) + epsilon)
                self.output_T = self.latent_vect['T'].permute(0, 2, 1).contiguous().squeeze().view(target_T.shape[0], -1, 3).contiguous().permute(0, 2, 1).contiguous()# [2048, 1, 3]
                target_T      = target_T.permute(0, 2, 1).contiguous()

            if self.config.pred_seg:
                self.output_C = self.net.classifier_seg(self.latent_vect['N'].squeeze().view(target_R.shape[0], -1, self.latent_vect['N'].shape[1]).contiguous().permute(0, 2, 1).contiguous())

            if self.config.pred_conf:
                self.output_Cf = self.net.regressor_confi(self.latent_vect['N'].squeeze().view(target_R.shape[0], -1, self.latent_vect['N'].shape[1]).contiguous().permute(0, 2, 1).contiguous())

            if self.config.pred_mode:
                self.output_M = self.net.classifier_mode(self.latent_vect['R0']).squeeze() # [BS * N, M, 1]-->[BS * N, M] --> [BS, M, N]
        else:
            # B, 3, N
            input_pts        = data['G'].ndata['x'].view(target_pts.shape[0], -1, 3).contiguous().permute(0, 2, 1).contiguous().cuda() #
            input_pts        = input_pts - input_pts.mean(dim=-1, keepdim=True)
            self.latent_vect = self.net.encoder(input_pts)
            if self.config.pred_nocs:
                self.output_N = self.latent_vect['N'] # assume to be B*N, 128? --> 3 channel
            else:
                target_R         = target_R.permute(0, 2, 1).contiguous() # B, 3, 1
                self.output_T    = self.latent_vect['T'] # 3, N, no activation
                target_T         = target_T.permute(0, 2, 1).contiguous()# 2, 3, 512
                if self.config.MODEL.num_channels_R > 1:
                    self.output_R  = self.latent_vect['R'].view(BS, self.config.MODEL.num_channels_R, 3, -1).contiguous().permute(0, 2, 3, 1).contiguous()# # [BS*N, C, 3] -> [Bs, N, C, 3] -> [Bs, 3, N, C]
                    self.output_R  = self.output_R/(torch.norm(self.output_R, dim=1, keepdim=True) + epsilon)
                    self.output_R_raw = self.output_R
                    self.degree_err = torch.acos( torch.sum(self.output_R*target_R.unsqueeze(-1), dim=1)) * 180 / np.pi
                else:
                    self.output_R = self.latent_vect['R'] # B, 3, N
                    self.output_R = self.output_R/(torch.norm(self.output_R, dim=1, keepdim=True) + epsilon)
                    self.degree_err = torch.acos( torch.sum(self.output_R*target_R, dim=1)) * 180 / np.pi

                # we have two ways to do the pooling
                self.output_R_pooled = self.output_R.mean(dim=2) # TODO
                self.output_R_pooled = self.output_R_pooled/(torch.norm(self.output_R_pooled, dim=1, keepdim=True) + epsilon)
            if self.config.pred_seg:
                self.output_C = self.latent_vect['C']
            if self.config.pred_mode:
                self.output_M = self.latent_vect['M'].permute(0, 2, 1).contiguous().view(-1, M).contiguous() # BS, 2, N

        #>>>>>>>>>>>>>>>>>> computing loss
        if self.config.pred_seg:
            self.seg_loss = compute_miou_loss(self.output_C, target_C).mean()

        if self.config.pred_nocs:
            self.nocs_loss= compute_1vN_nocs_loss(self.output_N, target_pts, target_category=self.config.target_category, num_parts=1, confidence=target_C)
            self.nocs_loss= self.nocs_loss.mean()
        else:
            confidence = target_C
            # if self.output_R.shape[-1] == 3:
            #     self.regressionR_loss = loss_geodesic(compute_rotation_matrix_from_ortho6d( self.output_R[:, :, :2].permute(0, 2, 1).contiguous().reshape(target_R.shape[0], -1).contiguous() ), target_R) #
            #     self.regressionR_loss = self.regressionR_loss.mean()
            if self.config.check_consistency: # on all points
                mean_R = torch.mean(self.output_R, dim=2, keepdim=True)
                variance = torch.norm(self.output_R - mean_R, dim=1).mean()
                self.consistency_loss = variance * self.config.consistency_loss_multiplier

            if self.config.MODEL.num_channels_R > 1: # apply to all
                if self.config.rotation_use_dense:
                    regressionR_loss = compute_vect_loss(self.output_R, target_R.unsqueeze(-1))
                else:
                    regressionR_loss = compute_vect_loss(self.output_R_pooled.unsqueeze(2), target_R.unsqueeze(2)) # [2, 3, 2], [2, 3, 1]
                min_loss, min_indices = torch.min(regressionR_loss, dim=-1)
                self.regressionR_loss = min_loss
                self.output_R   =  self.output_R[torch.arange(target_R.shape[0]), :, :, min_indices[:]]
                self.degree_err =  self.degree_err[torch.arange(target_R.shape[0]), :, min_indices[:]] # B, N, C
            else: # when only one mode, we have confidence for hand points
                if self.config.rotation_use_dense:
                    self.regressionR_loss = compute_vect_loss(self.output_R, target_R, confidence=confidence) # B
                else:
                    self.regressionR_loss = compute_vect_loss(self.output_R_pooled.unsqueeze(-1), target_R) # B, 3, N

            self.regressionR_loss = self.regressionR_loss.mean()

            if target_T is not None and self.config.use_objective_T:
                self.regressionT_loss = compute_vect_loss(self.output_T, target_T, confidence=confidence).mean() # TYPE_LOSS='SOFT_L1'

            dis = torch.norm(self.output_R - target_R, dim=1) # B, N
            correctness_mask = torch.zeros((dis.shape[0], dis.shape[1]), device=self.output_R.device)
            correctness_mask[dis<THRESHOLD_GOOD] = 1.0
            # good_pred_ratio = torch.sum(correctness_mask * confidence, dim=1) / (torch.sum(confidence, dim=1) + 1)
            good_pred_ratio = torch.mean(correctness_mask, dim=1) # TODO
            self.infos['good_pred_ratio'] = good_pred_ratio.mean() # scalar
            self.infos['consistency']     = self.consistency_loss

            if self.config.pred_mode and self.config.MODEL.num_channels_R > 1:
                self.target_M = min_indices.unsqueeze(1).repeat(1, self.output_R.shape[-1]).contiguous().view(-1).contiguous()
                self.classifyM_loss = compute_miou_loss(self.output_M, min_indices.unsqueeze(1).repeat(1, self.output_R.shape[-1]).contiguous().view(-1).contiguous(), loss_type='xentropy')

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

        save_pc = True
        degree_err = self.degree_err.cpu().detach().numpy()# B, N
        if len(degree_err.shape) < 3:
            degree_err = degree_err[:, :, np.newaxis]
        try:
            input_pts = data['G'].ndata['x'].view(target_pts.shape[0], -1, 3).contiguous().cpu().numpy()
        except:
            input_pts  = self.latent_vect['G'].ndata['x'].view(target_pts.shape[0], -1, 3).contiguous().cpu().numpy() # B, N, 3
        ids = data['id']
        idxs = data['idx']
        if save_pc:
            save_path = f'{self.config.log_dir}/generation'
            if not exists(save_path):
                print('making directories', save_path)
                makedirs(save_path)
            for j in range(degree_err.shape[-1]): # modals
                save_arr  = np.concatenate([input_pts, degree_err[:, :, j:j+1]], axis=-1)
                for k in range(degree_err.shape[0]): # batch
                    save_name = f'{save_path}/{mode}_{self.clock.step}_{ids[k]}_{idxs[k]}_{j}.txt' #
                    # print('--saving to ', save_name)
                    np.savetxt(save_name, save_arr[k])
        save_mode = True
        if save_mode and self.config.pred_mode:
            BS, NC = self.output_R.shape[0], self.config.MODEL.num_channels_R

            output_M = self.output_M.cpu().detach().numpy().reshape(BS, -1, NC)
            save_path = f'{self.config.log_dir}/mode'
            if not exists(save_path):
                print('making directories', save_path)
                makedirs(save_path)

            for j in range(degree_err.shape[-1]): # modals
                save_arr  = np.concatenate([input_pts, output_M[:, :, j:j+1]], axis=-1)
                for k in range(output_M.shape[0]): # batch
                    save_name = f'{save_path}/{mode}_{self.clock.step}_{ids[k]}_{idxs[k]}_{j}.txt' #
                    # print('--saving to ', save_name)
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
