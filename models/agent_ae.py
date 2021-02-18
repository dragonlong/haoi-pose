import torch
import torch.nn as nn
import numpy as np
import wandb
import __init__
from models.ae_gan import get_network
from models.base import BaseAgent
from utils.emd import earth_mover_distance
from models.losses import loss_geodesic, loss_vectors, compute_vect_loss
from common.d3_utils import compute_rotation_matrix_from_euler, compute_euler_angles_from_rotation_matrices, compute_rotation_matrix_from_ortho6d
#
def bp():
    import pdb;pdb.set_trace()

epsilon=1e-10
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

    def forward(self, data):
        input_pts = data['points'].cuda()
        target_pts = input_pts.clone().detach()
        target_R   = data['R'].cuda()
        if 'T' in data:
            target_T   = data['T'].cuda()
        else:
            target_T = None
        if 'Graph' in self.config.dataset_class:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            self.latent_vect = self.net.encoder(data['G'].to(device))
        else:
            self.latent_vect = self.net.encoder(input_pts)

        if 'pose' in self.config.task:
            if self.config.rotation_use_dense:
                self.output_R = self.latent_vect['R'].squeeze().view(target_R.shape[0], -1, 3).contiguous() # B, N, 3
                target_R = target_R.repeat(1, self.config.num_points, 1).contiguous()  # B, N, 3
            else:
                self.output_R = self.net.regressor(self.latent_vect['1'])

            if self.output_R.shape[-2] == 3:
                self.regressionR_loss = loss_geodesic(compute_rotation_matrix_from_ortho6d( self.output_R[:, :, :2].permute(0, 2, 1).contiguous().reshape(target_R.shape[0], -1).contiguous() ), target_R) #
                print('Pose error: ', self.regressionR_loss)
                self.regressionR_loss = self.regressionR_loss.mean()
            else:
                self.output_R = self.output_R/(torch.norm(self.output_R, dim=-1, keepdim=True) + epsilon) # B, 1, 3
                if self.config.rotation_loss_type == 0: # error in degrees
                    self.regressionR_loss = loss_vectors(self.output_R, target_R)
                elif self.config.rotation_loss_type == 1:
                    self.regressionR_loss = compute_vect_loss(self.output_R.permute(0, 2, 1).contiguous(), target_R.permute(0, 2, 1).contiguous()) #
                    self.regressionR_loss = self.regressionR_loss.mean()

            self.output_T = self.latent_vect['T'].permute(0, 2, 1).contiguous().squeeze() # [2048, 1, 3]
            if target_T is not None:
                self.regressionT_loss = compute_vect_loss(self.output_T.unsqueeze(-1), target_T.view(-1, 3).contiguous().unsqueeze(-1)).mean() # TYPE_LOSS='SOFT_L1'

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
            loss_dict["regressionR"]= self.regressionR_loss
            loss_dict["regressionT"]= self.regressionT_loss
        if 'completion' in self.config.task:
            loss_dict["emd"]=self.emd_loss
        return loss_dict

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
        if 'R' in pred_dict:
            self.latent_vect = pred_dict['R']
        if 'T' in pred_dict:
            self.output_T      = pred_dict['T']
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
