import os
import torch
import torch.optim as optim
import torch.nn as nn
from abc import abstractmethod
from tensorboardX import SummaryWriter
from common.train_utils import TrainClock
from utils.extensions.chamfer_dist import ChamferDistance
from utils.p2i_utils import ComputeDepthMaps
import vgtk
from vgtk.loss import CrossEntropyLoss
import vgtk.so3conv.functional as L
import wandb
def bp():
    import pdb;pdb.set_trace()

class BaseAgent(object):
    """Base trainer that provides common training behavior.
        All customized trainer should be subclass of this class.
    """
    def __init__(self, config):
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.clock      = TrainClock()
        self.batch_size = config.batch_size
        self.use_wandb  = config.use_wandb
        # build network
        self.net = self.build_net(config)
        self.config = config

        # set loss function
        self.set_loss_function()

        # set optimizer
        self.set_optimizer(config)

        # set lr scheduler
        self.set_scheduler(config)

        # set tensorboard writer
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'tb/train'))
        self.val_tb   = SummaryWriter(os.path.join(self.log_dir, 'tb/val'))

        # setup render
        self.render = ComputeDepthMaps(projection="perspective", eyepos_scale=1, image_size=256).float()

        self.is_testing = False  # e.g. use gt labels to choose during training, use predicted labels during testing

    @abstractmethod
    def build_net(self, config):
        raise NotImplementedError

    def set_loss_function(self):
        """set loss function used in training"""
        self.criterion = nn.MSELoss().cuda()
        if 'completion' in self.config.task:
            self.chamfer_dist = ChamferDistance()
        if 'ssl' in self.config.task or 'so3' in self.config.encoder_type:
            self.classifier = CrossEntropyLoss()
            self.anchors = torch.from_numpy(L.get_anchors(self.config.model.kanchor)).cuda()
        if self.config.pred_t:
            self.render_loss = torch.nn.L1Loss()

    def _setup_metric(self):
        # regressor + classifier
        anchors = torch.from_numpy(L.get_anchors(self.config.model.kanchor)).cuda()
        if self.config.model.representation == 'quat':
            out_channel = 4
        elif self.config.model.representation == 'ortho6d':
            out_channel = 6
        else:
            raise KeyError("Unrecognized representation of rotation: %s"%self.config.model.representation)
        print('---setting up metric!!!')
        self.metric = vgtk.MultiTaskDetectionLoss(anchors, nr=out_channel)

    @abstractmethod
    def collect_loss(self):
        """collect all losses into a dict"""
        raise NotImplementedError

    def set_optimizer(self, config):
        """set optimizer used in training"""
        self.base_lr = config.lr
        if config.TRAIN.train_batch == 1:
            self.optimizer = optim.SGD(
                self.net.parameters(),
                lr=config.lr,
                momentum=0.9,
                weight_decay=1e-4)
        else:
            self.optimizer = optim.Adam(self.net.parameters(), config.lr)

    def set_scheduler(self, config):
        """set lr scheduler used in training"""
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.lr_decay)

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.clock.epoch))
            print("Saving checkpoint epoch {}...".format(self.clock.epoch))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

        if isinstance(self.net, nn.DataParallel):
            model_state_dict = self.net.module.cpu().state_dict()
        else:
            model_state_dict = self.net.cpu().state_dict()

        torch.save({
            'clock': self.clock.make_checkpoint(),
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, save_path)

        self.net.cuda()

    def load_ckpt(self, name=None):
        """load checkpoint from saved checkpoint"""
        """load checkpoint from saved checkpoint"""
        if name == 'latest':
            pass
        elif name == 'best':
            pass
        else:
            name = "ckpt_epoch{}".format(name)
        load_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading checkpoint from {} ...".format(load_path))
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.clock.restore_checkpoint(checkpoint['clock'])

    @abstractmethod
    def forward(self, data):
        """forward logic for your network"""
        raise NotImplementedError

    def update_network(self, loss_dict):
        """update network by back propagation"""
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        """record and update learning rate"""
        self.train_tb.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.clock.epoch)
        if not self.optimizer.param_groups[-1]['lr'] < self.base_lr / 5.0:
            self.scheduler.step(self.clock.epoch)

    def record_losses(self, loss_dict, mode='train', infos_dict=None):
        """record loss to tensorboard"""
        losses_values = {k: v.item() for k, v in loss_dict.items()}

        tb = self.train_tb if mode == 'train' else self.val_tb
        for k, v in losses_values.items():
            tb.add_scalar(k, v, self.clock.step)
            if self.use_wandb:
                wandb.log({f'{mode}/{k}': v, 'step': self.clock.step})
        if infos_dict is not None:
            for k, v in infos_dict.items():
                tb.add_scalar(k, v, self.clock.step)
                if self.use_wandb:
                    wandb.log({f'{mode}/{k}': v, 'step': self.clock.step})
    def train_func(self, data):
        """one step of training"""
        self.net.train()
        self.is_testing = False

        self.forward(data)
        losses = self.collect_loss()
        infos  = self.collect_info()
        self.update_network(losses)
        self.record_losses(losses, 'train', infos_dict=infos)
        return losses, infos

    def val_func(self, data):
        """one step of validation"""
        self.net.eval()
        self.is_testing = True
        with torch.no_grad():
            self.forward(data)
        losses = self.collect_loss()
        infos  = self.collect_info()
        self.record_losses(losses, 'validation', infos_dict=infos)
        return losses, infos

    def visualize_batch(self, data, tb, **kwargs):
        """write visualization results to tensorboard writer"""
        raise NotImplementedError


class GANzEAgent(object):
    """Base GAN-zE trainer that provides common training behavior.
        All customized trainer should be subclass of this class.
    """
    def __init__(self, config):
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.clock = TrainClock()
        self.batch_size = config.batch_size
        self.use_wandb  = config.use_wandb
        self.z_dim = config.z_dim

        # build network
        self.build_net(config)
        self.config = config
        # set loss function
        self.set_loss_function()

        # set optimizer
        self.set_optimizer(config)

        # set tensorboard writer
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_tb = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

    @abstractmethod
    def build_net(self, config):
        """build network for netG, netD, netE"""
        raise NotImplementedError

    def set_loss_function(self):
        """set loss function used in training"""
        self.criterionGAN = nn.MSELoss().cuda()  # LSGAN
        self.criterionL1 = nn.L1Loss().cuda()

    @abstractmethod
    def collect_loss(self):
        """collect all losses into a dict"""
        raise NotImplementedError

    def set_optimizer(self, config):
        """set optimizer and lr scheduler used in training"""
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=config.lr, betas=(config.beta1_gan, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=config.lr, betas=(config.beta1_gan, 0.999))
        self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=config.lr, betas=(config.beta1_gan, 0.999))

    @abstractmethod
    def forward(self, data):
        """forward pass to set data"""
        raise NotImplementedError

    def update_G_and_E(self):
        """forward and backward pass for netG, netE"""
        raise NotImplementedError

    def update_D(self):
        """forward and backward pass for netD"""
        raise NotImplementedError

    @abstractmethod
    def optimize_network(self):
        """one step of optimization"""
        raise NotImplementedError

    def train_func(self, data):
        """one step of training"""
        self.forward(data)
        self.optimize_network()

        loss_dict = self.collect_loss()
        self.record_losses(loss_dict, "train")

    def val_func(self, data):
        """one step of validation"""
        with torch.no_grad():
            self.forward(data)

    def update_learning_rate(self):
        """record and update learning rate"""
        pass

    def eval(self):
        """set G, D, E to eval mode"""
        self.netG.eval()
        self.netD.eval()
        self.netE.eval()

    def get_random_noise(self, batch_size):
        """sample random z from gaussian"""
        z = torch.randn((batch_size, self.z_dim)).cuda()
        return z

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.clock.epoch))
            print("Saving checkpoint epoch {}...".format(self.clock.epoch))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

        torch.save({
            'clock': self.clock.make_checkpoint(),
            'netG_state_dict': self.netG.cpu().state_dict(),
            'netD_state_dict': self.netD.cpu().state_dict(),
            'netE_state_dict': self.netE.cpu().state_dict(),
            'optimizerG_state_dict': self.optimizer_G.state_dict(),
            'optimizerD_state_dict': self.optimizer_D.state_dict(),
            'optimizerE_state_dict': self.optimizer_E.state_dict(),
        }, save_path)

        self.netG.cuda()
        self.netD.cuda()
        self.netE.cuda()

    def load_ckpt(self, name=None):
        """load checkpoint from saved checkpoint"""
        if name == 'latest':
            pass
        elif name == 'best':
            pass
        else:
            name = "ckpt_epoch{}".format(name)
        load_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        checkpoint = torch.load(load_path)
        print("Loading checkpoint from {} ...".format(load_path))
        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        self.netD.load_state_dict(checkpoint['netD_state_dict'])
        self.netE.load_state_dict(checkpoint['netE_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizerD_state_dict'])
        self.optimizer_E.load_state_dict(checkpoint['optimizerE_state_dict'])
        self.clock.restore_checkpoint(checkpoint['clock'])

    def record_losses(self, loss_dict, mode='train', infos_dict=None):
        """record loss to tensorboard"""
        losses_values = {k: v.item() for k, v in loss_dict.items()}
        tb = self.train_tb if mode == 'train' else self.val_tb
        for k, v in losses_values.items():
            tb.add_scalar(k, v, self.clock.step)
            if self.use_wandb and self.clock.step % 10:
                wandb.log({f'{mode}/{k}': v, 'step': self.clock.step})
        if infos_dict is not None:
            for k, v in infos_dict.items():
                tb.add_scalar(k, v, self.clock.step)
                if self.use_wandb:
                    wandb.log({f'{mode}/{k}': v, 'step': self.clock.step})
    def visualize_batch(self, data, tb, **kwargs):
        """write visualization results to tensorboard writer"""
        raise NotImplementedError
