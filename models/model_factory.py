import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import __init__ as booger
from models.backbones import resnet, resnext, mobilenet, unet, pointnet
from models.decoders.flownet3d import FlowNet3D
from models.decoders.meteornet import MeteorNet
from models.decoders.deeppcn import DPCN
from models.decoders.pointnet2seq import PointMotionBaseModel
from models.decoders.pointnet2 import PointBaseModel
from models.decoders.pointnet_2 import PointNet2Segmenter

from common.debugger import bp
import global_info

setting = global_info.setting()
if setting.USE_MULTI_GPU:
    BatchNorm2d = nn.SyncBatchNorm
    BatchNorm1d = nn.SyncBatchNorm
else:
    BatchNorm2d = nn.BatchNorm2d
    BatchNorm1d = nn.BatchNorm1d

def eval_torch_func(key):
    if key == 'sigmoid':
        return nn.Sigmoid()
    elif key == 'tanh':
        return nn.Tanh()
    elif key == 'softmax':
        return nn.Softmax(1)
    else:
        return NotImplementedError

class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and classname.find('Set')== -1 and classname.find('Pc')== -1 and classname.find('Conv2d')== -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1 and classname.find('BatchNorm2d') ==-1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        elif classname.find('Linear') != -1:
           m.weight.data.normal_(0.0, 0.0001)

    @staticmethod
    def build_encoder(params, arch=None, fc_dim=None, weights=None):
        if params is not None:
            arch  = params.arch_encoder
            fc_dim= params.fc_dim
            weights= params.weights_encoder
        pretrained = False if len(weights) == 0 else True
        arch = arch.lower()
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)
        elif arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34dilated':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        elif arch == 'identical':
            net_encoder = None
        elif arch == 'pointnet':
            net_encoder = pointnet.PointNetEncoder()
        else:
            raise Exception('Architecture undefined!')

        if net_encoder:
            net_encoder.apply(ModelBuilder.weights_init)
            if len(weights) > 0:
                print('Loading weights for net_encoder')
                net_encoder.load_state_dict(
                    torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(params, options=None, arch=None, fc_dim=None, num_class=None, in_channels=None, stack_factor=None, depth=None, up_mode=None, weights=None, nframe=None, batch_norm=True, padding=True, use_softmax=False):
        # directly corresponds to cfg.MODEL
        if params is not None:
            arch      = params.arch_decoder
            fc_dim    = params.fc_dim
            num_class = params.num_classes
            if params.task == 'motion':
                num_class = 3
            # weights   = params.weights_decoder
            in_channels=params.in_channels
            stack_factor=params.stack_factor
            depth     = params.depth_decoder
            nframe    = params.n_frames
        arch = arch.lower()
        print('!!! Decoder using arch: ', arch)
        if arch == 'unet':
            net_decoder = unet.UNet(in_channels=in_channels, n_classes=num_class, stack_factor=stack_factor, depth=depth, batch_norm=batch_norm, padding=padding, up_mode=up_mode, use_softmax=use_softmax)
        elif arch == 'flownet3d':
            net_decoder = FlowNet3D()
        elif arch == 'meteornet':
            net_decoder = MeteorNet(nframe, radius_list=params.radius, num_class=num_class, knn=params.knn)
        elif arch == 'kaolin':
            net_decoder = PointNet2Segmenter(num_classes=params.num_classes, use_random_ball_query=False) # TODO
        elif arch == 'pointnet2':
            print(f'net_decoder has {num_class} classes')
            net_decoder = PointMotionBaseModel(options, 'pointnet2_charlesmsg', num_classes=num_class)
        elif arch == 'pointnet':
            net_decoder = pointnet.PointNetDecoder(k=num_class)
        elif arch == 'mlp':
            net_decoder = SimpleMLP(k=num_class)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0 and net_decoder:
            print(f'Loading weights {weights} for net_decoder {arch}')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder

    @staticmethod
    def build_header(layer_specs, weights=''):
        head = nn.ModuleList()
        names= []
        print('---head specs: \n ', layer_specs)
        for key, out_channels in layer_specs.items():
            names.append(key)
            layers = []#
            for i in range(1, len(out_channels)-1):
                if 'regression' in key:
                    layers += [nn.Linear(out_channels[i - 1], out_channels[i]), nn.BatchNorm1d(out_channels[i], eps=0.001), nn.ReLU()]
                else:
                    # layers += [nn.Conv1d(out_channels[i - 1], out_channels[i], 1, bias=True), nn.BatchNorm1d(out_channels[i], eps=0.001), nn.ReLU()]
                    layers += [nn.Conv1d(out_channels[i - 1], out_channels[i], 1, bias=True)]
            # hand last layer
            if 'regression' in key:
                layers += [nn.Linear(out_channels[-2], out_channels[-1])]
            if out_channels[-1] in ['sigmoid', 'tanh', 'relu', 'softmax']:
                layers +=[eval_torch_func(out_channels[-1])]
            head.append(nn.Sequential(*layers))
        head.apply(ModelBuilder.weights_init)
        if len(weights) > 0 and head:
            print(f'Loading weights {weights} for head with {len(layer_specs.keys())} branches')
            head.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return head, names


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


if __name__ == '__main__':
    gpu = 0
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch='identical',
        fc_dim=2048,
        weights='')
    net_decoder = ModelBuilder.build_decoder(
        arch='attent_pbr1',
        fc_dim=2048,
        num_class=20,
        weights='')
    print(net_decoder.parameters)
    pre_conv = nn.Conv2d(64, 64, 3, padding=1)
    pre_conv.cuda(gpu)
    net_decoder.cuda(gpu)

    # loss function
    crit = nn.NLLLoss(ignore_index=-1)

    BS, N, k  = 2, 150000, 20
    deploy_device   = torch.device('cuda:{}'.format(gpu))
    p_feat= torch.rand(BS, 128, N, requires_grad=True, device=deploy_device)
    v_feat= torch.rand(BS, 64, 480, 480, requires_grad=True, device=deploy_device)
    v_feat= pre_conv(v_feat)
    r_feat= torch.rand(BS, 20, 64, 2048, requires_grad=True, device=deploy_device)
    y_v     = torch.empty(BS, 480, 480, dtype=torch.long, device=deploy_device).random_(0, k)
    y_r     = torch.empty(BS, 64, 480, dtype=torch.long, device=deploy_device).random_(0, k)
    y_p     = torch.empty(BS, N, dtype=torch.long, device=deploy_device).random_(-1, k)

    v2p_ind = torch.empty(BS, N, 2, dtype=torch.long, device=deploy_device).random_(0, 470)
    r2p_ind = torch.empty(BS, N, 2, dtype=torch.long, device=deploy_device).random_(0, 64)
    #
    p_pred, v_pred, r_pred = net_decoder([p_feat, v_feat, r_feat], v2p_ind, r2p_ind)
    print('prediction: ', p_pred.size(), v_pred.size(), r_pred.size())

    loss = crit(v_pred, y_v)
    loss.backward()
    # print(v_feat.grad)
    # print(pre_conv.weight.grad)
