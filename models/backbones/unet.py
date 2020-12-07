# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import math
import torch.nn.functional as F

import __init__
from common.debugger import breakpoint
import global_info

setting = global_info.setting()
if setting.USE_MULTI_GPU:
    BatchNorm2d = nn.SyncBatchNorm
else:
    BatchNorm2d = nn.BatchNorm2d

# BatchNorm2d = nn.BatchNorm2d
__all__ = ['UNet']

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=2,
        stack_factor=1,
        depth=5,
        wf=6,
        padding=False,
        batch_norm=False,
        up_mode='upconv',
        use_softmax=False,
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding      = padding
        self.depth        = depth
        self.stack_factor = stack_factor
        self.use_softmax  = use_softmax
        self.n_classes    = n_classes
        prev_channels     = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes * stack_factor, kernel_size=1) # 1*1 enough? only for

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x, return_feature_maps=False, return_feature_banks=False):
        blocks = []
        # feat_dec = []
        feat_dict = {}
        c_feat = [512, 256, 128, 64]
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            feat_dict[c_feat[i]] = x
            x = up(x, blocks[-i - 1])

        feat_dict[64] = x
        # feat_last = x
        x = self.last(x)
        # reshape to high-dim
        if self.stack_factor > 1:
            bs, _, w, h = x.size()
            x = x.view(bs, self.n_classes, self.stack_factor, w, h)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # x = nn.functional.log_softmax(x, dim=1)
        if return_feature_maps:
            if return_feature_banks:
                return x, blocks, feat_dict
            else:
                return x, feat_dict # feat_dict[64]
        else:
            return x


class UNetConBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConBlock(in_size, out_size, padding, batch_norm)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


if __name__ == '__main__':
    gpu = 0
    # test.cuda(gpu)
    deploy_device   = torch.device('cuda:{}'.format(gpu))
    x = torch.randn(2, 14, 480, 480, requires_grad=True, device=deploy_device)
    y = torch.empty(2, 480, 480, dtype=torch.long, device=deploy_device).random_(0, 20)

    model = UNet(in_channels=14, n_classes=20, depth=4, batch_norm=True, padding=True, up_mode='upconv')
    model.cuda(gpu)

    pred, blocks, feat_dec = model.forward(x, return_feature_maps=True, return_feature_banks=True)
    print(pred.shape)
    for i, feat_layer in enumerate(blocks):
        print('Encoder {} has shape: {}'.format(i, feat_layer.size()))
    for i, feat_layer in enumerate(feat_dec.items()):
        print('Decoder {} has shape: {}'.format(i, feat_layer[1].size()))
