import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)

from point_transformer_modules import PointTransformerResBlock, PointTransformerDownBlock, PointTransformerUpBlock, MLP


class PointTransformer(nn.Module):
    """
    Output: 128 channels
    """
    def __init__(self, latent_channel=128, num_channels_R=2):
        super(PointTransformer, self).__init__()
        cfg = {
            "channel_mult": 4,
            "div": 4,
            "pos_mlp_hidden_dim": 64,
            "attn_mlp_hidden_mult": 4,
            "pre_module": {
                "channel": 16,
                "nsample": 16
            },
            "down_module": {
                "npoint": [256, 64, 32, 16],
                "nsample": [10, 16, 16, 16],
                "attn_channel": [16, 32, 64, 64],
                "attn_num": [2, 2, 2, 2]
            },
            "up_module": {
                "attn_num": [1, 1, 1, 1]
            },
            "heads": {
                "N": [latent_channel],
                "R": [num_channels_R * 3],
                "T": [3]
            }
        }
        k = cfg['channel_mult']
        div = cfg["div"]
        pos_mlp_hidden_dim = cfg["pos_mlp_hidden_dim"]
        attn_mlp_hidden_mult = cfg["attn_mlp_hidden_mult"]
        pre_module_channel = cfg["pre_module"]["channel"]
        pre_module_nsample = cfg["pre_module"]["nsample"]
        self.pre_module = nn.ModuleList([
            MLP(dim=1, in_channel=3, mlp=[pre_module_channel * k] * 2, use_bn=True, skip_last=False),
            PointTransformerResBlock(dim=pre_module_channel * k,
                                     div=div, pos_mlp_hidden_dim=pos_mlp_hidden_dim,
                                     attn_mlp_hidden_mult=attn_mlp_hidden_mult,
                                     num_neighbors=pre_module_nsample)
        ])
        self.down_module = nn.ModuleList()
        down_cfg = cfg["down_module"]

        last_channel = pre_module_channel
        attn_channel = down_cfg['attn_channel']
        down_sample = down_cfg['nsample']
        for i in range(len(attn_channel)):
            out_channel = attn_channel[i]
            self.down_module.append(PointTransformerDownBlock(npoint=down_cfg['npoint'][i],
                                                              nsample=down_sample[i],
                                                              in_channel=last_channel * k,
                                                              out_channel=out_channel * k,
                                                              num_attn=down_cfg['attn_num'][i],
                                                              div=div, pos_mlp_hidden_dim=pos_mlp_hidden_dim,
                                                              attn_mlp_hidden_mult=attn_mlp_hidden_mult))
            last_channel = out_channel
        up_channel = attn_channel[::-1] + [pre_module_channel]
        up_sample = down_sample[::-1]
        self.up_module = nn.ModuleList()
        up_cfg = cfg["up_module"]
        up_attn_num = up_cfg['attn_num']
        for i in range(len(attn_channel)):
            self.up_module.append(PointTransformerUpBlock(up_sample[i], up_channel[i] * k, up_channel[i + 1] * k, up_attn_num[i],
                                                          div=div, pos_mlp_hidden_dim=pos_mlp_hidden_dim,
                                                          attn_mlp_hidden_mult=attn_mlp_hidden_mult))

        self.heads = nn.ModuleDict()
        head_cfg = cfg['heads']
        for key, mlp in head_cfg.items():
            self.heads[key] = MLP(dim=1, in_channel=pre_module_channel * k, mlp=mlp, use_bn=True, skip_last=True)

    def forward(self, xyz):
        xyz_list, points_list = [], []
        points = self.pre_module[0](xyz)
        points = self.pre_module[1](xyz, points)
        xyz_list.append(xyz)
        points_list.append(points)

        for down in self.down_module:
            xyz, points = down(xyz, points)
            xyz_list.append(xyz)
            points_list.append(points)

        for i, up in enumerate(self.up_module):
            points = up(xyz_list[- (i + 1)], xyz_list[- (i + 2)], points, points_list[- (i + 2)])

        output = {}
        for key, head in self.heads.items():
            output[key] = head(points)

        return output


if __name__ == '__main__':
    model = PointTransformer()

    input = torch.randn((1, 3, 512))

    output = model(input)
    for key, value in output.items():
        print(key, value.shape)
