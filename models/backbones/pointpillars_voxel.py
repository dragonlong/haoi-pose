"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import torch
from torch import nn
from torch.nn import functional as F
# 
import __init__
import global_info
from torchplus.tools import change_default_args
from torch_scatter import scatter

setting = global_info.setting()
if setting.USE_MULTI_GPU:
    BatchNorm2d = nn.SyncBatchNorm
    BatchNorm1d = nn.SyncBatchNorm
else:
    BatchNorm2d = nn.BatchNorm2d
    BatchNorm1d = nn.BatchNorm1d

def breakpoint():
    import pdb; pdb.set_trace()

def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]

    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 height = 480, 
                 width = 480,
                 depth = 2,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        self.h = height
        self.w = width
        self.z = depth

        if use_norm:
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            Linear = change_default_args(bias=True)(nn.Linear)

        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs, indices):

        """
        inputs: [bs*seq, N, 9]
        coords: [bs*seq, N]
        """
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)

        # breakpoint()
        bs, n, c = x.size()
        x_max = torch.zeros(bs, self.h * self.w * self.z, c, device=x.device)
        x_max = scatter(x, indices, dim=1, out=x_max, reduce="max")
        # breakpoint()
        if self.last_vfe:
            # breakpoint()
            x_max = x_max.permute(0, 2, 1).contiguous().view(bs, c, self.h, self.w, self.z)
            x_max = x_max.permute(0, 4, 1, 2, 3).contiguous().view(bs, self.z*c, self.h, self.w)
            return x.permute(0, 2, 1).contiguous(), x_max
        else:
            x_repeat = torch.gather(x_max, 1, indices[:, :].unsqueeze(2).repeat(1, 1, self.units))
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 height=480, 
                 width=480,
                 depth=2,
                 with_distance=False):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNet'
        assert len(num_filters) > 0
        num_input_features += 0 # TODO
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        self.height = height
        self.width  = width
        self.depth  = depth

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, height=height, width=width, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

    def forward(self, features, coords):
        """
        features: [batch_size, seq, N, 9]
        coords:   [batch_size, seq, N, 2]
        """
        # Forward pass through PFNLayers
        bs, seq_len, n, c1 = features.size()
        bs, seq_len, n, c2 = coords.size()
        features = features.view(-1, n, c1).contiguous()
        coords  = coords.view(-1, n, c2).contiguous()
        indices = coords[:, :, 0] * self.width * self.depth + coords[:, :, 1] * self.depth + coords[:, :, 2]
        for pfn in self.pfn_layers:
            features = pfn(features, indices)

        return features[0], features[1].squeeze()


class PointPillar(nn.Module):
    def __init__(self, vfe_class_name, 
                num_input_features=4, 
                vfe_num_filters=[32, 128],
                use_norm=True, 
                with_distance=False,
                h=480, w=480):
        super().__init__()
        vfe_class_dict = {
            # "VoxelFeatureExtractor": VoxelFeatureExtractor,
            # "VoxelFeatureExtractorV2": VoxelFeatureExtractorV2,
            "PillarFeatureNet": PillarFeatureNet
        }
        vfe_class = vfe_class_dict[vfe_class_name]
        if vfe_class_name == "PillarFeatureNet":
            # breakpoint()
            self.voxel_feature_extractor = vfe_class(num_input_features, use_norm, num_filters=vfe_num_filters, \
                with_distance=with_distance, height=h, width=w,
            )

    def forward(self, x, coords):
        voxel_features = self.voxel_feature_extractor(x, coords)

        return voxel_features


if __name__ == '__main__':
    gpu = 0
    # test.cuda(gpu)
    deploy_device   = torch.device('cuda:{}'.format(gpu))
    N = 1500
    BS = 2
    seq = 3
    k  = 20
    x     = torch.rand(BS, seq, N, 9, requires_grad=True, device=deploy_device)
    coords= torch.empty(BS, seq, N, 3, dtype=torch.long, device=deploy_device).random_(0, 3)
    model = PointPillar('PillarFeatureNet').cuda(gpu)

    features_pesudo = model(x, coords)
    print(features_pesudo[0].size(), features_pesudo[1].size())
    