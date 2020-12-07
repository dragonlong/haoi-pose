import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time
import hydra
from omegaconf import DictConfig, ListConfig
from kaolin.models.PointNet2 import group_gather_by_index
from kaolin.models.PointNet2 import three_nn
from kaolin.models.PointNet2 import three_interpolate

import __init__
from modules.layers import Identity
from modules.blocks import GlobalDenseBaseModule, PointNetMSGDown, DenseFPModule, is_list

SPECIAL_NAMES = ["radius", "max_num_neighbors", "block_names"]

def breakpoint():
    import pdb;pdb.set_trace()
class PointBaseModel(nn.Module):
    """Create a Unet unwrapped generator
       referring to https://github.com/nicolas-chaulet/torch-points3d/blob/master/torch_points3d
    """
    def __init__(self,
        opt,
        model_type,
        in_features=1,
        num_classes=3,
        batchnorm=True,
        use_xyz_feature=True,
        motion_vector=False,
        use_random_ball_query=False):
        """Construct a Unet unwrapped generator

        The layers will be appended within lists with the following names
        * down_modules : Contains all the down module
        * inner_modules : Contain one or more inner modules
        * up_modules: Contains all the up module

        Parameters:
            opt - options for the network generation
            model_type - type of the model to be generated
            num_class - output of the network
            modules_lib - all modules that can be used in the UNet

        opt is expected to contains the following keys:
        * down_conv
        * up_conv
        * OPTIONAL: innermost
        """
        super(PointBaseModel, self).__init__()
        # detect which options format has been used to define the model
        self._spatial_ops_dict = {"neighbour_finder": [], "sampler": [], "upsample_op": []}

        self.down_modules   = nn.ModuleList()
        self.inner_modules  = nn.ModuleList()
        self.up_modules     = nn.ModuleList()

        # inner most modules
        contains_global = hasattr(opt, "innermost") and opt.innermost is not None
        if contains_global:
            args_innermost = opt.innermost
            if is_list(args_innermost):
                layers = []
                for inner_opt in args_innermost:
                    inner_layer = GlobalDenseBaseModule(**inner_opt)
                    layers.append(inner_layer)
                self.inner_modules.append(nn.Sequential(*layers))
            else:
                self.inner_modules.append(GlobalDenseBaseModule(**args_innermost))
        else:
            self.inner_modules.append(Identity())

        # Down modules
        for i in range(len(opt.down_conv.down_conv_nn)):
            args = self._fetch_arguments(opt.down_conv, i, "DOWN")
            down_module = PointNetMSGDown(**args)
            self.down_modules.append(down_module)

        # Up modules
        for i in range(len(opt.up_conv.up_conv_nn)):
            args = self._fetch_arguments(opt.up_conv, i, "UP")
            up_module = DenseFPModule(**args)
            self.up_modules.append(up_module)

        final_layer_modules = [
            module for module in [
                nn.Conv1d(128, 128, 1),
                nn.BatchNorm1d(128) if batchnorm else None,
                nn.ReLU(),
                nn.Dropout(0.5),
            ] if module is not None
        ]
        self.final_layers = nn.Sequential(*final_layer_modules)

    def _get_from_kwargs(self, kwargs, name):
        module = kwargs[name]
        kwargs.pop(name)
        return module

    def _create_inner_modules(self, args_innermost, modules_lib):
        inners = []
        if is_list(args_innermost):
            for inner_opt in args_innermost:
                module_name = self._get_from_kwargs(inner_opt, "module_name")
                inner_module_cls = getattr(modules_lib, module_name)
                inners.append(inner_module_cls(**inner_opt))

        else:
            module_name = self._get_from_kwargs(args_innermost, "module_name")
            inner_module_cls = getattr(modules_lib, module_name)
            inners.append(inner_module_cls(**args_innermost))

        return inners

    def _init_from_compact_format(self, opt, model_type):
        """Create a unetbasedmodel from the compact options format - where the
        same convolution is given for each layer, and arguments are given
        in lists
        """
        self.setDownConv   = nn.ModuleList()
        self.inner_modules = nn.ModuleList()
        self.setUpConv     = nn.ModuleList()

        contains_global = hasattr(opt, "innermost") and opt.innermost is not None
        if contains_global:
            inners = self._create_inner_modules(opt.innermost, modules_lib)
            for inner in inners:
                self.inner_modules.append(inner)
        else:
            self.inner_modules.append(Identity())

        # Down modules
        for i in range(len(opt.down_conv.down_conv_nn)):
            args = self._fetch_arguments(opt.down_conv, i, "DOWN")
            # conv_cls = self._get_from_kwargs(args, "conv_cls")
            down_module = conv_cls(**args)
            # self._save_sampling_and_search(down_module)
            self.down_modules.append(down_module)

        # Up modules
        for i in range(len(opt.up_conv.up_conv_nn)):
            args = self._fetch_arguments(opt.up_conv, i, "UP")
            conv_cls = self._get_from_kwargs(args, "conv_cls")
            up_module = conv_cls(**args)
            self._save_upsample(up_module)
            self.up_modules.append(up_module)

        self.metric_loss_module, self.miner_module = BaseModel.get_metric_loss_and_miner(
            getattr(opt, "loss", None), getattr(opt, "miner", None)
        )

    def _fetch_arguments_from_list(self, opt, index):
        """Fetch the arguments for a single convolution from multiple lists
        of arguments - for models specified in the compact format.
        """
        args = {}
        for o, v in opt.items():
            name = str(o)
            if is_list(v) and len(getattr(opt, o)) > 0:
                if name[-1] == "s" and name not in SPECIAL_NAMES:
                    name = name[:-1]
                v_index = v[index]
                if is_list(v_index):
                    v_index = list(v_index)
                args[name] = v_index
            else:
                if is_list(v):
                    v = list(v)
                args[name] = v
        return args

    def _fetch_arguments(self, conv_opt, index, flow):
        """ Fetches arguments for building a convolution (up or down)

        Arguments:
            conv_opt
            index in sequential order (as they come in the config)
            flow "UP" or "DOWN"
        """
        args = self._fetch_arguments_from_list(conv_opt, index)
        args["index"] = index
        return args

    def forward(self, xyz, x=None):
        """ This method does a forward on the PointNet++-like architecture
          xyz:
           x: original input for feature, not include xyz
        """
        stack_down = []
        feature = x
        stack_down.append([xyz, feature])
        for i in range(len(self.down_modules)):
            if feature is None:
                feature = xyz
            else:
                feature = torch.cat([feature, xyz], dim=1)
            xyz, feature= self.down_modules[i](xyz, xyz, feature) #
            stack_down.append([xyz, feature]) #[BS, C, N], TODO

        if not isinstance(self.inner_modules[0], Identity):
            feature = self.inner_modules[0](xyz, feature)[1] # return pos,
        #
        bottle_neck  = feature
        feature_prev = feature
        if feature_prev.size(-1) == 1:
            xyz_prev = None
        else:
            xyz_prev, feature_prev = stack_down.pop()
        for i in range(len(self.up_modules)):
            xyz, skip = stack_down.pop()
            xyz_prev, feature_prev = self.up_modules[i](xyz, skip, xyz_prev, feature_prev)
        pred = self.final_layers(feature_prev)

        return pred, bottle_neck

if __name__ == "__main__":
    from hydra.experimental import compose, initialize
    initialize(config_dir="../config/", strict=True)
    cfg = compose("config.yaml")
    opt = cfg.models.pointnet2_charlesmsg

    #>>>>> test input
    gpu = 0
    deploy_device   = torch.device('cuda:{}'.format(gpu))

    N     = 16384
    T     = 2
    xyz2  = torch.rand(1, T, 3, N, requires_grad=False, device=deploy_device).float() # [BS, T, C, N] --> later [BS, C, T, N]
    xyz1  = xyz2[:, 0, :, :]
    feat2 = torch.rand(1, T, 1, N, requires_grad=False, device=deploy_device).float()
    PM = PointMotionBaseModel(opt, 'pointnet2_charlesmsg').cuda(gpu)
    for i in range(10):
        pred  = PM(xyz1, xyz2, feat2)
    # for i in range(len(opt.down_conv.down_conv_nn)):
    # for i in range(1):
    #     args = PM._fetch_arguments(opt.down_conv, i, "DOWN")
    #     down_module = PointNetMSGDown3d(**args).cuda(gpu)
    #     xyz, feat   = down_module(xyz1, xyz2, feat2)
    #     print('output after one block has size: ', xyz.size, feat.size())



    print('Con!!!')
