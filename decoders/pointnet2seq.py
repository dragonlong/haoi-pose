"""
build a simple network mimicing pointnet2 to operate on point sequence video;
- T = 5 for conv3d, 2 layers with conv3d;
- sampling 1024, 256, 64, 16, upsampling ;
- features 
"""
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

# custom packages
import __init__
from modules.conv_layer import Identity
from modules.blocks import GlobalDenseBaseModule, PointNetMSGDown3d, DenseFPModule, is_list, breakpoint

SPECIAL_NAMES = ["radius", "max_num_neighbors", "block_names"]

class PointMotionBaseModel(nn.Module):
    """
        # pointnet++-like architecture
        referring to https://github.com/nicolas-chaulet/torch-points3d/blob/master/torch_points3d
    """
    def __init__(self, 
        opt, 
        model_type='pointnet2_charlesmsg',         
        num_classes=3,
        batchnorm=True,
        use_xyz=True,
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
            num_class  - how many output of the network
            modules_lib - all modules that can be used in the UNet
        """
        super(PointMotionBaseModel, self).__init__()
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
            args = self._fetch_arguments(opt.down_conv, i, "DOWN") # fetch args for current scale block, may have multiple branches with different radius and K;
            down_module = PointNetMSGDown3d(**args, scale_idx=i)
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
                nn.Conv1d(128, num_classes, 1)
            ] if module is not None
        ]
        self.final_layers = nn.Sequential(*final_layer_modules)
   
    def _get_from_kwargs(self, kwargs, name):
        module = kwargs[name]
        kwargs.pop(name)
        return module

    # ssfor
    def forward(self, xyz, feat, times=None, verbose=False):
        """ 
           xyz: [B, T, 3, N]
           feat: original input for feature: [B, T, 1, N]
        """
        stack_down = []
        feat = feat.permute(0, 2, 1, 3).contiguous() #TODO
        stack_down.append([xyz[:, 0, :, :].contiguous(), feat[:, :, 0, :].contiguous()]) # here we only care current frame point during upsampling
        for i in range(len(self.down_modules)):
            # output feat: [BS, C, T, N]
            xyz, times, feat= self.down_modules[i](xyz, times, feat) 
            stack_down.append([xyz[:, 0, :, :].contiguous(), feat[:, :, 0, :].contiguous()]) # [BS, C, T, N] -> [BS, C, N]

        if not isinstance(self.inner_modules[0], Identity):
            feat = self.inner_modules[0](xyz, feat)[1] # return pos, 
            xyz_last= None
        else:
            xyz_last, feat = stack_down.pop()

        for i in range(len(self.up_modules)):
            if verbose:
                print(f'entering {i}th up_module')
            xyz, skip = stack_down.pop()
            xyz_last, feat = self.up_modules[i](xyz, skip, xyz_last, feat)
        pred = self.final_layers(feat)

        return pred 

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
        # args["conv_cls"] = self._factory_module.get_module(flow)
        args["index"] = index
        return args

if __name__ == "__main__":
    from hydra.experimental import compose, initialize
    from omegaconf import DictConfig, ListConfig, OmegaConf
    initialize(config_dir="../config/", strict=False)
    cfg = compose("config.yaml")
    OmegaConf.set_struct(cfg, False)
    opt = cfg.models['pointnet2_meteornet']
    opt.down_conv.kernel_type= 'attention'

    #>>>>> test input 
    gpu = 1
    torch.cuda.set_device(gpu)
    deploy_device   = torch.device('cuda:{}'.format(gpu))

    B = 2
    N     = 16384
    T     = 2
    xyz2  = torch.rand(B, T, 3, N, requires_grad=False, device=deploy_device).float() # [BS, T, C, N] --> later [BS, C, T, N]
    xyz1  = xyz2[:, 0, :, :]
    feat2 = torch.rand(B, T, 1, N, requires_grad=False, device=deploy_device).float() 
    times = torch.rand(B, T, 1, N, requires_grad=False, device=deploy_device).float()
    PM = PointMotionBaseModel(opt, 'pointnet2_meteornet').cuda(gpu)
    for i in range(100):
        print(f'iterating step {i}')
        pred  = PM(xyz2, feat2, times=times)
    # for i in range(len(opt.down_conv.down_conv_nn)):
    # for i in range(1):
    #     args = PM._fetch_arguments(opt.down_conv, i, "DOWN")
    #     down_module = PointNetMSGDown3d(**args).cuda(gpu)
    #     xyz, feat   = down_module(xyz1, xyz2, feat2)
    #     print('output after one block has size: ', xyz.size, feat.size())
    
    print('Con!!!')
