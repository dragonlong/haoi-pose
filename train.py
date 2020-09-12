# System libs
import os
import time
import math
import random
import yaml
import GPUtil
import hydra
from hydra import utils
import logging
from omegaconf import DictConfig, ListConfig, OmegaConf

import torch
import torch.nn as nn

# Our libs
import __init__ as booger
import dataset.parser_vanilla as parserModule
from modules.network import Network
from global_info import global_info
from common.debugger import breakpoint

def main_worker(gpu, cfg):
    # >>>>>>>>>>>>>>>>> 1. create data loader;
    parser = parserModule.Parser(cfg)
    train_loader   = parser.get_train_set()
    # valid_loader   = parser.get_valid_set()
    # test_loader    = parser.get_test_set()

    cfg.TRAIN.epoch_iters = int(len(train_loader) / cfg.TRAIN.batch_size)
    cfg.TRAIN.max_iters   = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch

    if cfg.is_debug:
        dp = parser.train_dataset.__getitem__(100)

    # >>>>>>>>>>>>>>>>> 2. create model
    model = Network(gpu, cfg)

    # >>>>>>>>>>>>>>>>>>3. start train & evaluation
    #safe
    for epoch in range(cfg.TRAIN.num_epoch):
        train_infos = model.train_epoch(gpu, train_loader, model.modules['point'], epoch, cfg)

        # model.valid_epoch(valid_loader, cfg)
        # model.test_epoch(test_loader, cfg)

    # save to log, and save best model

@hydra.main(config_path="config/config.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly

    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(cfg.port)

    # category-wise training setup
    infos       = global_info()
    data_infos  = infos.datasets[cfg.item]
    cfg.n_max_parts = data_infos.num_parts
    cfg.HEAD.nocs_per_point[-2] = cfg.n_max_parts * 3
    cfg.root_data   = infos.second_path + '/data'
    cfg.log_dir     = infos.base_path + cfg.log_dir
    cfg.TRAIN.running_lr         = cfg.TRAIN.init_learning_rate
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    print(f' item: {cfg.item} , \n root_data: {cfg.root_data}, \
        \n, log_dir: {cfg.log_dir}')
    #>>>>>>>>>>>>>>>>>>>> setting ends here <<<<<<<<<< #

    #>>>>>>>>>>>>>>>>>> decide gpu, model weights <<<<<< #
    if cfg.TRAIN.start_epoch > 0:
        assert os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # evaluation
    if cfg.is_testing:
        cfg.TRAIN.num_epoch = 1
        cfg.MODEL.weights_decoder = cfg.log_dir + '/decoder_best_point_val.pth'

    cfg.num_gpus = torch.cuda.device_count()
    print(f'we have {cfg.num_gpus} gpus')
    if cfg.num_gpus > 0:
        gpus = GPUtil.getAvailable(order = 'load', limit = min(int(cfg.num_gpus), torch.cuda.device_count()), maxLoad=0.2, maxMemory=0.2, includeNan=False, excludeID=[], excludeUUID=[])
        cfg.num_gpus = len(gpus)
    # >>>>>>>>>>>>>>>>>> ends here. <<<<<<<<<<<<<<<<<<<< #

    # >>>>>>>>>>>> finalizing the config <<<<<<<<<<<<< #
    with open(os.path.join(cfg.log_dir, 'config.yaml'), 'w') as f:
        f.write(cfg.pretty())

    # set random seed
    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    if cfg.distributed:
        mp.spawn(main_worker, nprocs=cfg.num_gpus, cfg=(cfg))
    else:
        gpu = 0
        main_worker(gpu, cfg)

if __name__ == '__main__':
    main()
    print('')
