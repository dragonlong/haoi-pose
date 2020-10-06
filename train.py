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
import dataset.parser as parserModule
from dataset.hand_mano_regression import ManoRegressionDataset
from dataset.hand_shape2motion import ContactsVoteDataset
from modules.network import Network
from global_info import global_info
from common.debugger import breakpoint

def main_worker(gpu, cfg):
    # >>>>>>>>>>>>>>>>> 1. create data loader;
    # parser = parserModule.Parser(cfg, ContactsVoteDataset)
    parser = parserModule.Parser(cfg, ManoRegressionDataset)
    train_loader   = parser.get_train_set()
    valid_loader   = parser.get_valid_set()
    test_loader    = parser.get_test_set()

    cfg.TRAIN.epoch_iters = len(train_loader)
    cfg.TRAIN.max_iters   = max(500, cfg.TRAIN.epoch_iters) * cfg.TRAIN.num_epoch

    #
    if cfg.is_debug:
        dp = parser.train_dataset.__getitem__(100)

    # >>>>>>>>>>>>>>>>> 2. create model
    model = Network(gpu, cfg)
    best_train_iou = 0 #
    best_valid_iou = 0 #
    # >>>>>>>>>>>>>>>>>>3. start train & evaluation
    for epoch in range(cfg.TRAIN.num_epoch):
        if cfg.eval:
            break
        print('---epoch {}'.format(epoch))
        train_miou, train_loss  = model.train_epoch(gpu, train_loader, model.modules['point'], epoch, cfg)
        valid_miou, valid_loss  = model.valid_epoch(gpu, valid_loader, model.modules['point'], epoch, cfg, prefix='seen', save_pred=epoch%2==0)
        test_miou, test_loss    = model.valid_epoch(gpu, test_loader, model.modules['point'], epoch, cfg, prefix='unseen', save_pred=epoch%2==0)

        #>>>>>>>>>>>> save checkpoints <<<<<<<<<<<<<<<<<<#
        # if best_train_iou < train_miou:
        for key, nets in model.nets_dict.items():
            model.checkpoint(nets, cfg, epoch+1, suffix= key + '_train')
        # best_train_iou = train_miou
        #
        # if best_valid_iou < valid_miou:
        for key, nets in model.nets_dict.items():
            model.checkpoint(nets, cfg, epoch+1, suffix= key + '_valid')
        # best_valid_iou = valid_miou

    print('Training Done!')
    test_miou, test_loss    = model.valid_epoch(gpu, test_loader, model.modules['point'], epoch, cfg, prefix='unseen', save_pred=True)

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
    cfg.MODEL.num_classes = cfg.n_max_parts
    if 'nocs_per_point' in cfg.HEAD:
        cfg.HEAD.nocs_per_point[-2] = cfg.n_max_parts * 3
    cfg.root_data   = infos.second_path + '/data'
    cfg.log_dir     = infos.second_path + cfg.log_dir
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
        gpu = int(cfg.gpu)
        main_worker(gpu, cfg)

if __name__ == '__main__':
    main()
    print('')
