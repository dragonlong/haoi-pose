# ----------------------------------------------------------------------------------------------------------------------
#
#      Multi-sweep training on SemanticKitti dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Xiaolong Li - 06/20/2020
#
import os
import sys
import numpy as np
import signal
import hydra
import logging
import torch
import logging
from omegaconf import DictConfig, ListConfig, OmegaConf
from modules.train_helper import ModelTrainer
from dataset.kitti.SemanticKitti_motion import *
from decoders.kpconv import KPFCNN

def init_cfg(cfg):
    """
    Class Initialyser
    """

    # Number of layers
    cfg.num_layers = len([block for block in cfg.architecture if 'pool' in block or 'strided' in block]) + 1

    ###################
    # Deform layer list
    ###################
    #
    # List of boolean indicating which layer has a deformable convolution
    layer_blocks = []
    cfg.deform_layers = []
    arch = cfg.architecture
    for block_i, block in enumerate(arch):

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
            layer_blocks += [block]
            continue
        # Convolution neighbors indices
        # *****************************
        deform_layer = False
        if layer_blocks:
            if np.any(['deformable' in blck for blck in layer_blocks]):
                deform_layer = True

        if 'pool' in block or 'strided' in block:
            if 'deformable' in block:
                deform_layer = True

        cfg.deform_layers += [deform_layer]
        layer_blocks = []

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break
    return cfg

@hydra.main(config_path="config/motion_segmentation.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if cfg.pretty_print:
        print(cfg.pretty())
    if cfg.epoch_steps < 500:
        cfg.checkpoint_gap = 1
        cfg.validation_size = 100
    os.chdir(hydra.utils.get_original_cwd())
    cfg = init_cfg(cfg)

    # # Set GPU visible device
    if cfg.GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPU)

    logger = logging.getLogger(__name__)
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    logger.info("Outputing checkpoints to: {}".format(cfg.log_dir))

    with open(os.path.join(cfg.log_dir, 'multi_segmentation.yaml'), 'w') as f:
        f.write(cfg.pretty())

    # chkp_idx = 'chkp_0200.tar'
    chkp_idx = None
    if (cfg.use_pretrain or cfg.eval) and cfg.pretrained_path:
        chkp_path = os.path.join(cfg.pretrained_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']
        if chkp_idx is None:
            pretrained_weight = 'best_val_chkp.tar'
        else:
            pretrained_weight = chkp_idx
        pretrained_weight = os.path.join(cfg.pretrained_path, 'checkpoints', pretrained_weight)
        print('Initialization from :', pretrained_weight)
    else:
        pretrained_weight = None

    # Initialize datasets
    training_dataset = SemanticKittiDataset(cfg, set='training', balance_classes=True)
    valid_dataset    = SemanticKittiDataset(cfg, set='validation', balance_classes=False)
    test_dataset     = SemanticKittiDataset(cfg, set='test', balance_classes=False)

    # Initialize samplers
    training_sampler = SemanticKittiSampler(training_dataset)
    valid_sampler = SemanticKittiSampler(valid_dataset)
    test_sampler = SemanticKittiSampler(test_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=SemanticKittiCollate,
                                 num_workers=cfg.input_threads,
                                 pin_memory=True)

    if cfg.debug:
        datapoint = valid_dataset.__getitem__(200)
        for single_item in datapoint:
            try:
                print(len(single_item))
            except:
                pass
    valid_loader = DataLoader(valid_dataset,
                             batch_size=1,
                             sampler=valid_sampler,
                             collate_fn=SemanticKittiCollate,
                             num_workers=cfg.input_threads,
                             pin_memory=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=SemanticKittiCollate,
                             num_workers=cfg.input_threads,
                             pin_memory=True)
    # Calibrate max_in_point value
    training_sampler.calib_max_in(cfg, training_loader, verbose=True)
    valid_sampler.calib_max_in(cfg, valid_loader, verbose=True)

    # Calibrate samplers
    training_sampler.calibration(training_loader, verbose=True)
    valid_sampler.calibration(valid_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)
    # debug_timing(training_dataset, training_loader)
    # debug_timing(valid_dataset, valid_loader)
    # debug_class_w(training_dataset, training_loader)

    print('\nModel Preparation')
    print('*****************')

    t1  = time.time()
    # net = KPFCNN(cfg, training_dataset.label_values, training_dataset.ignored_labels)
    net = KPFCNN(cfg, training_dataset.motion_label_values, training_dataset.ignored_labels)

    debug = True
    if debug:
        print('\n*************************************\n')
        print(net)
        print('\n*************************************\n')
        for param in net.parameters():
            if param.requires_grad:
                print(param.shape)
        print('\n*************************************\n')
        print("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))
        print('\n*************************************\n')

    # Define a trainer class
    trainer = ModelTrainer(net, cfg, chkp_path=pretrained_weight)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')

    # Training
    trainer.train(net, training_loader, valid_loader, cfg, test_loader=test_loader)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)

if __name__ == '__main__':
    main()
