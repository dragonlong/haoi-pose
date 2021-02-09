import numpy as np
import os
from os import makedirs, remove
from os.path import exists, join
from datetime import datetime
from time import time
import matplotlib; matplotlib.use('Agg')
import hydra
from hydra import utils
import logging
from omegaconf import DictConfig, ListConfig, OmegaConf
import pickle
import wandb


import torch
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
from models.decoders.grnet import GRNet
from utils.extensions.chamfer_dist import ChamferDistance
from utils.extensions.gridding_loss import GriddingLoss
from common.train_utils import AverageMeter_gr
from utils.metric_util import Metrics
from common.train_utils import var_or_cuda, init_weights
from common.vis_utils import get_ptcloud_img

import common.loader_utils
import dataset
from dataset.obman_parser import ObmanParser
from global_info import global_info
from common.debugger import *


def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, grnet=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # # Set up data loader
        # dataset_loader = common.loader_utils.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        # test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
        #     common.loader_utils.DatasetSubset.TEST),
        #                                                batch_size=1,
        #                                                num_workers=cfg.CONST.NUM_WORKERS,
        #                                                collate_fn=common.loader_utils.collate_fn,
        #                                                pin_memory=True,
        #                                                shuffle=False)
        parser = ObmanParser(cfg)
        val_dataset   = parser.valid_dataset
        test_data_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, num_workers=cfg.CONST.NUM_WORKERS, shuffle=False,
                    collate_fn=common.loader_utils.collate_fn_obman,
                    worker_init_fn=dataset.worker_init_fn,
                                            pin_memory=True,
                                            drop_last=True)
    # Setup networks and initialize networks
    if grnet is None:
        grnet = GRNet(cfg)

        if torch.cuda.is_available():
            grnet = torch.nn.DataParallel(grnet).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        grnet.load_state_dict(checkpoint['grnet'])

    # Switch models to evaluation mode
    grnet.eval()

    # Set up loss functions
    chamfer_dist = ChamferDistance()
    gridding_loss = GriddingLoss(scales=cfg.NETWORK.GRIDDING_LOSS_SCALES,
                                 alphas=cfg.NETWORK.GRIDDING_LOSS_ALPHAS)    # lgtm [py/unused-import]

    # Testing loop
    n_samples = len(test_data_loader)
    test_losses = AverageMeter_gr(['SparseLoss', 'DenseLoss'])
    test_metrics = AverageMeter_gr(Metrics.names())
    category_metrics = dict()

    # Testing loop
    for model_idx, (taxonomy_id, model_id, data) in enumerate(test_data_loader):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        model_id = model_id[0]

        with torch.no_grad():
            for k, v in data.items():
                data[k] = var_or_cuda(v)

            sparse_ptcloud, dense_ptcloud = grnet(data)
            sparse_loss = chamfer_dist(sparse_ptcloud, data['gtcloud'])
            dense_loss = chamfer_dist(dense_ptcloud, data['gtcloud'])
            test_losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
            _metrics = Metrics.get(dense_ptcloud, data['gtcloud'])
            test_metrics.update(_metrics)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter_gr(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            if test_writer is not None and model_idx < 3:
                sparse_ptcloud = sparse_ptcloud.squeeze().cpu().numpy()
                # save_name = f'output/viz/pred_{epoch_idx}.npy'
                # save_for_viz(['points', 'labels'], [data['gtcloud'][m].cpu().numpy(), np.ones((data['gtcloud'][m].shape[0]))], save_name, type='np')
                sparse_ptcloud_img = get_ptcloud_img(sparse_ptcloud)
                test_writer.add_image('Model%02d/SparseReconstruction' % model_idx, sparse_ptcloud_img, epoch_idx)
                dense_ptcloud = dense_ptcloud.squeeze().cpu().numpy()
                dense_ptcloud_img = get_ptcloud_img(dense_ptcloud)
                test_writer.add_image('Model%02d/DenseReconstruction' % model_idx, dense_ptcloud_img, epoch_idx)
                gt_ptcloud = data['gtcloud'].squeeze().cpu().numpy()
                gt_ptcloud_img = get_ptcloud_img(gt_ptcloud)
                test_writer.add_image('Model%02d/GroundTruth' % model_idx, gt_ptcloud_img, epoch_idx)

            logging.info('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                         (model_idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()
                                                                            ], ['%.4f' % m for m in _metrics]))
            # save ply data
            save_offline =True
            if save_offline:
                m = 0
                save_name = f'{cfg.log_dir}/generation/input_{taxonomy_id}_{model_id}.npy'
                save_for_viz(['points', 'labels'], [data['partial_cloud'][m].cpu().numpy(), np.ones((data['partial_cloud'][m].shape[0]))], save_name, type='np')
                save_name = f'{cfg.log_dir}/generation/pred_{taxonomy_id}_{model_id}.npy'
                save_for_viz(['points', 'labels'], [dense_ptcloud[m].cpu().numpy(), np.ones((dense_ptcloud[m].cpu().numpy().shape[0]))], save_name, type='np')
                save_name = f'{cfg.log_dir}/generation/gt_{taxonomy_id}_{model_id}.npy'
                save_for_viz(['points', 'labels'], [data['gtcloud'][m].cpu().numpy(), np.ones((data['gtcloud'][m].shape[0]))], save_name, type='np')

    # Print testing results
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    for taxonomy_id in category_metrics:
        print(taxonomy_id, end='\t')
        print(category_metrics[taxonomy_id].count(0), end='\t')
        for value in category_metrics[taxonomy_id].avg():
            print('%.4f' % value, end='\t')
        print()

    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(1), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return Metrics(cfg.TEST.METRIC_NAME, test_metrics.avg())

def train_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    parser = ObmanParser(cfg)
    val_dataset   = parser.valid_dataset
    train_dataset = parser.train_dataset
    train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.CONST.NUM_WORKERS,
            collate_fn=common.loader_utils.collate_fn_obman,
            worker_init_fn=dataset.worker_init_fn,
                                    pin_memory=True,
                                    shuffle=True,
                                    drop_last=True)

    val_data_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=cfg.CONST.NUM_WORKERS, shuffle=False,
                collate_fn=common.loader_utils.collate_fn_obman,
                worker_init_fn=dataset.worker_init_fn,
                                        pin_memory=True,
                                        drop_last=True)

    # # Set up data loader
    # train_dataset_loader = common.loader_utils.DATASET_LOADER_MAPPING[cfg.DATASET.TRAIN_DATASET](cfg)
    # test_dataset_loader  = common.loader_utils.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    # train_data_loader    = torch.utils.data.DataLoader(dataset=train_dataset_loader.get_dataset(
    #     common.loader_utils.DatasetSubset.TRAIN),
    #                                                 batch_size=cfg.TRAIN.BATCH_SIZE,
    #                                                 num_workers=cfg.CONST.NUM_WORKERS,
    #                                                 collate_fn=common.loader_utils.collate_fn,
    #                                                 pin_memory=True,
    #                                                 shuffle=True,
    #                                                 drop_last=True)
    # val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader.get_dataset(
    #     common.loader_utils.DatasetSubset.VAL),
    #                                               batch_size=1,
    #                                               num_workers=cfg.CONST.NUM_WORKERS,
    #                                               collate_fn=common.loader_utils.collate_fn,
    #                                               pin_memory=True,
    #                                               shuffle=False)

    # Set up folders for logs and checkpoints
    output_dir = cfg.log_dir
    cfg.DIR.CHECKPOINTS = output_dir + '/checkpoints'
    cfg.DIR.LOGS = output_dir + '/logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)
        os.makedirs(cfg.DIR.LOGS)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer   = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    # Create the networks
    grnet = GRNet(cfg)
    grnet.apply(init_weights)
    logging.debug('Parameters in GRNet: %d.' % count_parameters(grnet))

    # Move the network to GPU if possible
    if torch.cuda.is_available():
        grnet = torch.nn.DataParallel(grnet).cuda()

    # Create the optimizers
    grnet_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, grnet.parameters()),
                                       lr=cfg.TRAIN.LEARNING_RATE,
                                       weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                       betas=(.9, .999))
    grnet_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(grnet_optimizer,
                                                              milestones=cfg.TRAIN.LR_MILESTONES,
                                                              gamma=cfg.TRAIN.GAMMA)
    # Set up loss functions
    chamfer_dist  = ChamferDistance()
    gridding_loss = GriddingLoss(    # lgtm [py/unused-local-variable]
        scales=cfg.NETWORK.GRIDDING_LOSS_SCALES,
        alphas=cfg.NETWORK.GRIDDING_LOSS_ALPHAS)

    # Load pretrained model if exists
    init_epoch = 0
    best_metrics = None
    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        best_metrics = Metrics(cfg.TEST.METRIC_NAME, checkpoint['best_metrics'])
        grnet.load_state_dict(checkpoint['grnet'])
        logging.info('Recover complete. Current epoch = #%d; best metrics = %s.' % (init_epoch, best_metrics))

    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()
        batch_time = AverageMeter_gr()
        data_time  = AverageMeter_gr()
        losses     = AverageMeter_gr(['SparseLoss', 'DenseLoss'])

        grnet.train()
        batch_end_time = time()
        n_batches = len(train_data_loader)
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(train_data_loader):
            data_time.update(time() - batch_end_time)
            if cfg.is_debug and batch_idx < 50 and epoch_idx < 1:
                print(f'---{batch_idx}', data['gtcloud'].shape, data['gtcloud'].shape)
                for m in range(1):
                    save_name = f'output/viz/input_{batch_idx}.npy'
                    save_for_viz(['points', 'labels'], [data['partial_cloud'][m].cpu().numpy(), np.ones((data['partial_cloud'][m].shape[0]))], save_name, type='np')
                    save_name = f'output/viz/gt_{batch_idx}.npy'
                    save_for_viz(['points', 'labels'], [data['gtcloud'][m].cpu().numpy(), np.ones((data['gtcloud'][m].shape[0]))], save_name, type='np')
            for k, v in data.items():
                data[k] = var_or_cuda(v)
            try:
                sparse_ptcloud, dense_ptcloud = grnet(data)
                sparse_loss = chamfer_dist(sparse_ptcloud, data['gtcloud'])
                dense_loss  = chamfer_dist(dense_ptcloud, data['gtcloud'])
            except:
                continue
            _loss = sparse_loss + dense_loss
            losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])

            grnet.zero_grad()
            _loss.backward()
            grnet_optimizer.step()

            n_itr = (epoch_idx - 1) * n_batches + batch_idx
            train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
            train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)

            batch_time.update(time() - batch_end_time)
            batch_end_time = time()
            logging.info('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s' %
                         (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches, batch_time.val(), data_time.val(),
                          ['%.4f' % l for l in losses.val()]))

        grnet_lr_scheduler.step()
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch_idx)
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]))

        # Validate the current model
        metrics = test_net(cfg, epoch_idx, val_data_loader, val_writer, grnet)

        # Save ckeckpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or metrics.better_than(best_metrics):
            file_name = 'ckpt-best.pth' if metrics.better_than(best_metrics) else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'epoch_index': epoch_idx,
                'best_metrics': metrics.state_dict(),
                'grnet': grnet.state_dict()
            }, output_path)  # yapf: disable

            logging.info('Saved checkpoint to %s ...' % output_path)
            if metrics.better_than(best_metrics):
                best_metrics = metrics

    train_writer.close()
    val_writer.close()

@hydra.main(config_path="config/completion.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(hydra.utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'

    # cfg = config.load_config(cfg.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not cfg.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")
    # Set t0
    t0 = time()
    # category-wise training setup
    infos       = global_info()
    data_infos  = infos.datasets[cfg.item]
    cfg.root_data   = infos.second_path + '/data'
    cfg.log_dir     = infos.second_path + cfg.log_dir
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    if cfg.use_wandb:
        wandb.init(project="haoi-pose", entity="teamname")
        wandb.init(config=cfg)
    # copy the whole project codes into log_dir
    # only need to do this for models
    if not cfg.debug:
        if not os.path.isdir(f'{cfg.log_dir}/code'):
            os.makedirs(f'{cfg.log_dir}/code')
            os.makedirs(f'{cfg.log_dir}/code/dataset')
        os.system('cp -r ./models {}/code'.format(cfg.log_dir))
        os.system('cp ./dataset/*py {}/code/dataset'.format(cfg.log_dir))
    # Shorthands
    out_dir    = cfg.log_dir
    print('Saving to ', out_dir)

    # training
    if cfg.eval:
        test_net(cfg)
    else:
        train_net(cfg)


if __name__ == '__main__':
    main()
