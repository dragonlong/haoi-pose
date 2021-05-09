import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import pickle
import wandb

import os
from os import makedirs, remove
from os.path import exists, join
import argparse
import time, datetime
import matplotlib; matplotlib.use('Agg')
import hydra
from hydra import utils
import logging
from omegaconf import DictConfig, ListConfig, OmegaConf

#
import dataset
from common.train_utils import CheckpointIO
from dataset.dataset_parser import DatasetParser
from utils import config
from collections import defaultdict
import shutil
from global_info import global_info

#
def bp():
    import pdb;pdb.set_trace()

# copy the code files into that folder as well
def save_for_viz(batch, config=None, index=0):
    save_viz = True
    save_path= './outputs/media'
    if save_viz:
        viz_dict = batch
        print('!!! Saving visualization data')
        save_viz_path = f'{save_path}/full_viz/'
        if not exists(save_viz_path):
            makedirs(save_viz_path)
        save_viz_name = f'{save_viz_path}{index}_data.npy'
        print('saving to ', save_viz_name)
        np.save(save_viz_name, arr=viz_dict)

@hydra.main(config_path="config/occupancy.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'

    # cfg = config.load_config(cfg.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not cfg.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")
    # Set t0
    t0 = time.time()
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
    batch_size = cfg['training']['batch_size']
    backup_every = cfg['training']['backup_every']
    vis_n_outputs = cfg['generation']['vis_n_outputs']
    exit_after = -1

    model_selection_metric = cfg['training']['model_selection_metric']
    if cfg['training']['model_selection_mode'] == 'maximize':
        model_selection_sign = 1
    elif cfg['training']['model_selection_mode'] == 'minimize':
        model_selection_sign = -1
    else:
        raise ValueError('model_selection_mode must be '
                         'either maximize or minimize.')

    # Output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    parser = DatasetParser(cfg)
    val_dataset   = parser.valid_dataset
    train_dataset = parser.train_dataset

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=cfg['training']['n_workers'], shuffle=True,
        collate_fn=dataset.collate_remove_none,
        worker_init_fn=dataset.worker_init_fn)

    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=cfg['training']['n_workers_val'], shuffle=False,
        collate_fn=dataset.collate_remove_none,
        worker_init_fn=dataset.worker_init_fn)
    # For visualizations
    vis_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=dataset.collate_remove_none,
        worker_init_fn=dataset.worker_init_fn)

    # Model
    model = config.get_model(cfg, device=device, dataset=train_dataset)
    if cfg.use_wandb:
        wandb.watch(model)
    # Generator
    generator = config.get_generator(model, cfg, device=device)

    # Intialize training
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    trainer = config.get_trainer(model, optimizer, cfg, device=device)

    checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
    try:
        load_dict = checkpoint_io.load('model.pt')
    except FileExistsError:
        load_dict = dict()
    epoch_it = load_dict.get('epoch_it', 0)
    it = load_dict.get('it', 0)
    metric_val_best = load_dict.get(
        'loss_val_best', -model_selection_sign * np.inf)

    if metric_val_best == np.inf or metric_val_best == -np.inf:
        metric_val_best = -model_selection_sign * np.inf
    print('Current best validation metric (%s): %.8f'
          % (model_selection_metric, metric_val_best))
    logger = SummaryWriter(os.path.join(out_dir, 'logs'))

    # Shorthands
    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    visualize_every = cfg['training']['visualize_every']

    # Print model
    nparameters = sum(p.numel() for p in model.parameters())
    print('Total number of parameters: %d' % nparameters)

    print('output path: ', out_dir)

    model_counter = defaultdict(int)
    data_vis_list = []

    if cfg.debug:
        dp1 = val_dataset.__getitem__(1)
        batch = {}
        for key, value in dp1.items():
            try:
                batch[key] = torch.from_numpy(value).unsqueeze(0)
            except:
                batch[key] = value
        loss = trainer.train_step(batch)
        # for validate
        eval_out = trainer.eval_step(batch)
        out = generator.generate_mesh(batch)
        try:
            mesh, stats_dict = out
        except TypeError:
            mesh, stats_dict = out, {}
        if not exists(os.path.join(out_dir, 'vis')):
            makedirs(os.path.join(out_dir, 'vis'))
        mesh.export(os.path.join(out_dir, 'vis', 'example.off'))
        print('saving mesh to ', os.path.join(out_dir, 'vis', 'example.off'))
        print('---passing debug')
    # Build a data dictionary for visualization
    iterator = iter(vis_loader)
    categories = val_dataset.categories
    print('Build a data dictionary for visualization')
    for i in range(len(vis_loader)):
        data_vis = next(iterator)
        idx = data_vis['idx'].item()
        model_path = val_dataset.pose_dataset.obj_paths[idx]
        category_name = model_path.split('/')[-4]
        instance_name = model_path.split('/')[-3]
        category_id = categories.index(category_name)
        c_it = model_counter[category_id]
        if c_it < vis_n_outputs:
            data_vis_list.append({'category': category_name, 'instance_name': instance_name, 'it': c_it, 'data': data_vis})
        #
        model_counter[category_id] += 1

    while True:
        epoch_it += 1
        for batch in train_loader:
            it += 1
            if it < 5:
                save_for_viz(batch, cfg, index=it)
            loss = trainer.train_step(batch)
            if type(loss) is list:
                logger.add_scalar('train/loss', loss[1], it)
                logger.add_scalar('train/hand_loss', loss[2], it)
                if cfg.use_wandb:
                    wandb.log({"train/loss": loss[1], 'train/hand_loss': loss[2]})
                loss = loss[0]
            else:
                logger.add_scalar('train/loss', loss, it)
                if cfg.use_wandb:
                    wandb.log({"train/loss": loss})
            # Print output
            if print_every > 0 and (it % print_every) == 0:
                t = datetime.datetime.now()
                print('[Epoch %02d] it=%03d, loss=%.4f, time: %.2fs, %02d:%02d'
                         % (epoch_it, it, loss, time.time() - t0, t.hour, t.minute))

            # Visualize output
            if visualize_every > 0 and (it % visualize_every) == 0:
                print('Visualizing')
                for data_vis in data_vis_list:
                    if cfg['generation']['sliding_window']:
                        out = generator.generate_mesh_sliding(data_vis['data'])
                    else:
                        out = generator.generate_mesh(data_vis['data'])
                    # Get statistics
                    try:
                        mesh, stats_dict = out
                    except TypeError:
                        mesh, stats_dict = out, {}
                    if not exists(os.path.join(out_dir, 'vis')):
                        makedirs(os.path.join(out_dir, 'vis'))
                    mesh.export(os.path.join(out_dir, 'vis', '{}_{}_{}_{}.off'.format(it, data_vis['category'], data_vis['instance_name'], data_vis['it'])))
                    np.save(os.path.join(out_dir, 'vis', '{}_{}_{}_{}.npy'.format(it, data_vis['category'], data_vis['instance_name'], data_vis['it'])), arr=data_vis['data'])
            # Save checkpoint
            if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
                print('Saving checkpoint')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

            # Backup if necessary
            if (backup_every > 0 and (it % backup_every) == 0):
                print('Backup checkpoint')
                checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)
            # Run validation
            if validate_every > 0 and (it % validate_every) == 0:
                eval_dict = trainer.evaluate(val_loader)
                metric_val = eval_dict[model_selection_metric]
                print('Validation metric (%s): %.4f'
                      % (model_selection_metric, metric_val))

                for k, v in eval_dict.items():
                    logger.add_scalar('val/%s' % k, v, it)

                if model_selection_sign * (metric_val - metric_val_best) > 0:
                    metric_val_best = metric_val
                    print('New best model (loss %.4f)' % metric_val_best)
                    checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                       loss_val_best=metric_val_best)

            # Exit if necessary
            if exit_after > 0 and (time.time() - t0) >= exit_after:
                print('Time limit reached. Exiting.')
                checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)
                exit(3)

if __name__ == '__main__':
    main()
