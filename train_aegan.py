from os import makedirs, remove
from os.path import exists, join
from time import time
import hydra
from hydra import utils
import logging
from omegaconf import DictConfig, ListConfig, OmegaConf
import pickle
import wandb
import torch

from collections import OrderedDict
from tqdm import tqdm
from dataset.obman_parser import ObmanParser
from common.train_utils import cycle
from models import get_agent

from common.debugger import *
from global_info import global_info

infos           = global_info()
my_dir          = infos.base_path
project_path    = infos.project_path
categories_id   = infos.categories_id
categories      = infos.categories

# training: ae_gan
@hydra.main(config_path="config/completion.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(hydra.utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    is_cuda = (torch.cuda.is_available() and not cfg.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")
    # Set t0
    t0 = time()
    # category-wise training setup
    infos       = global_info()
    data_infos  = infos.datasets[cfg.item]
    cfg.root_data   = infos.second_path + '/data'
    cfg.log_dir     = infos.second_path + cfg.log_dir
    cfg.model_dir   = cfg.log_dir + '/checkpoints'
    if not os.path.isdir(cfg.log_dir):
        os.makedirs(cfg.log_dir)
        os.makedirs(cfg.log_dir + '/checkpoints'
        )

    if cfg.use_wandb:
        wandb.init(project="haoi-pose", name=f'{cfg.exp_num}_{cfg.target_category}')
        wandb.init(config=cfg)
    # copy the project codes into log_dir
    if not cfg.debug:
        if not os.path.isdir(f'{cfg.log_dir}/code'):
            os.makedirs(f'{cfg.log_dir}/code')
            os.makedirs(f'{cfg.log_dir}/code/dataset')
        os.system('cp -r ./models {}/code'.format(cfg.log_dir))
        os.system('cp -r ./config {}/code'.format(cfg.log_dir))
        os.system('cp ./dataset/*py {}/code/dataset'.format(cfg.log_dir))
    # Shorthands
    out_dir = cfg.log_dir
    print('Saving to ', out_dir)

    # create network and training agent
    tr_agent = get_agent(cfg)
    if cfg.use_wandb:
        if cfg.module=='gan':
            wandb.watch(tr_agent.netG)
            wandb.watch(tr_agent.netD)
        else:
            wandb.watch(tr_agent.net)

    # load from checkpoint if provided
    if cfg.cont or cfg.eval:
        tr_agent.load_ckpt(cfg.ckpt)

    parser = ObmanParser(cfg)
    train_loader = parser.trainloader
    val_loader   = parser.validloader
    val_loader   = cycle(val_loader)
    # start training
    clock = tr_agent.clock #

    if cfg.eval:
        # data = next(val_loader)
        if cfg.set == 'val':
            pbar = tqdm(val_loader)
        else:
            pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            tr_agent.val_func(data)
            # if cfg.vis and clock.step % cfg.vis_frequency == 0:
            #     tr_agent.visualize_batch(data, "validation")
            save_offline =True
            if save_offline:
                if cfg.task == 'adversarial_adaptation':
                    outputs = tr_agent.fake_pc
                    for m in range(data["raw"].shape[0]):
                        model_id  = data["raw_id"][m]
                        taxonomy_id = categories[cfg.target_category]
                        save_name = f'{cfg.log_dir}/generation/{cfg.set}/input_{taxonomy_id}_{model_id}.npy'
                        save_for_viz(['points', 'labels'], [data["raw"][m].cpu().numpy().T, np.ones((data["raw"][m].shape[1]))], save_name, type='np')
                        save_name = f'{cfg.log_dir}/generation/{cfg.set}/{cfg.module}_{taxonomy_id}_{model_id}.npy'
                        save_for_viz(['points', 'labels'], [outputs[m].cpu().numpy().T, np.ones((outputs[m].cpu().numpy().shape[1]))], save_name, type='np')
                else:
                    outputs = tr_agent.output_pts
                    if cfg.use_wandb:
                        for m in range(data["points"].shape[0]):
                            target_pts  = data["points"][m].cpu().numpy().T
                            outputs_pts = outputs[m].cpu().numpy().T
                            outputs_pts = outputs_pts + np.array([0, 1, 0]).reshape(1, -1)
                            pts = np.concatenate([target_pts, outputs_pts], axis=0)
                            wandb.log({"input+AE_GAN_output": [wandb.Object3D(pts)], 'step': b*data["points"].shape[0] + m})

                    for m in range(data["points"].shape[0]):
                        model_id  = data['id'][m]
                        taxonomy_id = categories[cfg.target_category]
                        save_name = f'{cfg.log_dir}/generation/{cfg.set}/input_{taxonomy_id}_{model_id}.npy'
                        save_for_viz(['points', 'labels'], [data["points"][m].cpu().numpy().T, np.ones((data["points"][m].shape[1]))], save_name, type='np')
                        save_name = f'{cfg.log_dir}/generation/{cfg.set}/{cfg.module}_{taxonomy_id}_{model_id}.npy'
                        save_for_viz(['points', 'labels'], [outputs[m].cpu().numpy().T, np.ones((outputs[m].cpu().numpy().shape[1]))], save_name, type='np')
        return

    for e in range(clock.epoch, cfg.nr_epochs):
        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            torch.cuda.empty_cache()
            tr_agent.train_func(data)

            # visualize
            if cfg.vis and clock.step % cfg.vis_frequency == 0:
                tr_agent.visualize_batch(data, "train")

            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            losses = tr_agent.collect_loss()
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            # validation step
            if clock.step % cfg.val_frequency == 0:
                torch.cuda.empty_cache()
                data = next(val_loader)
                tr_agent.val_func(data)

                if cfg.vis and clock.step % cfg.vis_frequency == 0:
                    tr_agent.visualize_batch(data, "validation")

            clock.tick()

        tr_agent.update_learning_rate()
        clock.tock()

        if clock.epoch % cfg.save_frequency == 0:
            tr_agent.save_ckpt()
        tr_agent.save_ckpt('latest')


if __name__ == '__main__':
    main()
