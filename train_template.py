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
import dgl

from collections import OrderedDict
from tqdm import tqdm
from dataset.obman_parser import ObmanParser
from dataset.modelnet40_parser import ModelParser
from common.train_utils import cycle
from models import get_agent

from common.debugger import *
from evaluation.pred_check import post_summary, prepare_pose_eval
from common.algorithms import compute_pose_ransac
from common.d3_utils import axis_diff_degree, rot_diff_rad
from common.vis_utils import plot_distribution
from global_info import global_info

infos           = global_info()
my_dir          = infos.base_path
project_path    = infos.project_path
categories_id   = infos.categories_id
categories      = infos.categories

whole_obj = infos.whole_obj
part_obj  = infos.part_obj
obj_urdf  = infos.obj_urdf
categories_id = infos.categories_id
project_path  = infos.project_path

def check_r_bb(test_data, tr_agent):
    BS = 2
    input_pts     = test_data['G'].ndata['x'].view(BS, -1, 3).contiguous().cpu().numpy() # B, N, 3
    pred_vertices = tr_agent.output_R.permute(0, 2, 3, 1).cpu().numpy() + input_pts[:, :, np.newaxis, :]# Bs, 3, N, M --> B, N, M, 3
    target_R = test_data['R'].numpy() + input_pts[:, :, np.newaxis, :] # B, N, M, 3
    return np.linalg.norm(pred_vertices - target_R, axis=-1).mean(axis=0).mean(axis=0)

def check_t(data, tr_agent):
    BS = 2
    input_pts     = test_data['G'].ndata['x'].view(BS, -1, 3).contiguous().cpu().numpy() # B, N, 3
    gt_center     = input_pts - test_data['T'].cpu().numpy()
    pred_center   = input_pts - tr_agent.output_T.cpu().detach().numpy().transpose(0, 2, 1)

    return np.linalg.norm(np.mean(pred_center, axis=1) - np.mean(gt_center, axis=1))

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
        if cfg.eval:
            run_name = f'{cfg.exp_num}_{cfg.target_category}_eval'
        else:
            run_name = f'{cfg.exp_num}_{cfg.target_category}'
        wandb.init(project="haoi-pose", name=run_name)
        wandb.init(config=cfg)
    # copy the project codes into log_dir
    if (not cfg.eval) and (not cfg.debug):
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
    if cfg.use_pretrain or cfg.eval:
        tr_agent.load_ckpt(cfg.ckpt)

    if cfg.name_dset == 'obman':
        parser = ObmanParser(cfg)
    else:
        parser = ModelParser(cfg)
    train_loader = parser.trainloader
    val_loader   = parser.validloader
    test_loader  = parser.validloader
    dset         = parser.valid_dataset
    dp = dset.__getitem__(0)



    # start training
    clock = tr_agent.clock #
    best_R_error = 100
    val_loader   = cycle(val_loader)
    for e in range(clock.epoch, cfg.nr_epochs):
        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            torch.cuda.empty_cache()
            tr_agent.train_func(data)
            pbar.set_description("EPOCH[{}][{}]".format(e, b))
            losses = tr_agent.collect_loss()
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))

            # validation step
            if clock.step % cfg.val_frequency == 0:
                torch.cuda.empty_cache()
                data = next(val_loader)
                tr_agent.val_func(data)
            if clock.step % cfg.eval_frequency == 0:
                r_mean_err = []
                t_mean_err = []
                for num, test_data in enumerate(test_loader):
                    if num > 100: # we only evaluate 100 data every 1000 steps
                        break
                    losses, infos = tr_agent.eval_func(test_data)
                    # r
                    if cfg.pred_bb:
                        r_mean_err.append(check_r_bb(test_data, tr_agent))
                    elif cfg.pred_6d:
                        pass
                    else:
                        pass
                    # t
                    t_mean_err.append(check_t(test_data, tr_agent))

                r_err = np.array(r_mean_err).mean(axis=0)
                t_err = np.array(t_mean_err).mean(axis=0)

                if cfg.use_wandb:
                    if cfg.pred_bb:
                        wandb.log({f'test/vertice1_err': r_err[0], 'step': clock.step})
                        wandb.log({f'test/vertice2_err': r_err[1], 'step': clock.step}) # max
                    elif cfg.pred_6d:
                        pass
                    else:
                        pass
                    if cfg.use_objective_T:
                        wandb.log({f'test/center_err': t_err, 'step': clock.step})
            clock.tick()
            if clock.step % cfg.save_step_frequency == 0:
                tr_agent.save_ckpt('latest')

        tr_agent.update_learning_rate()
        clock.tock()

        if clock.epoch % cfg.save_frequency == 0:
            tr_agent.save_ckpt()
        tr_agent.save_ckpt('latest')


if __name__ == '__main__':
    main()
