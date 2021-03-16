from os import makedirs, remove
from os.path import exists, join
from time import time
import hydra
from hydra import utils
import logging
from omegaconf import DictConfig, ListConfig, OmegaConf
import pickle
import wandb
import random
import torch
import dgl

from collections import OrderedDict
from tqdm import tqdm
from sklearn.externals import joblib
from dataset.obman_parser import ObmanParser
from dataset.modelnet40_parser import ModelParser
from common.train_utils import cycle
from models import get_agent
from models.ae_gan.networks_ae import BuildGraph
from models.losses import loss_geodesic
from common.d3_utils import compute_rotation_matrix_from_ortho6d

from common.debugger import *
from evaluation.pred_check import post_summary, prepare_pose_eval
from common.algorithms import compute_pose_ransac
from common.d3_utils import axis_diff_degree, rot_diff_rad, rotate_about_axis
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

THRESHOLD_GOOD = 5 # 5 degrees
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

# only get one instance
def get_test_data(nx=8, ny=8, nz=8):
    fixed_sampling = True
    category_name = 'airplane'
    instance_name = '002'

    fn  = [category_name, '/groups/CESCA-CV/ICML2021/data/modelnet40/airplane/0_0_0.txt']
    BS  = 1
    train_pts= []
    targets = []
    builder = BuildGraph(num_samples=10)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                theta_x = 360/8 * int(i)
                Rx = rotate_about_axis(theta_x / 180 * np.pi, axis='x')
                theta_y = 360/8 * int(j)
                Ry = rotate_about_axis(theta_y / 180 * np.pi, axis='y')
                theta_z = 360/8 * int(k)
                Rz = rotate_about_axis(theta_z / 180 * np.pi, axis='z')
                r = np.matmul(Ry, Rx).astype(np.float32)
                r = np.matmul(Rz, r).astype(np.float32)
                point_normal_set = np.loadtxt(fn[1].replace('0_0_0', f'{i}_{j}_{k}'), delimiter=' ').astype(np.float32)
                full_pts  = np.copy(point_normal_set[:, 0:3])
                train_pts.append(full_pts)
                targets.append(r)

    return train_pts, targets

def get_category_data():
    fpath  = '/groups/CESCA-CV/ICML2021/data/modelnet40'
    f_train= f'{fpath}/airplane_train_2048.pk'
    f_test = f'{fpath}/airplane_test_2048.pk'

    with open(f_train, "rb") as f:
       train_pts = joblib.load(f)

    with open(f_test, "rb") as obj_f:
        test_pts  = joblib.load(obj_f)

    print(train_pts.shape, test_pts.shape)
    return train_pts, test_pts

def app_func(tr_agent, g, target_R, cfg):
    BS     = 2
    N  = cfg.num_points
    M  = cfg.num_modes_R
    CS = cfg.MODEL.num_channels_R
    torch.cuda.empty_cache()
    tr_agent.net.train()
    latent_vect = tr_agent.net.encoder(g)
    # [B*N, C, 3],  --> [B*N, M, 2, 3] --> B*N, M, 6 --> [B*N*M, 6]
    tr_agent.output_R = latent_vect['R'].view(-1, 6).contiguous()
    tr_agent.output_R = compute_rotation_matrix_from_ortho6d(tr_agent.output_R)
    # [B*M, 6]
    tr_agent.output_R_pooled = latent_vect['1'].view(-1, 6).contiguous() # dense by default!!! B, 2*C, 3
    tr_agent.output_R_pooled = compute_rotation_matrix_from_ortho6d(tr_agent.output_R_pooled) #

    # B, 3, 3
    target_R_tiled  = target_R.unsqueeze(1).contiguous().repeat(1, N, 1, 1).contiguous()
    geodesic_loss   = loss_geodesic(tr_agent.output_R, target_R_tiled.view(-1, 3, 3).contiguous().float())  # BS*N* C/2
    tr_agent.degree_err_full = geodesic_loss.view(BS, N).contiguous()       # BS, N
    tr_agent.output_R_full   = tr_agent.output_R.view(BS, N, 3, 3).contiguous() # BS, N, M, 3, 3
    tr_agent.output_R   = tr_agent.output_R_full.squeeze()
    tr_agent.norm_err   = torch.norm(tr_agent.output_R.view(BS, N, 9).contiguous() - target_R_tiled.view(BS, N, 9).contiguous(), dim=-1)

    loss_dict = {'regressionR': tr_agent.norm_err.mean()}
    tr_agent.degree_err = tr_agent.degree_err_full
    correctness_mask = torch.zeros((tr_agent.degree_err.shape[0], tr_agent.degree_err.shape[1]), device=tr_agent.output_R.device) # B, N
    correctness_mask[tr_agent.degree_err<THRESHOLD_GOOD] = 1.0
    good_pred_ratio = torch.mean(correctness_mask, dim=1) # TODO
    info_dict = {'5deg': good_pred_ratio.mean(), 'rdiff': tr_agent.degree_err_full.mean()}

    return loss_dict, info_dict

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

    # train_pts, targets = get_test_data()
    train_pts, test_pts = get_category_data()
    builder = BuildGraph(num_samples=10)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    inputs = []
    BS     = 2
    N  = cfg.num_points
    M  = cfg.num_modes_R
    CS = cfg.MODEL.num_channels_R
    npoints= cfg.num_points
    fixed_sampling = cfg.fixed_sampling
    target_R = torch.from_numpy(np.stack([np.eye(3).astype(np.float32), np.eye(3).astype(np.float32)], axis=0)).cuda()

    # start training
    clock = tr_agent.clock #
    pbar = tqdm(range(0, int(train_pts.shape[0]/2)))
    for e in range(clock.epoch, 200):
        for _, b in enumerate(pbar):
            input_pts = np.copy(train_pts[2*b:2*b+2].astype(np.float32))
            if fixed_sampling:
                input_pts = np.random.permutation(input_pts)
            xyz1 = torch.from_numpy(input_pts[:, :npoints]).cuda()
            g, _     = builder(xyz1)

            # error
            loss_dict, info_dict = app_func(tr_agent, g, target_R, cfg)
            loss = sum(loss_dict.values())
            tr_agent.optimizer.zero_grad()
            loss.backward()
            tr_agent.optimizer.step()

            pbar.set_description("EPOCH[{}][{}]".format(0, e))
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in  {**loss_dict, **info_dict}.items()}))
            if cfg.use_wandb:
                for k, v in loss_dict.items():
                    wandb.log({f'train/{k}': v, 'step': clock.step})
                for k, v in info_dict.items():
                    wandb.log({f'train/{k}': v, 'step': clock.step})

            with torch.no_grad():
                torch.cuda.empty_cache()
                tr_agent.net.eval()
                # randomly get 2 ints
                inds = random.sample(range(1, test_pts.shape[0]), 2)
                input_pts = np.copy(test_pts[inds].astype(np.float32))
                if fixed_sampling:
                    input_pts = np.random.permutation(input_pts)
                xyz1 = torch.from_numpy(input_pts[:, :npoints]).cuda()
                g1, _     = builder(xyz1)
                latent_vect = tr_agent.net.encoder(g1)
                # [B*N, C, 3],  --> [B*N, M, 2, 3] --> B*N, M, 6 --> [B*N*M, 6]
                tr_agent.output_R = latent_vect['R'].view(-1, 6).contiguous()
                tr_agent.output_R = compute_rotation_matrix_from_ortho6d(tr_agent.output_R)
                # [B*M, 6]
                tr_agent.output_R_pooled = latent_vect['1'].view(-1, 6).contiguous() # dense by default!!! B, 2*C, 3
                tr_agent.output_R_pooled = compute_rotation_matrix_from_ortho6d(tr_agent.output_R_pooled) #

                # B, 3, 3
                target_R_tiled  = target_R.unsqueeze(1).contiguous().repeat(1, N, 1, 1).contiguous()
                geodesic_loss   = loss_geodesic(tr_agent.output_R, target_R_tiled.view(-1, 3, 3).contiguous().float())  # BS*N* C/2
                tr_agent.degree_err_full = geodesic_loss.view(BS, N).contiguous()       # BS, N
                tr_agent.output_R_full   = tr_agent.output_R.view(BS, N, 3, 3).contiguous() # BS, N, M, 3, 3
                tr_agent.output_R   = tr_agent.output_R_full.squeeze()
                tr_agent.norm_err   = torch.norm(tr_agent.output_R.view(BS, N, 9).contiguous() - target_R_tiled.view(BS, N, 9).contiguous(), dim=-1)

                loss_dict = {'regressionR': tr_agent.norm_err.mean()}
                tr_agent.degree_err = tr_agent.degree_err_full
                correctness_mask = torch.zeros((tr_agent.degree_err.shape[0], tr_agent.degree_err.shape[1]), device=tr_agent.output_R.device) # B, N
                correctness_mask[tr_agent.degree_err<THRESHOLD_GOOD] = 1.0
                good_pred_ratio = torch.mean(correctness_mask, dim=1) # TODO
                info_dict = {'5deg': good_pred_ratio.mean(), 'rdiff': tr_agent.degree_err_full.mean()}

                if cfg.use_wandb:
                    for k, v in loss_dict.items():
                        wandb.log({f'test/{k}': v, 'step': clock.step})
                    for k, v in info_dict.items():
                        wandb.log({f'test/{k}': v, 'step': clock.step})

            clock.tick()
            if clock.step % cfg.save_step_frequency == 0:
                tr_agent.save_ckpt('latest')

        tr_agent.update_learning_rate()
        clock.tock()

        if clock.epoch % 10 == 0:
            tr_agent.save_ckpt()
    # best_R_error = 100
    # val_loader   = cycle(val_loader)
    # for e in range(clock.epoch, cfg.nr_epochs):
    #     # begin iteration
    #     pbar = tqdm(train_loader)
    #     for b, data in enumerate(pbar):
    #         # train step
    #         torch.cuda.empty_cache()
    #         tr_agent.train_func(data)
    #         pbar.set_description("EPOCH[{}][{}]".format(e, b))
    #         losses = tr_agent.collect_loss()
    #         pbar.set_postfix(OrderedDict({k: v.item() for k, v in losses.items()}))
    #
    #         # validation step
    #         if clock.step % cfg.val_frequency == 0:
    #             torch.cuda.empty_cache()
    #             data = next(val_loader)
    #             tr_agent.val_func(data)
    #         if clock.step % cfg.eval_frequency == 0:
    #             r_mean_err = []
    #             t_mean_err = []
    #             for num, test_data in enumerate(test_loader):
    #                 if num > 100: # we only evaluate 100 data every 1000 steps
    #                     break
    #                 losses, infos = tr_agent.eval_func(test_data)
    #                 # r
    #                 if cfg.pred_bb:
    #                     r_mean_err.append(check_r_bb(test_data, tr_agent))
    #                 elif cfg.pred_6d:
    #                     pass
    #                 else:
    #                     pass
    #                 # t
    #                 t_mean_err.append(check_t(test_data, tr_agent))
    #
    #             r_err = np.array(r_mean_err).mean(axis=0)
    #             t_err = np.array(t_mean_err).mean(axis=0)
    #
    #             if cfg.use_wandb:
    #                 if cfg.pred_bb:
    #                     wandb.log({f'test/vertice1_err': r_err[0], 'step': clock.step})
    #                     wandb.log({f'test/vertice2_err': r_err[1], 'step': clock.step}) # max
    #                 elif cfg.pred_6d:
    #                     pass
    #                 else:
    #                     pass
    #                 if cfg.use_objective_T:
    #                     wandb.log({f'test/center_err': t_err, 'step': clock.step})
    #         clock.tick()
    #         if clock.step % cfg.save_step_frequency == 0:
    #             tr_agent.save_ckpt('latest')
    #
    #     tr_agent.update_learning_rate()
    #     clock.tock()
    #
    #     if clock.epoch % cfg.save_frequency == 0:
    #         tr_agent.save_ckpt()
    #     tr_agent.save_ckpt('latest')


if __name__ == '__main__':
    main()
