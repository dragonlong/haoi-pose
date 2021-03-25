from os import makedirs, remove
from os.path import exists, join
from copy import deepcopy
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
from equivariant_attention.from_se3cnn.SO3 import rot
from se3net import equivariance_test
from collections import OrderedDict
from tqdm import tqdm
from sklearn.externals import joblib
from dataset.dataset_parser import DatasetParser
from dataset.modelnet40_parser import ModelParser
from common.train_utils import cycle

from models.ae_gan.networks_ae import BuildGraph
from models.base import BaseAgent
from models.ae_gan import get_network
from utils.emd import earth_mover_distance
from models.losses import loss_geodesic, compute_vect_loss, compute_1vN_nocs_loss, compute_miou_loss
from common.yj_pose import compute_pose_diff, rot_diff_degree

from common.debugger import *
from evaluation.pred_check import post_summary, prepare_pose_eval
from common.algorithms import compute_pose_ransac
from common.d3_utils import compute_rotation_matrix_from_ortho6d, axis_diff_degree, rot_diff_rad, rotate_about_axis
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
def set_feat(G, R, num_features=1):
    G.edata['d'] = G.edata['d'] @ R
    if 'w' in G.edata:
        G.edata['w'] = torch.rand((G.edata['w'].size(0), 0))

    G.ndata['x'] = G.ndata['x'] @ R
    G.ndata['f'] = torch.ones((G.ndata['f'].size(0), num_features, 1))
    # # print(G)

    features = {'0': G.ndata['f']}
    return G, features

def check_r_bb(test_data, tr_agent):
    BS = 2
    input_pts     = test_data['G'].ndata['x'].view(BS, -1, 3).contiguous().cpu().numpy() # B, N, 3
    pred_vertices = tr_agent.output_R.permute(0, 2, 3, 1).cpu().numpy() + input_pts[:, :, np.newaxis, :]# Bs, 3, N, M --> B, N, M, 3
    target_R      = test_data['R'].numpy() + input_pts[:, :, np.newaxis, :] # B, N, M, 3
    return np.linalg.norm(pred_vertices - target_R, axis=-1).mean(axis=0).mean(axis=0)

def check_t(data, tr_agent):
    BS = 2
    input_pts     = test_data['G'].ndata['x'].view(BS, -1, 3).contiguous().cpu().numpy() # B, N, 3
    gt_center     = input_pts - test_data['T'].cpu().numpy()
    pred_center   = input_pts - tr_agent.output_T.cpu().detach().numpy().transpose(0, 2, 1)

    return np.linalg.norm(np.mean(pred_center, axis=1) - np.mean(gt_center, axis=1))

def get_single_data(category_name='bowl', augment=False):

    instance_name = '002'
    train_pts= []
    targets = []
    num = 2
    if augment:
        num = 20
    for i in range(num):
        # fn  = [category_name, f'{my_dir}/data/modelnet40_normal_resampled/airplane/airplane_0002.txt']
        fn  = [category_name, f'{my_dir}/data/modelnet40_normal_resampled/{category_name}/{category_name}_0002.txt']
        point_normal_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        full_pts  = np.copy(point_normal_set[:, 0:3])
        r = np.eye(3).astype(np.float32)

        if augment:
            tx, ty, tz = np.random.rand(1, 3)[0] * 180
            r = rot(tx, ty, tz).cpu().numpy()
            full_pts = np.matmul(full_pts, r.T)

        train_pts.append(full_pts)
        targets.append(r.T) # r.T
    print('data loading ...  ready')
    return np.stack(train_pts, axis=0), np.stack(targets, axis=0)

def get_chirality_data(augment=False):
    category_name = 'airplane'
    instance_name = '0002'
    num_points    = 256
    train_pts= []
    targets  = []
    for m in range(4):
        fn  = [category_name, f'{my_dir}/data/modelnet40_normal_resampled/{category_name}/airplane_{instance_name}.txt']
        point_normal_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        # xyz  = np.random.permutation(point_normal_set[:, 0:3])[:256]
        xyz  = point_normal_set[:, 0:3][:256] # fixed
        xyz1 = np.concatenate([xyz, xyz * np.array([[1, 1, -1]])], axis=0)
        r = np.eye(3).astype(np.float32)

        train_pts.append(xyz1)
        targets.append(r)
    print('data loading ...  ready')
    return np.stack(train_pts, axis=0), np.stack(targets, axis=0)

def getitem(pts, targets, builder, npoints=512, fixed_sampling=True):
    input_pts = np.copy(pts).astype(np.float32)
    target_R = torch.from_numpy(targets).cuda()

    if not fixed_sampling:
        input_pts[0] = np.random.permutation(input_pts[0])
        input_pts[1] = np.random.permutation(input_pts[1])
    xyz1 = torch.from_numpy(input_pts[:, :npoints]).cuda()
    g, _     = builder(xyz1)
    data = {}
    data['G'] = g
    data['R'] = target_R
    data['points'] = xyz1
    data['T'] = xyz1
    data['id']= ['0002', '0002']
    data['idx'] = ['0', '0']
    return data, xyz1, target_R

# only get one instance
def get_test_data(nx=8, ny=8, nz=8):
    category_name = 'airplane'
    instance_name = '002'
    fn  = [category_name, f'{my_dir}/data/airplane/airplane/0_0_0.txt']
    train_pts= []
    targets = []
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
                # targets.append(r)
                targets.append(np.eye(3).astype(np.float32))
    print('data loading ...  ready')
    return train_pts, targets

def get_category_data():
    fpath  = f'{my_dir}/data/modelnet40'
    f_train= f'{fpath}/airplane_train_2048.pk'
    f_test = f'{fpath}/airplane_test_2048.pk'

    with open(f_train, "rb") as f:
       train_pts = joblib.load(f)

    with open(f_test, "rb") as obj_f:
        test_pts  = joblib.load(obj_f)

    print(train_pts.shape, test_pts.shape)
    return train_pts, test_pts


def examine_equivariance(x1, x2, f1, f2, target_R):
    x_diff_rotation = x2 - torch.matmul(x1, target_R)
    f_diff_rotation = f2 - f1

    print('x diff mean:', torch.mean(torch.abs(x_diff_rotation)))
    print('x diff max:', torch.max(torch.abs(x_diff_rotation)))
    print('f diff mean:', torch.mean(torch.abs(f_diff_rotation)))
    print('f diff max:', torch.max(torch.abs(f_diff_rotation)))
    # for key, value in xf1.items():
    #     x1_1 = value['1']
    #     x2_1 = xf2[key]['1']
    #     x1_0 = value['0']
    #     x2_0 = xf2[key]['0']
    #     N = int(x1_1.shape[0]/2)
    #     target_R_tiled  = target_R.unsqueeze(1).contiguous().repeat(1, N, 1, 1).contiguous().view(-1, 3, 3).contiguous()
    #     print(f'{key} x diff mean:', torch.mean(torch.abs(x2_1 - torch.matmul(x1_1, target_R_tiled))))
    #     print(f'{key} x diff max:', torch.mean(torch.abs(x2_1 - torch.matmul(x1_1, target_R_tiled))))
    #     print(f'{key} f diff mean:', torch.mean(torch.abs(x2_0 - x1_0)))
    #     print(f'{key} f diff max:', torch.max(torch.abs(x2_0 - x1_0)))
    #     print('')

def examine_unit(data, tr_agent):
    def create_se3_data(data):
        tx, ty, tz = np.random.rand(1, 3)[0] * 180
        R2 = rot(tx, ty, tz).cuda()
        G2, features = set_feat(data['G'], R2)
        xyz2         = G2.ndata['x'].view(2, 512, 3).contiguous()
        target_R     = R2.unsqueeze(0).repeat(2, 1, 1).contiguous()
        print('xyz diff max:', torch.max(torch.abs(xyz2 - torch.matmul(xyz1, target_R))))
        data['G'] = G2
        data['R'] = target_R
        return data, xyz2, target_R
    torch.cuda.empty_cache()
    loss_dict, info_dict = tr_agent.val_func(data)
    x1  = tr_agent.output_R
    f1  = tr_agent.latent_vect['N']
    torch.cuda.empty_cache()
    data, xyz2, target_R = create_se3_data(data)
    loss_dict, info_dict = tr_agent.val_func(data)
    x2 = tr_agent.output_R
    f2 = tr_agent.latent_vect['N']
    examine_equivariance(x1[0], x2[0], f1, f2, target_R[0])

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
    # tr_agent = PointAEPoseAgent(cfg)
    equivariance_test(tr_agent.net.encoder)
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
    # train_pts, test_pts = get_category_data()
    train_pts, train_targets = get_single_data('airplane')
    # test_pts, test_targets = get_test_data(nx=8, ny=1, nz=1)
    test_pts, test_targets =  get_single_data('airplane', augment=True)

    builder = BuildGraph(num_samples=10)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    inputs = []
    BS     = 2
    N  = cfg.num_points
    M  = cfg.num_modes_R
    CS = cfg.MODEL.num_channels_R
    npoints= cfg.num_points
    fixed_sampling = cfg.fixed_sampling

    # start training
    clock = tr_agent.clock #
    epoch_size = max(int(len(train_pts)/2), 1000)
    pbar = tqdm(range(0, epoch_size))
    for e in range(clock.epoch, 200):
        for _, b in enumerate(pbar):
            data, xyz1, target_R = getitem(train_pts[:2], train_targets[:2], builder, npoints=N, fixed_sampling=fixed_sampling)
            torch.cuda.empty_cache()
            loss_dict, info_dict = tr_agent.train_func(data)
            pbar.set_description("EPOCH[{}][{}]".format(0, e)) #
            pbar.set_postfix(OrderedDict({k: v.item() for k, v in  {**loss_dict, **info_dict}.items()}))

            if cfg.vis and clock.step % cfg.vis_frequency == 0:
                tr_agent.visualize_batch(data, "train")

            # if clock.step % cfg.eval_frequency == 0:
            #     track_dict = {'rdiff': [], 'tdiff': [], 'sdiff': [],
            #                   '5deg': [], '5cm': [], '5deg5cm': []}
            #     for num in range(10):
            #         data, xyz1, target_R = getitem(train_pts[:2], train_targets[:2], builder, npoints=N, fixed_sampling=fixed_sampling)
            #         data['idx'] = [b, b]
            #         tr_agent.eval_func(data)
            #         pose_diff = tr_agent.pose_err
            #         for key in ['rdiff', 'tdiff', 'sdiff']:
            #             track_dict[key].append(pose_diff[key].cpu().numpy().mean())
            #         deg = pose_diff['rdiff'] <= 5.0
            #         cm = pose_diff['tdiff'] <= 0.05
            #         degcm = deg & cm
            #         for key, value in zip(['5deg', '5cm', '5deg5cm'], [deg, cm, degcm]):
            #             track_dict[key].append(value.float().cpu().numpy().mean())
            #     if cfg.use_wandb:
            #         for key, value in track_dict.items():
            #             wandb.log({f'test/{key}': np.array(value).mean(), 'step': clock.step})

            if clock.step % cfg.val_frequency == 0:
                inds = random.sample(range(0, len(test_pts)-1), BS)
                data2, xyz2, target_R2 = getitem(test_pts[inds[:]], test_targets[inds[:]], builder, npoints=N, fixed_sampling=fixed_sampling)
                loss_dict, info_dict = tr_agent.val_func(data2)
                if cfg.vis and clock.step % cfg.vis_frequency == 0:
                    tr_agent.visualize_batch(data2, "validation")

            clock.tick()
            if clock.step % cfg.save_step_frequency == 0:
                tr_agent.save_ckpt('latest')

        tr_agent.update_learning_rate()
        clock.tock()

        if clock.epoch % 10 == 0:
            tr_agent.save_ckpt()

if __name__ == '__main__':
    main()
