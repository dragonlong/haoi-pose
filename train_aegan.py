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
from dataset.dataset_parser import DatasetParser
from common.train_utils import cycle
from models import get_agent
from se3net import equivariance_test

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

def set_feat(G, R, num_features=1):
    G.edata['d'] = G.edata['d'] @ R
    if 'w' in G.edata:
        G.edata['w'] = torch.rand((G.edata['w'].size(0), 0))

    G.ndata['x'] = G.ndata['x'] @ R
    G.ndata['f'] = torch.ones((G.ndata['f'].size(0), num_features, 1))

    features = {'0': G.ndata['f']}
    return G, features

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

    #>>>>>>>>>>>>>>>>>>>>>> create network and training agent
    tr_agent = get_agent(cfg)
    if 'se3' in cfg.encoder_type:
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

    #>>>>>>>>>>>>>>>>>>>> dataset
    parser = DatasetParser(cfg)
    train_loader = parser.trainloader
    val_loader   = parser.validloader
    test_loader  = parser.validloader
    valid_dataset= parser.valid_dataset
    dp = valid_dataset.__getitem__(0)
    if cfg.eval_mini or cfg.eval:
        all_rts, file_name, mean_err, r_raw_err, t_raw_err, s_raw_err = prepare_pose_eval(cfg.exp_num, cfg)
        # file_name = file_name.replace('unseen', 'seen180') #'/groups/CESCA-CV/ICML2021/results/test_pred/oracle/0.64_unseen_part_rt_pn_general.npy'
        infos_dict = {'basename': [], 'in': [], 'r_raw': [],
                      'r_gt': [], 't_gt': [], 's_gt': [],
                      'r_pred': [], 't_pred': [], 's_pred': []}
        track_dict = {'rdiff': [], 'tdiff': [], 'sdiff': [],
                      '5deg': [], '5cm': [], '5deg5cm': []}
        num_iteration = 1
        if 'partial' not in cfg.task:
            num_iteration = 2
        for iteration in range(num_iteration):
            cfg.iteration = iteration
            # for num, data in enumerate(train_loader):
            for num, data in enumerate(test_loader):
                if num % 10 == 0:
                    print('checking batch ', num)
                BS = data['points'].shape[0]
                idx = data['idx']
                torch.cuda.empty_cache()
                tr_agent.eval_func(data)
                pose_diff = tr_agent.pose_err
                for key in ['rdiff', 'tdiff', 'sdiff']:
                    track_dict[key] += pose_diff[key].float().cpu().numpy().tolist()
                deg = pose_diff['rdiff'] <= 5.0
                cm = pose_diff['tdiff'] <= 0.05
                degcm = deg & cm
                for key, value in zip(['5deg', '5cm', '5deg5cm'], [deg, cm, degcm]):
                    track_dict[key] += value.float().cpu().numpy().tolist()
                for key, value in tr_agent.pose_info.items():
                    infos_dict[key] += value.float().cpu().numpy().tolist()
                input_pts  = data['G'].ndata['x'].view(BS, -1, 3).contiguous() # B, N, 3
                for m in range(BS):
                    basename   = f'{cfg.iteration}_' + data['id'][m] + f'_' + data['class'][m]
                    infos_dict['basename'].append(basename)
                    infos_dict['in'].append(input_pts[m].cpu().numpy())
                # tr_agent.visualize_batch(data, "test")
        # print
        for key, value in track_dict.items():
            print(key, ':', np.array(value).mean())

        # save
        if cfg.save:
            print('--saving to ', file_name)
            np.save(file_name, arr={'info': infos_dict, 'err': track_dict})

        # visualize distribution
        for key, value in track_dict.items():
            value = np.array(value)
            plot_distribution(value.reshape(-1), labelx=key, labely='frequency', title_name=f'{key}', sub_name=cfg.exp_num, save_fig=True)

        return

    # >>>>>>>>>>> main training
    clock = tr_agent.clock #
    val_loader   = cycle(val_loader)
    for e in range(clock.epoch, cfg.nr_epochs):
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

            if clock.step % cfg.eval_frequency == 0:
                track_dict = {'rdiff': [], 'tdiff': [], 'sdiff': [],
                              '5deg': [], '5cm': [], '5deg5cm': []}
                if cfg.num_modes_R > 1:
                    track_dict.update({'mode_accuracy': [], 'chosenR': []})

                for num, test_data in enumerate(test_loader):
                    if num > 100:
                        break
                    tr_agent.eval_func(test_data)
                    pose_diff = tr_agent.pose_err
                    for key in ['rdiff', 'tdiff', 'sdiff']:
                        track_dict[key].append(pose_diff[key].cpu().numpy().mean())
                    deg = pose_diff['rdiff'] <= 5.0
                    cm = pose_diff['tdiff'] <= 0.05
                    degcm = deg & cm
                    for key, value in zip(['5deg', '5cm', '5deg5cm'], [deg, cm, degcm]):
                        track_dict[key].append(value.float().cpu().numpy().mean())
                if cfg.use_wandb:
                    for key, value in track_dict.items():
                        wandb.log({f'test/{key}': np.array(value).mean(), 'step': clock.step})

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
