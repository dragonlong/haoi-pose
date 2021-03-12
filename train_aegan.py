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
# training: ae_gan
def vote_rotation_ransac_6d(out_R, out_M, gt_R=None, gt_M=None, degree_err=None, use_base=False, thres=5, verbose=True):
    """
    out_R is : [3, N, M]; or [N, M, 3, 3]
    out_M    : [N, M] with scores;
    gt_R     : [3, 1]
    but we only need to output 3;
    voting!!!
    a baseline would be voting for mode + averaging for R;
    """
    # with confidence estimation
    N, M, _, _ = out_R.shape
    # [N, N, M]
    pairwise_dis = np.linalg.norm((out_R[:, np.newaxis, :, :, :] - out_R[np.newaxis, :, :, :, :]).reshape(N, N, M, 9), axis=-1)
    inliers_mask = np.zeros_like(pairwise_dis)
    inliers_mask[pairwise_dis<thres] = 1.0 # [N, N, M]
    #
    score = np.mean(inliers_mask, axis=1) # N, M
    score_weighted = score * out_M # assume we have good predictions, and good predictions are close to each other, while bad predictions just randomly scatter around

    R_best1 = out_R.reshape(-1, 9)[np.argmax(score_weighted), :].reshape(3, 3)
    pred_M  = np.argmax(out_M, axis=-1) # N
    select_M  = np.bincount(pred_M).argmax()
    R_best2 = out_R[:, select_M, :, :][np.argmax(score[:, select_M]), :, :]

    # 1. mode accuracy, best_pred's mode score
    mode_acc =  len(np.where(pred_M==gt_M)[0])/N
    degree_err[degree_err<thres] = 1.0
    degree_err[degree_err>thres] = 0.0       # N, M
    good_ratio = np.mean(degree_err, axis=0) # M

    pred_R = out_R[:, select_M, :, :]
    R_best3= pred_R.mean(axis=0)

    # GT_M + ransac
    pred_R = out_R[:, gt_M[0], :, :]
    R_best4= pred_R.mean(axis=0)

    R_best5 = out_R[:, gt_M[0], :, :][np.argmax(score[:, gt_M[0]]), :, :]

    if verbose:
        print('---mode accuracy is ', mode_acc)
        print('---per mode accuracy: ', good_ratio)
        print('---selected mode is ', select_M)
        print('---r_diff is ', rot_diff_rad(R_best1, gt_R) / np.pi * 180, rot_diff_rad(R_best2, gt_R) / np.pi * 180, rot_diff_rad(R_best3, gt_R) / np.pi * 180, rot_diff_rad(R_best4, gt_R) / np.pi * 180, rot_diff_rad(R_best5, gt_R) / np.pi * 180)
    return R_best1, R_best2, R_best3, R_best4, R_best5, mode_acc, good_ratio

def vote_rotation_ransac(out_R, out_M, gt_R=None, gt_M=None, use_base=False, thres=5, verbose=True):
    """
    out_R is : [3, N, M];
    out_M    : [N, M] with scores;
    gt_R     : [3, 1]
    but we only need to output 3;
    voting!!!
    a baseline would be voting for mode + averaging for R;
    """
    # with confidence estimation
    _, N, M = out_R.shape
    pairwise_dis = np.arccos(np.sum(out_R[:, np.newaxis, :, :] * out_R[:, :, np.newaxis, :], axis=0))* 180 / np.pi
    pairwise_dis[np.isnan(pairwise_dis)] = 0.0
    inliers_mask = np.zeros_like(pairwise_dis)
    inliers_mask[pairwise_dis<thres] = 1.0 # [N, N, M]
    #
    score = np.mean(inliers_mask, axis=1) # N, M
    score_weighted = score * out_M # assume we have good predictions, and good predictions are close to each other, while bad predictions just randomly scatter around
    R_best1 = out_R.reshape(3, -1)[:, np.argmax(score_weighted)]
    pred_M  = np.argmax(out_M, axis=-1) # N
    select_M  = np.bincount(pred_M).argmax()
    R_best2 = out_R[:, :, select_M][:, np.argmax(score[:, select_M])]
    # we could evalaute the voted R error, best_R ratio,
    degree_err  = np.arccos(np.sum(out_R * (gt_R.T)[:, :, np.newaxis], axis=0))* 180 / np.pi
    degree_err[np.isnan(degree_err)] = 0.0
    degree_err[degree_err<thres] = 1.0
    degree_err[degree_err>thres] = 0.0 # N, M
    good_ratio = np.mean(degree_err, axis=0) # M

    mode_acc =  len(np.where(pred_M==gt_M)[0])/N

    pred_R  = out_R[:, :, select_M]
    R_best3 = pred_R.mean(axis=-1)

    if verbose:
        print('---mode accuracy is ', mode_acc)
        print('---per mode good_ratio: ', good_ratio)
        print('---averaged mode is ', select_M)
        print('---r_diff is ', axis_diff_degree(R_best1, gt_R), axis_diff_degree(R_best2, gt_R),  axis_diff_degree(R_best3, gt_R))

    return R_best1, R_best2, R_best3, mode_acc, good_ratio

# given the agent, we could decide the evaluation methods
def eval_func(tr_agent, data, all_rts, cfg, r_raw_err=None, t_raw_err=None):
    tr_agent.visualize_batch(data, "test") # save nocs or degree error
    #
    target_pts = data['points'].numpy().transpose(0, 2, 1)
    input_pts  = data['G'].ndata['x'].view(target_pts.shape[0], -1, 3).contiguous().permute(0, 2, 1).contiguous().cpu().numpy().transpose(0, 2, 1)
    target_R   = data['R'].cpu().numpy()
    target_T   = data['T'].cpu().numpy()
    idx        = data['idx']
    infos_dict = {}
    if 'C' in data:
        target_C   = data['C'].cpu().numpy()
    pred_dict = {}
    infos_dict['mode_acc']      = []
    infos_dict['good_per_mode'] = []
    if cfg.pred_nocs:
        print('Not implemented yet')
    elif cfg.pred_6d:
        if cfg.num_modes_R > 1: # use modes here, default to be dense
            output_R = tr_agent.output_R_full.cpu().detach().numpy() # [BS, 3, N, C]
            BS, M    = output_R.shape[0], cfg.num_modes_R
            gt_M     = tr_agent.target_M.cpu().detach().numpy().reshape(BS, -1)
            output_M = tr_agent.output_M.cpu().detach().numpy().reshape(BS, -1, M)
            degree_err = tr_agent.degree_err_full.cpu().detach().numpy()
            pred_dict['R'] = []
            for m in range(BS):
                bp()
                R1, R2, R3, R4, R5, mode_acc, good_ratio = vote_rotation_ransac_6d(output_R[m], output_M[m], gt_R=target_R[m], gt_M=gt_M[m], degree_err=degree_err[m], use_base=False, thres=0.5, verbose=cfg.eval)
                if cfg.eval_mode_r==0:
                    pred_dict['R'].append(R1) # mixed mode score ransac
                elif cfg.eval_mode_r==1:
                    pred_dict['R'].append(R2) # ransac in best mode
                elif cfg.eval_mode_r==2:
                    pred_dict['R'].append(R3) # mean in best mode
                elif cfg.eval_mode_r==3:
                    pred_dict['R'].append(R4) # mean in best mode
                elif cfg.eval_mode_r==4:
                    pred_dict['R'].append(R5) # mean in best mode

                infos_dict['mode_acc'].append(mode_acc)
                infos_dict['good_per_mode'].append(good_ratio)
    else: # predict R/T
        if cfg.num_modes_R > 1: # use modes here, default to be dense
            output_R = tr_agent.output_R_full.cpu().detach().numpy() # [BS, 3, N, C]
            BS, M    = output_R.shape[0], cfg.num_modes_R
            assert cfg.num_modes_R == cfg.MODEL.num_channels_R, "modes should be equal to channels"
            gt_M     = tr_agent.target_M.cpu().detach().numpy().reshape(BS, -1)
            output_M = tr_agent.output_M.cpu().detach().numpy().reshape(BS, -1, M)
            pred_dict['R'] = []
            for m in range(BS):
                R1, R2, R3, mode_acc, good_ratio = vote_rotation_ransac(output_R[m], output_M[m], gt_R=target_R[m], gt_M=gt_M[m], use_base=False, thres=5, verbose=cfg.eval)
                if cfg.eval_mode_r==0:
                    pred_dict['R'].append(R1) # ransac in best mode
                elif cfg.eval_mode_r==1:
                    pred_dict['R'].append(R2) # mixed mode score ransac
                elif cfg.eval_mode_r==2:
                    pred_dict['R'].append(R3) # mixed mode score ransac
                infos_dict['mode_acc'].append(mode_acc)
                infos_dict['good_per_mode'].append(good_ratio)
        else:
            if cfg.MODEL.rotation_use_dense:
                output_R = tr_agent.output_R_full.cpu().detach().numpy() # [BS, 3, N, C]
                pred_dict['R'] = output_R.mean(axis=-1)                  # all as B, 3, N
            else:
                pred_dict['R'] = tr_agent.output_R_pooled.cpu().detach().numpy() # use pooled
            # output_R = tr_agent.output_R_pooled.cpu().detach().numpy()
            # BS, NC = output_R.shape[0], cfg.MODEL.num_channels_R
            # output_M = tr_agent.output_M.cpu().detach().numpy()
            # pred_dict['M'] = np.argmax(output_M.reshape(BS, -1, NC), axis=-1)
            # pred_dict['M'] = (pred_dict['M'].mean(axis=-1) + 0.5).astype(np.int8)
            # pred_dict['R'] = output_R[np.arange(BS), :, pred_dict['M'][:]]
    pred_dict['T'] = tr_agent.output_T.cpu().detach().numpy().transpose(0, 2, 1)

    # voting to get the final predictions
    for m in range(target_R.shape[0]):
        basename   = f'{cfg.iteration}_' + data['id'][m] + f'{idx[m]}_' + data['class'][m]
        # basename  = f'{cfg.iteration}_' + data['id'][m] + '_' + data['class'][m]
        scale_dict = {'gt': [], 'baseline': [], 'nonlinear': []}
        r_dict     = {'gt': [], 'baseline': [], 'nonlinear': []}
        t_dict     = {'gt': [], 'baseline': [], 'nonlinear': []}
        xyz_err    = {'baseline': [], 'nonlinear': []}
        rpy_err    = {'baseline': [], 'nonlinear': []}
        scale_err  = {'baseline': [], 'nonlinear': []}

        # for r, s
        r_dict['baseline'].append(pred_dict['R'][m])
        scale_dict['gt'].append(1)
        if cfg.pred_6d:
            rpy_err['baseline'] = rot_diff_rad(pred_dict['R'][m], target_R[m], category=categories_id[data['class'][m]]) / np.pi * 180
        else:
            rpy_err['baseline'] = axis_diff_degree(pred_dict['R'][m], target_R[m], category=categories_id[data['class'][m]])

        # for t
        if input_pts[m].shape[0] == pred_dict['T'][m].shape[0]:
            if cfg.pred_center:
                pred_center= pred_dict['T'][m]
                gt_center  = target_T[m]
            else:
                pred_center= input_pts[m] - pred_dict['T'][m]
                gt_center  = input_pts[m] - target_T[m]
            xyz_err['baseline'] = np.linalg.norm(np.mean(pred_center, axis=0) - np.mean(gt_center, axis=0))
            t_dict['baseline'].append(pred_center)
        else:
            xyz_err['baseline'] = 0
            t_dict['baseline'].append(pred_dict['T'][m])

        rts_dict = {}
        rts_dict['scale']     = scale_dict
        rts_dict['rotation']      = r_dict
        rts_dict['translation']   = t_dict
        rts_dict['xyz_err']   = xyz_err
        rts_dict['rpy_err']   = rpy_err
        rts_dict['scale_err'] = scale_err
        rts_dict['in']     = input_pts[m]
        all_rts[basename]  = rts_dict #
        return infos_dict

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
    if cfg.eval:
        # data = next(val_loader)
        """
        """
        num_parts = 2
        all_rts, file_name, mean_err, r_raw_err, t_raw_err, s_raw_err = prepare_pose_eval(cfg.exp_num, cfg)
        infos_all = {}
        pbar = tqdm(val_loader)
        if 'partial' in cfg.task:
            num_iteration = 1
        else:
            num_iteration = 10
        for iteration in range(num_iteration):
            cfg.iteration = iteration
            for b, data in enumerate(pbar):
                if cfg.target_category == 'bottle' and b > 769:
                    break
                tr_agent.val_func(data)
                infos_dict = eval_func(tr_agent, data, all_rts, cfg, r_raw_err, t_raw_err)
                # additional infos to plot hist
                for key, value in infos_dict.items():
                    if key not in infos_all:
                        infos_all[key] = []
                    infos_all[key].append(value)

                save_offline =False
                if save_offline:
                    if cfg.task == 'adversarial_adaptation':
                        outputs = tr_agent.fake_pc
                        for m in range(data["raw"].shape[0]):
                            model_id  = data["raw_id"][m]
                            taxonomy_id = data['class'][m]
                            save_name = f'{cfg.log_dir}/generation/{cfg.split}/input_{taxonomy_id}_{model_id}.npy'
                            save_for_viz(['points', 'labels'], [data["raw"][m].cpu().numpy().T, np.ones((data["raw"][m].shape[1]))], save_name, type='np')
                            save_name = f'{cfg.log_dir}/generation/{cfg.split}/{cfg.module}_{taxonomy_id}_{model_id}.npy'
                            save_for_viz(['points', 'labels'], [outputs[m].cpu().numpy().T, np.ones((outputs[m].cpu().numpy().shape[1]))], save_name, type='np')
                    else:
                        outputs = tr_agent.output_pts
                        latent_vect = tr_agent.latent_vect
                        graphs = dgl.unbatch(data["G"])
                        if cfg.use_wandb:
                            for m in range(data["points"].shape[0]):
                                # target_pts  = data["points"][m].cpu().numpy().T
                                target_pts  = graphs[m].ndata['x'].cpu().numpy()
                                outputs_pts = outputs[m].cpu().numpy().T
                                outputs_pts = outputs_pts + np.array([0, 1, 0]).reshape(1, -1)
                                pts = np.concatenate([target_pts, outputs_pts], axis=0)
                                wandb.log({"input+AE_GAN_output": [wandb.Object3D(pts)], 'step': b*data["points"].shape[0] + m})

                        for m in range(data["points"].shape[0]):
                            model_id  = data['id'][m]
                            taxonomy_id = data['class'][m]
                            save_name = f'{cfg.log_dir}/generation/{cfg.split}/input_{taxonomy_id}_{model_id}.npy'
                            save_for_viz(['points', 'labels'], [data["points"][m].cpu().numpy().T, np.ones((data["points"][m].shape[1]))], save_name, type='np')
                            save_name = f'{cfg.log_dir}/generation/{cfg.split}/{cfg.module}_{taxonomy_id}_{model_id}.npy'
                            save_for_viz(['points', 'labels'], [outputs[m].cpu().numpy().T, np.ones((outputs[m].cpu().numpy().shape[1]))], save_name, type='np')

        post_summary(all_rts, file_name, args=cfg, extra_key=f'{cfg.ckpt}_{cfg.eval_mode_r}')
        for key, value in infos_all.items():
            value = np.array(value)
            if len(value.shape)>2:
                for j in range(value.shape[-1]):
                    plot_distribution(value[:, :, j].reshape(-1), labelx=key, labely='frequency', title_name=f'rotation_error_{key}_{j}', sub_name=cfg.exp_num, save_fig=True)
            else:
                plot_distribution(value.reshape(-1), labelx=key, labely='frequency', title_name=f'rotation_error_{key}', sub_name=cfg.exp_num, save_fig=True)
        return
    best_R_error = 100
    val_loader   = cycle(val_loader)
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

            if clock.step % cfg.eval_frequency == 0:
                track_dict = {'averageR': [], '100bestR': []}
                all_rts, file_name, mean_err, r_raw_err, t_raw_err, s_raw_err = prepare_pose_eval(cfg.exp_num, cfg)
                if cfg.num_modes_R > 1:
                    track_dict.update({'mode_accuracy': [], 'chosenR': []})

                for num, test_data in enumerate(test_loader):
                    if num > 100: # we only evaluate 100 data every 1000 steps
                        break
                    losses, infos = tr_agent.eval_func(test_data)
                    if cfg.pred_nocs:
                        pass
                    if cfg.use_objective_R:
                        degree_err    = tr_agent.degree_err.cpu().detach().numpy().squeeze()
                        best100_ind   = np.argsort(degree_err, axis=1)  # [B, N]
                        best100_ind = best100_ind[:, :100]
                        best100_err = degree_err[np.arange(best100_ind.shape[0]).reshape(-1, 1), best100_ind].mean()
                        track_dict['averageR'].append(degree_err.mean())
                        track_dict['100bestR'].append(best100_err)
                        # 3. more confident estimations
                        if cfg.num_modes_R > 1:
                            mode_acc = tr_agent.classifyM_acc.cpu().detach().numpy().mean()
                            chosen_deg_err = tr_agent.degree_err_chosen.cpu().detach().numpy().mean()
                            track_dict['mode_accuracy'].append(mode_acc)
                            track_dict['chosenR'].append(chosen_deg_err)

                    infos_dict = eval_func(tr_agent, test_data, all_rts, cfg, r_raw_err, t_raw_err) # we will keep adding new evaluations
                # evaluate per category as well
                xyz_err_dict = {}
                rpy_err_dict = {}
                for key, value_dict in all_rts.items():
                    category_name = key.split('_')[-1]
                    if category_name not in xyz_err_dict:
                        xyz_err_dict[category_name] = []
                        rpy_err_dict[category_name] = []
                    if isinstance(value_dict['rpy_err']['baseline'], list):
                        rpy_err_dict[category_name].append(value_dict['rpy_err']['baseline'][0])
                    else:
                        rpy_err_dict[category_name].append(value_dict['rpy_err']['baseline'])
                    if isinstance(value_dict['xyz_err']['baseline'], list):
                        xyz_err_dict[category_name].append(value_dict['xyz_err']['baseline'][0])
                    else:
                        xyz_err_dict[category_name].append(value_dict['xyz_err']['baseline'])

                all_categorys = xyz_err_dict.keys()
                test_infos    = {}
                num_parts     = 1
                for category in all_categorys:
                    print(f'{category}\t{np.array(rpy_err_dict[category]).mean():0.4f}\t{np.array(xyz_err_dict[category]).mean():0.4f}')
                    r_err = np.array(rpy_err_dict[category])
                    num_valid = r_err.shape[0]
                    r_acc = []
                    for j in range(num_parts):
                        idx = np.where(r_err < 5)[0]
                        acc   = len(idx) / num_valid
                        r_acc.append(acc)
                    if cfg.use_objective_R or cfg.pred_nocs:
                        test_infos['rdiff'] = r_err.mean()
                        test_infos['5deg']  = np.array(r_acc).mean()
                    # 5 degrees 5 cm
                    rt_acc = []
                    t_err  = np.array(xyz_err_dict[category])
                    for j in range(num_parts):
                        idx = np.where(r_err < 5)[0]
                        acc   = len(np.where( t_err[idx] < 0.05 )[0]) / num_valid
                        rt_acc.append(acc) # two modes
                    if cfg.use_objective_T:
                        test_infos['tdiff'] = t_err.mean()
                        test_infos['5cm']   = len(np.where(t_err < 0.05)[0]) / t_err.shape[0]

                    if cfg.pred_nocs:
                        test_infos['5deg5cm']  = np.array(rt_acc).mean()

                if np.array(track_dict['averageR']).mean() < best_R_error:
                    best_R_error = np.array(track_dict['averageR']).mean()
                    tr_agent.save_ckpt('best')

                if cfg.use_wandb:
                    for key, value in track_dict.items():
                        wandb.log({f'test/{key}': np.array(value).mean(), 'step': clock.step})
                    for key, value in test_infos.items():
                        wandb.log({f'test/{key}': value, 'step': clock.step})
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
