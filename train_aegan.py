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
from common.train_utils import cycle
from models import get_agent

from common.debugger import *
from evaluation.pred_check import post_summary, prepare_pose_eval
from common.algorithms import compute_pose_ransac
from common.d3_utils import axis_diff_degree
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

# given the agent, we could decide the evaluation methods
def eval_func(tr_agent, data, all_rts, cfg):
    tr_agent.val_func(data)
    # print(data.keys())
    target_pts = data['points'].numpy().transpose(0, 2, 1)
    input_pts  = data['G'].ndata['x'].view(target_pts.shape[0], -1, 3).contiguous().permute(0, 2, 1).contiguous().cpu().numpy().transpose(0, 2, 1)
    target_R   = data['R'].cpu().numpy()
    target_T   = data['T'].cpu().numpy()
    # print(input_pts.shape, target_pts.shape, target_R.shape, target_T.shape)
    if 'C' in data:
        target_C   = data['C'].cpu().numpy()
    if 'pose' in cfg.task:
        pred_dict = {}
        if cfg.pred_nocs:
            pred_dict['N'] = tr_agent.output_N.cpu().detach().numpy().transpose(0, 2, 1) # B, N, 3
            # ransac for pose
            for m in range(target_R.shape[0]):
                basename  = f'{cfg.iteration}_' + data['id'][m] + '_' + categories[cfg.target_category]
                nocs_gt   = target_pts[m]
                nocs_pred = np.concatenate([2*np.ones_like(pred_dict['N'][m]), pred_dict['N'][m]], axis=1)
                part_idx_list_pred = [None, np.arange(nocs_pred.shape[0])]
                part_idx_list_gt   = [None, np.arange(nocs_gt.shape[0])]
                rts_dict = compute_pose_ransac(nocs_gt, nocs_pred, input_pts[m], part_idx_list_pred, num_parts, basename, r_raw_err, t_raw_err, s_raw_err, \
                        partidx_gt=part_idx_list_gt, target_category=cfg.target_category, is_special=cfg.is_special, verbose=False)
                rts_dict['pred']   = nocs_pred[:, 3:]
                rts_dict['gt']     = nocs_gt
                rts_dict['in']     = input_pts[m]
                all_rts[basename]  = rts_dict
        else:
            if cfg.rotation_use_dense:
                pred_dict['R'] = tr_agent.output_R.cpu().detach().numpy()
            else:
                pred_dict['R'] = tr_agent.output_R_pooled.cpu().detach().numpy()
            pred_dict['T'] = tr_agent.output_T.cpu().detach().numpy().transpose(0, 2, 1)

            # voting to get the final predictions
            for m in range(target_R.shape[0]):
                basename  = f'{cfg.iteration}_' + data['id'][m] + '_' + categories[cfg.target_category]
                scale_dict = {'gt': [], 'baseline': [], 'nonlinear': []}
                r_dict     = {'gt': [], 'baseline': [], 'nonlinear': []}
                t_dict     = {'gt': [], 'baseline': [], 'nonlinear': []}
                xyz_err    = {'baseline': [], 'nonlinear': []}
                rpy_err    = {'baseline': [], 'nonlinear': []}
                scale_err  = {'baseline': [], 'nonlinear': []}

                rpy_err['baseline'] = axis_diff_degree(pred_dict['R'][m], target_R[m])
                pred_center= input_pts[m] - pred_dict['T'][m]
                gt_center  = input_pts[m] - target_T[m]
                xyz_err['baseline'] = np.linalg.norm(np.mean(pred_center, axis=0) - np.mean(gt_center, axis=0))
                r_dict['baseline'].append(pred_dict['R'][m])
                t_dict['baseline'].append(pred_center)
                scale_dict['gt'].append(1)

                rts_dict = {}
                rts_dict['scale']   = scale_dict
                rts_dict['rotation']      = r_dict
                rts_dict['translation']   = t_dict
                rts_dict['xyz_err']   = xyz_err
                rts_dict['rpy_err']   = rpy_err
                rts_dict['scale_err'] = scale_err
                rts_dict['in']     = input_pts[m]
                all_rts[basename]  = rts_dict

        return rts_dict, pred_dict, basename

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

    parser = ObmanParser(cfg)
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
        pbar = tqdm(val_loader)
        if 'partial' in cfg.task:
            num_iteration = 1
        else:
            num_iteration = 10
        for iteration in range(num_iteration):
            cfg.iteration = iteration
            for b, data in enumerate(pbar):
                rts_dict, pred_dict, basename = eval_func(tr_agent, data, all_rts, cfg)
                save_offline =False
                if save_offline:
                    if cfg.task == 'adversarial_adaptation':
                        outputs = tr_agent.fake_pc
                        for m in range(data["raw"].shape[0]):
                            model_id  = data["raw_id"][m]
                            taxonomy_id = categories[cfg.target_category]
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
                            taxonomy_id = categories[cfg.target_category]
                            save_name = f'{cfg.log_dir}/generation/{cfg.split}/input_{taxonomy_id}_{model_id}.npy'
                            save_for_viz(['points', 'labels'], [data["points"][m].cpu().numpy().T, np.ones((data["points"][m].shape[1]))], save_name, type='np')
                            save_name = f'{cfg.log_dir}/generation/{cfg.split}/{cfg.module}_{taxonomy_id}_{model_id}.npy'
                            save_for_viz(['points', 'labels'], [outputs[m].cpu().numpy().T, np.ones((outputs[m].cpu().numpy().shape[1]))], save_name, type='np')
        post_summary(all_rts, file_name, args=cfg)
        return

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

                if cfg.MODEL.num_channels_R > 1:
                    track_dict.update({'mode_accuracy': [], 'chosenR': []})

                for num, test_data in enumerate(test_loader):
                    # print('--going over ', num)
                    if num > 100: # we only evaluate 100 data every 1000 steps
                        break
                    losses, infos = tr_agent.eval_func(test_data)
                    degree_err    = tr_agent.degree_err.cpu().detach().numpy()
                    best100_ind   = np.argsort(degree_err, axis=1)  # [B, N]
                    best100_ind = best100_ind[:, :100]
                    best100_err = degree_err[np.arange(best100_ind.shape[0]).reshape(-1, 1), best100_ind].mean()
                    # best100_err   = degree_err[0, best100_ind[0][:100]].mean() + degree_err[1, best100_ind[1][:100]].mean()
                    # 1. whole R loss in degree;
                    track_dict['averageR'].append(degree_err.mean())
                    # 2. better R estimation;
                    track_dict['100bestR'].append(best100_err)
                    # 3. more confident estimations
                    if cfg.MODEL.num_channels_R > 1:
                        mode_acc = tr_agent.classifyM_acc.cpu().detach().numpy().mean()
                        chosen_deg_err = tr_agent.degree_err_chosen.cpu().detach().numpy().mean()
                        track_dict['mode_accuracy'].append(mode_acc)
                        track_dict['chosenR'].append(chosen_deg_err)
                # print('>>>>>>during testing: ', np.array(track_dict['averageR']).mean(), np.array(track_dict['100bestR']).mean())
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
