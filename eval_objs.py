import argparse
import os
import time
from tqdm import tqdm
import pandas as pd
import trimesh
import torch
import hydra
from hydra import utils
import logging
from omegaconf import DictConfig, ListConfig, OmegaConf

# custom

import __init__
from global_info import global_info
from dataset.obman_parser import ObmanParser
from utils import config
import dataset
from dataset.obman_parser import ObmanParser
from utils.eval import MeshEvaluator
from utils.external.io import load_pointcloud
from common import bp

def breakpoint():
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

    # Shorthands
    out_dir    = cfg.log_dir
    print('Saving to ', out_dir)
    generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
    if not cfg.eval_input:
        out_file = os.path.join(generation_dir, 'eval_meshes_full.pkl')
        out_file_class = os.path.join(generation_dir, 'eval_meshes.csv')
    else:
        out_file = os.path.join(generation_dir, 'eval_input_full.pkl')
        out_file_class = os.path.join(generation_dir, 'eval_input.csv')

    parser = ObmanParser(cfg)
    val_dataset   = parser.valid_dataset

    # Evaluator
    evaluator = MeshEvaluator(n_points=100000)

    # Loader
    test_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=8, shuffle=False)

    # Evaluate all classes
    eval_dicts = []
    categories = val_dataset.categories
    print(f'Evaluating meshes... {categories}')
    for it, data in enumerate(tqdm(test_loader)):
        if data is None:
            print('Invalid data.')
            continue

        # Output folders
        if not cfg.eval_input:
            mesh_dir = os.path.join(generation_dir, 'meshes')
            pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
        else:
            mesh_dir = os.path.join(generation_dir, 'input')
            pointcloud_dir = os.path.join(generation_dir, 'input')

        # Get index etc.
        idx = data['idx'].item()
        try:
            model_path = val_dataset.pose_dataset.obj_paths[idx]
            category_name = model_path.split('/')[-4]
            modelname = model_path.split('/')[-3]
            category_id = categories.index(category_name)
        except AttributeError:
            category_name = 'n/a'
            modelname = 'n/a'

        if category_id != 'n/a':
            mesh_dir = os.path.join(mesh_dir, str(category_id))
            pointcloud_dir = os.path.join(pointcloud_dir, str(category_id))

        # Evaluate
        pointcloud_tgt = data['pointcloud_chamfer'].squeeze(0).numpy()
        normals_tgt = data['pointcloud_chamfer.normals'].squeeze(0).numpy()
        points_tgt = data['points_iou'].squeeze(0).numpy()
        occ_tgt = data['points_iou.occ'].squeeze(0).numpy()
        # Evaluating mesh and pointcloud
        # Start row and put basic informatin inside
        eval_dict = {
            'idx': idx,
            'class id': category_id,
            'class name': category_name,
            'modelname': modelname,
        }
        eval_dicts.append(eval_dict)

        # Evaluate mesh
        if cfg['test']['eval_mesh']:
            mesh_file = os.path.join(mesh_dir, '%s.off' % modelname)
            if os.path.exists(mesh_file):
                # try:
                mesh = trimesh.load(mesh_file, process=False)
                eval_dict_mesh = evaluator.eval_mesh(mesh, pointcloud_tgt, normals_tgt, points_tgt, occ_tgt, remove_wall=cfg['test']['remove_wall'])
                for k, v in eval_dict_mesh.items():
                    eval_dict[k + ' (mesh)'] = v
                # except Exception as e:
                #     print("Error: Could not evaluate mesh: %s" % mesh_file)
            else:
                print('Warning: mesh does not exist: %s' % mesh_file)

        # Evaluate point cloud
        if cfg['test']['eval_pointcloud']:
            pointcloud_file = os.path.join(
                pointcloud_dir, '%s.ply' % modelname)

            if os.path.exists(pointcloud_file):
                pointcloud = load_pointcloud(pointcloud_file)
                eval_dict_pcl = evaluator.eval_pointcloud(
                    pointcloud, pointcloud_tgt)
                for k, v in eval_dict_pcl.items():
                    eval_dict[k + ' (pcl)'] = v
            else:
                print('Warning: pointcloud does not exist: %s'
                        % pointcloud_file)

    # Create pandas dataframe and save
    eval_df = pd.DataFrame(eval_dicts)
    eval_df.set_index(['idx'], inplace=True)
    eval_df.to_pickle(out_file)

    # Create CSV file  with main statistics
    eval_df_class = eval_df.groupby(by=['class name']).mean()
    eval_df_class.to_csv(out_file_class)

    # Print results
    eval_df_class.loc['mean'] = eval_df_class.mean()
    print(eval_df_class)

if __name__ == '__main__':
    main()
