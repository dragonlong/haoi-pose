import torch
import os
import shutil
import argparse
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd
import hydra
from hydra import utils
import logging
from omegaconf import DictConfig, ListConfig, OmegaConf
# custom
from global_info import global_info
from utils import config
from dataset.dataset_parser import DatasetParser
from common.train_utils import CheckpointIO
from utils.external.io import export_pointcloud
from utils.external.visualize import visualize_data
from utils.external.voxels import VoxelGrid
from common import bp
#
def breakpoint():
    import pdb;pdb.set_trace()

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
    out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
    out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')

    input_type = cfg['data']['input_type']
    vis_n_outputs = cfg['generation']['vis_n_outputs']
    if vis_n_outputs is None:
        vis_n_outputs = -1

    parser = DatasetParser(cfg)
    val_dataset   = parser.valid_dataset

    # Model
    model = config.get_model(cfg, device=device, dataset=val_dataset)

    checkpoint_io = CheckpointIO(out_dir, model=model)
    checkpoint_io.load(cfg['test']['model_file'])

    # Generator
    generator = config.get_generator(model, cfg, device=device)

    # Determine what to generate
    generate_mesh = cfg['generation']['generate_mesh']
    generate_pointcloud = cfg['generation']['generate_pointcloud']

    if generate_mesh and not hasattr(generator, 'generate_mesh'):
        generate_mesh = False
        print('Warning: generator does not support mesh generation.')

    if generate_pointcloud and not hasattr(generator, 'generate_pointcloud'):
        generate_pointcloud = False
        print('Warning: generator does not support pointcloud generation.')

    # Loader
    test_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=8, shuffle=False)

    # Statistics
    time_dicts = []

    # Generate
    model.eval()

    # Count how many models already created
    model_counter = defaultdict(int)

    for it, data in enumerate(tqdm(test_loader)):
        # Output folders
        mesh_dir = os.path.join(generation_dir, 'meshes')
        pointcloud_dir = os.path.join(generation_dir, 'pointcloud')
        in_dir = os.path.join(generation_dir, 'input')
        generation_vis_dir = os.path.join(generation_dir, 'vis')

        # Get index etc.
        categories = val_dataset.categories
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
            in_dir = os.path.join(in_dir, str(category_id))

            folder_name = str(category_id)
            if category_name != 'n/a':
                folder_name = str(folder_name) + '_' + category_name.split(',')[0]

            generation_vis_dir = os.path.join(generation_vis_dir, folder_name)

        # Create directories if necessary
        if vis_n_outputs >= 0 and not os.path.exists(generation_vis_dir):
            os.makedirs(generation_vis_dir)

        if generate_mesh and not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)

        if generate_pointcloud and not os.path.exists(pointcloud_dir):
            os.makedirs(pointcloud_dir)

        if not os.path.exists(in_dir):
            os.makedirs(in_dir)

        # Timing dict
        time_dict = {
            'idx': idx,
            'class id': category_id,
            'class name': category_name,
            'modelname': modelname,
        }
        time_dicts.append(time_dict)

        # Generate outputs
        out_file_dict = {}

        # Also copy ground truth
        if cfg['generation']['copy_groundtruth']:
            modelpath = os.path.join(
                dataset.dataset_folder, category_id, modelname,
                cfg['data']['watertight_file'])
            out_file_dict['gt'] = modelpath

        if generate_mesh:
            t0 = time.time()
            if cfg['generation']['sliding_window']:
                if it == 0:
                    print('Process scenes in a sliding-window manner')
                out = generator.generate_mesh_sliding(data)
            else:
                out = generator.generate_mesh(data)
            time_dict['mesh'] = time.time() - t0

            # Get statistics
            try:
                mesh, stats_dict = out
            except TypeError:
                mesh, stats_dict = out, {}
            time_dict.update(stats_dict)

            # Write output
            mesh_out_file = os.path.join(mesh_dir, '%s.off' % modelname)
            mesh.export(mesh_out_file)
            out_file_dict['mesh'] = mesh_out_file

        if generate_pointcloud:
            t0 = time.time()
            pointcloud = generator.generate_pointcloud(data)
            time_dict['pcl'] = time.time() - t0
            pointcloud_out_file = os.path.join(
                pointcloud_dir, '%s.ply' % modelname)
            export_pointcloud(pointcloud, pointcloud_out_file)
            out_file_dict['pointcloud'] = pointcloud_out_file

        if cfg['generation']['copy_input']:
            # Save inputs
            if input_type == 'voxels':
                inputs_path = os.path.join(in_dir, '%s.off' % modelname)
                inputs = data['inputs'].squeeze(0).cpu()
                voxel_mesh = VoxelGrid(inputs).to_mesh()
                voxel_mesh.export(inputs_path)
                out_file_dict['in'] = inputs_path
            elif input_type ==  'pointcloud_crop':
                inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
                inputs = data['inputs'].squeeze(0).cpu().numpy()
                export_pointcloud(inputs, inputs_path, False)
                out_file_dict['in'] = inputs_path
            elif input_type == 'pointcloud' or 'partial_pointcloud':
                inputs_path = os.path.join(in_dir, '%s.ply' % modelname)
                inputs = data['inputs'].squeeze(0).cpu().numpy()
                export_pointcloud(inputs, inputs_path, False)
                out_file_dict['in'] = inputs_path

        # Copy to visualization directory for first vis_n_output samples
        c_it = model_counter[category_id]
        if c_it < vis_n_outputs:
            # Save output files
            img_name = '%02d.off' % c_it
            for k, filepath in out_file_dict.items():
                ext = os.path.splitext(filepath)[1]
                out_file = os.path.join(generation_vis_dir, '%02d_%s%s'
                                        % (c_it, k, ext))
                shutil.copyfile(filepath, out_file)

        model_counter[category_id] += 1

    # Create pandas dataframe and save
    time_df = pd.DataFrame(time_dicts)
    time_df.set_index(['idx'], inplace=True)
    time_df.to_pickle(out_time_file)

    # Create pickle files  with main statistics
    time_df_class = time_df.groupby(by=['class name']).mean()
    time_df_class.to_pickle(out_time_file_class)

    # Print results
    time_df_class.loc['mean'] = time_df_class.mean()
    print('Timings [s]:')
    print(time_df_class)
    print(categories)
#
if __name__ == '__main__':
    main()
