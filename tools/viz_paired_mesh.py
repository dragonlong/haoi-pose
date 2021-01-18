import os
import glob
import time
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import hydra
from hydra import utils
from omegaconf import DictConfig, ListConfig, OmegaConf
import torch
import igl
import meshplot
from meshplot import plot, subplot, interact

# custom
import __init__
from global_info import global_info
from dataset.obman_parser import ObmanParser
from utils.checkpoints import CheckpointIO
from common import bp

# try using custom packages
infos     = global_info()
my_dir    = infos.base_path
group_path= infos.group_path
second_path = infos.second_path
render_path = infos.render_path
viz_path  = infos.viz_path
hand_mesh = infos.hand_mesh
hand_urdf = infos.hand_urdf
grasps_meta  = infos.grasps_meta
mano_path    = infos.mano_path

whole_obj = infos.whole_obj
part_obj  = infos.part_obj
obj_urdf  = infos.obj_urdf

categories = infos.categories
categories_list = infos.categories_list

#
# def get_pairs(index_pair):
#     # fetch all needed
#     v, f = igl.read_triangle_mesh()
#     A1, B1 = 1, 2
#
#     return [[A, B], [A1, B1]]

def get_index_per_category(generation_dir, ShapeNetCore_dir, id):
    # pred_path
    # /groups/CESCA-CV/ICML2021/model/obman/2.06/generation/meshes/0
    pred_path = f'{generation_dir}/meshes/{categories_list.index(id)}'
    mesh_names= os.listdir(pred_path)

    # data path
    gt_path = f'{ShapeNetCore_dir}/{id}'

    # loop by predictions
    pairs_dict = {}
    for mesh_name in mesh_names:
        key = mesh_name.split('.')[0]
        pred_obj = pred_path + f'/{mesh_name}'
        gt_obj   = gt_path + f'/{key}/models/model_normalized.obj'
        pairs_dict[key] = [pred_obj, gt_obj]

    return pairs_dict

@hydra.main(config_path="../config/occupancy.yaml")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    #>>>>>>>>>>>>>>>>> setting <<<<<<<<<<<<<<<<<<< #
    os.chdir(utils.get_original_cwd())
    os.environ['MASTER_ADDR'] = '127.0.0.1'

    is_cuda = (torch.cuda.is_available() and not cfg.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")
    t0 = time.time()
    infos       = global_info()
    data_infos  = infos.datasets[cfg.item]

    cfg.root_data   = infos.second_path + '/data'
    cfg.log_dir     = infos.second_path + cfg.log_dir

    # Shorthands
    out_dir    = cfg.log_dir
    print('checking out_dir:  ', out_dir)
    generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
    if cfg.target_category:
        category_ids    = [categories[cfg.target_category]]
    else:
        category_ids    = [id for key, id in categories.items()]
    print(category_ids)
    ShapeNetCore_dir    = group_path + '/external/ShapeNetCore.v2'
    for id in category_ids:
        pairs_dict = get_index_per_category(generation_dir, ShapeNetCore_dir, id)
        for instance_id, index_pair in pairs_dict.items():
            v1, f1 = igl.read_triangle_mesh(index_pair[0])
            v2, f2 = igl.read_triangle_mesh(index_pair[1])
            meshplot.offline()
            # p = plot(v1, f, shading={"point_size": 0.2})
            # p.add_points(pts, shading={"point_size": 0.02, "point_color": "blue"})
            p = subplot(v1, f1, c=None, shading={"point_size": 0.2}, s=[1, 2, 0])
            subplot2 = subplot(v2, f2, c=None, shading={"point_size": 0.2}, s=[1, 2, 1], data=p)
            p.save("test2.html")
            bp()
            # p.add_edges(v_box, f_box, shading={"line_color": "red"});
            # add_edges(vertices, edges, shading={}, obj=None)
            # add_lines(beginning, ending, shading={}, obj=None)
            # add_mesh(v, f, c=None, uv=None, shading={})
            # add_points(points, shading={}, obj=None)
            # add_text(text, shading={})
            # remove_object(obj_id)

            # reset()
            # to_html()
            # update()




if __name__ == '__main__':
    main()
