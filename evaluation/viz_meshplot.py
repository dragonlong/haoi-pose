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

def get_index_per_category(generation_dir, ShapeNetCore_dir, id):
    # pred_path
    # /groups/CESCA-CV/ICML2021/model/obman/2.06/generation/meshes/0
    input_path = f'{generation_dir}/input/{categories_list.index(id)}'
    pred_path = f'{generation_dir}/meshes/{categories_list.index(id)}'
    mesh_names= os.listdir(pred_path)

    # data path
    gt_path = f'{ShapeNetCore_dir}/{id}'

    # loop by predictions
    pairs_dict = {}
    for mesh_name in mesh_names:
        key = mesh_name.split('.')[0]
        input_ply= input_path + f'/{key}.ply'
        pred_obj = pred_path + f'/{mesh_name}'
        gt_obj   = gt_path + f'/{key}/models/model_normalized.obj'
        pairs_dict[key] = [input_ply, pred_obj, gt_obj]

    return pairs_dict

if __name__ == '__main__':
    generation_dir = '/home/dragon/Documents/ICML2021/model/obman/2.06/generation'
    use_category_id= True
    if use_category_id:
        category_ids    = [categories['bottle']]
    else:
        category_ids    = [id for key, id in categories.items()]
    print(category_ids)
    ShapeNetCore_dir    = group_path + '/external/ShapeNetCore.v2'
    num = 0
    start_num = 20
    end_num   = 30
    for id in category_ids:
        pairs_dict = get_index_per_category(generation_dir, ShapeNetCore_dir, id)
        for instance_id, index_pair in pairs_dict.items():
            num +=1
            if num < start_num:
                continue
            if num > end_num:
                break
            input_pts, _ = igl.read_triangle_mesh(index_pair[0])
            v1, f1 = igl.read_triangle_mesh(index_pair[1])
            v2, f2 = igl.read_triangle_mesh(index_pair[2])
            p = subplot(input_pts, c=None, shading={"point_size": 0.2}, s=[1, 3, 0])
            subplot1 = subplot(v1, f1, c=None, shading={"point_size": 0.2}, s=[1, 3, 1], data=p)
            subplot2 = subplot(v2, f2, c=None, shading={"point_size": 0.2}, s=[1, 3, 2], data=p)
    #         p.save("test2.html")
