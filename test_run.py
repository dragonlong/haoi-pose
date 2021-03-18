import wandb
import glob
import torch
import igl
import meshplot
import time
import numpy as np


import __init__
from global_info import global_info
from common.debugger import *

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

def get_index_per_category(generation_dir, ShapeNetCore_dir, id, module='ae'):
    # pred_path
    # /groups/CESCA-CV/ICML2021/model/obman/2.06/generation/meshes/0
    pred_files = glob.glob(f'{generation_dir}/{module}*')
    mesh_names = [x.split('/')[-1] for x in pred_files]
    print('we have {} data in {}'.format(len(mesh_names), generation_dir))
    gt_path = f'{ShapeNetCore_dir}/{id}'
    pairs_dict = {}
    for mesh_name in mesh_names:
        key = mesh_name.split('.')[0].split('_')[-1]
        input_ply= generation_dir + '/' + mesh_name.replace(module, 'input')
        pred_obj = generation_dir + f'/{mesh_name}'
        gt_obj   = gt_path + f'/{key}/models/model_normalized.obj'
        pairs_dict[key] = [input_ply, pred_obj, gt_obj]

    return pairs_dict

if __name__ == '__main__':
    # exp = '2.6'
    # target_category = 'bottle'
    # exp = '2.7'
    # target_category = 'bottle'
    exp = '2.71'
    target_category = 'jar'
    module= 'gan'
    wandb.init(
      project="haoi-pose",
      notes="check GAN",
      tags=["AE_GAN", "paper2"],
      name=f'{exp}_viz',
      sync_tensorboard=False
    )

    for set_name in ['val', 'train']:
        # 3. Log metrics over time to visualize performance
        generation_dir = f'/groups/CESCA-CV/ICML2021/model/obman/{exp}/generation/{set_name}'
        use_category_id= True
        if use_category_id:
            category_ids    = [categories[target_category]]
        else:
            category_ids    = [id for key, id in categories.items()]
        ShapeNetCore_dir    = group_path + '/external/ShapeNetCore.v2'
        num = 0
        start_num = 0
        end_num   = 100
        for id in category_ids:
            pairs_dict = get_index_per_category(generation_dir, ShapeNetCore_dir, id, module=module)
            for instance_id, index_pair in pairs_dict.items():
                num +=1
                if num < start_num:
                    continue
                if num > end_num:
                    break
                print(index_pair)
                input_pts, _ = igl.read_triangle_mesh(index_pair[0])
                v1, f1 = igl.read_triangle_mesh(index_pair[1])
                v1 = v1 + np.array([0, 1, 0]).reshape(1, -1)
                pts = np.concatenate([input_pts, v1], axis=0)
                bp()
                wandb.log({f"{set_name}: input": wandb.Object3D('points': pts, "vectors": [{"start": [0,0,0], "end": [0.1,0.2,0.5]}] )})
                # wandb.log({f"{set_name}_input": [wandb.Object3D(input_pts)], f"{set_name}_AE_GAN_output": [wandb.Object3D(v1)]})
