import hydra
from PIL import Image
import os
import os.path
from os.path import join as pjoin
import numpy as np
import numpy.ma as ma
import scipy.io as scio
import sys
from tqdm import tqdm
import argparse
import torch
sys.path.insert(0, '/orion/u/yijiaw/projects/haoi/code/haoi-pose')
from models.pointnet_lib.pointnet2_modules import farthest_point_sample

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


def mean_shift(data, radius=5.0):
    clusters = []
    for i in range(len(data)):
        cluster_centroid = data[i]
        cluster_frequency = np.zeros(len(data))
        # Search points in circle
        while True:
            temp_data = []
            for j in range(len(data)):
                v = data[j]
                # Handle points in the circles
                if np.linalg.norm(v - cluster_centroid) <= radius:
                    temp_data.append(v)
                    cluster_frequency[i] += 1
            # Update centroid
            old_centroid = cluster_centroid
            new_centroid = np.average(temp_data, axis=0)
            cluster_centroid = new_centroid
            # Find the mode
            if np.array_equal(new_centroid, old_centroid):
                break
        # Combined 'same' clusters
        has_same_cluster = False
        for cluster in clusters:
            if np.linalg.norm(cluster['centroid'] - cluster_centroid) <= radius:
                has_same_cluster = True
                cluster['frequency'] = cluster['frequency'] + cluster_frequency
                break
        if not has_same_cluster:
            clusters.append({
                'centroid': cluster_centroid,
                'frequency': cluster_frequency
            })

    print('clusters (', len(clusters), '): ', clusters)
    clustering(data, clusters)
    return clusters


# Clustering data using frequency
def clustering(data, clusters):
    t = []
    for cluster in clusters:
        cluster['data'] = []
        t.append(cluster['frequency'])
    t = np.array(t)
    # Clustering
    for i in range(len(data)):
        column_frequency = t[:, i]
        cluster_index = np.where(column_frequency == np.max(column_frequency))[0][0]
        clusters[cluster_index]['data'].append(data[i])


def get_per_instance_lists(mode, root, minimum_num_pt=50):
    cur_path = os.path.abspath(os.path.dirname(__file__))
    config_path = '/'.join(cur_path.split('/')[:-1] + ['config/datasets/ycb_config'])
    data_list_path = pjoin(config_path, f'{mode}_data_list.txt')
    input_file = open(data_list_path)
    list = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]
        list.append(input_line)
    input_file.close()

    list_dict = {i: [] for i in range(1, 22)}

    for filename in list:
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(root, filename)))
        label = np.array(Image.open('{0}/{1}-label.png'.format(root, filename)))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(root, filename))
        obj = meta['cls_indexes'].flatten().astype(np.int32)

        for idx in range(len(obj)):
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            if len(mask.nonzero()[0]) > minimum_num_pt:
                list_dict[obj[idx]].append(filename)

    per_instance_dir = pjoin(config_path, f'per_instance_{mode}_list')
    if not os.path.exists(per_instance_dir):
        os.makedirs(per_instance_dir)
    for instance, data_list in list_dict.items():
        with open(pjoin(per_instance_dir, f'{instance}.txt'), 'w') as f:
            for filename in data_list:
                print(filename, file=f)


def get_per_instance_pc(mode, root, instance, num_pt=2048, minimum_num_pt=50):
    cam_cx_1 = 312.9869
    cam_cy_1 = 241.3109
    cam_fx_1 = 1066.778
    cam_fy_1 = 1067.487

    cam_cx_2 = 323.7872
    cam_cy_2 = 279.6921
    cam_fx_2 = 1077.836
    cam_fy_2 = 1078.189

    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    cur_path = os.path.abspath(os.path.dirname(__file__))
    config_path = '/'.join(cur_path.split('/')[:-1] + ['config/datasets/ycb_config'])
    # data_list_path = pjoin(config_path, f'{mode}_data_list.txt')
    data_list_path = pjoin(config_path, 'back_up', f'per_instance_{mode}_list', f'{instance}.txt')
    input_file = open(data_list_path)
    list = [line for line in [line.strip() for line in input_file.readlines()] if len(line)]
    input_file.close()

    valid_list = []

    output_path = pjoin(root, 'cloud', f'{instance}')
    os.makedirs(output_path, exist_ok=True)

    for filename in tqdm(list):
        if filename[:8] != 'data_syn' and int(filename[5:9]) >= 60:
            cam_cx = cam_cx_2
            cam_cy = cam_cy_2
            cam_fx = cam_fx_2
            cam_fy = cam_fy_2
        else:
            cam_cx = cam_cx_1
            cam_cy = cam_cy_1
            cam_fx = cam_fx_1
            cam_fy = cam_fy_1

        depth = np.array(Image.open('{0}/{1}-depth.png'.format(root, filename)))
        label = np.array(Image.open('{0}/{1}-label.png'.format(root, filename)))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(root, filename))
        obj = meta['cls_indexes'].flatten().astype(np.int32)
        ins_idx = [idx for idx in range(len(obj)) if obj[idx] == instance]
        if len(ins_idx) == 0:
            continue
        ins_idx = ins_idx[0]

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(label, instance))
        mask = mask_label * mask_depth
        if len(mask.nonzero()[0]) < minimum_num_pt:
            continue

        valid_list.append(filename)

        rmin, rmax, cmin, cmax = get_bbox(mask_label)

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

        cam_scale = meta['factor_depth'][0][0]
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        if len(cloud) > num_pt:
            torch_cloud = torch.from_numpy(cloud).float().to(device).unsqueeze(0)  # [1, N, 3]
            fps_idx = farthest_point_sample(torch_cloud, num_pt)  # [B, M]
            torch_cloud = torch_cloud[0][fps_idx[0]]  # [M, 3]
            cloud = torch_cloud.cpu().numpy()

        output_folders, output_name = filename.split('/')[:-1], filename.split('/')[-1]
        cur_output_path = pjoin(output_path, *output_folders)
        os.makedirs(cur_output_path, exist_ok=True)
        np.savez_compressed(pjoin(cur_output_path, f'{output_name}.npz'), points=cloud)

    data_list_path = pjoin(config_path, f'per_instance_{mode}_list', f'{instance}.txt')
    with open(data_list_path, 'w') as f:
        for filename in valid_list:
            print(filename, file=f)


@hydra.main(config_path="../config/completion.yaml")
def main(cfg):
    for mode in ['train', 'test']:
        get_per_instance_pc(mode, root=cfg.ycb_root, instance=cfg.instance)
    # get_per_instance_lists(mode, root=cfg.ycb_root)


if __name__ == '__main__':
    main()
