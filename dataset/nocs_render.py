import trimesh
import numpy as np
import os
import multiprocessing as mp
import gc
import argparse
from tqdm import tqdm
import pyrender
import matplotlib.pyplot as plt
import cv2
from os.path import join as pjoin
import pickle

def bp():
    import pdb;pdb.set_trace()


def get_normalized_mesh(read_path, transform=False):
    m = trimesh.load(read_path)
    if isinstance(m, trimesh.Trimesh):
        m = trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
    else:
        try:
            m = trimesh.util.concatenate([trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in m.geometry.values()])
        except:
            print('obj path', read_path)
            print(m)
    if transform:
        trans = np.eye(4)
        trans[:3, :3] = np.asarray([[0, 1, 0],
                                    [0, 0, 1],
                                    [1, 0, 0]])
        m.apply_transform(trans)
    bbox_up, bbox_down = np.max(m.vertices, axis=0, keepdims=True), np.min(m.vertices, axis=0, keepdims=True)
    center = 0.5 * (bbox_up + bbox_down)
    scale = np.sqrt(np.sum((bbox_up - bbox_down) ** 2, axis=1))
    # centralization
    trans = np.eye(4)
    trans[:3, 3] = -center
    m.apply_transform(trans)
    trans = np.eye(4)
    trans[:3, :3] = np.eye(3) / scale
    m.apply_transform(trans)

    pts = m.vertices

    pmin, pmax = pts.min(axis=0), pts.max(axis=0)
    assert not (np.any(pmin < -0.5) or np.any(pmax > 0.5)), f'{pmin}, {pmax} out of range'

    """
    center = (pmin + pmax) * 0.5
    lim = max(pmax - pmin) * 0.5 + 0.2

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.axis('off')
    ax = plt.subplot(1, 1, 1, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.8, s=1)
    ax.set_xlim3d([center[0] - lim, center[0] + lim])
    ax.set_ylim3d([center[1] - lim, center[1] + lim])
    ax.set_zlim3d([center[2] - lim, center[2] + lim])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    """

    return m


def create_partial(save_folder, m, ins_num, pose_list,
                   yfov=np.deg2rad(60), pw=640, ph=480, near=0.1, far=10):
    scene = pyrender.Scene()

    mesh = pyrender.Mesh.from_trimesh(m)
    node = pyrender.Node(mesh=mesh, matrix=np.eye(4))

    scene.add_node(node)

    camera_pose = np.eye(4)
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=pw / ph, znear=near, zfar=far)
    projection = camera.get_projection_matrix()
    scene.add(camera, camera_pose)
    r = pyrender.OffscreenRenderer(pw, ph)

    depth_path = pjoin(save_folder, ins_num, 'depth')
    os.makedirs(depth_path, exist_ok=True)
    gt_path = pjoin(save_folder, ins_num, 'gt')
    os.makedirs(gt_path, exist_ok=True)
    for i, pose_dict in enumerate(pose_list):
        pose = np.eye(4)
        pose[:3, :3] = pose_dict['rotation']
        pose[:3, 3:4] = pose_dict['translation'] / pose_dict['scale']
        pose[3, 3] = 1.0 / pose_dict['scale']
        scene.set_pose(node, pose)
        depth_buffer = r.render(scene, flags=pyrender.constants.RenderFlags.DEPTH_ONLY)
        # pts = backproject(depth_buffer, projection, near, far, from_image=False)
        mask = depth_buffer > 0
        depth_z = buffer_depth_to_ndc(depth_buffer, near, far)  # [-1, 1]
        depth_image = depth_z * 0.5 + 0.5  # [0, 1]
        depth_image = linearize_img(depth_image, near, far)  # [0, 1]
        depth_image = np.uint16((depth_image * mask) * ((1 << 16) - 1))
        cv2.imwrite(pjoin(depth_path, f'{i:03}.png'), depth_image)
        np.save(pjoin(gt_path, f'{i:03}.npy'), pose)
    # backproject(depth_image, projection, near, far, from_image=True, vis=True)
    del(r)
    del(camera)
    del(scene)

    return projection, near, far


def ndc_depth_to_buffer(z, near, far):  # z in [-1, 1]
    return 2 * near * far / (near + far - z * (far - near))


def buffer_depth_to_ndc(d, near, far):  # d in (0, +
    return ((near + far) - 2 * near * far / np.clip(d, a_min=1e-6, a_max=1e6)) / (far - near)


def linearize_img(d, near, far):  # for visualization only
    return 2 * near / (near + far - d * (far - near))


def inv_linearize_img(d, near, far):  # for visualziation only
    return (near + far - 2 * near / d) / (far - near)


def backproject(depth, projection, near, far, from_image=False, vis=False):
    proj_inv = np.linalg.inv(projection)
    height, width = depth.shape
    non_zero_mask = (depth > 0)
    idxs = np.where(non_zero_mask)
    depth_selected = depth[idxs[0], idxs[1]].astype(np.float32).reshape((1, -1))
    if from_image:
        z = depth_selected / ((1 << 16) - 1)  # [0, 1]
        z = inv_linearize_img(z, near, far)  # [0, 1]
        z = z * 2 - 1.0  # [-1, 1]
        d = ndc_depth_to_buffer(z, near, far)
    else:
        d = depth_selected
        z = buffer_depth_to_ndc(d, near, far)

    grid = np.array([idxs[1] / width * 2 - 1, 1 - idxs[0] / height * 2])  # ndc [-1, 1]

    ones = np.ones_like(z)
    pts = np.concatenate((grid, z, ones), axis=0) * d  # before dividing by w, w = -z_world = d

    pts = proj_inv @ pts
    pts = np.transpose(pts)

    pts = pts[:, :3]

    if vis:
        pmin, pmax = pts.min(axis=0), pts.max(axis=0)
        center = (pmin + pmax) * 0.5
        lim = max(pmax - pmin) * 0.5 + 0.2

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(depth, cmap=plt.cm.gray_r)
        ax = plt.subplot(1, 2, 2, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.8, s=1)
        ax.set_xlim3d([center[0] - lim, center[0] + lim])
        ax.set_ylim3d([center[1] - lim, center[1] + lim])
        ax.set_zlim3d([center[2] - lim, center[2] + lim])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    return pts


def get_modelnet_objects(root_path, category, split):
    read_folder = pjoin(root_path, category, split)
    path_list = [pjoin(read_folder, i) for i in os.listdir(read_folder) if i.endswith('off')]
    mesh_dict = {}
    for read_path in path_list:
        instance_name = read_path.split('.')[-2].split('/')[-1]   # 'aiplane_0001'
        instance_name = instance_name.split('_')[-1]
        mesh_dict[instance_name] = get_normalized_mesh(read_path, transform=True)
    return mesh_dict


def get_shapenet_objects(root_path, category, split):
    cat2id = {
        'bottle': '02876657',
        'bowl': '02880940',
        'camera': '02942699',
        'can': '02946921',
        'laptop': '03642806',
        'mug': '03797390'
    }
    split = 'val' if split == 'test' else split
    read_folder = pjoin(root_path, split, cat2id[category])
    path_list = [path for path in [pjoin(read_folder, i, 'model.obj') for i in os.listdir(read_folder)]
                 if os.path.exists(path)]
    mesh_dict = {}
    for read_path in path_list:
        instance_name = read_path.split('.')[-2].split('/')[-2]
        mesh_dict[instance_name] = get_normalized_mesh(read_path)
    return mesh_dict


def proc_render(is_first, save_folder, mesh_dict, pose_dict, render_instances,
                yfov=np.deg2rad(60), pw=640, ph=480, near=0.1, far=10):
    for instance in tqdm(render_instances):
        pose_list = pose_dict[instance]
        projection, near, far = create_partial(save_folder, mesh_dict[instance], instance, pose_list,
                                               yfov=yfov, pw=pw, ph=ph, near=near, far=far)
    if is_first:
        meta_path = pjoin(save_folder, 'meta.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump({'near': near, 'far': far, 'projection': projection}, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--pose_input', type=str)
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--category', type=str, default='airplane')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num_proc', type=int, default=8)
    parser.add_argument('--random_ins', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.dataset == 'modelnet':
        mesh_dict = get_modelnet_objects(args.input, args.category, args.split)
    elif args.dataset == 'shapenet':
        mesh_dict = get_shapenet_objects(args.input, args.category, args.split)
    else:
        assert 0, f'Unsupported dataset {args.dataset}'

    mp.set_start_method('spawn')

    save_folder = pjoin(args.output, args.category, args.split)

    pose_dict = np.load(args.pose_input, allow_pickle=True)['data'].item()
    new_instances = list(mesh_dict.keys())
    if args.random_ins:
        new_pose_dict = {ins: [] for ins in new_instances}
        for instance, pose_list in pose_dict.items():
            for pose in pose_list:
                cur_ins = new_instances[np.random.randint(0, len(new_instances))]
                new_pose_dict[cur_ins].append(pose)
    else:
        new_pose_dict = pose_dict

    num_per_ins = (len(new_instances) - 1) // args.num_proc + 1
    processes = []
    for i in range(args.num_proc):
        st = num_per_ins * i
        ed = min(st + num_per_ins, len(new_instances))
        p = mp.Process(target=proc_render, args=(i == 0, save_folder, mesh_dict, new_pose_dict,
                                                 new_instances[st: ed],
                                                 np.deg2rad(60), 640, 480, 0.01, 10))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
