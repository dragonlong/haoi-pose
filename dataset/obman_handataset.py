import random
import traceback
import csv

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image, ImageFilter
import pickle
from tqdm import tqdm
import os
from os import makedirs, remove
from os.path import exists, join
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as func_transforms
import __init__
from global_info import global_info
from common import data_utils, handutils, vis_utils, bp
from common.d3_utils import align_rotation # 4 * 4
from common.queries import (
    BaseQueries,
    TransQueries,
    one_query_in,
    no_query_in,
)
def bp():
    import pdb;pdb.set_trace()

infos           = global_info()
my_dir          = infos.base_path
project_path    = infos.project_path

def bbox_from_joints(joints):
    x_min, y_min = joints.min(0)
    x_max, y_max = joints.max(0)
    bbox = [x_min, y_min, x_max, y_max]
    return bbox

class HandDataset(Dataset):
    """Class inherited by hands datasets
    hands datasets must implement the following methods:
    - get_image
    that respectively return a PIL image and a numpy array
    - the __len__ method

    and expose the following attributes:
    - the cache_folder : the path to the cache folder of the dataset
    """
    def __init__(
        self,
        pose_dataset,
        cfg=None,
        center_idx=9,
        point_nb=600,
        inp_res=256,
        max_rot=np.pi,
        normalize_img=False,
        split="train",
        scale_jittering=0.3,
        center_jittering=0.2,
        train=True,
        hue=0.15,
        saturation=0.5,
        contrast=0.5,
        brightness=0.5,
        blur_radius=0.5,
        queries=[
            BaseQueries.images,
            TransQueries.joints2d,
            TransQueries.verts3d,
            TransQueries.joints3d,
            TransQueries.depth,
            TransQueries.sdf,
            TransQueries.sdf_points,
        ],
        sides="both",
        block_rot=False,
        black_padding=False,
        as_obj_only=False,
        is_testing=False
    ):
        """
        Args:
        center_idx: idx of joint on which to center 3d pose
        as_obj_only: apply same centering and scaling as when objects are
            not present
        sides: if both, don't flip hands, if 'right' flip all left hands to
            right hands, if 'left', do the opposite
        """
        # Dataset attributes
        self.num_points   = cfg.num_points # fixed for category with < 5 parts
        self.J_num        = 21
        self.batch_size   = cfg.batch_size
        self.n_max_parts  = 2
        self.task       = cfg.task
        if self.task == 'category_pose':
            self.is_gen       = cfg.is_gen
            self.is_debug     = cfg.is_debug
            self.nocs_type    = cfg.nocs_type

            # controls
            self.pred_mano  = cfg.pred_mano
            self.rot_align  = cfg.rot_align
            self.hand_only  = cfg.hand_only

        self.pose_dataset = pose_dataset
        self.cfg = cfg
        self.as_obj_only = as_obj_only
        self.inp_res = inp_res
        self.point_nb = point_nb
        self.normalize_img = normalize_img
        self.center_idx = center_idx
        self.sides = sides
        self.black_padding = black_padding

        self.is_testing   = is_testing
        # Color jitter attributes
        self.hue = hue
        self.contrast = contrast
        self.brightness = brightness
        self.saturation = saturation
        self.blur_radius = blur_radius

        self.max_rot   = max_rot
        self.block_rot = block_rot

        # Training attributes
        self.train = train
        self.scale_jittering = scale_jittering
        self.center_jittering = center_jittering

        self.queries = queries
        self.stddev  = 0.005

        selected_csv = f'{project_path}/haoi-pose/scripts/shapenet_select.csv'
        shapenet_info = {}
        with open(selected_csv, 'r') as csv_f:
            reader = csv.DictReader(csv_f)
            for row_idx, row in enumerate(reader):
                shapenet_info[row['class']] = row['path']

        sample_paths = []
        self.models  = []
        self.categories = []
        self.class2category = {}
        self.category2ind = {}
        # self.categories = ['02876657', '03797390', '02880940', '02946921', '03593526', '03624134', '02992529', '02942699', '04074963']
        for class_id, class_path in tqdm(shapenet_info.items(), desc='class'):
            try:
                samples = sorted(os.listdir(class_path))
                self.categories.append(class_path.split('/')[-1])
                self.class2category[class_id] = self.categories[-1]
                self.models.append(samples)
            except:
                class_path = class_path.replace('/groups/CESCA-CV/external', '/home/dragon/Documents/external')
                samples = sorted(os.listdir(class_path))
                self.categories.append(class_path.split('/')[-1])
                self.class2category[class_id] = self.categories[-1]
                self.models.append(samples)
        print('we have categories: \n', self.categories)
        for ind, category in enumerate(self.categories):
            self.category2ind[category]= ind

        self.list_frame = [fname.split('/')[-1].split('.')[0] for fname in self.pose_dataset.image_names]
        self.arr_category = np.array([self.category2ind[model_path.split('/')[-4]] for model_path in self.pose_dataset.obj_paths])
        self.list_instance = ['{}_{}'.format(model_path.split('/')[-3], model_path.split('/')[-4]) for model_path in self.pose_dataset.obj_paths]
        self.ids_per_category = {}
        for ind, category in enumerate(self.categories):
            self.ids_per_category[category] = np.where(self.arr_category==ind)[0].tolist()
        self.basename_list = ['{}_{}'.format(self.list_frame[j], self.list_instance[j]) for j in range(len(self.list_frame))]
        if len(cfg.target_category) > 0:
            # target_categorys = ['camera', 'knife', 'jar', 'cellphone', 'mug', 'remote']
            # if type(target_categorys) is list:
            #     self.all_ids = []
            #     for target in target_categorys:
            #         target_category = self.class2category[target]
            #         self.all_ids += self.ids_per_category[target_category]
            # else:
            #     target_category = self.class2category[self.cfg.target_category] # # jar, 03593526
            #     self.all_ids = self.ids_per_category[target_category]
            target_category = self.class2category[self.cfg.target_category] # # jar, 03593526
            self.all_ids = self.ids_per_category[target_category]
        else:
            self.all_ids = np.arange(0, len(self.pose_dataset)) # add one more level of packaging to change choice base;
    def get_model_dict(self, idx):
        return self.models[idx]

    def __len__(self):
        return len(self.all_ids)

    def get_sample(self, idx, query=None, debug=False):
        if query is None:
            query = self.queries
        sample = {}

        if BaseQueries.images in query or TransQueries.images in query:
            center, scale = self.pose_dataset.get_center_scale(idx)
            needs_center_scale = True
        else:
            needs_center_scale = False

        # Get sides
        if BaseQueries.sides in query:
            hand_side = self.pose_dataset.get_sides(idx)
            # Flip if needed
            if self.sides == "right" and hand_side == "left":
                flip = True
                hand_side = "right"
            elif self.sides == "left" and hand_side == "right":
                flip = True
                hand_side = "left"
            else:
                flip = False
            sample[BaseQueries.sides] = hand_side
        else:
            flip = False

        # Get original image
        if BaseQueries.images in query or TransQueries.images in query:
            img = self.pose_dataset.get_image(idx)
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if BaseQueries.images in query:
                sample[BaseQueries.images] = img

        # Flip and image 2d if needed
        if flip:
            center[0] = img.size[0] - center[0]
        # Data augmentation
        if self.train and needs_center_scale:
            # Randomly jitter center
            # Center is located in square of size 2*center_jitter_factor
            # in center of cropped image
            center_offsets = (
                self.center_jittering
                * scale
                * np.random.uniform(low=-1, high=1, size=2)
            )
            center = center + center_offsets.astype(int)

            # Scale jittering
            scale_jittering = self.scale_jittering * np.random.randn() + 1
            scale_jittering = np.clip(
                scale_jittering,
                1 - self.scale_jittering,
                1 + self.scale_jittering,
            )
            scale = scale * scale_jittering

            rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
        else:
            rot = 0
        if self.block_rot:
            rot = self.max_rot
        rot_mat = np.array(
            [
                [np.cos(rot), -np.sin(rot), 0],
                [np.sin(rot), np.cos(rot), 0],
                [0, 0, 1],
            ]
        ).astype(np.float32)

        # Get 2D hand joints
        if (TransQueries.joints2d in query) or (TransQueries.images in query):
            affinetrans, post_rot_trans = handutils.get_affine_transform(
                center, scale, [self.inp_res, self.inp_res], rot=rot
            )
            if TransQueries.affinetrans in query:
                sample[TransQueries.affinetrans] = torch.from_numpy(
                    affinetrans
                )
        if BaseQueries.joints2d in query or TransQueries.joints2d in query:
            joints2d = self.pose_dataset.get_joints2d(idx)
            if flip:
                joints2d = joints2d.copy()
                joints2d[:, 0] = img.size[0] - joints2d[:, 0]
            if BaseQueries.joints2d in query:
                sample[BaseQueries.joints2d] = torch.from_numpy(joints2d)
        if TransQueries.joints2d in query:
            rows = handutils.transform_coords(joints2d, affinetrans)
            sample[TransQueries.joints2d] = torch.from_numpy(np.array(rows))

        if BaseQueries.camintrs in query or TransQueries.camintrs in query:
            camintr = self.pose_dataset.get_camintr(idx)
            if BaseQueries.camintrs in query:
                sample[BaseQueries.camintrs] = camintr
            if TransQueries.camintrs in query:
                # Rotation is applied as extr transform
                new_camintr = post_rot_trans.dot(camintr)
                sample[TransQueries.camintrs] = new_camintr
        else:
            camintr = None
        # Get 2D object points
        if BaseQueries.objpoints2d in query or (
            TransQueries.objpoints2d in query
        ):
            objpoints2d = self.pose_dataset.get_objpoints2d(idx)
            if flip:
                objpoints2d = objpoints2d.copy()
                objpoints2d[:, 0] = img.size[0] - objpoints2d[:, 0]
            if BaseQueries.objpoints2d in query:
                sample[BaseQueries.objpoints2d] = torch.from_numpy(objpoints2d)
            if TransQueries.objpoints2d in query:
                transobjpoints2d = handutils.transform_coords(
                    objpoints2d, affinetrans
                )
                sample[TransQueries.objpoints2d] = torch.from_numpy(
                    np.array(transobjpoints2d)
                )

        # Get segmentation
        if BaseQueries.segms in query or TransQueries.segms in query:
            segm = self.pose_dataset.get_segm(idx)
            if flip:
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)
            if BaseQueries.segms in query:
                sample[BaseQueries.segms] = segm
            if TransQueries.segms in query:
                segm = handutils.transform_img(
                    segm, affinetrans, [self.inp_res, self.inp_res]
                )
                segm = segm.crop((0, 0, self.inp_res, self.inp_res))
                segm = func_transforms.to_tensor(segm)
                sample[TransQueries.segms] = segm
        else:
            segm = None

        # Get 3D hand joints
        if (
            (BaseQueries.joints3d in query)
            or (TransQueries.joints3d in query)
            or (TransQueries.verts3d in query)
            or (TransQueries.objverts3d in query)
            or (TransQueries.objpoints3d in query)
        ):
            # Center on root joint
            center3d_queries = [
                TransQueries.joints3d,
                BaseQueries.joints3d,
                TransQueries.verts3d,
            ]
            obj_only = (
                (
                    TransQueries.objverts3d in query
                    or TransQueries.objpoints3d in query
                )
                and no_query_in(
                    center3d_queries, self.pose_dataset.all_queries
                )
                or self.as_obj_only
            )
            if not obj_only:
                if one_query_in(
                    [TransQueries.objpoints3d]
                    + [TransQueries.objverts3d]
                    + center3d_queries,
                    query,
                ):
                    joints3d = self.pose_dataset.get_joints3d(idx)
                    if flip:
                        joints3d[:, 0] = -joints3d[:, 0]

                    if BaseQueries.joints3d in query:
                        sample[BaseQueries.joints3d] = joints3d
                    if self.train:
                        joints3d = rot_mat.dot(
                            joints3d.transpose(1, 0)
                        ).transpose()
                    # Compute 3D center
                    if self.center_idx is not None:
                        if self.center_idx == -1:
                            center3d = (joints3d[9] + joints3d[0]) / 2
                        else:
                            center3d = joints3d[self.center_idx]
                    if TransQueries.joints3d in query and (
                        self.center_idx is not None
                    ):
                        joints3d = joints3d - center3d
                    if TransQueries.joints3d in query:
                        sample[TransQueries.joints3d] = torch.from_numpy(
                            joints3d
                        )

        # Get 3D hand vertices
        if TransQueries.verts3d in query:
            hand_verts3d = self.pose_dataset.get_verts3d(idx)
            if flip:
                hand_verts3d[:, 0] = -hand_verts3d[:, 0]
            hand_verts3d = rot_mat.dot(
                hand_verts3d.transpose(1, 0)
            ).transpose()
            if self.center_idx is not None:
                hand_verts3d = hand_verts3d - center3d
            sample[TransQueries.verts3d] = hand_verts3d

        # Get 3D object points
        if TransQueries.objpoints3d in query and (
            BaseQueries.objpoints3d in self.pose_dataset.all_queries
        ):
            points3d, canon_pts = self.pose_dataset.get_objpoints3d(
                idx, point_nb=self.point_nb
            )
            if flip:
                points3d[:, 0] = -points3d[:, 0]
            points3d = rot_mat.dot(points3d.transpose(1, 0)).transpose()
            obj_verts3d = points3d
        elif (
            TransQueries.objpoints3d in query
            or BaseQueries.objverts3d in query
            or TransQueries.objverts3d in query
        ) and (BaseQueries.objverts3d in self.pose_dataset.all_queries):
            obj_verts3d, obj_faces, canon_pts = self.pose_dataset.get_obj_verts_faces(idx)
            if flip:
                obj_verts3d[:, 0] = -obj_verts3d[:, 0]
            if BaseQueries.objverts3d in query:
                sample[BaseQueries.objverts3d] = obj_verts3d
            if TransQueries.objverts3d in query:
                origin_trans_mesh = rot_mat.dot(
                    obj_verts3d.transpose(1, 0)
                ).transpose()
                if self.center_idx is not None:
                    origin_trans_mesh = origin_trans_mesh - center3d
                sample[TransQueries.objverts3d] = origin_trans_mesh

            if BaseQueries.objfaces in query:
                sample[BaseQueries.objfaces] = obj_faces
            obj_verts3d = data_utils.points_from_mesh(
                obj_faces,
                obj_verts3d,
                show_cloud=False,
                vertex_nb=self.point_nb,
            ).astype(np.float32)
            obj_verts3d = rot_mat.dot(obj_verts3d.transpose(1, 0)).transpose()

        elif TransQueries.objpoints3d in query:
            raise ValueError(
                "Requested TransQueries.objpoints3d for dataset "
                "without BaseQueries.objpoints3d and BaseQueries.objverts3d"
            )
        else:
            model_path = self.pose_dataset.obj_paths[idx].replace(
                "model_normalized.pkl", "surface_points.pkl"
            )
            with open(model_path, "rb") as obj_f:
                canon_pts = pickle.load(obj_f)

        # Center object on hand or center of object if no hand present
        if TransQueries.objpoints3d in query:
            if obj_only:
                center3d = (obj_verts3d.max(0) + obj_verts3d.min(0)) / 2
            if self.center_idx is not None or obj_only:
                obj_verts3d = obj_verts3d - center3d
            if obj_verts3d.max() > 5000:
                print("BIIIG problem with sample")
                print(self.pose_dataset.image_names[idx])
            if obj_only:
                # Inscribe into sphere of radius 1
                radius = np.linalg.norm(obj_verts3d, 2, 1).max()
                obj_verts3d = obj_verts3d / radius
            sample[TransQueries.objpoints3d] = torch.from_numpy(obj_verts3d)

        if TransQueries.center3d in query:
            sample[TransQueries.center3d] = center3d

        if BaseQueries.manoidxs in query:
            sample[BaseQueries.manoidxs] = self.pose_dataset.get_manoidxs(idx)

        # Get rgb image
        if TransQueries.images in query:
            # Data augmentation
            if self.train:
                blur_radius = random.random() * self.blur_radius
                img = img.filter(ImageFilter.GaussianBlur(blur_radius))
                img = data_utils.color_jitter(
                    img,
                    brightness=self.brightness,
                    saturation=self.saturation,
                    hue=self.hue,
                    contrast=self.contrast,
                )
            # Transform and crop
            img = handutils.transform_img(
                img, affinetrans, [self.inp_res, self.inp_res]
            )
            img = img.crop((0, 0, self.inp_res, self.inp_res))

            # Tensorize and normalize_img
            img = func_transforms.to_tensor(img).float()
            if self.black_padding:
                padding_ratio = 0.2
                padding_size = int(self.inp_res * padding_ratio)
                img[:, 0:padding_size, :] = 0
                img[:, -padding_size:-1, :] = 0
                img[:, :, 0:padding_size] = 0
                img[:, :, -padding_size:-1] = 0

            if self.normalize_img:
                img = func_transforms.normalize(img, self.mean, self.std)
            else:
                img = func_transforms.normalize(
                    img, [0.5, 0.5, 0.5], [1, 1, 1]
                )
            if TransQueries.images in query:
                sample[TransQueries.images] = img

        # Add meta information
        if BaseQueries.meta in query:
            meta = self.pose_dataset.get_meta(idx)
            sample[BaseQueries.meta] = meta

        if BaseQueries.depth in query:
            depth = self.pose_dataset.get_depth(idx)
            sample[BaseQueries.depth] = depth
        else:
            depth = None

        if BaseQueries.pcloud in query or TransQueries.pcloud in query:
            if depth is None:
                depth = self.pose_dataset.get_depth(idx)
            if camintr is None:
                camintr = self.pose_dataset.get_camintr(idx)
            cloud = self.pose_dataset.get_pcloud(depth, camintr) # * 1000 - center3d

            obj_segm = self.pose_dataset.get_segm(idx, ext_cmd='obj', debug=False) # only TODO
            obj_segm[obj_segm>1.0] = 1.0
            obj_hand_segm = (np.asarray(self.pose_dataset.get_segm(idx, debug=False)) / 255).astype(np.int)
            full_segm = obj_hand_segm[:, :, 0] | obj_hand_segm[:, :, 1]
            if self.cfg.use_transform_hand:
                segm = full_segm
            else:
                segm = obj_segm
            pcloud = cloud[np.where(segm>0)[0], np.where(segm>0)[1], :]
            cls_arr = obj_segm[np.where(full_segm>0)[0], np.where(full_segm>0)[1]] # hand is 0, obj =1
            if debug:
                sample[BaseQueries.pcloud] = pcloud
                full_segm = self.pose_dataset.get_segm(idx, debug=False) # TODO
                obj_hand_segm = (np.asarray(self.pose_dataset.get_segm(idx, debug=False)) / 255).astype(np.int)
                segm = obj_hand_segm[:, :, 0] | obj_hand_segm[:, :, 1]
                full_pcloud = cloud[np.where(segm>0)[0], np.where(segm>0)[1], :]
                sample['full_pcloud'] = full_pcloud

        if BaseQueries.nocs in query or TransQueries.nocs in query:
            if self.cfg.oracle_nocs:
                # change target points
                boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]
                center_pt = (boundary_pts[0] + boundary_pts[1])/2
                length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
                nocs = (canon_pts - center_pt.reshape(1, 3)) / length_bb + 0.5
            else:
                boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]
                nocs = self.pose_dataset.get_nocs(idx, pcloud, boundary_pts)
                if debug:
                    sample[BaseQueries.nocs] = nocs
            inds = np.random.choice(nocs.shape[0], 1024)
            if inds.shape[0] < 1024:
                return None
            sample['inputs'] = nocs[inds].astype(np.float32)
            if self.cfg.use_transform_hand:
                sample['inputs'] = np.concatenate([sample['inputs'], cls_arr[inds].reshape(-1, 1)], axis=1)
        if debug:
            sample['canon_pts'] = canon_pts
            sample['model_path']= self.pose_dataset.obj_paths[idx]
        sample['idx'] = idx
        if BaseQueries.occupancy in query or TransQueries.occupancy in query:
            points, occupancy_labels = self.pose_dataset.get_occupany_pts(idx, canon_pts)
            sample['points'] = points.astype(np.float32)
            sample['points.occ'] = occupancy_labels.astype(np.float32)

        model_path    = self.pose_dataset.obj_paths[idx]
        category_name = model_path.split('/')[-4]
        instance_name = model_path.split('/')[-3]
        category_id   = self.categories.index(category_name)
        category_code = np.zeros((len(self.categories)), dtype=np.float32)
        category_code[category_id] = 1.0
        sample['code'] = category_code

        if self.pose_dataset.split == 'val' or self.pose_dataset.split == 'test':
            # get points 'points_iou', 'points_iou.occ', 'idx', 'inputs.ind', 'points_iou.normalized'
            points_iou, points_iou_occ = self.pose_dataset.get_volume_pts(idx, canon_pts)
            if points_iou is None:
                return None
            sample['points_iou']     = points_iou.astype(np.float32)
            sample['points_iou.occ'] = points_iou_occ.astype(np.float32)

        if self.cfg.eval:
            pointcloud_chamfer, chamfer_normals = self.pose_dataset.get_surface_pts(idx, canon_pts)
            sample['pointcloud_chamfer'] = pointcloud_chamfer.astype(np.float32)
            sample['pointcloud_chamfer.normals'] = chamfer_normals.astype(np.float32)

        return sample

    def get_sample_mine(self, idx, debug=False):
        n_parts = self.n_max_parts
        assert n_parts == 2
        depth   = self.pose_dataset.get_depth(idx)
        camintr = self.pose_dataset.get_camintr(idx)
        cloud = self.pose_dataset.get_pcloud(depth, camintr) # * 1000 - center3d
        obj_segm = self.pose_dataset.get_segm(idx, ext_cmd='obj', debug=False) # only TODO
        obj_segm[obj_segm>1.0] = 1.0
        # obj_pcloud = cloud[np.where(obj_segm>0)[0], np.where(obj_segm>0)[1], :] # object cloud
        obj_hand_segm = (np.asarray(self.pose_dataset.get_segm(idx, debug=False)) / 255).astype(np.int)
        segm = obj_hand_segm[:, :, 0] | obj_hand_segm[:, :, 1]

        model_path = self.pose_dataset.obj_paths[idx].replace(
            "model_normalized.pkl", "surface_points.pkl"
        )
        with open(model_path, "rb") as obj_f:
            canon_pts = pickle.load(obj_f)

        boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]

        pts_arr = cloud[np.where(segm>0)[0], np.where(segm>0)[1], :]
        cls_arr = obj_segm[np.where(segm>0)[0], np.where(segm>0)[1]] # hand is 0, obj =1
        nocs    = self.pose_dataset.get_nocs(idx, pts_arr, boundary_pts)
        n_total_points = pts_arr.shape[0]
        output_arr     = [pts_arr, cls_arr, nocs]
        if self.hand_only:
            n_total_points   = np.where(cls_arr==0)[0].shape[0]

        if n_total_points < self.num_points:
            tile_n = int(self.num_points/n_total_points) + 1
            n_total_points = tile_n * n_total_points
            for j in range(len(output_arr)):
                arr_tiled = np.concatenate([np.copy(output_arr[j])] * tile_n, axis=0)
                output_arr[j] = arr_tiled

        pts_arr, cls_arr, p_arr = output_arr
        # use furthest point sampling
        if self.hand_only:
            hand_cls   = 0
            perm       = np.random.permutation(n_total_points)
            hand_inds  = np.where(cls_arr[perm]==hand_cls)[0]
            sample_ind = hand_inds[:self.num_points].tolist()
        else:
            perm       = np.random.permutation(n_total_points)
            sample_ind = np.arange(0, self.num_points).tolist()

        pts_arr       = pts_arr[perm][sample_ind] # norm_factors[0] # by default is 1
        cls_arr       = cls_arr[perm][sample_ind]
        p_arr         = p_arr[perm][sample_ind]   # norm
        mask_array    = np.zeros([self.num_points, n_parts], dtype=np.float32)
        mask_array[np.arange(self.num_points), cls_arr.astype(np.int8)] = 1.00 #
        mask_array[np.arange(self.num_points), 0] = 0.0
        # mask_array[np.arange(self.num_points), n_parts-1] = 0.0 # ignore hand points for nocs prediction
        if not self.is_testing:
            pts_arr = torch.from_numpy(pts_arr.astype(np.float32).transpose(1, 0))
            cls_arr = torch.from_numpy(cls_arr.astype(np.float32))
            mask_array = torch.from_numpy(mask_array.astype(np.float32).transpose(1, 0))
            p_arr = torch.from_numpy(p_arr.astype(np.float32).transpose(1, 0))
        data_dict = {'P': pts_arr,
                    'partcls_per_point': cls_arr, #
                    'part_mask': mask_array,
                    'nocs_per_point' : p_arr}
        data_dict['index'] = idx
        return data_dict

    def __getitem__(self, idx):
        try:
            idx = self.all_ids[idx]
            if self.task == 'category_pose':
                sample = self.get_sample_mine(idx)
            else:
                sample = self.get_sample(idx, self.queries)
        except Exception:
            traceback.print_exc()
            print("Encountered error processing sample {}".format(idx))
            random_idx = random.randint(0, len(self))
            idx = self.all_ids[random_idx]
            if self.task == 'category_pose':
                sample = self.get_sample_mine(idx)
            else:
                sample = self.get_sample(idx, self.queries)

        return sample

    def display_3d(self, ax, sample, proj="z", joint_idxs=False, axis_off=False):
        # Scatter  projection of 3d vertices
        pts = []
        if TransQueries.verts3d in sample:
            verts3d = sample[TransQueries.verts3d]
            pts.append(verts3d)
            ax.scatter(verts3d[:, 0], verts3d[:, 1], verts3d[:, 2], s=1)

        # Scatter projection of object 3d vertices
        if TransQueries.objpoints3d in sample:
            obj_verts3d = sample[TransQueries.objpoints3d]
            pts.append(obj_verts3d)
            ax.scatter(obj_verts3d[:, 0], obj_verts3d[:, 1], obj_verts3d[:, 2], s=1)

        # Scatter  projection of 3d vertices
        if BaseQueries.verts3d in sample:
            verts3d = sample[BaseQueries.verts3d]
            pts.append(verts3d)
            ax.scatter(verts3d[:, 0], verts3d[:, 1], verts3d[:, 2], s=1)

        # Scatter projection of object 3d vertices
        if BaseQueries.objpoints3d in sample:
            obj_verts3d = sample[BaseQueries.objpoints3d]
            pts.append(obj_verts3d)
            ax.scatter(obj_verts3d[:, 0], obj_verts3d[:, 1], obj_verts3d[:, 2], s=1)

        # Scatter projection of object 3d vertices
        if BaseQueries.pcloud in sample:
            obj_verts3d = sample[BaseQueries.pcloud]
            pts.append(obj_verts3d)
            ax.scatter(obj_verts3d[:, 0], obj_verts3d[:, 1], obj_verts3d[:, 2], s=1)

        # Scatter projection of object 3d vertices
        if TransQueries.pcloud in sample:
            obj_verts3d = sample[TransQueries.pcloud]
            pts.append(obj_verts3d)
            ax.scatter(obj_verts3d[:, 0], obj_verts3d[:, 1], obj_verts3d[:, 2], s=1)

        cam_equal_aspect_3d(ax, np.concatenate(pts, axis=0))
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        if axis_off:
            plt.axis('off')

    def display_nocs(self, ax, sample, proj="z", joint_idxs=False, axis_off=False):
        # Scatter  projection of 3d vertices
        pts = []
        if TransQueries.nocs in sample:
            verts3d = sample[TransQueries.nocs]
            pts.append(verts3d)
            ax.scatter(verts3d[:, 0], verts3d[:, 1], verts3d[:, 2], s=1)

        # Scatter  projection of 3d vertices
        if BaseQueries.nocs in sample:
            verts3d = sample[BaseQueries.nocs]
            pts.append(verts3d)
            ax.scatter(verts3d[:, 0], verts3d[:, 1], verts3d[:, 2], s=1)

        if 'canon_pts' in sample:
            obj_verts3d = sample['canon_pts']
            pts.append(obj_verts3d)
            ax.scatter(obj_verts3d[:, 0], obj_verts3d[:, 1], obj_verts3d[:, 2], s=1)

        cam_equal_aspect_3d(ax, np.concatenate(pts, axis=0))
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        if axis_off:
            plt.axis('off')

    def display_proj(self, ax, sample, proj="z", joint_idxs=False):

        if proj == "z":
            proj_1 = 0
            proj_2 = 1
            ax.invert_yaxis()
        elif proj == "y":
            proj_1 = 0
            proj_2 = 2
        elif proj == "x":
            proj_1 = 1
            proj_2 = 2

        if TransQueries.joints3d in sample:
            joints3d = sample[TransQueries.joints3d]
            vis_utils.visualize_joints_2d(
                ax,
                np.stack([joints3d[:, proj_1], joints3d[:, proj_2]], axis=1),
                joint_idxs=joint_idxs,
                links=self.pose_dataset.links,
            )
        # Scatter  projection of 3d vertices
        if TransQueries.verts3d in sample:
            verts3d = sample[TransQueries.verts3d]
            ax.scatter(verts3d[:, proj_1], verts3d[:, proj_2], s=1)

        # Scatter projection of object 3d vertices
        if TransQueries.objpoints3d in sample:
            obj_verts3d = sample[TransQueries.objpoints3d]
            ax.scatter(obj_verts3d[:, proj_1], obj_verts3d[:, proj_2], s=1)
        ax.set_aspect("equal")  # Same axis orientation as imshow

    def visualize_3d_proj(self, idx, joint_idxs=False):
        queries = [
            BaseQueries.sides,
            BaseQueries.images,
            TransQueries.joints3d,
            TransQueries.images,
            TransQueries.objpoints3d,
            TransQueries.verts3d,
            BaseQueries.joints2d,
        ]

        sample_queries = []
        for query in queries:
            if query in self.pose_dataset.all_queries:
                sample_queries.append(query)

        sample = self.get_sample(idx, query=sample_queries)
        fig = plt.figure()

        # Display transformed image
        ax = fig.add_subplot(121)
        img = sample[TransQueries.images].numpy().transpose(1, 2, 0)
        if not self.normalize_img:
            img += 0.5
        ax.imshow(img)

        # Add title
        if BaseQueries.sides in sample:
            side = sample[BaseQueries.sides]
            ax.set_title("{} hand".format(side))

        if TransQueries.objpoints3d in sample:
            ax = fig.add_subplot(122, projection="3d")
            objpoints3d = sample[TransQueries.objpoints3d].numpy()
            ax.scatter(objpoints3d[:, 0], objpoints3d[:, 1], objpoints3d[:, 2])
            ax.view_init(elev=90, azim=-90)
            cam_equal_aspect_3d(ax, objpoints3d)
        plt.show()

    def visualize_3d_transformed(self, idx, joint_idxs=False):
        queries = [
            BaseQueries.images,
            BaseQueries.occupancy,
            BaseQueries.sides,
            BaseQueries.objpoints3d,
            BaseQueries.verts3d,
            BaseQueries.joints2d,
            BaseQueries.joints3d,
            BaseQueries.depth,
            BaseQueries.pcloud,
            BaseQueries.nocs,
            TransQueries.joints3d,
            TransQueries.images,
            TransQueries.objpoints3d,
            TransQueries.verts3d,
            TransQueries.nocs,
        ]
        sample_queries = []
        for query in queries:
            if query in self.pose_dataset.all_queries:
                sample_queries.append(query)
        sample = self.get_sample(idx, query=sample_queries, debug=True)
        # print('sample has ', sample.keys())
        # self.save_for_viz(sample, index=idx)

        # vis_utils.visualize_pointcloud(sample['inputs'], title_name='inputs', backend='pyrender')
        # vis_utils.visualize_pointcloud(sample['points'], title_name='points', labels=sample['points.occ'], backend='pyrender')
        # img = sample[TransQueries.images].numpy().transpose(1, 2, 0)

        canon_pts = sample['canon_pts']
        boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1])/2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
        canon_pts_normalized = (canon_pts - center_pt.reshape(1, 3)) / length_bb + 0.5
        # vis_utils.visualize_pointcloud([sample[BaseQueries.pcloud], sample['full_pcloud']], title_name='input depth', backend='pyrender')
        # vis_utils.visualize_pointcloud([sample[TransQueries.verts3d], sample[TransQueries.objpoints3d]], title_name='verts & object', backend='pyrender')
        vis_utils.visualize_pointcloud([sample[BaseQueries.nocs], canon_pts_normalized], title_name='points', labels=sample['points.occ'], backend='pyrender')
        #
        # # Display XY projection
        # ax = fig.add_subplot(142)
        # self.display_proj(ax, sample, proj="z", joint_idxs=joint_idxs)
        #
        # # Display YZ projection
        # ax = fig.add_subplot(121, projection='3d')
        # self.display_nocs(ax, sample, proj="y", joint_idxs=joint_idxs)
        #
        # # Display XZ projection
        # ax = fig.add_subplot(122, projection='3d')
        # self.display_3d(ax, sample, proj="y", joint_idxs=joint_idxs)
        #
        # show 3d points

    def save_for_viz(self, batch, config=None, index=0):
        save_viz = True
        save_path= './media'
        if save_viz:
            viz_dict = batch
            print('!!! Saving visualization data')
            save_viz_path = f'{save_path}/full_viz/'
            if not exists(save_viz_path):
                makedirs(save_viz_path)
            save_viz_name = f'{save_viz_path}{index}_data.npy'
            print('saving to ', save_viz_name)
            np.save(save_viz_name, arr=viz_dict)

    def visualize_transformed(self, idx, joint_idxs=False):
        queries = [
            BaseQueries.images,
            BaseQueries.joints2d,
            BaseQueries.sides,
            BaseQueries.objverts3d,
            TransQueries.images,
            TransQueries.joints2d,
            TransQueries.objverts3d,
            BaseQueries.objfaces,
            TransQueries.camintrs,
            TransQueries.center3d,
            TransQueries.objpoints3d,
        ]
        sample_queries = []
        for query in queries:
            if query in self.pose_dataset.all_queries:
                sample_queries.append(query)
        sample = self.get_sample(idx, query=sample_queries)
        fig = plt.figure(figsize=(50, 20))
        ax = fig.add_subplot(111)
        img = sample[TransQueries.images].numpy().transpose(1, 2, 0)
        if not self.normalize_img:
            img += 0.5
        ax.imshow(img)
        # Add title
        if BaseQueries.sides in sample:
            side = sample[BaseQueries.sides]
            ax.set_title("{} hand".format(side))

        if TransQueries.joints2d in sample:
            joints2d = sample[TransQueries.joints2d]
            vis_utils.visualize_joints_2d(
                ax,
                joints2d,
                joint_idxs=joint_idxs,
                links=self.pose_dataset.links,
            )
        if (
            TransQueries.camintrs in sample
            and (TransQueries.objverts3d in sample)
            and BaseQueries.objfaces in sample
        ):
            verts = (
                torch.from_numpy(sample[TransQueries.objverts3d])
                .unsqueeze(0)
                .cuda()
            )
            center3d = (
                torch.from_numpy(sample[TransQueries.center3d])
                .cuda()
                .unsqueeze(0)
            )
            verts = center3d.unsqueeze(1) / 1000 + verts / 1000
        plt.show()


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)
