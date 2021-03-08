import os
import pickle

import cv2
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
from igl import read_triangle_mesh, signed_distance
from meshplot import plot, subplot, interact

import __init__
from global_info import global_info
from common.queries import BaseQueries, get_trans_queries
from common import handutils
from common.data_utils import fast_load_obj
from common.d3_utils import calculate_3d_backprojections, align_rotation
from common.vis_utils import plot_imgs
from common.yj_utils import print_composite

from utils.external import binvox_rw, voxels
from utils.external.libmesh import check_mesh_contains
ImageFile.LOAD_TRUNCATED_IMAGES = True
def bp():
    import pdb;pdb.set_trace()

# try using custom packages
infos     = global_info()
my_dir    = infos.base_path
group_path= infos.group_path
second_path = infos.second_path
render_path = infos.render_path
project_path= infos.project_path
categories  = infos.categories
categories_id = infos.categories_id
symmetry_dict = infos.symmetry_dict


class ObMan:
    def __init__(
        self,
        split="train",
        mode="obj",
        joint_nb=21,
        mini_factor=None,
        use_cache=False,
        root_palm=False,
        segment=False,
        override_scale=False,
        use_external_points=True,
        apply_obj_transform=True,
        segmented_depth=True,
        shapenet_root="datasymlinks/ShapeNetCore.v2",
        obman_root="datasymlinks/obman",
    ):
        # Set cache path
        self.split = split
        self.mode  = mode
        obman_root = os.path.join(obman_root, split)
        self.override_scale = override_scale  # Use fixed scale
        self.root_palm = root_palm
        self.obman_root= obman_root
        self.segment = segment
        self.apply_obj_transform = apply_obj_transform
        self.segmented_depth = segmented_depth

        self.use_external_points = use_external_points
        if mode == "all" and not self.override_scale:
            self.all_queries = [
                BaseQueries.images,
                BaseQueries.joints2d,
                BaseQueries.joints3d,
                BaseQueries.sides,
                BaseQueries.segms,
                BaseQueries.verts3d,
                BaseQueries.hand_pcas,
                BaseQueries.hand_poses,
                BaseQueries.camintrs,
                BaseQueries.depth,
                BaseQueries.nocs,
                BaseQueries.meta,
                BaseQueries.pcloud,
                BaseQueries.occupancy,
            ]
            if use_external_points:
                self.all_queries.append(BaseQueries.objpoints3d)
            else:
                self.all_queries.append(BaseQueries.objverts3d)
                self.all_queries.append(BaseQueries.objfaces)
            self.rgb_folder = os.path.join(obman_root, "rgb")
        elif mode == "obj" or (self.mode == "all" and self.override_scale):
            self.all_queries = [BaseQueries.images, BaseQueries.camintrs]
            if use_external_points:
                self.all_queries.append(BaseQueries.objpoints3d)
            else:
                self.all_queries.append(BaseQueries.objpoints3d)
                self.all_queries.append(BaseQueries.objverts3d)
                self.all_queries.append(BaseQueries.objfaces)
            if mode == "obj":
                self.rgb_folder = os.path.join(obman_root, "rgb_obj")
            else:
                self.rgb_folder = os.path.join(obman_root, "rgb")
        elif mode == "hand":
            self.all_queries = [
                BaseQueries.images,
                BaseQueries.joints2d,
                BaseQueries.joints3d,
                BaseQueries.sides,
                BaseQueries.segms,
                BaseQueries.verts3d,
                BaseQueries.hand_pcas,
                BaseQueries.hand_poses,
                BaseQueries.camintrs,
                BaseQueries.depth,
            ]
            self.rgb_folder = os.path.join(obman_root, "rgb_hand")
        else:
            raise ValueError(
                "Mode should be in [all|obj|hand], got {}".format(mode)
            )

        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.extend(trans_queries)

        # Cache information
        self.use_cache = use_cache
        self.name = "obman"
        self.cache_folder = os.path.join(project_path, "haoi-pose/dataset/data", "cache", self.name)
        os.makedirs(self.cache_folder, exist_ok=True)
        self.mini_factor = mini_factor
        self.cam_intr = np.array(
            [[480.0, 0.0, 128.0], [0.0, 480.0, 128.0], [0.0, 0.0, 1.0]]
        ).astype(np.float32)

        self.cam_extr = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
            ]
        ).astype(np.float32)

        self.joint_nb = joint_nb
        self.segm_folder = os.path.join(obman_root, "segm")

        self.prefix_template = "{:08d}"
        self.meta_folder = os.path.join(obman_root, "meta")
        self.coord2d_folder = os.path.join(obman_root, "coords2d")

        # Define links on skeleton
        self.links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]
        # Load mano faces
        self.faces = {}
        for side in ['left', 'right']:
            with open(f'{project_path}/haoi-pose/dataset/mano_faces_{side}.pkl', 'rb') as p_f:
                self.faces[side] = pickle.load(p_f)

        # Object info
        self.shapenet_template = os.path.join(
            shapenet_root, "{}/{}/models/model_normalized.pkl"
        )
        self.load_dataset()

    def _get_image_path(self, prefix):
        image_path = os.path.join(self.rgb_folder, "{}.jpg".format(prefix))

        return image_path

    def load_dataset(self):
        # pkl_path = "/sequoia/data1/yhasson/code/\
        #             pose_3d/mano_render/mano/models/MANO_RIGHT_v1.pkl"
        pkl_path = '../../manopth/mano/models/MANO_RIGHT.pkl'
        if not os.path.exists(pkl_path):
            # print('---not existing !!! ')
            pkl_path = "/home/lxiaol9/3DGenNet2019/manopth/mano/models/MANO_RIGHT.pkl"

        cache_path = os.path.join(
            self.cache_folder,
            "{}_{}_mode_{}.pkl".format(
                self.split, self.mini_factor, self.mode
            ),
        )
        file_to_match = [] if not os.path.exists(self.cache_folder) else os.listdir(self.cache_folder)
        cache_chunk_prefix = f'{self.split}_{self.mini_factor}_{self.mode}'
        cache_chunk_files = [file for file in file_to_match if file.startswith(cache_chunk_prefix) and file.endswith('.pkl')]
        cache_chunk_files = sorted(cache_chunk_files, key=lambda file: int(file.split('.')[-2].split('_')[-1]))
        if os.path.exists(cache_path) and self.use_cache:
            with open(cache_path, "rb") as cache_f:
                annotations = pickle.load(cache_f)
            print(
                "Cached information for dataset {} loaded from {}".format(
                    self.name, cache_path
                )
            )
        elif len(cache_chunk_files) > 0:
            annotations = {}
            for filename in cache_chunk_files:
                with open(os.path.join(self.cache_folder, filename), "rb") as cache_f:
                    cur_anno = pickle.load(cache_f)
                for key, value in cur_anno.items():
                    if key not in annotations:
                        annotations[key] = []
                    annotations[key] += value
            print(
                "Cached information for dataset {} loaded from {}".format(
                    self.name, cache_chunk_files
                )
            )
        else:
            idxs = [
                int(imgname.split(".")[0])
                for imgname in sorted(os.listdir(self.meta_folder))
            ]

            if self.mini_factor:
                mini_nb = int(len(idxs) * self.mini_factor)
                idxs = idxs[:mini_nb]

            prefixes = [self.prefix_template.format(idx) for idx in idxs]
            print(
                "Got {} samples for split {}, generating cache !".format(
                    len(idxs), self.split
                )
            )

            image_names = []
            all_joints2d = []
            all_joints3d = []
            hand_sides = []
            hand_poses = []
            hand_pcas = []
            hand_verts3d = []
            obj_paths = []
            obj_transforms = []
            meta_infos = []
            depth_infos = []

            # for debugging
            for idx, prefix in enumerate(tqdm(prefixes)):
                meta_path = os.path.join(
                    self.meta_folder, "{}.pkl".format(prefix)
                )
                with open(meta_path, "rb") as meta_f:
                    meta_info = pickle.load(meta_f)
                image_path = self._get_image_path(prefix)
                image_names.append(image_path)
                all_joints2d.append(meta_info["coords_2d"])
                all_joints3d.append(meta_info["coords_3d"])
                hand_verts3d.append(meta_info["verts_3d"])
                hand_sides.append(meta_info["side"])
                hand_poses.append(meta_info["hand_pose"])
                hand_pcas.append(meta_info["pca_pose"])
                depth_infos.append(
                    {
                        "depth_min": meta_info["depth_min"],
                        "depth_max": meta_info["depth_max"],
                        "hand_depth_min": meta_info["hand_depth_min"],
                        "hand_depth_max": meta_info["hand_depth_max"],
                        "obj_depth_min": meta_info["obj_depth_min"],
                        "obj_depth_max": meta_info["obj_depth_max"],
                    }
                )
                obj_path = self._get_obj_path(
                    meta_info["class_id"], meta_info["sample_id"]
                )

                obj_paths.append(obj_path)
                obj_transforms.append(meta_info["affine_transform"])
                meta_info_full = {
                    "obj_scale": meta_info["obj_scale"],
                    "obj_class_id": meta_info["class_id"],
                    "obj_sample_id": meta_info["sample_id"],
                }
                if "grasp_quality" in meta_info:
                    meta_info_full["grasp_quality"] = meta_info[
                        "grasp_quality"
                    ]
                    meta_info_full["grasp_epsilon"] = meta_info[
                        "grasp_epsilon"
                    ]
                    meta_info_full["grasp_volume"] = meta_info["grasp_volume"]
                meta_infos.append(meta_info_full)

            annotations = {
                "depth_infos": depth_infos,
                "image_names": image_names,
                "joints2d": all_joints2d,
                "joints3d": all_joints3d,
                "hand_sides": hand_sides,
                "hand_poses": hand_poses,
                "hand_pcas": hand_pcas,
                "hand_verts3d": hand_verts3d,
                "obj_paths": obj_paths,
                "obj_transforms": obj_transforms,
                "meta_infos": meta_infos,
            }
            print(
                "class_nb: {}".format(
                    np.unique(
                        [
                            (meta_info["obj_class_id"])
                            for meta_info in meta_infos
                        ],
                        axis=0,
                    ).shape
                )
            )
            print(
                "sample_nb : {}".format(
                    np.unique(
                        [
                            (
                                meta_info["obj_class_id"],
                                meta_info["obj_sample_id"],
                            )
                            for meta_info in meta_infos
                        ],
                        axis=0,
                    ).shape
                )
            )
            if group_path.startswith('/orion'):
                total = len(prefixes)
                chunk_size = 20000
                num_chunks = (total + chunk_size - 1) // chunk_size
                for i_chunk in range(num_chunks):
                    end = min(total, (i_chunk + 1) * chunk_size)
                    cur_anno = {key: value[i_chunk * chunk_size: end] for key, value in annotations.items()}
                    cache_chunk_path = os.path.join(self.cache_folder,
                                                    f'{self.split}_{self.mini_factor}_{self.mode}_{i_chunk}.pkl')
                    with open(cache_chunk_path, "wb") as fid:
                        pickle.dump(cur_anno, fid)
                    print(
                        "Wrote cache [{}:{}] for dataset {} to {}".format(
                            i_chunk * chunk_size, end,
                            self.name, cache_chunk_path
                        )
                    )
            else:
                with open(cache_path, "wb") as fid:
                    pickle.dump(annotations, fid)
                print(
                    "Wrote cache for dataset {} to {}".format(
                        self.name, cache_path
                    )
                )

        # Set dataset attributes
        all_objects = [
            obj[:-7].split("/")[-1].split("_")[0]
            for obj in annotations["obj_paths"]
        ]
        selected_idxs = list(range(len(all_objects)))
        obj_paths = [annotations["obj_paths"][idx] for idx in selected_idxs]
        image_names = [
            annotations["image_names"][idx] for idx in selected_idxs
        ]
        joints3d = [annotations["joints3d"][idx] for idx in selected_idxs]
        joints2d = [annotations["joints2d"][idx] for idx in selected_idxs]
        hand_sides = [annotations["hand_sides"][idx] for idx in selected_idxs]
        hand_pcas = [annotations["hand_pcas"][idx] for idx in selected_idxs]
        hand_verts3d = [
            annotations["hand_verts3d"][idx] for idx in selected_idxs
        ]
        obj_transforms = [
            annotations["obj_transforms"][idx] for idx in selected_idxs
        ]
        #
        meta_infos = [annotations["meta_infos"][idx] for idx in selected_idxs]
        if "depth_infos" in annotations:
            has_depth_info = True
            depth_infos = [
                annotations["depth_infos"][idx] for idx in selected_idxs
            ]
        else:
            has_depth_info = False
        if has_depth_info:
            self.depth_infos = depth_infos
        self.image_names = image_names
        self.joints2d = joints2d
        self.joints3d = joints3d
        self.hand_sides = hand_sides
        self.hand_pcas = hand_pcas
        self.hand_verts3d = hand_verts3d
        self.obj_paths = obj_paths
        self.obj_transforms = obj_transforms
        self.meta_infos = meta_infos
        # Initialize cache for center and scale in case objects are used
        self.center_scale_cache = {}

    def get_meta(self, idx):
        pass

    def get_image(self, idx):
        image_path = self.image_names[idx]
        side = self.get_sides(idx)
        if self.segment:
            if self.mode == "all":
                segm_path = image_path.replace("rgb", "segm").replace(
                    "jpg", "png"
                )
            elif self.mode == "hand":
                segm_path = image_path.replace("rgb_hand", "segm").replace(
                    "jpg", "png"
                )
            elif self.mode == "obj":
                segm_path = image_path.replace("rgb_obj", "segm").replace(
                    "jpg", "png"
                )

            img = cv2.imread(image_path, 1)
            if img is None:
                raise ValueError("cv2 could not open {}".format(image_path))
            segm_img = cv2.imread(segm_path, 1)
            if segm_img is None:
                raise ValueError("cv2 could not open {}".format(segm_path))
            if self.mode == "all":
                segm_img = segm_img[:, :, 0]
            elif self.mode == "hand":
                segm_img = segm_img[:, :, 1]
            elif self.mode == "obj":
                segm_img = segm_img[:, :, 2]
            segm_img = _get_segm(segm_img, side=side)
            segm_img = segm_img.sum(2)[:, :, np.newaxis]
            # blacken not segmented
            img[~segm_img.astype(bool).repeat(3, 2)] = 0
            img = Image.fromarray(img[:, :, ::-1])
        else:
            img = Image.open(image_path)
            img = img.convert("RGB")
        return img

    def get_segm(self, idx, ext_cmd=None, pil_image=True, debug=False):
        side = self.get_sides(idx)
        image_path = self.image_names[idx]

        if self.mode == "all":
            image_path = image_path.replace("rgb", "segm").replace(
                "jpg", "png"
            )
        elif self.mode == "hand":
            image_path = image_path.replace("rgb_hand", "segm").replace(
                "jpg", "png"
            )
        elif self.mode == "obj":
            image_path = image_path.replace("rgb_obj", "segm").replace(
                "jpg", "png"
            )

        img = cv2.imread(image_path, 1)
        if debug:
            plot_imgs([img[:, :, 0], img[:, :, 1], img[:, :, 2]], ['all', 'hand', 'obj'])
        if img is None:
            raise ValueError("cv2 could not open {}".format(image_path))
        if ext_cmd=='obj':
            full_seg = img[:, :, 0]
            full_seg[np.where(full_seg!=100)] = 0
            return full_seg

        use_key = self.mode

        if use_key == "all":
            segm_img = _get_segm(img[:, :, 0], side=side)
        elif use_key == "hand":
            segm_img = _get_segm(img[:, :, 1], side=side)
        elif use_key == "obj":
            segm_img = _get_segm(img[:, :, 2], side=side)
        if pil_image:
            segm_img = Image.fromarray((255 * segm_img).astype(np.uint8))

        return segm_img

    def get_verts2d(self, idx):
        verts3d = self.get_verts3d(idx)
        verts2d = get_coords_2d(
            verts3d, cam_extr=None, cam_calib=self.cam_intr)
        return verts2d


    def get_joints2d(self, idx):
        return self.joints2d[idx].astype(np.float32)

    def get_joints3d(self, idx):
        joints3d = self.joints3d[idx]
        if self.root_palm:
            # Replace wrist with palm
            verts3d = self.hand_verts3d[idx]
            palm = (verts3d[95] + verts3d[218]) / 2
            joints3d = np.concatenate([palm[np.newaxis, :], joints3d[1:]])
        # No hom coordinates needed because no translation
        assert (
            np.linalg.norm(self.cam_extr[:, 3]) == 0
        ), "extr camera should have no translation"

        joints3d = self.cam_extr[:3, :3].dot(joints3d.transpose()).transpose()
        return 1000 * joints3d

    def get_verts3d(self, idx):
        verts3d = self.hand_verts3d[idx]
        if self.apply_obj_transform:
            verts3d = self.cam_extr[:3, :3].dot(verts3d.transpose()).transpose()
        else:
            obj_transform = np.linalg.pinv(self.obj_transforms[idx])
            hom_verts = np.concatenate([verts3d, np.ones([verts3d.shape[0], 1])],
                                       axis=1)
            verts3d = obj_transform.dot(hom_verts.T).T[:, :3]
        # return 1000 * verts3d
        return verts3d

    def get_surface_pts(self, idx, canon_pts=None, debug=False):
        # read obj
        model_path = self.obj_paths[idx]
        surface_pts_path = model_path.replace("model_normalized", "surface_points")
        all_pts = np.load(surface_pts_path, allow_pickle=True)
        # change target points
        boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1])/2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])

        # we care about the nomralized shape in NOCS
        points = (all_pts - center_pt.reshape(1, 3)) / length_bb + 0.5
        nomral_vectors = np.ones_like(points)
        return points, nomral_vectors

    def get_occupany_pts(self, idx, canon_pts=None, npt=2048, thres=0.01, debug=False):
        # read obj
        model_path = self.obj_paths[idx]
        outside_pts_path = model_path.replace("model_normalized", "outside_points")
        inside_pts_path  = model_path.replace("model_normalized", "inside_points")
        surface_pts_path = model_path.replace("model_normalized", "surface_points")
        # if os.path.exists(model_path):
        #     with open(model_path, "rb") as obj_f:
        #         mesh = pickle.load(obj_f)

        all_pts = []
        filenames = [inside_pts_path, surface_pts_path, outside_pts_path]
        for filename in filenames:
            if os.path.exists(filename):
                # with open(filename, "r") as m_f:
                #     all_pts.append(pickle.load(m_f))
                all_pts.append(np.load(filename, allow_pickle=True))
        if len(all_pts) < 3:
            return None, None

        length_list = [512, 512, 1024]
        final_pts = []
        for j in range(3):
            inds = np.random.choice(all_pts[j].shape[0], min(length_list[j], all_pts[j].shape[0]))
            final_pts.append(all_pts[j][inds])
            if len(inds) < length_list[j]:
                length_list[j+1] += length_list[j] - len(inds)

        occupancy_labels = np.ones((2048))
        occupancy_labels[-length_list[-1]:] = 0
        points = np.concatenate(final_pts, axis=0)

        # change target points
        boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1])/2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])

        # we care about the nomralized shape in NOCS
        points = (points - center_pt.reshape(1, 3)) / length_bb + 0.5

        # # v, f = read_triangle_mesh(model_path_obj)
        # if os.path.exists(model_path):
        #     with open(model_path, "rb") as obj_f:
        #         mesh = pickle.load(obj_f)
        # elif os.path.exists(model_path_obj):
        #     with open(model_path_obj, "r") as m_f:
        #         mesh = fast_load_obj(m_f)[0]
        # v, f = mesh['vertices'], mesh['faces']
        #
        # # random pick pts
        # pts = np.random.rand(npt2, 3) # 0-1
        #
        # # mix with surface points
        # if canon_pts is None:
        #     model_path = self.obj_paths[idx].replace(
        #         "model_normalized.pkl", "surface_points.pkl"
        #     )
        #     with open(model_path, "rb") as obj_f:
        #         canon_pts = pickle.load(obj_f)
        #
        # boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]
        # center_pt = (boundary_pts[0] + boundary_pts[1])/2
        # length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
        #
        # # decide sdf
        # v = (v - center_pt.reshape(1, 3)) / length_bb + 0.5
        # # S, _, _ = signed_distance(pts, v, f, return_normals=False)
        #
        # full_nocs = (canon_pts - center_pt.reshape(1, 3)) / length_bb + 0.5
        # idxs = np.random.choice(full_nocs.shape[0], npt1)
        # surface_pts = full_nocs[idxs]
        #
        # # mix pts
        # points = np.concatenate([pts, surface_pts], axis=0)
        # occupancy_labels = np.ones((npt))
        # occupancy_labels[np.where(S>thres)[0]] = 0

        return points, occupancy_labels

    def get_volume_pts(self, idx, canon_pts=None, npt=2048, thres=0.01):
        model_path = self.obj_paths[idx]
        outside_pts_path = model_path.replace("model_normalized", "outside_points")
        inside_pts_path  = model_path.replace("model_normalized", "inside_points")
        # if os.path.exists(model_path):
        #     with open(model_path, "rb") as obj_f:
        #         mesh = pickle.load(obj_f)

        all_pts = []
        filenames = [inside_pts_path, outside_pts_path]
        for filename in filenames:
            if os.path.exists(filename):
                # with open(filename, "r") as m_f:
                #     all_pts.append(pickle.load(m_f))
                all_pts.append(np.load(filename, allow_pickle=True))

        occupancy_labels0 = np.ones((all_pts[0].shape[0]))
        occupancy_labels1 = np.zeros((all_pts[1].shape[0]))
        occupancy_labels = np.concatenate([occupancy_labels0, occupancy_labels1], axis=0)
        points = np.concatenate(all_pts, axis=0)

        # change target points
        boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1])/2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])

        # we care about the nomralized shape in NOCS
        points = (points - center_pt.reshape(1, 3)) / length_bb + 0.5

        return points, occupancy_labels

    def get_obj_verts_faces(self, idx):
        model_path = self.obj_paths[idx]
        model_path_obj = model_path.replace('.pkl', '.obj')
        if os.path.exists(model_path):
            print('---loading ', model_path)
            with open(model_path, 'rb') as obj_f:
                mesh = pickle.load(obj_f)
        elif os.path.exists(model_path_obj):
            print('---loading ', model_path_obj)
            with open(model_path_obj, 'r') as m_f:
                mesh = fast_load_obj(m_f)[0]
        else:
            raise ValueError(
                'Could not find model pkl or obj file at {}'.format(
                    model_path.split('.')[-2]))

        verts = mesh['vertices']
        canon_pts = np.copy(verts)
        # Apply transforms
        if self.apply_obj_transform:
            obj_scale = self.meta_infos[idx]["obj_scale"]
            obj_transform = self.obj_transforms[idx]
            hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])],
                                       axis=1)
            trans_verts = obj_transform.dot(hom_verts.T).T[:, :3]
            trans_verts = self.cam_extr[:3, :3].dot(
                trans_verts.transpose()).transpose()
        else:
            trans_verts = verts
        return np.array(trans_verts).astype(np.float32), np.array(
            mesh['faces']).astype(np.int16), canon_pts

    # def get_obj_verts_faces(self, idx):
    #     model_path = self.obj_paths[idx]
    #     model_path_obj = model_path.replace(".pkl", ".obj")
    #     if os.path.exists(model_path):
    #         with open(model_path, "rb") as obj_f:
    #             mesh = pickle.load(obj_f)
    #     elif os.path.exists(model_path_obj):
    #         with open(model_path_obj, "r") as m_f:
    #             mesh = fast_load_obj(m_f)[0]
    #     else:
    #         raise ValueError(
    #             "Could not find model pkl or obj file at {}".format(
    #                 model_path.split(".")[-2]
    #             )
    #         )
    #
    #     obj_scale = self.meta_infos[idx]["obj_scale"]
    #     if self.mode == "obj" or self.override_scale:
    #         verts = mesh["vertices"] * 0.18
    #     else:
    #         verts = mesh["vertices"] * obj_scale
    #     canon_pts = np.copy(verts)
    #     # boundary_pts = [np.min(verts, axis=0), np.max(verts, axis=0)]
    #     # Apply transforms
    #     if self.apply_obj_transform:
    #         obj_transform = self.obj_transforms[idx]
    #         hom_verts = np.concatenate(
    #             [verts, np.ones([verts.shape[0], 1])], axis=1
    #         )
    #         trans_verts = obj_transform.dot(hom_verts.T).T[:, :3]
    #         trans_verts = (
    #             self.cam_extr[:3, :3].dot(trans_verts.transpose()).transpose()
    #         )
    #     else:
    #         trans_verts = verts
    #     return (
    #         np.array(trans_verts).astype(np.float32) * 1000,
    #         np.array(mesh["faces"]).astype(np.int16),
    #         canon_pts
    #     )

    def get_objpoints3d(self, idx, point_nb=600):
        model_path = self.obj_paths[idx].replace(
            "model_normalized.pkl", "surface_points.pkl"
        )
        with open(model_path, "rb") as obj_f:
            points = pickle.load(obj_f)
        # Apply scaling
        if self.mode == "obj" or self.override_scale:
            points = points * 0.18
        else:
            points = points
        canon_pts = np.copy(points)
        # obj_scale = self.meta_infos[idx]["obj_scale"]
        # print('obj_scale ', obj_scale)
        # canon_pts = canon_pts * obj_scale
        # boundary_pts = [np.min(points, axis=0), np.max(points, axis=0)]
        # Filter very far outlier points from modelnet/shapenet !!
        point_nb_or = points.shape[0]
        points = points[
            np.linalg.norm(points, 2, 1)
            < 20 * np.median(np.linalg.norm(points, 2, 1))
        ]
        if points.shape[0] < point_nb_or:
            print(
                "Filtering {} points out of {} "
                "for sample {} from split {}".format(
                    point_nb_or - points.shape[0],
                    point_nb_or,
                    self.image_names[idx],
                    self.split,
                )
            )
        # Sample points
        idxs   = np.random.choice(points.shape[0], point_nb)
        points = points[idxs]
        # Apply transforms
        if self.apply_obj_transform:
            obj_transform = self.obj_transforms[idx]
            hom_points = np.concatenate(
                [points, np.ones([points.shape[0], 1])], axis=1
            )
            trans_points = obj_transform.dot(hom_points.T).T[:, :3]
            trans_points = (
                self.cam_extr[:3, :3].dot(trans_points.transpose()).transpose()
            )
        else:
            trans_points = points

        return trans_points.astype(np.float32) * 1000, canon_pts

    def get_nocs(self, idx, points, boundary_pts, sym_aligned_nocs=False):
        # cam2world
        # boundary_pts = [np.min(canon_pts, axis=0), np.max(canon_pts, axis=0)]
        c2w_mat = np.linalg.pinv(self.cam_extr[:3, :3])
        trans_points = (
            c2w_mat.dot(points.transpose()).transpose()
        )
        # world2canonical
        obj_transform = self.obj_transforms[idx]

        # consider_symmetry:
        class_id      = self.meta_infos[idx]['obj_class_id']
        category_name = categories_id[class_id]
        instance_id   = self.meta_infos[idx]['obj_sample_id']

        if sym_aligned_nocs and symmetry_dict[f'{class_id}_{instance_id}']: # : # remove any y rotation
            # print('---', class_id, ' transformed!!')
            obj_transform = align_rotation(obj_transform)

        obj_transform_inv = np.linalg.pinv(obj_transform)
        hom_points = np.concatenate([trans_points, np.ones([trans_points.shape[0], 1])], axis=1)
        trans_points = obj_transform_inv.dot(hom_points.T).T[:, :3]

        # normalize
        # print('cannonical min', np.min(trans_points, axis=0), boundary_pts[0])
        # print('cannonical max', np.max(trans_points, axis=0), boundary_pts[1])
        center_pt = (boundary_pts[0] + boundary_pts[1])/2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])
        nocs = (trans_points - center_pt.reshape(1, 3)) / length_bb + 0.5

        # return nocs
        return nocs

    def get_sides(self, idx):
        return self.hand_sides[idx]

    def get_camintr(self, idx):
        return self.cam_intr

    def get_pcloud(self, depth, camintr):
        cloud = calculate_3d_backprojections(depth, camintr, height=depth.shape[0], width=depth.shape[1], verbose=False)

        return cloud

    def get_faces3d(self, idx):
        faces = self.faces[self.get_sides(idx)]
        return faces

    def get_depth(self, idx):
        image_path = self.image_names[idx]
        if self.mode == "all":
            image_path = image_path.replace("rgb", "depth")
        elif self.mode == "hand":
            image_path = image_path.replace("rgb_hand", "depth")
        elif self.mode == "obj":
            image_path = image_path.replace("rgb_obj", "depth")
        image_path = image_path.replace("jpg", "png")

        img = cv2.imread(image_path, 1)
        img_hand = img[:, :, 1]
        img_obj  = img[:, :, 2]
        if img is None:
            raise ValueError("cv2 could not open {}".format(image_path))

        depth_info = self.depth_infos[idx]
        if self.mode == "all":
            img = img[:, :, 0]
            depth_max = depth_info["depth_max"]
            depth_min = depth_info["depth_min"]
        elif self.mode == "hand":
            img = img[:, :, 1]
            depth_max = depth_info["hand_depth_max"]
            depth_min = depth_info["hand_depth_min"]
        elif self.mode == "obj":
            img = img[:, :, 2]
            depth_max = depth_info["obj_depth_max"]
            depth_min = depth_info["obj_depth_min"]
        assert (
            img.max() == 255
        ), "Max value of depth jpg should be 255, not {}".format(img.max())

        img = (img - 1) / 254 * (depth_min - depth_max) + depth_max
        # img_hand = (img_hand - 1) / 254 * (depth_info["hand_depth_min"] - depth_info["hand_depth_max"]) + depth_info["hand_depth_max"]
        # img_obj = (img_obj - 1) / 254 * (depth_info["obj_depth_min"] - depth_info["obj_depth_max"]) + depth_info["obj_depth_max"]
        # print(np.max(img), np.min(img))
        if self.segmented_depth:
            obj_hand_segm = (np.asarray(self.get_segm(idx)) / 255).astype(
                np.int
            )
            segm = obj_hand_segm[:, :, 0] | obj_hand_segm[:, :, 1]
            img = img * segm
        return img

    def get_center_scale(self, idx, scale_factor=2.2):
        if self.mode == "obj" or self.override_scale:
            if idx not in self.center_scale_cache:
                segm = self.get_segm(idx, pil_image=False)
                min_y = np.nonzero(segm[:, :, 1].sum(1))[0].min()
                max_y = np.nonzero(segm[:, :, 1].sum(1))[0].max()
                min_x = np.nonzero(segm[:, :, 1].sum(0))[0].min()
                max_x = np.nonzero(segm[:, :, 1].sum(0))[0].max()
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                scale = scale_factor * np.max([max_y - min_y, max_x - min_x])
                center = np.array([center_x, center_y])
                self.center_scale_cache[idx] = (center, scale)
            else:
                center, scale = self.center_scale_cache[idx]
        else:
            joints2d = self.get_joints2d(idx)
            center = handutils.get_annot_center(joints2d)
            scale = handutils.get_annot_scale(
                joints2d, scale_factor=scale_factor
            )
        return center, scale

    def _get_obj_path(self, class_id, sample_id):
        shapenet_path = self.shapenet_template.format(class_id, sample_id)
        return shapenet_path

    def __len__(self):
        return len(self.image_names)

    def get_obj_verts2d(self, idx):
        verts3d, _, _ = self.get_obj_verts_faces(idx)
        verts3d = verts3d
        verts2d = get_coords_2d(
            verts3d, cam_extr=None, cam_calib=self.cam_intr)
        return verts2d

def _get_segm(img, side="left"):
    if side == "right":
        hand_segm_img = (img == 22).astype(float) + (img == 24).astype(float)
    elif side == "left":
        hand_segm_img = (img == 21).astype(float) + (img == 23).astype(float)
    else:
        raise ValueError("Got side {}, expected [right|left]".format(side))

    obj_segm_img = (img == 100).astype(float)
    segm_img = np.stack(
        [hand_segm_img, obj_segm_img, np.zeros_like(hand_segm_img)], axis=2
    )
    return segm_img

def get_coords_2d(coords3d, cam_extr=None, cam_calib=None):
    if cam_extr is None:
        coords2d_hom = np.dot(cam_calib, coords3d.transpose())
    else:
        coords3d_hom = np.concatenate(
            [coords3d, np.ones((coords3d.shape[0], 1))], 1)
        coords3d_hom = coords3d_hom.transpose()
        coords2d_hom = np.dot(cam_calib, np.dot(cam_extr, coords3d_hom))
    coords2d = coords2d_hom / coords2d_hom[2, :]
    coords2d = coords2d[:2, :]
    return coords2d.transpose()
