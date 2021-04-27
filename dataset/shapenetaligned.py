import numpy as np
import trimesh
import os
import glob
import scipy.io as sio
import torch
import torch.utils.data as data
import __init__
import vgtk.pc as pctk
import vgtk.point3d as p3dtk
import vgtk.so3conv.functional as L
from vgtk.functional import rotation_distance_np, label_relative_rotation_np
from scipy.spatial.transform import Rotation as sciR

def bp():
    import pdb;pdb.set_trace()

class ShapeNetH5(data.Dataset):
    def __init__(self, opt, mode=None, npoints=2048, novel_input=True, novel_input_only=False):
        super(ShapeNetH5, self).__init__()
        self.opt = opt

        self.mode = opt.mode if mode is None else mode
        if 'val' in self.mode:
            self.mode = 'test'
        self.anchors = L.get_anchors()

        dataset_path    = '/groups/arcadm/xiaolong/mvp'
        self.input_path = f'{dataset_path}/mvp_{self.mode}_input.h5'
        self.gt_path = f'{dataset_path}/mvp_{self.mode}_gt_{npoints}pts.h5'

        self.npoints = npoints
        bp()
        input_file = h5py.File(self.input_path, 'r')
        self.input_data = np.array((input_file['incomplete_pcds'][()]))
        # self.labels = np.array((input_file['labels'][()]))
        # self.novel_input_data = np.array((input_file['novel_incomplete_pcds'][()]))
        # self.novel_labels = np.array((input_file['novel_labels'][()]))
        input_file.close()

        gt_file = h5py.File(self.gt_path, 'r')
        self.gt_data = np.array((gt_file['complete_pcds'][()]))
        self.novel_gt_data = np.array((gt_file['novel_complete_pcds'][()]))
        gt_file.close()

        if novel_input_only:
            self.input_data = self.novel_input_data
            self.gt_data = self.novel_gt_data
            self.labels = self.novel_labels
        elif novel_input:
            self.input_data = np.concatenate((self.input_data, self.novel_input_data), axis=0)
            self.gt_data = np.concatenate((self.gt_data, self.novel_gt_data), axis=0)
            self.labels = np.concatenate((self.labels, self.novel_labels), axis=0)

        # print(self.input_data.shape)
        # print(self.gt_data.shape)
        # print(self.labels.shape)
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))
        # complete = torch.from_numpy((self.gt_data[index // 26]))
        # label = (self.labels[index])
        data = {}
        data['pc'] = partial


        _, pc = pctk.uniform_resample_np(data['pc'], self.opt.model.input_num)
        # pc = p3dtk.normalize_np(pc.T)
        # pc = pc.T
        # pc_canon = np.copy(pc)/2 + 0.5
        boundary_pts = [np.min(pc, axis=0), np.max(pc, axis=0)]
        center_pt = (boundary_pts[0] + boundary_pts[1])/2
        length_bb = np.linalg.norm(boundary_pts[0] - boundary_pts[1])

        # all normalize into 0
        pc_canon = (pc - center_pt.reshape(1, 3))/length_bb
        pc = np.copy(pc_canon)    # centered at 0
        pc_canon = pc_canon + 0.5 # NOCS space

        R = np.eye(3)
        R_label = 29
        t = np.random.rand(1, 3)
        if self.opt.augment:
            if 'R' in data.keys() and self.mode != 'train':
                pc, R = pctk.rotate_point_cloud(pc, data['R'])
            else:
                pc, R = pctk.rotate_point_cloud(pc)
            R_gt = np.copy(R)
        else:
            R_gt = np.copy(R)
        _, R_label, R0 = rotation_distance_np(R, self.anchors)

        return {'xyz': torch.from_numpy(pc.astype(np.float32)),
                'points': torch.from_numpy(pc_canon.astype(np.float32)),
                'label': torch.from_numpy(data['label'].flatten()).long(),
                'R_gt' : torch.from_numpy(R_gt.astype(np.float32)),
                'R_label': torch.Tensor([R_label]).long(),
                'R': R0,
                'T': torch.from_numpy(t.astype(np.float32)),
                'fn': data['name'][0],
                'id': index,
                'idx': index,
               }

# for relative rotation alignment
class Dataloader_ModelNet40Alignment(data.Dataset):
    def __init__(self, opt, mode=None):
        super(Dataloader_ModelNet40Alignment, self).__init__()
        self.opt = opt

        # 'train' or 'eval'
        self.mode = opt.mode if mode is None else mode

        # attention method: 'attention | rotation'
        self.flag = opt.model.flag
        self.anchors = L.get_anchors(self.opt.model.kanchor)

        cats = ['airplane']
        print(f"[Dataloader]: USING ONLY THE {cats[0]} CATEGORY!!")

        self.dataset_path = opt.DATASET.dataset_path
        self.all_data = []
        for cat in cats:
            for fn in glob.glob(os.path.join(opt.DATASET.dataset_path, cat, self.mode, "*.mat")):
                self.all_data.append(fn)
        print("[Dataloader] : Training dataset size:", len(self.all_data))


    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        data = sio.loadmat(self.all_data[index])
        _, pc = pctk.uniform_resample_np(data['pc'], self.opt.model.input_num)

        # normalization
        pc = p3dtk.normalize_np(pc.T)
        pc = pc.T
        pc_src, R_src = pctk.rotate_point_cloud(pc)
        pc_tgt = pc

        # if self.mode == 'test':
        #     data['R'] = R
        #     output_path = os.path.join(self.dataset_path, data['cat'][0], 'testR')
        #     os.makedirs(output_path,exist_ok=True)
        #     sio.savemat(os.path.join(output_path, data['name'][0] + '.mat'), data)
        # _, R_label, R0 = rotation_distance_np(R, self.anchors)

        # T = R_src @ R_tgt.T
        T = R_src # @ R_tgt.T

        # RR_regress = np.einsum('abc,bj,ijk -> aick', self.anchors, T, self.anchors)
        # R_label = np.argmax(np.einsum('abii->ab', RR_regress),axis=1)
        # idxs = np.vstack([np.arange(R_label.shape[0]), R_label]).T
        # R = RR_regress[idxs[:,0], idxs[:,1]]
        R, R_label = label_relative_rotation_np(self.anchors, T)
        pc_tensor = np.stack([pc_src, pc_tgt])

        return {'pc':torch.from_numpy(pc_tensor.astype(np.float32)),
                'fn': data['name'][0],
                'T' : torch.from_numpy(T.astype(np.float32)),
                'R': torch.from_numpy(R.astype(np.float32)),
                'R_label': torch.Tensor([R_label]).long(),
               }

if __name__ == '__main__':
    from models.spconv.options import opt
    BS = 2
    N  = 1024
    C  = 3
    device = torch.device("cuda:0")
    x = torch.randn(BS, N, 3).to(device)
    opt.model.model = 'inv_so3net'
    dset = ShapeNetH5(opt, mode='test')
    for i in range(10):
        dp = dset.__getitem__(i)
        print(dp)
    print('Con!')
