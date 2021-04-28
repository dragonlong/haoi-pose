import numpy as np
import os
import glob
import h5py
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
    def __init__(self, opt, root=None, mode=None, npoints=16384, novel_input=False, novel_input_only=False):
        super(ShapeNetH5, self).__init__()
        self.opt = opt

        self.mode = opt.mode if mode is None else mode
        if 'val' in self.mode:
            self.mode = 'test'
        self.anchors = L.get_anchors()
        if root is None:
            dataset_path    = '/groups/arcadm/xiaolong/mvp'
        else:
            data_path = root
        self.input_path = f'{dataset_path}/mvp_{self.mode}_input.h5'
        self.gt_path = f'{dataset_path}/mvp_{self.mode}_gt_{npoints}pts.h5'

        self.npoints = npoints
        input_file = h5py.File(self.input_path, 'r')
        self.input_data = np.array((input_file['incomplete_pcds'][()]))
        self.labels = np.array((input_file['labels'][()]))
        self.novel_input_data = np.array((input_file['novel_incomplete_pcds'][()]))
        self.novel_labels = np.array((input_file['novel_labels'][()]))
        input_file.close()

        gt_file = h5py.File(self.gt_path, 'r')
        self.gt_data = np.array((gt_file['complete_pcds'][()]))
        self.novel_gt_data = np.array((gt_file['novel_complete_pcds'][()]))
        gt_file.close()

        if opt.target_category == 'airplane':
            select_id = 0
        elif opt.target_category == 'chair':
            select_id = 3
        else:
            select_id= -1
        if select_id > -1:
            self.idxs = np.where(self.labels==select_id)[0]
        else:
            self.idxs = np.arange(self.labels.shape[0])
        # if novel_input_only:
        #     self.input_data = self.novel_input_data
        #     self.gt_data = self.novel_gt_data
        #     self.labels = self.novel_labels
        # elif novel_input:
        #     self.input_data = np.concatenate((self.input_data, self.novel_input_data), axis=0)
        #     self.gt_data = np.concatenate((self.gt_data, self.novel_gt_data), axis=0)
        #     self.labels = np.concatenate((self.labels, self.novel_labels), axis=0)

        print(self.input_data.shape)
        print(self.gt_data.shape)
        print(self.labels.shape)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        index   = self.idxs[idx]
        # complete = torch.from_numpy((self.gt_data[index // 26]))
        data = {}
        data['pc'] = self.input_data[index]
        data['label'] = self.labels[index]
        data['name']  = ['0']
        _, pc = pctk.uniform_resample_np(data['pc'], self.opt.model.input_num)
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

if __name__ == '__main__':
    from models.spconv.options import opt
    BS = 2
    N  = 1024
    C  = 3
    device = torch.device("cuda:0")
    x = torch.randn(BS, N, 3).to(device)
    opt.model.model = 'inv_so3net'
    opt.target_category ='airplane'
    opt.augment = True
    dset = ShapeNetH5(opt, mode='test')
    for i in range(10):
        dp = dset.__getitem__(i)
        print(dp)
    print('Con!')
