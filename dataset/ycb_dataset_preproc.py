import hydra
from PIL import Image
import os
import os.path
from os.path import join as pjoin
import numpy as np
import numpy.ma as ma
import scipy.io as scio
import argparse


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

@hydra.main(config_path="../config/completion.yaml")
def main(cfg):
    for mode in ['train', 'test']:
        get_per_instance_lists(mode, root=cfg.ycb_root)

if __name__ == '__main__':
    main()