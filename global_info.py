import os
import sys
import collections
import os
import sys
import platform
import numpy as np
import glob

def bp():
    import pdb;pdb.set_trace()

class setting():
    def __init__(self):
        self.USE_MULTI_GPU=True
        self.USE_RV_PRED  =True
        self.USE_BEV_PRED =False
        self.USE_PT_PRED  =False

DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['dataset_name', 'num_object', 'parts_map', 'num_parts', 'train_size', 'test_size', 'train_list', 'test_list', 'spec_list', 'spec_map', 'exp', 'baseline', 'joint_baseline',  'style']
)

TaskData= collections.namedtuple('TaskData', ['query', 'target'])

""" obman meta infos
(Pdb) print(annotations.keys())
dict_keys(['depth_infos', 'image_names', 'joints2d', 'joints3d', 'hand_sides', 'hand_poses', 'hand_pcas', 'hand_verts3d', 'obj_paths', 'obj_transforms', 'meta_infos'])
(Pdb) print(annotations['meta_infos'].keys())
*** AttributeError: 'list' object has no attribute 'keys'
(Pdb) print(annotations['meta_infos'][0])
{'obj_scale': 0.2, 'obj_class_id': '02876657', 'obj_sample_id': '860c81982a1703d35a7ce4c111af2db', 'grasp_quality': 0.3023727713275639, 'grasp_epsilon': 0.015956883019473657, 'grasp_volume': 0.03792773308413238}
"""

""" meta info
(Pdb) meta_info.keys()
dict_keys(['affine_transform', 'bg_path', 'grasp_epsilon', 'grasp_volume', 'obj_depth_min', 'coords_2d', 'obj_visibility_ratio', 'body_tex', 'sample_id', 'obj_scale', 'class_id', 'coords_3d', 'depth_min', 'obj_texture', 'side', 'obj_path', 'pose', 'trans', 'depth_max', 'z', 'grasp_quality', 'hand_depth_max', 'verts_3d', 'shape', 'hand_pose', 'hand_depth_min', 'obj_depth_max', 'pca_pose', 'sh_coeffs'])
(Pdb) meta_info['trans']
array([ 0.18896857,  0.46480602, -1.1284928 ], dtype=float32)
(Pdb) meta_info['pose'].shape
(156,)
(Pdb) meta_info['sh_coeffs'].shape
(9,)
(Pdb) meta_info['hand_pose'].shape
(45,)
(Pdb) meta_info['pca_pose'].shape
(45,)
(Pdb) meta_info['affine_transform']
array([[-0.01125364,  0.17572469,  0.09483772, -0.02914107],
       [ 0.14346975, -0.05895266,  0.12625773,  0.02036591],
       [ 0.13888767,  0.07513601, -0.12273872, -0.82511777],
       [ 0.        ,  0.        ,  0.        ,  1.        ]],
      dtype=float32)
"""
_DATASETS = dict(
    eyeglasses=DatasetInfo(
        dataset_name='shape2motion',
        num_object=24,
        parts_map=[[0], [1], [2], [3]],
        num_parts=4,
        train_size=13000,
        test_size=3480,
        train_list=None,
        test_list=['0007', '0016', '0036'],
        spec_list=['0006'],
        spec_map=None,
        exp='8.1',
        baseline='8.11',
        joint_baseline='8.12',
        style='new'
       ),
    obman=DatasetInfo(
        dataset_name='obman',
        num_object=3200,
        parts_map=[[0], [1]],
        num_parts=2,
        train_size=None,
        test_size=None,
        train_list=None,
        test_list=None,
        spec_list=None,
        spec_map=None,
        exp='8.1',
        baseline='8.11',
        joint_baseline='8.12',
        style='new'
       ),
    modelnet40=DatasetInfo(
        dataset_name='modelnet40',
        num_object=10000,
        parts_map=[[0], [1]],
        num_parts=2,
        train_size=None,
        test_size=None,
        train_list=None,
        test_list=None,
        spec_list=None,
        spec_map=None,
        exp='8.1',
        baseline='8.11',
        joint_baseline='8.12',
        style='new'
       ),
    modelnet40aligned=DatasetInfo(
        dataset_name='modelnet40aligned',
        num_object=10000,
        parts_map=[[0], [1]],
        num_parts=2,
        train_size=None,
        test_size=None,
        train_list=None,
        test_list=None,
        spec_list=None,
        spec_map=None,
        exp='8.1',
        baseline='8.11',
        joint_baseline='8.12',
        style='new'
       ),
    shapenetaligned=DatasetInfo(
        dataset_name='shapenetaligned',
        num_object=10000,
        parts_map=[[0], [1]],
        num_parts=2,
        train_size=None,
        test_size=None,
        train_list=None,
        test_list=None,
        spec_list=None,
        spec_map=None,
        exp='8.1',
        baseline='8.11',
        joint_baseline='8.12',
        style='new'
       ),
    oracle=DatasetInfo(
        dataset_name='oracle',
        num_object=10000,
        parts_map=[[0], [1]],
        num_parts=2,
        train_size=None,
        test_size=None,
        train_list=None,
        test_list=None,
        spec_list=None,
        spec_map=None,
        exp='8.1',
        baseline='8.11',
        joint_baseline='8.12',
        style='new'
       ),
    shapenet=DatasetInfo(
        dataset_name='shapenet',
        num_object=10000,
        parts_map=[[0], [1]],
        num_parts=2,
        train_size=None,
        test_size=None,
        train_list=None,
        test_list=None,
        spec_list=None,
        spec_map=None,
        exp='8.1',
        baseline='8.11',
        joint_baseline='8.12',
        style='new'
       ),
    nocs_synthetic=DatasetInfo(
        dataset_name='nocs_synthetic',
        num_object=10000,
        parts_map=[[0]],
        num_parts=1,
        train_size=None,
        test_size=None,
        train_list=None,
        test_list=None,
        spec_list=None,
        spec_map=None,
        exp='8.1',
        baseline='8.11',
        joint_baseline='8.12',
        style='new'
       ),
    humanhand=DatasetInfo(
        dataset_name='shape2motion',
        num_object=1,
        parts_map=[[0]],
        num_parts=1,
        train_size=13000,
        test_size=3480,
        train_list=None,
        test_list=['0006'],
        spec_list=['0007'],
        spec_map=None,
        exp=None,
        baseline=None,
        joint_baseline=None,
        style='new'
       ),
    # eyeglasses_hand=DatasetInfo(
    #     dataset_name='shape2motion',
    #     num_object=24,
    #     parts_map=[[0], [1], [2]],
    #     num_parts=3,
    #     train_size=13000,
    #     test_size=3480,
    #     train_list=None,
    #     test_list=['0005', '0016', '0036'],
    #     spec_list=['0006'],
    #     spec_map=None,
    #     exp='3.9',
    #     baseline='3.91',
    #     joint_baseline='5.0',
    #     style='new'
    #    ),

    oven=DatasetInfo(
        dataset_name='shape2motion',
        num_object=42,
        parts_map=[[0], [1]],
        num_parts=2,
        train_size=25000,
        test_size=5480,
        train_list=None,
        test_list=['0003', '0016', '0029'], # for dataset.py
        spec_list=['0006', '0015', '0035', '0038'], # for dataset.py
        spec_map=None,
        exp='3.0',
        baseline='3.01',
        joint_baseline='5.2',
        style='old'
       ),

    laptop=DatasetInfo(
        dataset_name='shape2motion',
        num_object=86,
        parts_map=[[0], [1]],
        num_parts=2,
        train_size=67603,
        test_size=5036,
        train_list=None,
        test_list=['0004', '0008', '0069'],
        spec_list=['0003', '0006', '0041', '0080', '0081'],
        spec_map=None,
        exp='3.6',
        baseline='3.61',
        joint_baseline='5.1',
        style='new'
       ),

    washing_machine=DatasetInfo(
        dataset_name='shape2motion',
        num_object=62,
        parts_map=[[0], [1]],
        num_parts=2,
        train_size=43000,
        test_size=3480,
        train_list=None,
        test_list=['0003', '0029'], # for dataset.py
        spec_list=['0001', '0002', '0006', '0007', '0010',
                   '0027', '0031', '0040', '0050', '0009',
                   '0029', '0038', '0039', '0041', '0046',
                   '0052', '0058'], # for dataset.py
       spec_map=None,
       exp='3.1',
       baseline='3.11',
       joint_baseline='5.3',
       style='old'
       ),

    Laptop=DatasetInfo(
        dataset_name='BMVC15',
        num_object=1,
        parts_map=[[0], [1]],
        num_parts=2,
        train_size=13000,
        test_size=3480,
        train_list=['0001'],
        test_list=['0006'],
        spec_list=[],
        spec_map=None,
        exp=None,
        baseline=None,
        joint_baseline=None,
        style=None
       ),

    Cabinet=DatasetInfo(
        dataset_name='BMVC15',
        num_object=1,
        parts_map=[[0], [1], [2]], # (001)base + (002)drawer + (000)door
        num_parts=3,
        train_size=13000,
        test_size=3480,
        train_list=['0001'],
        test_list=['0006'],
        spec_list=[],
        spec_map={'0001': [1, 2, 0], '0006':[1, 2, 0]},
        exp=None,
        baseline=None,
        joint_baseline=None,
        style=None
       ),

    Cupboard=DatasetInfo(
        dataset_name='BMVC15',
        num_object=1,
        parts_map=[[0], [1]], # base(000) + drawer(001)
        num_parts=2,
        train_size=13000,
        test_size=3480,
        train_list=['0001'],
        test_list=['0006'],
        spec_list=[],
        spec_map={'0001':[0, 1], '0006':[0, 1]},
        exp=None,
        baseline=None,
        joint_baseline=None,
        style=None
       ),

    Train=DatasetInfo(
        dataset_name='BMVC15',
        num_object=1,
        parts_map=[[0], [1], [2], [3]],
        num_parts=4,
        train_size=13000,
        test_size=3480,
        train_list=['0001'],
        test_list=['0006'],
        spec_list=[],
        spec_map={'0001':[0, 1, 2, 3], '0006':[0, 1, 2, 3]},
        exp=None,
        baseline=None,
        joint_baseline=None,
        style=None
       ),

    drawer=DatasetInfo(
        dataset_name='sapien',
        num_object=1,
        parts_map=[[0], [1], [2], [3]],
        num_parts=4,
        train_size=13000,
        test_size=3480,
        train_list=['40453', '44962', '45132',
                    '45290', '46130', '46334',  '46462',
                    '46537', '46544', '46641', '47178', '47183',
                    '47296', '47233', '48010', '48253',  '48517',
                    '48740', '48876', '46230', '44853', '45135',
                    '45427', '45756', '46653', '46879', '47438', '47711', '48491'],
        test_list=[ '46123',  '45841', '46440'],
        spec_list=[],
        spec_map={  '40453':[3, 0, 1, 2], '44962':[3, 0, 1, 2], '45132':[3, 0, 1, 2], '45290':[3, 0, 1, 2], '46123':[3, 0, 1, 2],
                    '46130':[3, 0, 1, 2], '46334':[3, 0, 1, 2], '46440':[3, 0, 1, 2], '46462':[3, 0, 1, 2], '46537':[3, 0, 1, 2],
                    '46544':[3, 0, 1, 2], '46641':[3, 0, 1, 2], '47178':[3, 0, 1, 2], '47183':[3, 0, 1, 2], '47296':[3, 0, 1, 2],
                    '47233':[3, 0, 1, 2], '48010':[3, 0, 1, 2], '48253':[3, 0, 1, 2], '48517':[3, 0, 1, 2], '48740':[3, 0, 1, 2],
                    '48876':[3, 0, 1, 2], '46230':[3, 0, 1, 2],
                    '44853':[3, 1, 2, 0], '45135':[3, 1, 0, 2], '45427':[3, 2, 0, 1], '45756':[3, 1, 2, 0], '45841':[0, 1, 2, 3],
                    '46653':[0, 1, 2, 3], '46879':[3, 1, 2, 0], '47438':[3, 2, 1, 0], '47711':[0, 1, 2, 3], '48491':[0, 1, 2, 3]},
        exp='3.3',
        baseline='3.31',
        joint_baseline='5.4',
        style='new'
       ),
)

class global_info(object):
    def __init__(self):
        self.name      = 'art6d'
        self.datasets  = _DATASETS
        self.model_type= 'pointnet++'
        self.group_path= None
        self.name_dataset = 'shape2motion'
        # print('---leveraging platform-wise global infos')
        # check dataset_name automactically
        group_path = None
        project_path = None
        if platform.uname()[0] == 'Darwin':
            print("Now it knows it's in my local Mac")
            base_path = '/Users/DragonX/Downloads/ARC/6DPOSE'
        elif platform.uname()[1] == 'viz1':
            base_path = '/home/xiaolong/Downloads/6DPOSE'
        elif platform.uname()[1] == 'vllab3':
            base_path = '/mnt/data/lxiaol9/rbo'
        elif platform.uname()[1] == 'dragon':
            base_path = '/home/dragon/Documents/ICML2021'
            second_path = '/home/dragon/Documents/ICML2021'
            group_path= '/home/dragon/Documents'
            project_path = '/home/dragon/Dropbox/ICML2021/code'
            # second_path = '/home/dragon/ARCwork'
            # mano_path = '/home/dragon/Downloads/ICML2021/YCB_Affordance/data/mano'
            mano_path = '/home/dragon/Dropbox/ICML2021/code/manopth/mano/models'
        elif platform.uname()[1].endswith('stanford.edu'):
            base_path = '/orion/u/yijiaw/projects/haoi/base'
            second_path = '/orion/u/yijiaw/projects/haoi/base'
            group_path ='/orion/u/yijiaw/projects/haoi'
            project_path = '/orion/u/yijiaw/projects/haoi/code'
            mano_path = '/orion/u/yijiaw/projects/haoi/code/manopth/mano/models'
        else:
            base_path = '/groups/CESCA-CV/ICML2021'
            group_path= '/groups/CESCA-CV'
            second_path = '/groups/CESCA-CV/ICML2021'
            mano_path = '/home/lxiaol9/3DGenNet2019/manopth/mano/models'
            project_path = '/home/lxiaol9/3DGenNet2019'
        self.platform_name = platform.uname()[1]
        self.render_path = second_path + '/data/render'
        self.viz_path  = second_path + '/data/images'
        self.hand_mesh = second_path + '/data/hands'
        self.hand_urdf = second_path + '/data/urdfs'
        self.grasps_meta = second_path + '/data/grasps'
        self.mano_path   = mano_path
        self.hand_path = self.hand_mesh
        self.urdf_path = self.hand_urdf

        self.whole_obj = second_path + '/data/objs'
        self.part_obj  = base_path + '/dataset/{}/objects'.format(self.name_dataset)
        self.obj_urdf  = base_path + '/dataset/{}/urdf'.format(self.name_dataset)
        self.second_path = second_path
        self.base_path = base_path
        self.group_path= group_path
        self.project_path= project_path
        self.categories_list = ['02876657', '03797390', '02880940', '02946921', '03593526', '03624134', '02992529', '02942699', '04074963']
        self.categories = { 'bottle': '02876657',   # 498
                            'mug': '03797390',      # 214
                            'bowl': '02880940',     # 186
                            'can': '02946921',      # 108
                            'jar':  '03593526',     # 596
                            'knife': '03624134',    # 424
                            'cellphone': '02992529',# 831
                            'camera': '02942699',   # 113,
                            'remote': '04074963',   # 66
                            'airplane': 'airplane',
                            'chair': 'chair',
                            'car': 'car'
                            }
        self.categories_id = { '02876657': 'bottle', # 498
                            '03797390': 'mug',  # 214, pointNet++
                            '02880940': 'bowl', # 186, vectors partial-dataset, try
                            '02946921': 'can' , # 108, vector 2-vector
                            '03593526': 'jar'  ,  # 596
                            '03624134': 'knife' , # 424
                            '02992529': 'cellphone' ,# 831
                            '02942699': 'camera', # 113,
                            '04074963': 'remote', # 66
                            'airplane': 'airplane',
                            'chair': 'chair',
                            'car': 'car'
                            } # need further classification
        self.symmetry_dict = np.load(f'{self.project_path}/haoi-pose/dataset/data/symmetry.npy', allow_pickle=True).item()
        sym_type = {}
        sym_type['bottle'] = {'y': 36} # use up axis
        sym_type['bowl']   = {'y': 36} # up axis!!!
        sym_type['can']    = {'y': 36, 'x': 2, 'z': 2} # up axis could be also 180 upside down
        sym_type['jar']    = {'y': 36, 'x': 2} # up axis only, + another axis? or 2 * 6D representations
        sym_type['mug']    = {'y': 1}  # up axis + ;
        sym_type['knife']  = {'y': 2, 'x': 2}  # up axis + ;
        sym_type['camera'] = {'y': 1}  # no symmetry; 6D predictions? or addtional axis!!!
        sym_type['remote'] = {'y': 2, 'x': 2}  # symmetric setting, 180 apply to R,
        sym_type['cellphone'] = {'y': 2, 'x': 2} # up axis has 2 groups, x axis has
        sym_type['airplane']= {'y': 1}
        sym_type['chair']= {'y': 1}
        sym_type['car']= {'y': 1}
        sym_type['laptop']= {'y': 1}
        self.sym_type = sym_type

        delta_R = {}
        delta_T = {}
        delta_R['modelnet40aligned_airplane'] =  np.array([[[ 0.2543,  0.4519,  0.8551], \
                 [ 0.8555, -0.5174,  0.0190], \
                 [ 0.4510,  0.7267, -0.5182]]])
        # train set
        delta_R['modelnet40aligned_chair'] =  np.array([[[ 0.0589, -0.9726, -0.2248],
                     [-0.3055,  0.1969, -0.9316],
                     [ 0.9504,  0.1235, -0.2855]]])
        delta_R['modelnet40aligned_car'] =  np.array([[[ 0.8886, -0.3725,  0.2675],
                     [-0.0436, -0.6493, -0.7592],
                     [ 0.4565,  0.6630, -0.5933]]])
        delta_R['modelnet40aligned_sofa'] =  np.array([[[ 0.5100,  0.8591, -0.0439],
                     [-0.4315,  0.2113, -0.8770],
                     [-0.7441,  0.4662,  0.4784]]])
        delta_R['modelnet40aligned_bowl'] =  np.array([[[ 0.7004, -0.6336,  0.3287],
                     [-0.0676,  0.3995,  0.9142],
                     [-0.7106, -0.6625,  0.2370]]])
        delta_R['modelnet40aligned_laptop'] =  np.array([[[ 0.1526,  0.2332,  0.9604],
                     [-0.4046, -0.8719,  0.2760],
                     [ 0.9017, -0.4307, -0.0387]]])
        delta_R['modelnet40aligned_cup'] =  np.array([[[ 0.2633,  0.9625,  0.0660],
                     [ 0.9440, -0.2711,  0.1880],
                     [ 0.1988,  0.0128, -0.9800]]])
        delta_R['modelnet40aligned_bottle'] =  np.array([[[ 0.4411, -0.8931, -0.0879],
                     [-0.2806, -0.2303,  0.9318],
                     [-0.8525, -0.3864, -0.3522]]])
        delta_R['modelnet40aligned_table'] =  np.array([[[ 0.4539, -0.3162, -0.8331],
                     [-0.8494,  0.1290, -0.5117],
                     [ 0.2693,  0.9399, -0.2101]]])

        delta_R['shapenetaligned_airplane']   = np.array([[[-0.0329,  0.8528, -0.5212],
                 [-0.7577,  0.3187,  0.5695],
                 [ 0.6518,  0.4136,  0.6357]]])
        delta_R['0.845_shapenetaligned_airplane']  = np.array([[[-0.2837, -0.9514, -0.1195],
                 [ 0.6972, -0.2902,  0.6555],
                 [-0.6584,  0.1027,  0.7456]]])
        delta_R['0.8451_shapenetaligned_airplane']  = np.array([[[ 0.4569,  0.8890,  0.0311],
                 [ 0.3217, -0.1325, -0.9375],
                 [-0.8293,  0.4384, -0.3465]]])
        delta_R['0.8475_shapenetaligned_airplane']  = np.array([[[ 0.4590, -0.6603, -0.5944],
                 [-0.7184, -0.6695,  0.1889],
                 [-0.5227,  0.3404, -0.7816]]])

        delta_R['0.84a_shapenetaligned_airplane']  = np.array([[[ 0.5541,  0.6258, -0.5490],
                 [ 0.0681, -0.6913, -0.7194],
                 [-0.8297,  0.3612, -0.4256]]])

        delta_R['0.84b_shapenetaligned_airplane']  = np.array([[[ 0.3676,  0.6781,  0.6364],
         [-0.3663,  0.7346, -0.5711],
         [-0.8548, -0.0232,  0.5185]]])

        delta_R['0.84c_shapenetaligned_airplane']  = np.array([[[-0.4660, -0.8710, -0.1553],
                     [-0.5925,  0.1768,  0.7860],
                     [-0.6571,  0.4583, -0.5985]]])
        delta_R['0.86_nocs_synthetic_laptop'] =  np.array([[[ 0.7726,  0.4386, -0.4590],
                 [-0.6077,  0.7203, -0.3345],
                 [ 0.1839,  0.5374,  0.8230]]])
        delta_R['0.861_nocs_synthetic_bowl'] =  np.array([[[ 0.4852, -0.6475, -0.5876],
                 [-0.7801, -0.6241,  0.0436],
                 [-0.3950,  0.4372, -0.8079]]])
        delta_R['0.862_nocs_synthetic_mug'] =  np.array([[[ 0.1988, -0.8803,  0.4308],
                 [-0.9784, -0.1533,  0.1385],
                 [-0.0559, -0.4490, -0.8918]]])
        delta_R['0.863_nocs_synthetic_laptop'] =  np.array([[[-0.0354,  0.3212, -0.9464],
                 [-0.5487,  0.7852,  0.2871],
                 [ 0.8353,  0.5294,  0.1484]]])


        # find all pre-computed delta_R, delta_T
        rt_files = glob.glob(f'{project_path}/haoi-pose/evaluation/infos/*.npy')
        for rt_file in rt_files:
            exp_num, name_dset, target_category = rt_file.split('/')[-1].split('.npy')[0].split('_')[:3]
            rt_dict = np.load(rt_file, allow_pickle=True).item()
            delta_R[f'{exp_num}_{name_dset}_{target_category}'] = rt_dict['delta_r']
            if 'delta_t' in rt_dict:
                delta_T[f'{exp_num}_{name_dset}_{target_category}'] = rt_dict['delta_t'].reshape(1, 3)
        self.delta_R = delta_R
        self.delta_T = delta_T

        # bottle 33278
        # mug 0
        # bowl 10301
        # can 5921
        # jar 37828
        # knife 9558
        # cellphone 34109
        # camera 7208
        # remote 3347
        # Final dataset size: 141550
        # >>>>> validation
        # bottle 1541
        # mug 0
        # bowl 432
        # can 242
        # jar 1728
        # knife 461
        # cellphone 1659
        # camera 244
        # remote 156

        # num = 0
        # for key, value in self.symmetry_dict.items():
        #     if value:
        #         num +=1
        # print(f'we have {num} symmetric objects')

if __name__ == '__main__':
    infos = global_info()
    print(infos.datasets['bike'].dataset_name)
