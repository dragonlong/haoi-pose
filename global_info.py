import os
import sys
import collections
import os
import sys
import platform

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
        exp='8.2',
        baseline='8.21',
        joint_baseline='8.22',
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

        # check dataset_name automactically
        group_path = None
        if platform.uname()[0] == 'Darwin':
            print("Now it knows it's in my local Mac")
            base_path = '/Users/DragonX/Downloads/ARC/6DPOSE'
        elif platform.uname()[1] == 'viz1':
            base_path = '/home/xiaolong/Downloads/6DPOSE'
        elif platform.uname()[1] == 'vllab3':
            base_path = '/mnt/data/lxiaol9/rbo'
        elif platform.uname()[1] == 'dragon':
            base_path = '/home/dragon/Documents/CVPR2020'
            second_path = '/home/dragon/Documents/ICML2021'
            # second_path = '/home/dragon/ARCwork'
            # mano_path = '/home/dragon/Downloads/ICML2021/YCB_Affordance/data/mano'
            mano_path = '/home/dragon/Dropbox/ICML2021/code/manopth/mano/models'
        else:
            base_path = '/work/cascades/lxiaol9/6DPOSE'
            group_path= '/groups/CESCA-CV'
            second_path = '/groups/CESCA-CV/ICML2021'
            mano_path = '/home/lxiaol9/3DGenNet2019/manopth/mano/models'

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

if __name__ == '__main__':
    infos = global_info()
    print(infos.datasets['bike'].dataset_name)
