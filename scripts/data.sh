python dataset_parser.py dataset_class='HandDatasetAEGraph' target_category='bottle' augment=True num_points=1024 single_instance=True fetch_cache=True
python dataset_parser.py dataset_class='HandDatasetAEGraph' target_category='bottle' augment=True num_points=1024 task='partial_pcloud_pose' \
use_hand=True
#
python dataset_parser.py dataset_class='HandDatasetAEGraph' target_category='bottle' augment=True num_points=1024 task='partial_pcloud_pose' \
models=en3 encoder_type='en3' \
pred_6d=True
use_hand=True
#
python dataset_parser.py dataset_class='HandDatasetAEGraph' target_category='bottle' augment=True num_points=1024 task='pcloud_pose' \
use_hand=True

# preprocess
python dataset_parser.py dataset_class='HandDatasetAEGraph' target_category='bottle' preprocess=True split='val' num_points=1024

# check airplane
python dataset_parser.py task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' dataset_class='AEGraph' augment=True num_points=1024 \

# check camera
python dataset_parser.py task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' dataset_class='AEGraph' augment=True num_points=1024 \

# check NOCS datasets
python dataset_parser.py datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' target_category='camera' dataset_class='AEGraph' augment=True num_points=512 \

# with background
python dataset_parser.py datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' target_category='laptop' dataset_class='AEGraph' augment=True num_points=1024 use_background=True \

# 3. bottleneck concat,
python modelnet40.py datasets=modelnet40

python modelnet40_parser.py task='pcloud_pose' datasets='modelnet40' item='modelnet40' name_dset='modelnet40' DATASET.num_of_class=40 target_category='airplane'

# 3. bottleneck concat,
python nocs_synthetic_simple.py datasets=nocs_synthetic target_category='laptop'

python shapenetaligned.py datasets=shapenetaligned target_category='laptop'

python modelnet40new.py datasets=modelnet40new target_category='bowl'
