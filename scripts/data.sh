python obman_parser.py dataset_class='HandDatasetAEGraph' target_category='bottle' augment=True num_points=1024 single_instance=True fetch_cache=True
python obman_parser.py dataset_class='HandDatasetAEGraph' target_category='bottle' augment=True num_points=1024 task='partial_pcloud_pose' \
use_hand=True
#
python obman_parser.py dataset_class='HandDatasetAEGraph' target_category='bottle' augment=True num_points=1024 task='partial_pcloud_pose' \
models=en3 encoder_type='en3' \
pred_6d=True
use_hand=True
#
python obman_parser.py dataset_class='HandDatasetAEGraph' target_category='bottle' augment=True num_points=1024 task='pcloud_pose' \
use_hand=True

# preprocess
python obman_parser.py dataset_class='HandDatasetAEGraph' target_category='bottle' preprocess=True split='val' num_points=1024

big enough, and also could pass to next layer, and even final layer;

# 1. receptive field, global info not here;
  - lossless;
  - jump
# 2. confidence, A/B test;
  -  180, ambiguity;

# 3. bottleneck concat,
python modelnet40.py datasets=modelnet40

python modelnet40_parser.py task='pcloud_pose' datasets='modelnet40' item='modelnet40' name_dset='modelnet40' DATASET.num_of_class=40 target_category='airplane'
