python obman_parser.py dataset_class='HandDatasetAEGraph' target_category='bottle' augment=True num_points=1024 single_instance=True fetch_cache=True
python obman_parser.py dataset_class='HandDatasetAEGraph' target_category='bottle' augment=True num_points=1024 task='partial_pcloud_pose' \
use_hand=True

python obman_parser.py dataset_class='HandDatasetAEGraph' target_category='bottle' augment=True num_points=1024 task='pcloud_pose' \
use_hand=True
# preprocess
python obman_parser.py dataset_class='HandDatasetAEGraph' target_category='bottle' preprocess=True split='val' num_points=1024
