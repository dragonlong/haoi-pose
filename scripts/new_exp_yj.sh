
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='y0.15' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
use_wandb=True