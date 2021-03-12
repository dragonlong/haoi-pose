2.40916211:  # bottle, 4 heads, modal=2,
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40916211' DATASET.train_batch=2 DATASET.test_batch=2 \
vis_frequency=1000 vis=True \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True check_consistency=True consistency_loss_multiplier=0.1 \
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_2.4091621.log
use_wandb=True

2.4171:  # jar, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='jar' exp_num='2.4171' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True \
eval=True save=True ckpt=latest eval_mode_r=1 2>&1 | tee results/eval_2.4171_latest_upside.log
use_wandb=True \
# to eval on full test set, just add below

2.471:  # bowl, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bowl' exp_num='2.471' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True \
eval=True save=True eval_mode_r=1 2>&1 | tee results/eval_2.471.log
use_wandb=True \

2.482:  # can, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='can' exp_num='2.482' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True \
eval=True save=True ckpt=best eval_mode_r=0 2>&1 | tee results/eval_2.482_best_0.log
use_wandb=True \

2.4546:  # camera, use R, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='2.45461' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True use_adaptive_mode=False \
pred_6d=True num_modes_R=5 MODEL.num_channels_R=10 mode_features='[5, 'softmax']' \
eval=True save=True ckpt=best eval_mode_r=0 2>&1 | tee results/eval_2.45461.log
use_wandb=True \

2.4547:  # camera, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='2.4547' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_adaptive_mode=True \
pred_6d=True num_modes_R=5 MODEL.num_channels_R=10 mode_features='[5, 'softmax']' \
eval=True save=True ckpt=best eval_mode_r=3 2>&1 | tee results/eval_2.4547_3.log
use_wandb=True \

2.434:  # remote, use R, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='remote' exp_num='2.434' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True use_adaptive_mode=True \
pred_6d=True num_modes_R=5 MODEL.num_channels_R=10 mode_features='[5, 'softmax']' \
eval=True save=True ckpt=latest eval_mode_r=3 2>&1 | tee results/eval_2.434_3.log
use_wandb=True \

2.443:  # cellphone, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='cellphone' exp_num='2.443' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True \
pred_6d=True num_modes_R=5 MODEL.num_channels_R=10 mode_features='[5, 'softmax']' \
eval=True save=True ckpt=latest eval_mode_r=3 2>&1 | tee results/eval_2.443_3.log
use_wandb=True \

2.461:  # ca226, 0 knife, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='knife' exp_num='2.461' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True \
pred_6d=True num_modes_R=2 MODEL.num_channels_R=4 mode_features='[2, 'softmax']' \
eval=True save=True ckpt=latest eval_mode_r=3 2>&1 | tee results/eval_2.461_3.log # 1 is better
use_wandb=True \


# complete shape

# a for R, sub number for different experiments, b for
# x0.1a bottle, R
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='x0.1a' DATASET.train_batch=2 DATASET.test_batch=2 \
vis_frequency=1000 vis=True \
models=en3 encoder_type='en3' \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True num_modes_R=2 MODEL.num_channels_R=2  \
eval_mode_r=2 use_wandb=True
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_x0.1.log

# x0.11a bottle, R, up axis, 2 modes
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='x0.11a' DATASET.train_batch=2 DATASET.test_batch=2 \
vis_frequency=1000 vis=True \
models=en3 encoder_type='en3' \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
num_modes_R=1 MODEL.num_mid_channels=1 MODEL.num_channels_R=1 \
eval_mode_r=2 use_wandb=True
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_x0.1.log

# x0.12a bottle, R, airplane, complete shape, up axis, 12
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' datasets='modelnet40' item='modelnet40' name_dset='modelnet40' DATASET.num_of_class=40 target_category='airplane' exp_num='x0.12a' DATASET.train_batch=2 DATASET.test_batch=2 \
vis_frequency=1000 vis=True \
models=en3 encoder_type='en3' \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
num_modes_R=1 MODEL.num_mid_channels=1 MODEL.num_channels_R=1 use_objective_V=True \
eval_mode_r=2 use_wandb=True
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_x0.1.log

# x0.121a bottle, R, airplane, complete shape, up axis, 12
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' datasets='modelnet40' item='modelnet40' name_dset='modelnet40' DATASET.num_of_class=40 target_category='airplane' \
exp_num='x0.121a' DATASET.train_batch=2 DATASET.test_batch=2 \
vis_frequency=1000 vis=True \
models=en3 encoder_type='en3' \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
num_modes_R=1 MODEL.num_mid_channels=1 MODEL.num_channels_R=1 use_objective_V=True \
eval_mode_r=2 use_wandb=True

# x0.122a bottle, R, airplane, complete shape, up axis, 12
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' datasets='modelnet40' item='modelnet40' name_dset='modelnet40' DATASET.num_of_class=40 target_category='airplane' \
exp_num='x0.122a' DATASET.train_batch=2 DATASET.test_batch=2 \
vis_frequency=1000 vis=True \
models=en3 encoder_type='en3' \
single_instance=True augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
num_modes_R=1 MODEL.num_mid_channels=1 MODEL.num_channels_R=1 use_objective_V=True \
eval_mode_r=2 use_wandb=True

# x0.13a bottle, R, airplan, complete shape, complete shape
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='x0.13a' DATASET.train_batch=2 DATASET.test_batch=2 \
vis_frequency=1000 vis=True \
models=en3 encoder_type='en3' \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_mid_channels=10 MODEL.num_channels_R=2 \
eval_mode_r=2 use_wandb=True
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_x0.1.log


num_channels_R: 1 # or

# x0.1b bottle, T
TRAIN_OBJ='python train_aegan.py training=ae_gan num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='x0.1b' DATASET.train_batch=2 DATASET.test_batch=2 \
vis_frequency=1000 vis=True \
models=en3 encoder_type='en3' \
use_objective_T=True pred_center=True \
use_wandb=True
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_x0.1.log

# # x0.1c, SE3 transformer
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='x0.1c' DATASET.train_batch=2 DATASET.test_batch=2 \
# vis_frequency=1000 vis=True \
# models=se3_transformer_default \
# use_objective_T=True \
# use_wandb=True

# x0.2 jar, T


bowl
#
can
#
camera : 0.5
#
0.5:  # camera, use R, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='x0.5a' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True use_adaptive_mode=False \
pred_bb=True num_modes_R=1 MODEL.num_channels_R=1 \
use_wandb=True \
eval=True save=True ckpt=best eval_mode_r=0 2>&1 | tee results/eval_2.45461.log
remote
#
cellphone
#
knife
