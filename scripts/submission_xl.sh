2.40916211:  # bottle, 4 heads, modal=2,
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40916211' TRAIN.train_batch=2 TRAIN.test_batch=2 \
vis_frequency=1000 vis=True \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True check_consistency=True consistency_loss_multiplier=0.1 \
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_2.4091621.log
use_wandb=True

2.4171:  # jar, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='jar' exp_num='2.4171' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True \
eval=True save=True ckpt=latest eval_mode_r=1 2>&1 | tee results/eval_2.4171_latest_upside.log
use_wandb=True \
# to eval on full test set, just add below

2.471:  # bowl, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bowl' exp_num='2.471' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True \
eval=True save=True eval_mode_r=1 2>&1 | tee results/eval_2.471.log
use_wandb=True \

2.482:  # can, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='can' exp_num='2.482' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True \
eval=True save=True ckpt=best eval_mode_r=0 2>&1 | tee results/eval_2.482_best_0.log
use_wandb=True \

2.4546:  # camera, use R, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='2.45461' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True use_adaptive_mode=False \
pred_6d=True num_modes_R=5 MODEL.num_channels_R=10 mode_features='[5, 'softmax']' \
eval=True save=True ckpt=best eval_mode_r=0 2>&1 | tee results/eval_2.45461.log
use_wandb=True \

2.4547:  # camera, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='2.4547' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_adaptive_mode=True \
pred_6d=True num_modes_R=5 MODEL.num_channels_R=10 mode_features='[5, 'softmax']' \
eval=True save=True ckpt=best eval_mode_r=3 2>&1 | tee results/eval_2.4547_3.log
use_wandb=True \

2.434:  # remote, use R, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='remote' exp_num='2.434' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True use_adaptive_mode=True \
pred_6d=True num_modes_R=5 MODEL.num_channels_R=10 mode_features='[5, 'softmax']' \
eval=True save=True ckpt=latest eval_mode_r=3 2>&1 | tee results/eval_2.434_3.log
use_wandb=True \

2.443:  # cellphone, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='cellphone' exp_num='2.443' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True \
pred_6d=True num_modes_R=5 MODEL.num_channels_R=10 mode_features='[5, 'softmax']' \
eval=True save=True ckpt=latest eval_mode_r=3 2>&1 | tee results/eval_2.443_3.log
use_wandb=True \

2.461:  # ca226, 0 knife, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='knife' exp_num='2.461' TRAIN.train_batch=2 TRAIN.test_batch=2 \
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
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='x0.1a' TRAIN.train_batch=2 TRAIN.test_batch=2 \
vis_frequency=1000 vis=True \
models=en3 encoder_type='en3' \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True num_modes_R=2 MODEL.num_channels_R=2  \
eval_mode_r=2 use_wandb=True
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_x0.1.log

# x0.11a bottle, R, up axis, 2 modes
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='x0.11a' TRAIN.train_batch=2 TRAIN.test_batch=2 \
vis_frequency=1000 vis=True \
models=en3 encoder_type='en3' \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
num_modes_R=1 MODEL.num_mid_channels=1 MODEL.num_channels_R=1 \
eval_mode_r=2 use_wandb=True
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_x0.1.log

# x0.12a bottle, R, airplane, complete shape, up axis, 12
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' datasets='modelnet40' item='modelnet40' name_dset='modelnet40' DATASET.num_of_class=40 target_category='airplane' exp_num='x0.12a' TRAIN.train_batch=2 TRAIN.test_batch=2 \
vis_frequency=1000 vis=True \
models=en3 encoder_type='en3' \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
num_modes_R=1 MODEL.num_mid_channels=1 MODEL.num_channels_R=1 use_objective_V=True \
eval_mode_r=2 use_wandb=True
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_x0.1.log

# x0.121a bottle, R, airplane, complete shape, up axis, 12
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' datasets='modelnet40' item='modelnet40' name_dset='modelnet40' DATASET.num_of_class=40 target_category='airplane' \
exp_num='x0.121a' TRAIN.train_batch=2 TRAIN.test_batch=2 \
vis_frequency=1000 vis=True \
models=en3 encoder_type='en3' \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
num_modes_R=1 MODEL.num_mid_channels=1 MODEL.num_channels_R=1 use_objective_V=True \
eval_mode_r=2 use_wandb=True

# x0.122a bottle, R, airplane, complete shape, up axis, 12
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' datasets='modelnet40' item='modelnet40' name_dset='modelnet40' DATASET.num_of_class=40 target_category='airplane' \
exp_num='x0.122a' TRAIN.train_batch=2 TRAIN.test_batch=2 \
vis_frequency=1000 vis=True \
models=en3 encoder_type='en3' \
single_instance=True augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
num_modes_R=1 MODEL.num_mid_channels=1 MODEL.num_channels_R=1 use_objective_V=True \
eval_mode_r=2 use_wandb=True

# x0.13a bottle, R, airplan, complete shape, complete shape
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='x0.13a' TRAIN.train_batch=2 TRAIN.test_batch=2 \
vis_frequency=1000 vis=True \
models=en3 encoder_type='en3' \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_mid_channels=10 MODEL.num_channels_R=2 \
eval_mode_r=2 use_wandb=True
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_x0.1.log

# x0.14a

num_channels_R: 1 # or

# x0.1b bottle, T
TRAIN_OBJ='python train_aegan.py training=ae_gan num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='x0.1b' TRAIN.train_batch=2 TRAIN.test_batch=2 \
vis_frequency=1000 vis=True \
models=en3 encoder_type='en3' \
use_objective_T=True pred_center=True \
use_wandb=True
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_x0.1.log

# # x0.1c, SE3 transformer
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='x0.1c' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# vis_frequency=1000 vis=True \
# models=se3_transformer_default \
# use_objective_T=True \
# use_wandb=True

# x0.2 jar, T
jar:
#

bowl:
#


can
#

0.5: >>>>>>>>>>>>>>>>>>>>>>>>>>>> camera + airplane <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
0.5:  # camera, use R, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=256 n_pts=256 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='camera' exp_num='x0.5a' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True use_adaptive_mode=False \
pred_bb=True num_modes_R=1 MODEL.num_channels_R=1 \
use_wandb=True \
eval=True save=True ckpt=best eval_mode_r=0 2>&1 | tee results/eval_2.45461.log

0.51a :  # camera, use R, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='0.51a' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
pred_bb=True num_modes_R=1 MODEL.num_channels_R=2 \
use_objective_R=True \
use_wandb=True
eval=True save=True ckpt=best eval_mode_r=0 2>&1 | tee results/eval_0.51a.log
use_wandb=True \

0.52a :  # camera, use R, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='0.52a' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
pred_bb=True num_modes_R=1 MODEL.num_channels_R=2 \
use_objective_R=True use_objective_T=True \
use_wandb=True eval_frequency=500 \
eval=True save=True ckpt=best eval_mode_r=0 2>&1 | tee results/eval_0.51a.log
use_wandb=True \

0.52a :  # camera, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='2.4547' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_adaptive_mode=True \
pred_6d=True num_modes_R=5 MODEL.num_channels_R=10 mode_features='[5, 'softmax']' \
eval=True save=True ckpt=best eval_mode_r=3 2>&1 | tee results/eval_2.4547_3.log
use_wandb=True \

0.53 :  # single instance, airplane, fixed sampling
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='0.53' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
eval=True save=True ckpt=best eval_mode_r=3 2>&1 | tee results/eval_2.4547_3.log
use_wandb=True \

0.54 :  # single instance, airplane, random sampling
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='airplane' exp_num='0.54' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
use_wandb=True
eval=True save=True ckpt=best eval_mode_r=3 2>&1 | tee results/eval_2.4547_3.log
use_wandb=True \

0.55 :  # airplane, fixed_sampling, all 3k instances
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='airplane' exp_num='0.55' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
fixed_sampling=True use_wandb=True
eval=True save=True ckpt=best eval_mode_r=3 2>&1 | tee results/eval_2.4547_3.log
use_wandb=True \

0.56 :  # airplane, random sampling, all 3k instances
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='airplane' exp_num='0.56' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
fixed_sampling=False use_wandb=True
eval=True save=True ckpt=best eval_mode_r=3 2>&1 | tee results/eval_2.4547_3.log
use_wandb=True \

0.57 :  # airplane, fixed_sampling, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
# single complete shape--> partial data test
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True vis_frequency=10 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='airplane' exp_num='0.57' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
fixed_sampling=True use_wandb=True
eval=True save=True ckpt=best eval_mode_r=3 2>&1 | tee results/eval_2.4547_3.log
use_wandb=True \

#
0.58 :  # airplane, fixed_sampling, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
# single complete shape--> partial data test, random sampling of input pts
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True vis_frequency=10 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='airplane' exp_num='0.58' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
fixed_sampling=False use_wandb=True

# check chirality,
0.59 :  # airplane, fixed_sampling, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
# single complete shape--> partial data test, random sampling of input pts
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True vis_frequency=10 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='airplane' exp_num='0.59' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
fixed_sampling=True use_wandb=True

# 3850 * 125 data over partial airplane, 500k data?? train on complete data? would it work?
0.6: # single instance, first check whether test works for 6d r regression
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='airplane' exp_num='0.6' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
eval_frequency=10 vis_frequency=10 val_frequency=10 \
fixed_sampling=False use_wandb=True

0.61: # single instance, check equivalence, with rotated train data, predict r
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='airplane' exp_num='0.61' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
eval_frequency=10 vis_frequency=10 val_frequency=10 \
fixed_sampling=False use_wandb=True

0.611: # single instance, check equivalence, with rotated train data, predict r^T
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='airplane' exp_num='0.611' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
eval_frequency=10 vis_frequency=10 val_frequency=10 \
fixed_sampling=True use_wandb=True

0.62: # single instance bowl, check equivalence, with fixed training data, predict NOCS
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='bowl' exp_num='0.62' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
pred_nocs=True use_objective_N=True \
eval_frequency=10 vis_frequency=10 val_frequency=10 \
fixed_sampling=False use_wandb=True

0.621: # single instance bowl, check equivalence, with fixed training data, predict NOCS
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='airplane' exp_num='0.621' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
pred_nocs=True use_objective_N=True \
eval_frequency=10 vis_frequency=10 val_frequency=10 \
fixed_sampling=False use_wandb=True

0.63: # single instance bowl, check equivalence, with rotated train data, predict r
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='bowl' exp_num='0.63' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
eval_frequency=10 vis_frequency=10 val_frequency=10 \
fixed_sampling=False use_wandb=True

0.64: # try airplane category-level data; R
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6401' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
save=True eval=True
use_wandb=True

# 0.641: # R regression with en3-Transformer, 360 degrees
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='pcloud_pose'  item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.641e' TRAIN.train_batch=12 TRAIN.test_batch=12 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_6d=True num_modes_R=1 MODEL.num_in_channels=1 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True

0.641: # R regression with en3-Transformer, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose'  item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.641e' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=2 MODEL.num_in_channels=1 use_adaptive_mode=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.641t: # T regression with en3-Transformer, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose'  item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.641t' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_T=True \
pred_6d=True num_modes_R=2 MODEL.num_in_channels=1 use_adaptive_mode=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6411e:
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose'  item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6411e' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True single_instance=True \
pred_6d=True num_modes_R=2 MODEL.num_in_channels=1 use_adaptive_mode=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6412e:
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose'  item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6412e' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True single_instance=True \
pred_6d=True num_modes_R=4 MODEL.num_in_channels=1 use_adaptive_mode=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6413: # original dgcnn for R regression, mode=2
TRAIN_OBJ='python train_aegan.py training=ae_gan models=dgcnn encoder_type=dgcnn vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6413' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=2 MODEL.num_in_channels=1 use_adaptive_mode=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6414: # original dgcnn for R regression, mode=1
TRAIN_OBJ='python train_aegan.py training=ae_gan models=dgcnn encoder_type=dgcnn vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6414' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_in_channels=1 use_adaptive_mode=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6415: # adapted dgcnn for R regression, mode=2
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6415' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=2 MODEL.num_in_channels=3 use_adaptive_mode=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6416: # adapted dgcnn for R regression, mode=1
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6416' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_in_channels=3 use_adaptive_mode=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.64161: # R regression with en3-dgcnn, 360, mode=1
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.64161' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_in_channels=3 use_adaptive_mode=True \
MODEL.m_pool_method='max' MODEL.knn_method='max' \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.64162: # R regression with en3-dgcnn, 360, mode=2
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.64162' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=2 MODEL.num_in_channels=3 use_adaptive_mode=True \
MODEL.m_pool_method='max' MODEL.knn_method='max' \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.64163: # R regression with en3-dgcnn, 360, mode=1, use_equivariance R
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.64163' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_in_channels=3 use_adaptive_mode=True \
MODEL.m_pool_method='max' MODEL.knn_method='max' MODEL.is_equivalence=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.64164: # R regression with en3-dgcnn, 360, mode=1, use_equivariance R
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.64164' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=2 MODEL.num_in_channels=3 use_adaptive_mode=True \
MODEL.m_pool_method='max' MODEL.knn_method='max' MODEL.is_equivalence=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True
#     0.641: # R regression with PointTransformer, 180
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='pcloud_pose'  item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.641' TRAIN.train_batch=12 TRAIN.test_batch=12 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_6d=True num_modes_R=1 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True
#
#     0.642: # R regression with PointNet++, ca226, 180
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ task='pcloud_pose' datasets=modelnet40 item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.642' TRAIN.train_batch=12 TRAIN.test_batch=12 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_6d=True num_modes_R=1 HEAD.R='[128, 128, 6, None]' \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# eval=True save=True
#   use_wandb=True
#
#   0.643: # R regression with PointTransformer, 360 degrees
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='pcloud_pose'  item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.643' TRAIN.train_batch=12 TRAIN.test_batch=12 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_6d=True num_modes_R=1 \
#
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True

0.6431: # R regression with PointTransformer, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose'  item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6431' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

  0.644: # R regression with PointNet++, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' datasets=modelnet40 item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.644' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 HEAD.R='[128, 128, 6, None]' \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

>>>>>>>>>>>. NOCS prediction

0.65: # try airplane category-level data; NOCS
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6501' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

#   0.651: # NOCS regression with PointTransformer, 360
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.651' TRAIN.train_batch=12 TRAIN.test_batch=12 \
# augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True

  0.651: # NOCS regression with en3, 360
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.651e' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6511: # NOCS regression with original dgcnn, 360
TRAIN_OBJ='python train_aegan.py training=ae_gan models=dgcnn encoder_type=dgcnn vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6511' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6512: # NOCS regression with adpated dgcnn, 360, use feature knn
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6512' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True MODEL.num_in_channels=3 use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6513: # NOCS regression with adpated dgcnn, 360, use xyz knn
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6513' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True MODEL.num_in_channels=3 use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6514: # NOCS regression with en3-dgcnn, 360, use xyz knn
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6514' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True MODEL.num_in_channels=3 use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

# 0.1 cleaned EGNN;
# 1. use largest=True, knn
# 2. further remove rela_dist
# 3. replace e_mlp with the same in dgcnn: bn, leaky_relu, better?

0.6515: # NOCS regression with en3-dgcnn, 360, use xyz knn
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6515' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True MODEL.num_in_channels=3 use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True
# 0 only remove batch_norm:       not working;
# 1 replace leaky_relu with silu: almost as good
# 2 change 64 --> 14?     :       almost as good
# 3 only change 128 --> 64:       guess to be same good!
# 4 MLP + layernorm:      : not working
# 5 cleaned conv2d        : working!!!
# 6 with additional relative dist
# 7 use additional layer with f_mlp,
# 8 use additional layer with f_mlp, bias=False, with leaky relu

0.6516: # NOCS regression with en3-dgcnn, 360, use xyz knn, inherit from 6
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6516' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True MODEL.num_in_channels=3 use_objective_N=True \
MODEL.m_pool_method='max' MODEL.knn_method='max' \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.65161: # NOCS regression with en3-dgcnn, 360, use xyz knn, inherit from 6
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.65161' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True MODEL.num_in_channels=3 use_objective_N=True \
MODEL.m_pool_method='mean' MODEL.knn_method='max' \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.65162: # NOCS regression with en3-dgcnn, 360, use xyz knn, inherit from 6
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.65162' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True MODEL.num_in_channels=3 use_objective_N=True \
MODEL.m_pool_method='max' MODEL.knn_method='min' \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.65163: # NOCS regression with en3-dgcnn, 360, use xyz knn, inherit from 6
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.65163' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True MODEL.num_in_channels=3 use_objective_N=True \
MODEL.m_pool_method='mean' MODEL.knn_method='min' \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True


#
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='pcloud_pose'  item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.641e' TRAIN.train_batch=12 TRAIN.test_batch=12 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_6d=True num_modes_R=2 MODEL.num_in_channels=1 use_adaptive_mode=True \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True

#
#   0.652: # NOCS regression with PointNet++, ca226 1
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ task='pcloud_pose' datasets=modelnet40 item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.652' TRAIN.train_batch=12 TRAIN.test_batch=12 \
# augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# eval=True save=True
#   use_wandb=True

  0.653: # NOCS regression with PointTransformer, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.653' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

  0.654: # NOCS regression with PointNet++, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' datasets=modelnet40 item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.654' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

>>>>>>>>>>>>>>>>>>>>>> partial data
0.66: # camera, RT
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.66' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

  0.661: (wrong!) # camera, RT, point-transformer
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.661' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
  use_wandb=True

  0.662: # NOCS camera, RT, pointnet++
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.662' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 HEAD.R='[128, 128, 6, None]' \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
  use_wandb=True

  0.663: # NOCS, SE3Transformer, predict RT, add RGB feature
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.663' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.664: # NOCS, SE3Transformer, predict RT, add RGB feature
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.664' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.67: # camera, NOCS, ca206 0
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.67' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True pred_nocs=True use_objective_N=True MODEL.num_in_channels=1 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.671: (wrong! fixed ) # camera, NOCS, NOCS, point_transformer
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.671' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.672: # NOCS regression with PointNet++, camera
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.672' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
save=True eval=True
use_wandb=True

  0.673: # need further waiting # camera, NOCS, with RGB, ca202
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.673' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
save=True eval=True
use_wandb=True

0.674:  # camera, NOCS, with RGB, ca202
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.674' TRAIN.train_batch=2 TRAIN.test_batch=2 use_background=True \
models=se3_transformer_default \
augment=True pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.68: # laptop, RT
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.68' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

  0.681: # laptop, RT, point-transformer
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.681' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
  use_wandb=True

  0.6811: # laptop, RT, point-transformer, add RGB feature
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6811' TRAIN.train_batch=12 TRAIN.test_batch=12 use_rgb=True \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

# 0.6812: # laptop, RT, point-transformer, add RGB feature
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
# target_category='laptop' exp_num='0.6812' TRAIN.train_batch=12 TRAIN.test_batch=12 use_rgb=True \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_6d=True num_modes_R=1 MODEL.num_in_channels=6 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True

  0.682: # laptop, RT, pointnet++
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.682' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 HEAD.R='[128, 128, 6, None]' \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
  use_wandb=True

  0.6821: # laptop, RT, pointnet++, add RGB
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6821' TRAIN.train_batch=12 TRAIN.test_batch=12 use_rgb=True \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 HEAD.R='[128, 128, 6, None]' MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

  0.683: # laptop, RT, SE3Transformer, no background points, add RGB
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.683' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.6841: # laptop, RT, SE3Transformer, add background points, use RGB
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6841' TRAIN.train_batch=2 TRAIN.test_batch=2 use_background=True \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6851: #  laptop, RT, SE3Transformer, add background points, without RGB
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6851' TRAIN.train_batch=2 TRAIN.test_batch=2 use_background=True \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 MODEL.num_in_channels=1 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

#
# 0.684: # add background points, use 1024 points, with RGB,
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
# target_category='laptop' exp_num='0.684' TRAIN.train_batch=1 TRAIN.test_batch=1 use_background=True \
# models=se3_transformer_default \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 MODEL.num_in_channels=3 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# save=True eval=True
# use_wandb=True
#
# 0.685: # add background points, use 1024 points, without RGB
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
# target_category='laptop' exp_num='0.685' TRAIN.train_batch=1 TRAIN.test_batch=1 use_background=True \
# models=se3_transformer_default \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 MODEL.num_in_channels=1 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# save=True eval=True
# use_wandb=True


0.686: # laptop, RT, point-transformer, add background points
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.686' TRAIN.train_batch=12 TRAIN.test_batch=12 use_background=True \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
save=True eval=True
use_wandb=True

0.6861: # laptop, RT, point-transformer, add background points, add RGB feature, 3
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6861' TRAIN.train_batch=12 TRAIN.test_batch=12 use_background=True use_rgb=True \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
save=True eval=True
use_wandb=True

# 0.6862: # laptop, RT, point-transformer, add background points, add RGB feature, 6
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
# target_category='laptop' exp_num='0.6862' TRAIN.train_batch=12 TRAIN.test_batch=12 use_background=True use_rgb=True \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_6d=True num_modes_R=1 MODEL.num_in_channels=6 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True

  0.687: # laptop, RT, pointnet++
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.687' TRAIN.train_batch=12 TRAIN.test_batch=12 use_background=True \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 HEAD.R='[128, 128, 6, None]' \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
save=True eval=True
use_wandb=True

0.6871: # laptop, RT, pointnet++
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6871' TRAIN.train_batch=12 TRAIN.test_batch=12 use_background=True use_rgb=True \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 HEAD.R='[128, 128, 6, None]' MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
save=True eval=True
use_wandb=True

0.688: # laptop, R, spconv ca219 0
python train_epn.py datasets='nocs_synthetic' item='nocs_synthetic' name_dset='nocs_synthetic' target_category='laptop' \
models=epn exp_num='0.688' task='pcloud_pose' \
TRAIN.train_batch=2 TRAIN.test_batch=2 num_points=1024 model.input_num=1024 \
use_wandb=True

# try remove T
# try 512 points instead,
# understand how the group conv works, and how to get invariant feature
# 0.6881: # laptop, R, spconv, use 512 pts
# python train_epn.py datasets='nocs_synthetic' item='nocs_synthetic' name_dset='nocs_synthetic' target_category='laptop' \
# models=epn exp_num='0.6881' task='pcloud_pose' \
# TRAIN.train_batch=2 TRAIN.test_batch=2 num_points=512 model.input_num=512 \
# use_wandb=True

0.6881: # laptop, R, spconv, use maxpooling
python train_epn.py datasets='nocs_synthetic' item='nocs_synthetic' name_dset='nocs_synthetic' target_category='laptop' \
models=epn exp_num='0.6881' task='pcloud_pose' \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
num_modes_R=1 model.pooling_method='max' \
use_wandb=True

0.68811: # laptop, R, spconv, use pointnet
python train_epn.py datasets='nocs_synthetic' item='nocs_synthetic' name_dset='nocs_synthetic' target_category='laptop' \
models=epn exp_num='0.68811' task='pcloud_pose' \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
num_modes_R=1 model.pooling_method='pointnet' \
use_wandb=True

0.68812: # laptop, R, spconv, use maxpooling, use kpconv feature, ca219 0
python train_epn.py datasets='nocs_synthetic' item='nocs_synthetic' name_dset='nocs_synthetic' target_category='laptop' \
models=epn exp_num='0.68812' task='pcloud_pose' \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
num_modes_R=1 model.pooling_method='max' model.kpconv=True \
use_wandb=True

0.6882: # laptop, R, spconv, add RGB points, batch_size=4

0.6883: # laptop, R, spconv, add background points and direct pooling,batch_size=4
python train_epn.py datasets='nocs_synthetic' item='nocs_synthetic' name_dset='nocs_synthetic' target_category='laptop' \
models=epn exp_num='0.6883' task='pcloud_pose' \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 use_background=True \
use_wandb=True

0.68831: # laptop, R, spconv, add background points, max
python train_epn.py datasets='nocs_synthetic' item='nocs_synthetic' name_dset='nocs_synthetic' target_category='laptop' \
models=epn exp_num='0.68831' task='pcloud_pose' \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 use_background=True \
num_modes_R=1 model.pooling_method='max' \
use_wandb=True

0.68832: # laptop, R, spconv, add background points, pointnet
python train_epn.py datasets='nocs_synthetic' item='nocs_synthetic' name_dset='nocs_synthetic' target_category='laptop' \
models=epn exp_num='0.68832' task='pcloud_pose' \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 use_background=True \
num_modes_R=1 model.pooling_method='pointnet' \
use_wandb=True

0.68833: # laptop, R, spconv, add background points, max, use Kp-Conv
python train_epn.py datasets='nocs_synthetic' item='nocs_synthetic' name_dset='nocs_synthetic' target_category='laptop' \
models=epn exp_num='0.68833' task='pcloud_pose' \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 use_background=True \
num_modes_R=1 model.pooling_method='max' model.kpconv=True \
use_wandb=True

0.6884: # laptop, R, spconv, add background points and RGB

0.6885: #laptop, R, spconv, no rgb, no background points, use two modes for R regression?? or just try two-GT, first how it considers 180 degrees;
python train_epn.py datasets='nocs_synthetic' item='nocs_synthetic' name_dset='nocs_synthetic' target_category='laptop' \
models=epn exp_num='0.6885' task='pcloud_pose' \
TRAIN.train_batch=2 TRAIN.test_batch=2 num_points=1024 model.input_num=1024 \
num_modes_R=2 \
use_wandb=True

0.694 # try NOCS prediction;(on complete shape as well)


0.69: # laptop, NOCS, ca219
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.69' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.691: # camera, NOCS, point_transformer, run!
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.691' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.6911: # laptop, NOCS, point-transformer, add RGB feature, 3
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6911' TRAIN.train_batch=12 TRAIN.test_batch=12 use_rgb=True \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

# 0.6912: # laptop, NOCS, point-transformer, add RGB feature, 6
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
# target_category='laptop' exp_num='0.6912' TRAIN.train_batch=12 TRAIN.test_batch=12 use_rgb=True \
# augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=6 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# eval=True save=True
# use_wandb=True

  0.692: # laptp, NOCS, PointNet++,
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.692' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.6921:# laptp, NOCS, PointNet++, add RGB
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6921' TRAIN.train_batch=12 TRAIN.test_batch=12 use_rgb=True \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.693: # laptopp, NOCS, SE3Transformer
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.693' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.694: # laptop, NOCS, spconv, points only, Torun
python train_aegan.py models=epn exp_num='0.694' task='pcloud_pose' model.model='dir_so3net' encoder_type=dir_so3net name_model=ae \
datasets='nocs_synthetic' item='nocs_synthetic' name_dset='nocs_synthetic' target_category='laptop'  dataset_class=AEGraph \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True pred_nocs=True use_objective_N=True MODEL.num_in_channels=1 \
use_wandb=True
#
# 0.694:(wrong) # laptop, NOCS, num_in_channels=3, use_background
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
# target_category='laptop' exp_num='0.694' TRAIN.train_batch=1 TRAIN.test_batch=1 use_background=True \
# models=se3_transformer_default \
# augment=True pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# eval=True save=True
# use_wandb=True

# 0.695: # (wrong) laptop, NOCS, num_in_channels=1, use_background
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
# target_category='laptop' exp_num='0.695' TRAIN.train_batch=1 TRAIN.test_batch=1 use_background=True \
# models=se3_transformer_default \
# augment=True pred_nocs=True use_objective_N=True MODEL.num_in_channels=1 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# eval=True save=True
# use_wandb=True

0.6951: # laptop, SE3Transformer, no RGB, use_background
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6951' TRAIN.train_batch=2 TRAIN.test_batch=2 use_background=True \
models=se3_transformer_default \
augment=True pred_nocs=True use_objective_N=True MODEL.num_in_channels=1 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

# 0.696: #(partially wrong) laptop, NOCS, num_in_channels=3, not use_background, simply have batch_size = 1 and use SGD optimizer

0.697: # laptop, NOCS, SE3Transformer, with RGB, use_background, 512 points, batch_size = 2
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.697' TRAIN.train_batch=2 TRAIN.test_batch=2 use_background=True \
models=se3_transformer_default \
augment=True pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.698: # laptop, NOCS, point_transformer, run!
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.698' TRAIN.train_batch=1 TRAIN.test_batch=1 use_background=True \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.6981: # laptop, NOCS, point_transformer,with RGB, use_background
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6981' TRAIN.train_batch=12 TRAIN.test_batch=12 use_background=True use_rgb=True \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

# 0.6982: # laptop, NOCS, point_transformer, run! add RGB, 6
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
# target_category='laptop' exp_num='0.6982' TRAIN.train_batch=12 TRAIN.test_batch=12 use_background=True use_rgb=True \
# augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=6 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True

0.6983: # laptop, NOCS, point_transformer, run! add background points, add RGB, 3, 512
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6983' TRAIN.train_batch=1 TRAIN.test_batch=1 use_background=True use_rgb=True \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

# 0.6984: # laptop, NOCS, point_transformer, run! add RGB, 6, 512
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
# target_category='laptop' exp_num='0.6984' TRAIN.train_batch=12 TRAIN.test_batch=12 use_background=True use_rgb=True \
# augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=6 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True

0.699: # laptop, NOCS, PointNet++, with background
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.699' TRAIN.train_batch=12 TRAIN.test_batch=12 use_background=True \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.6991: # laptop, NOCS, PointNet++, background points, add rgb
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6991' TRAIN.train_batch=1 TRAIN.test_batch=1 use_background=True use_rgb=True \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.6992: # laptop, NOCS, PointNet++, background points, add rgb, 512 points instead
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6992' TRAIN.train_batch=1 TRAIN.test_batch=1 use_background=True use_rgb=True \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

remote
#
cellphone
#
knife

#
bottle.tar  2_bowl.tar  3  3_camera.tar  4_can.tar  5_laptop.tar  6_mug

0.74: # try camera category-level data; R
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.74' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
eval=True save=True
use_wandb=True

0.741: # R regression with PointTransformer, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.741' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
use_wandb=True

0.742: # R regression with PointNet++, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.742' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 HEAD.R='[128, 128, 6, None]' \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
use_wandb=True

0.743: # R regression with PointTransformer, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.743' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
use_wandb=True

0.744: # R regression with PointNet++, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.744' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 HEAD.R='[128, 128, 6, None]' \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
use_wandb=True

0.75: # try airplane category-level data; NOCS
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.75' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
use_wandb=True

0.751: # NOCS regression with PointTransformer, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.751' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
use_wandb=True

0.752: # NOCS regression with PointNet++, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.752' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
use_wandb=True

0.753: # NOCS regression with PointTransformer, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.753' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
use_wandb=True

0.754: # NOCS regression with PointNet++, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.754' TRAIN.train_batch=12 TRAIN.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
use_wandb=True

#>>>>>>>>>>>>>>>>>>>> self-supervised reconstruction of canonical shape
0.8: # airplane, spconv, points only
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae \
models=epn exp_num='0.8' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
use_wandb=True

0.81: # supervision in camera space, random R
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True \
models=epn exp_num='0.81' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True
use_wandb=True


0.811: # supervision in nocs space
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True \
models=epn exp_num='0.811a' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 \
use_wandb=True


0.812: # supervision in nocs space, random R
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True \
models=epn exp_num='0.812b' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
use_wandb=True

0.813(work!): # supervision in camera space, random R, with adaptive R label cls loss
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.813' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_M=True \
eval=True

use_pretrain=True \
use_wandb=True

0.814(work!): # supervision in camera space, random R, with adaptive R label cls loss, 0.001
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=10 \
models=epn exp_num='0.814' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_M=True modecls_loss_multiplier=0.001 \
eval=True save=True
use_wandb=True

0.8141: # supervision in camera space, random R, with adaptive R label cls loss, 0.01
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=10 \
models=epn exp_num='0.8141' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_M=True modecls_loss_multiplier=0.01 \
use_wandb=True

0.8142: # supervision in camera space, random R, with adaptive R label cls loss
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=10 \
models=epn exp_num='0.8142' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_M=True modecls_loss_multiplier=0.1 \
use_wandb=True

0.815: # supervision in camera space, random R, airplane, use_symmetry
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=10 \
models=epn exp_num='0.815' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_symmetry_loss=True \
eval=True
use_wandb=True

0.816: # supervision in camera space, random R, use atlas deformation
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net decoder_type='atlas' name_model=ae vis=True \
models=epn exp_num='0.816' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True
use_wandb=True

0.8161: # supervision in camera space, random R, use atlas deformation, but choose a different sphere 'uniform_sphere'
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net decoder_type='atlas' template_shape='uniform_sphere' name_model=ae vis=True \
models=epn exp_num='0.8161' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True
use_wandb=True

0.817: # supervision in camera space, random R, use atlas deformation
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net decoder_type='tree' name_model=ae vis=True \
models=epn exp_num='0.817' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
use_wandb=True

0.818: # supervision in camera space, random R, add random_T
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True \
models=epn exp_num='0.818' model.model='enc_so3net' model.pooling_method='pointnet' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True \
eval=True \
use_wandb=True

0.819: # supervision in camera space, random R, add random_T, no sigmoid
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True \
models=epn exp_num='0.819' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True \
use_wandb=True
#>>>>>>>>>>>>>>>>>>>>>>>>>>> use partial point cloud <<<<<<<<<<<<<<<<<<<<<<<<<<<#
0.82: # supervision in camera space, random R, on all category
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=10 \
models=epn exp_num='0.82' model.model='enc_so3net' model.pooling_method='max' \
datasets=shapenetaligned item=shapenetaligned name_dset=shapenetaligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
use_wandb=True

0.821: # supervision in nocs space, on all category
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=1 \
models=epn exp_num='0.821' model.model='enc_so3net' model.pooling_method='max' \
datasets=shapenetaligned item=shapenetaligned name_dset=shapenetaligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 use_objective_canon=True \
use_wandb=True

0.822: # supervision in nocs space, random R, on all category
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=1 \
models=epn exp_num='0.822' model.model='enc_so3net' model.pooling_method='max' \
datasets=shapenetaligned item=shapenetaligned name_dset=shapenetaligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_canon=True \
use_wandb=True

# chair
0.83: # supervision in camera space, random R
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=1 \
models=epn exp_num='0.83' model.model='enc_so3net' model.pooling_method='max' \
datasets=shapenetaligned item=shapenetaligned name_dset=shapenetaligned target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
use_wandb=True

0.831: # supervision in nocs space
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=1 \
models=epn exp_num='0.831' model.model='enc_so3net' model.pooling_method='max' \
datasets=shapenetaligned item=shapenetaligned name_dset=shapenetaligned target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 use_objective_canon=True \
use_wandb=True

0.832: # supervision in nocs space, random R
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=1 \
models=epn exp_num='0.832' model.model='enc_so3net' model.pooling_method='max' \
datasets=shapenetaligned item=shapenetaligned name_dset=shapenetaligned target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_canon=True \
use_wandb=True

0.833: # supervision in camera space, random R, use atlas decoder
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net decoder_type=atlas name_model=ae vis=True save_frequency=1 \
models=epn exp_num='0.833' model.model='enc_so3net' model.pooling_method='max' \
datasets=shapenetaligned item=shapenetaligned name_dset=shapenetaligned target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
use_wandb=True

# airplane
0.84: # supervision in camera space, random R, on airplane,
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.84' model.model='enc_so3net' model.pooling_method='max' \
datasets=shapenetaligned item=shapenetaligned name_dset=shapenetaligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True pre_compute_delta=True
use_wandb=True

# 0.841: # supervision in nocs space, on airplane,
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=1 \
# models=epn exp_num='0.821' model.model='enc_so3net' model.pooling_method='max' \
# datasets=shapenetaligned item=shapenetaligned name_dset=shapenetaligned target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 use_objective_canon=True \
# use_wandb=True

0.841: #  supervision in camera space, random R, on airplane, with symmetry loss
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.841' model.model='enc_so3net' model.pooling_method='max' \
datasets=shapenetaligned item=shapenetaligned name_dset=shapenetaligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_symmetry_loss=True \
eval=True \
use_wandb=True

0.842: # supervision in nocs space, random R, airplane
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.842' model.model='enc_so3net' model.pooling_method='max' \
datasets=shapenetaligned item=shapenetaligned name_dset=shapenetaligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_canon=True \
use_wandb=True

0.843: # supervision in camera space, random R, airplane, decoder_type='atlas'
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net decoder_type='atlas' name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.843' model.model='enc_so3net' model.pooling_method='max' \
datasets=shapenetaligned item=shapenetaligned name_dset=shapenetaligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True
use_wandb=True

0.8431: # supervision in camera space, random R, airplane, decoder_type='atlas', template_shape='uniform_sphere'
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net decoder_type='atlas' template_shape='uniform_sphere' name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8431' model.model='enc_so3net' model.pooling_method='max' \
datasets=shapenetaligned item=shapenetaligned name_dset=shapenetaligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True
use_wandb=True

0.844: # supervision in camera space, random R, airplane, decoder_type='tree'
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net decoder_type='tree' name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.844' model.model='enc_so3net' model.pooling_method='max' \
datasets=shapenetaligned item=shapenetaligned name_dset=shapenetaligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
use_wandb=True

0.845: # supervision in camera space, random R, airplane, but add T estimation
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.845' model.model='enc_so3net' model.pooling_method='max' \
datasets=shapenetaligned item=shapenetaligned name_dset=shapenetaligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.8451: # supervision in camera space, random R, airplane, but add T estimation
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8451' model.model='enc_so3net' model.pooling_method='pointnet' \
datasets=shapenetaligned item=shapenetaligned name_dset=shapenetaligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.846: # supervision in camera space, random R, airplane, but add T estimation, no sigmoid with random size
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.846' model.model='enc_so3net' model.pooling_method='max' \
datasets=shapenetaligned item=shapenetaligned name_dset=shapenetaligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True \
use_wandb=True

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 0.85, for modelnet40
airplane  bench      bowl   cone     desk     flower_pot  keyboard  mantel       person  radio       sofa    table   tv_stand  xbox
bathtub   bookshelf  car    cup      door     glass_box   lamp      monitor      piano   range_hood  stairs  tent    vase
bed       bottle     chair  curtain  dresser  guitar      laptop    night_stand  plant   sink        stool   toilet  wardrobe

0.85: # supervision in camera space, random R, chair, ca227
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.85' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True \
use_pretrain=True
use_wandb=True

0.851: # car, ca202
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.851' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True
use_pretrain=True \
use_wandb=True

0.852: # sofa, ca228
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.852' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='sofa' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True
use_pretrain=True \
use_wandb=True

0.853(wrong!!!): # table,  ca204,
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.853' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='table' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True
use_pretrain=True \
use_wandb=True

0.854:(wrong!!!) # laptop, ca234
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.854' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True
use_pretrain=True \
use_wandb=True

0.855:(check!!) # bowl, ca230
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.855' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True
use_pretrain=True \
use_wandb=True

0.856: (wrong!!!) # bottle, ca221
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.856' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True
use_pretrain=True \
use_wandb=True

0.857: (check!!)  # cup, ca223
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.857' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='cup' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True pre_compute_delta=True
use_pretrain=True \
use_wandb=True

>>>>>>>>>>>>>>>>>>>>>>>>>> partial dataset NOCS
# different partail like chair, or like bed, NOCS dataset;
0.86: # supervision in camera space, random R,laptop
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.86' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.8601: # supervision in camera space, random R,laptop
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8601' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True \
use_wandb=True

0.861: # supervision in camera space, random R, bowl
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.861' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.8611: # supervision in camera space, random R, bowl, add T, s
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8611' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True \
use_wandb=True

0.862: # supervision in camera space, random R, mug
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.862' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='mug' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True \
use_wandb=True

0.8621: # supervision in camera space, random R, mug, add T, s
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8621' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='mug' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True \
use_wandb=True

0.863: # supervision in camera space, random R,laptop, add t, seems better
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.863' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True \
eval=True save=True pre_compute_delta=True
use_wandb=True


# 0.864: # supervision in camera space, random R,laptop, add t
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.864' model.model='enc_so3net' model.pooling_method='pointnet' \
# datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='laptop' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# augment=True MODEL.num_in_channels=1 pred_t=True \
# use_wandb=True

0.865: # using linear layer
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.865' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True \
use_wandb=True
