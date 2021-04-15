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

# x0.14a

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
jar:
#

bowl:
#


can
#

0.5: >>>>>>>>>>>>>>>>>>>>>>>>>>>> camera + airplane <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
0.5:  # camera, use R, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=256 n_pts=256 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='camera' exp_num='x0.5a' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True use_adaptive_mode=False \
pred_bb=True num_modes_R=1 MODEL.num_channels_R=1 \
use_wandb=True \
eval=True save=True ckpt=best eval_mode_r=0 2>&1 | tee results/eval_2.45461.log

0.51a :  # camera, use R, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='0.51a' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
pred_bb=True num_modes_R=1 MODEL.num_channels_R=2 \
use_objective_R=True \
use_wandb=True
eval=True save=True ckpt=best eval_mode_r=0 2>&1 | tee results/eval_0.51a.log
use_wandb=True \

0.52a :  # camera, use R, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='0.52a' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
pred_bb=True num_modes_R=1 MODEL.num_channels_R=2 \
use_objective_R=True use_objective_T=True \
use_wandb=True eval_frequency=500 \
eval=True save=True ckpt=best eval_mode_r=0 2>&1 | tee results/eval_0.51a.log
use_wandb=True \

0.52a :  # camera, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='2.4547' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_adaptive_mode=True \
pred_6d=True num_modes_R=5 MODEL.num_channels_R=10 mode_features='[5, 'softmax']' \
eval=True save=True ckpt=best eval_mode_r=3 2>&1 | tee results/eval_2.4547_3.log
use_wandb=True \

0.53 :  # single instance, airplane, fixed sampling
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='0.53' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
eval=True save=True ckpt=best eval_mode_r=3 2>&1 | tee results/eval_2.4547_3.log
use_wandb=True \

0.54 :  # single instance, airplane, random sampling
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='airplane' exp_num='0.54' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
use_wandb=True
eval=True save=True ckpt=best eval_mode_r=3 2>&1 | tee results/eval_2.4547_3.log
use_wandb=True \

0.55 :  # airplane, fixed_sampling, all 3k instances
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='airplane' exp_num='0.55' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
fixed_sampling=True use_wandb=True
eval=True save=True ckpt=best eval_mode_r=3 2>&1 | tee results/eval_2.4547_3.log
use_wandb=True \

0.56 :  # airplane, random sampling, all 3k instances
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='airplane' exp_num='0.56' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
fixed_sampling=False use_wandb=True
eval=True save=True ckpt=best eval_mode_r=3 2>&1 | tee results/eval_2.4547_3.log
use_wandb=True \

0.57 :  # airplane, fixed_sampling, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
# single complete shape--> partial data test
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True vis_frequency=10 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='airplane' exp_num='0.57' DATASET.train_batch=2 DATASET.test_batch=2 \
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
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='airplane' exp_num='0.58' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
fixed_sampling=False use_wandb=True

# check chirality,
0.59 :  # airplane, fixed_sampling, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
# single complete shape--> partial data test, random sampling of input pts
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True vis_frequency=10 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='airplane' exp_num='0.59' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
fixed_sampling=True use_wandb=True

# 3850 * 125 data over partial airplane, 500k data?? train on complete data? would it work?
0.6: # single instance, first check whether test works for 6d r regression
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='airplane' exp_num='0.6' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
eval_frequency=10 vis_frequency=10 val_frequency=10 \
fixed_sampling=False use_wandb=True

0.61: # single instance, check equivalence, with rotated train data, predict r
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='airplane' exp_num='0.61' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
eval_frequency=10 vis_frequency=10 val_frequency=10 \
fixed_sampling=False use_wandb=True

0.611: # single instance, check equivalence, with rotated train data, predict r^T
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='airplane' exp_num='0.611' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
eval_frequency=10 vis_frequency=10 val_frequency=10 \
fixed_sampling=True use_wandb=True

0.62: # single instance bowl, check equivalence, with fixed training data, predict NOCS
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='bowl' exp_num='0.62' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
pred_nocs=True use_objective_N=True \
eval_frequency=10 vis_frequency=10 val_frequency=10 \
fixed_sampling=False use_wandb=True

0.621: # single instance bowl, check equivalence, with fixed training data, predict NOCS
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='airplane' exp_num='0.621' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
pred_nocs=True use_objective_N=True \
eval_frequency=10 vis_frequency=10 val_frequency=10 \
fixed_sampling=False use_wandb=True

0.63: # single instance bowl, check equivalence, with rotated train data, predict r
TRAIN_OBJ='python train_aegan_mini.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='bowl' exp_num='0.63' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
eval_frequency=10 vis_frequency=10 val_frequency=10 \
fixed_sampling=False use_wandb=True

0.64: # try airplane category-level data; R
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6401' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
save=True eval=True
use_wandb=True

# 0.641: # R regression with en3-Transformer, 360 degrees
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='pcloud_pose'  item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.641e' DATASET.train_batch=12 DATASET.test_batch=12 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_6d=True num_modes_R=1 MODEL.num_in_channels=1 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True

0.641: # R regression with en3-Transformer, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose'  item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.641e' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=2 MODEL.num_in_channels=1 use_adaptive_mode=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.641t: # T regression with en3-Transformer, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose'  item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.641t' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_T=True \
pred_6d=True num_modes_R=2 MODEL.num_in_channels=1 use_adaptive_mode=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6411e:
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose'  item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6411e' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True single_instance=True \
pred_6d=True num_modes_R=2 MODEL.num_in_channels=1 use_adaptive_mode=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6412e:
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose'  item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6412e' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True single_instance=True \
pred_6d=True num_modes_R=4 MODEL.num_in_channels=1 use_adaptive_mode=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6413: # dgcnn for R regression, mode=2
TRAIN_OBJ='python train_aegan.py training=ae_gan models=dgcnn encoder_type=dgcnn vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6413' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=2 MODEL.num_in_channels=1 use_adaptive_mode=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6414: # dgcnn for R regression, mode=1
TRAIN_OBJ='python train_aegan.py training=ae_gan models=dgcnn encoder_type=dgcnn vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6414' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_in_channels=1 use_adaptive_mode=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6415: # dgcnn for R regression, mode=2
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6415' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=2 MODEL.num_in_channels=3 use_adaptive_mode=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6416: # dgcnn for R regression, mode=1
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6416' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_in_channels=3 use_adaptive_mode=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

#     0.641: # R regression with PointTransformer, 180
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='pcloud_pose'  item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.641' DATASET.train_batch=12 DATASET.test_batch=12 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_6d=True num_modes_R=1 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True
#
#     0.642: # R regression with PointNet++, ca226, 180
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ task='pcloud_pose' datasets=modelnet40 item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.642' DATASET.train_batch=12 DATASET.test_batch=12 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_6d=True num_modes_R=1 HEAD.R='[128, 128, 6, None]' \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# eval=True save=True
#   use_wandb=True
#
#   0.643: # R regression with PointTransformer, 360 degrees
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='pcloud_pose'  item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.643' DATASET.train_batch=12 DATASET.test_batch=12 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_6d=True num_modes_R=1 \
#
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True

0.6431: # R regression with PointTransformer, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose'  item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6431' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

  0.644: # R regression with PointNet++, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' datasets=modelnet40 item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.644' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 HEAD.R='[128, 128, 6, None]' \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.65: # try airplane category-level data; NOCS
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6501' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

#   0.651: # NOCS regression with PointTransformer, 360
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.651' DATASET.train_batch=12 DATASET.test_batch=12 \
# augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True

  0.651: # NOCS regression with en3, 360
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.651e' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6511: # NOCS regression with dgcnn, 360
TRAIN_OBJ='python train_aegan.py training=ae_gan models=dgcnn encoder_type=dgcnn vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6511' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6512: # NOCS regression with en3, 360, use feature knn
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6512' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True MODEL.num_in_channels=3 use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6513: # NOCS regression with en3, 360, use xyz knn
TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.6513' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True MODEL.num_in_channels=3 use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True
#
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=en3 encoder_type=en3 vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='pcloud_pose'  item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.641e' DATASET.train_batch=12 DATASET.test_batch=12 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_6d=True num_modes_R=2 MODEL.num_in_channels=1 use_adaptive_mode=True \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True

#
#   0.652: # NOCS regression with PointNet++, ca226 1
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ task='pcloud_pose' datasets=modelnet40 item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.652' DATASET.train_batch=12 DATASET.test_batch=12 \
# augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# eval=True save=True
#   use_wandb=True

  0.653: # NOCS regression with PointTransformer, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.653' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

  0.654: # NOCS regression with PointNet++, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' datasets=modelnet40 item='oracle' name_dset='oracle' target_category='airplane' exp_num='0.654' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

>>>>>>>>>>>>>>>>>>>>>> partial data
0.66: # camera, RT
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.66' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

  0.661: (wrong!) # camera, RT, point-transformer
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.661' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
  use_wandb=True

  0.662: # NOCS camera, RT, pointnet++
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.662' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 HEAD.R='[128, 128, 6, None]' \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
  use_wandb=True

  0.663: # NOCS, SE3Transformer, predict RT, add RGB feature
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.663' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.664: # NOCS, SE3Transformer, predict RT, add RGB feature
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.664' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.67: # camera, NOCS, ca206 0
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.67' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True pred_nocs=True use_objective_N=True MODEL.num_in_channels=1 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.671: (wrong! fixed ) # camera, NOCS, NOCS, point_transformer
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.671' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.672: # NOCS regression with PointNet++, camera
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.672' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
save=True eval=True
use_wandb=True

  0.673: # need further waiting # camera, NOCS, with RGB, ca202
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.673' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
save=True eval=True
use_wandb=True

0.674:  # camera, NOCS, with RGB, ca202
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='camera' exp_num='0.674' DATASET.train_batch=2 DATASET.test_batch=2 use_background=True \
models=se3_transformer_default \
augment=True pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.68: # laptop, RT
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.68' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

  0.681: # laptop, RT, point-transformer
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.681' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
  use_wandb=True

  0.6811: # laptop, RT, point-transformer, add RGB feature
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6811' DATASET.train_batch=12 DATASET.test_batch=12 use_rgb=True \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

# 0.6812: # laptop, RT, point-transformer, add RGB feature
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
# target_category='laptop' exp_num='0.6812' DATASET.train_batch=12 DATASET.test_batch=12 use_rgb=True \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_6d=True num_modes_R=1 MODEL.num_in_channels=6 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True

  0.682: # laptop, RT, pointnet++
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.682' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 HEAD.R='[128, 128, 6, None]' \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
  use_wandb=True

  0.6821: # laptop, RT, pointnet++
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6821' DATASET.train_batch=12 DATASET.test_batch=12 use_rgb=True \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 HEAD.R='[128, 128, 6, None]' MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

  0.683: # add RGB
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.683' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.6841: # add background points, use 1024 points, with RGB,
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6841' DATASET.train_batch=2 DATASET.test_batch=2 use_background=True \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

0.6851: # add background points, use 1024 points, without RGB
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6851' DATASET.train_batch=2 DATASET.test_batch=2 use_background=True \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 MODEL.num_in_channels=1 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True


#
# 0.684: # add background points, use 1024 points, with RGB,
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
# target_category='laptop' exp_num='0.684' DATASET.train_batch=1 DATASET.test_batch=1 use_background=True \
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
# target_category='laptop' exp_num='0.685' DATASET.train_batch=1 DATASET.test_batch=1 use_background=True \
# models=se3_transformer_default \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 MODEL.num_in_channels=1 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# save=True eval=True
# use_wandb=True


0.686: # laptop, RT, point-transformer, add background points
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.686' DATASET.train_batch=12 DATASET.test_batch=12 use_background=True \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
save=True eval=True
use_wandb=True

0.6861: # laptop, RT, point-transformer, add background points, add RGB feature, 3
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6861' DATASET.train_batch=12 DATASET.test_batch=12 use_background=True use_rgb=True \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
save=True eval=True
use_wandb=True

# 0.6862: # laptop, RT, point-transformer, add background points, add RGB feature, 6
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
# target_category='laptop' exp_num='0.6862' DATASET.train_batch=12 DATASET.test_batch=12 use_background=True use_rgb=True \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_6d=True num_modes_R=1 MODEL.num_in_channels=6 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True

  0.687: # laptop, RT, pointnet++
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.687' DATASET.train_batch=12 DATASET.test_batch=12 use_background=True \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 HEAD.R='[128, 128, 6, None]' \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
save=True eval=True
use_wandb=True

0.6871: # laptop, RT, pointnet++
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6871' DATASET.train_batch=12 DATASET.test_batch=12 use_background=True use_rgb=True \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 HEAD.R='[128, 128, 6, None]' MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
save=True eval=True
use_wandb=True

0.69: # laptop, NOCS, ca219
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.69' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.691: # camera, NOCS, point_transformer, run!
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.691' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.6911: # laptop, NOCS, point-transformer, add RGB feature, 3
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6911' DATASET.train_batch=12 DATASET.test_batch=12 use_rgb=True \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

# 0.6912: # laptop, NOCS, point-transformer, add RGB feature, 6
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
# target_category='laptop' exp_num='0.6912' DATASET.train_batch=12 DATASET.test_batch=12 use_rgb=True \
# augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=6 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# eval=True save=True
# use_wandb=True

  0.692: # laptp, NOCS regression with PointNet++,
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.692' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.6921:# laptp, NOCS regression with PointNet++, add RGB
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6921' DATASET.train_batch=12 DATASET.test_batch=12 use_rgb=True \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.693: # NOCS
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.693' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

# 0.694:(wrong) # laptop, NOCS, num_in_channels=3, use_background
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
# target_category='laptop' exp_num='0.694' DATASET.train_batch=1 DATASET.test_batch=1 use_background=True \
# models=se3_transformer_default \
# augment=True pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# eval=True save=True
# use_wandb=True

# 0.695: # (wrong) laptop, NOCS, num_in_channels=1, use_background
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
# target_category='laptop' exp_num='0.695' DATASET.train_batch=1 DATASET.test_batch=1 use_background=True \
# models=se3_transformer_default \
# augment=True pred_nocs=True use_objective_N=True MODEL.num_in_channels=1 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# eval=True save=True
# use_wandb=True

0.6951: # batch size=2, num_in_channels=1, use_background
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6951' DATASET.train_batch=2 DATASET.test_batch=2 use_background=True \
models=se3_transformer_default \
augment=True pred_nocs=True use_objective_N=True MODEL.num_in_channels=1 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

# 0.696: #(partially wrong) laptop, NOCS, num_in_channels=3, not use_background, simply have batch_size = 1 and use SGD optimizer
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
# target_category='laptop' exp_num='0.696' DATASET.train_batch=1 DATASET.test_batch=1 \
# models=se3_transformer_default \
# augment=True pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True

0.697: # laptop, NOCS, num_in_channels=3, use_background, 512 points, batch_size = 2, add RGB
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.697' DATASET.train_batch=2 DATASET.test_batch=2 use_background=True \
models=se3_transformer_default \
augment=True pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.698: # laptop, NOCS, point_transformer, run!
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.698' DATASET.train_batch=1 DATASET.test_batch=1 use_background=True \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.6981: # laptop, NOCS, point_transformer, run! add RGB, 3
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6981' DATASET.train_batch=12 DATASET.test_batch=12 use_background=True use_rgb=True \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
use_wandb=True

# 0.6982: # laptop, NOCS, point_transformer, run! add RGB, 6
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
# target_category='laptop' exp_num='0.6982' DATASET.train_batch=12 DATASET.test_batch=12 use_background=True use_rgb=True \
# augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=6 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True

0.6983: # laptop, NOCS, point_transformer, run! add background points, add RGB, 3, 512
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6983' DATASET.train_batch=1 DATASET.test_batch=1 use_background=True use_rgb=True \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

# 0.6984: # laptop, NOCS, point_transformer, run! add RGB, 6, 512
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
# $TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
# target_category='laptop' exp_num='0.6984' DATASET.train_batch=12 DATASET.test_batch=12 use_background=True use_rgb=True \
# augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=6 \
# eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
# use_wandb=True

0.699: # laptop, NOCS, PointNet++,
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.699' DATASET.train_batch=12 DATASET.test_batch=12 use_background=True \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.6991: # laptop, NOCS, PointNet++, background points, add rgb
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=1024 n_pts=1024 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6991' DATASET.train_batch=1 DATASET.test_batch=1 use_background=True use_rgb=True \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 \
eval=True save=True
use_wandb=True

0.6992: # laptop, NOCS, PointNet++, 512
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ datasets='nocs_synthetic' task='pcloud_pose' item='nocs_synthetic' name_dset='nocs_synthetic' \
target_category='laptop' exp_num='0.6992' DATASET.train_batch=1 DATASET.test_batch=1 use_background=True use_rgb=True \
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
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.74' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
eval=True save=True
use_wandb=True

0.741: # R regression with PointTransformer, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.741' DATASET.train_batch=2 DATASET.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
use_wandb=True

0.742: # R regression with PointNet++, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.742' DATASET.train_batch=2 DATASET.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 HEAD.R='[128, 128, 6, None]' \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
use_wandb=True

0.743: # R regression with PointTransformer, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.743' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
use_wandb=True

0.744: # R regression with PointNet++, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.744' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_6d=True num_modes_R=1 HEAD.R='[128, 128, 6, None]' \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
use_wandb=True

0.75: # try airplane category-level data; NOCS
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.75' DATASET.train_batch=2 DATASET.test_batch=2 \
models=se3_transformer_default \
augment=True pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
use_wandb=True

0.751: # NOCS regression with PointTransformer, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.751' DATASET.train_batch=2 DATASET.test_batch=2 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
use_wandb=True

0.752: # NOCS regression with PointNet++, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.752' DATASET.train_batch=2 DATASET.test_batch=2 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
use_wandb=True

0.753: # NOCS regression with PointTransformer, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=ptrans encoder_type=point_transformer vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.753' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True MODEL.num_in_channels=3 \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
use_wandb=True

0.754: # NOCS regression with PointNet++, 360 degrees
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type=pnet2plusplus vis=True num_points=512 n_pts=512 name_model=ae dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='shapenet' name_dset='shapenet' target_category='camera' exp_num='0.754' DATASET.train_batch=12 DATASET.test_batch=12 \
augment=True MODEL.num_channels_R=1 pred_nocs=True use_objective_N=True \
eval_frequency=1000 vis_frequency=500 val_frequency=100 save_frequency=10 nr_epochs=1000 \
use_wandb=True
