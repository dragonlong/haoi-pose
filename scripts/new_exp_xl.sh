export PATH="/home/lxiaol9/anaconda3/bin:$PATH"
cd
. scripts/ai_power1.sh
module load cuda/10.1.168
source activate merl


# 3.11 different categories
# airplane   bottle  cup      filelist.txt  lamp                        modelnet10_train.txt        night_stand  range_hood  table     wardrobe
# bathtub    bowl    curtain  flower_pot    laptop                      modelnet40_shape_names.txt  person       sink        tent      xbox
# bed        car     desk     glass_box     mantel                      modelnet40_test.txt         piano        sofa        toilet
# bench      chair   door     guitar        modelnet10_shape_names.txt  modelnet40_train.txt        plant        stairs      tv_stand
# bookshelf  cone    dresser  keyboard      modelnet10_test.txt         monitor                     radio        stool       vase




>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2.405: bottle complete <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# # 2.4058 multiple instance, dynamic graph, with augmentation, L2
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4058' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# use_objective_R=True augment=True rotation_loss_type=1 eval=True save=True
# use_wandb=True
#
# # 2.40581 encoder-only v3, dense one-mode prediction
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.40581' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# vis_frequency=1000 vis=True \
# models=se3_transformerv3 \
# MODEL.num_channels_R=1 MODEL.encoder_only=True \
# eval=True save=True
# use_wandb=True
#
# # 2.40582 encoder-only v3, dense two-mode prediction
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.40582' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# vis_frequency=1000 vis=True \
# models=se3_transformerv3 \
# MODEL.num_channels_R=2 MODEL.encoder_only=True \
# pred_mode=True use_objective_M=True \
# use_wandb=True
#
# # 2.40583 (later for dense prediction visualization & evaluation) v3, use pooled predictions
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.40583' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True \
# vis_frequency=1000 vis=True \
# models=se3_transformerv3 \
# MODEL.num_channels_R=1 MODEL.encoder_only=True \
# use_wandb=True

# 2.40584, encoder-decoder full training, use 2 mode, dense prediction, check visulizations and log
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.40584' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=2 MODEL.encoder_only=False \
pred_mode=True use_objective_M=True \
eval=True save=True \
use_wandb=True

# 2.40585, encoder-decoder full training, pooled prediction, 1 mode
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.40585' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=1 MODEL.encoder_only=False \
eval=True save=True 2>&1 | tee results/eval.2.40585.log
use_wandb=True

# # 2.406971, multiple instance, dynamic graph, with augmentation, T voting
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.406971' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# use_objective_T=True augment=True eval=True save=True
# use_wandb=True

2.4073
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4073' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
eval=True save=True

use_wandb=True
#
# 2.40731 # ca211 0 general training,  with pooling
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40731' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True MODEL.num_channels_R=1 \
# use_wandb=True
#
# # 2.40732 # general training, dense training
# # TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# # $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40732' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# # augment=True rotation_loss_type=1 use_objective_R=True MODEL.num_channels_R=1 rotation_use_dense=True \
# # use_wandb=True
#
# 2.40733: # ca210 1 add two modes, dense
# TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40733' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# MODEL.num_channels_R=2 HEAD.R='[128, 128, 6, None]' \
# pred_mode=True use_objective_M=True use_objective_V=True consistency_loss_multiplier=0.1 \
# use_wandb=True

2.40734 # ca236 1 general training,  with pooling, knn
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40734' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True MODEL.num_channels_R=1 MODEL.use_ball_query=False \
use_wandb=True

2.40735: # (killed) ca210 0 add two modes, dense, knn
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40735' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
MODEL.num_channels_R=2 HEAD.R='[128, 128, 6, None]' \
pred_mode=True use_objective_M=True use_objective_V=True consistency_loss_multiplier=0.1 MODEL.use_ball_query=False \
use_wandb=True

2.4074: # pointnet++ on complete data for NOCS
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4074' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True pred_nocs=True use_objective_N=True  \
eval=True save=True
use_wandb=True

TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.4075' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 pred_nocs=True use_objective_N=True MODEL.use_ball_query=False \
use_wandb=True

TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='jar' exp_num='2.4076' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 pred_nocs=True use_objective_N=True MODEL.use_ball_query=False \
use_wandb=True

TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='can' exp_num='2.4077' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 pred_nocs=True use_objective_N=True MODEL.use_ball_query=False \
use_wandb=True

TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bowl' exp_num='2.4078' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 pred_nocs=True use_objective_N=True MODEL.use_ball_query=False \
use_wandb=True

TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='2.4079' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 pred_nocs=True use_objective_N=True MODEL.use_ball_query=False \
use_wandb=True

TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='remote' exp_num='2.40791' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 pred_nocs=True use_objective_N=True MODEL.use_ball_query=False \
use_wandb=True

TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='cellphone' exp_num='2.40792' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 pred_nocs=True use_objective_N=True MODEL.use_ball_query=False \
use_wandb=True

TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='knife' exp_num='2.40793' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 pred_nocs=True use_objective_N=True MODEL.use_ball_query=False \
use_wandb=True
#>>>>>>>>>>>>>>>>>>>>>>>>






>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> partial shape <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
2.409:
  2.4091: # train for partial shape, bottle, object pts only, R, L2, dense
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.4091' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
use_wandb=True

#   2.40911: # train for partial shape, bottle, object pts only, R, L2, dense, use confidence type 1, confidence_loss_multiplier=0.1
#   # later any with clean confidence loss, but show the prediction confidence
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40911' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_conf=True use_confidence_R=True confidence_loss_type=1 confidence_loss_multiplier=0.1 \
# use_wandb=True
#
#   2.40912: # train for partial shape, bottle, object pts only, R, L2, dense, use confidence type 1, confidence_loss_multiplier=0.1
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40912' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_conf=True use_confidence_R=True confidence_loss_type=1 confidence_loss_multiplier=0.1 \
# vis_frequency=100 vis=True \
# eval=True save=True
# use_wandb=True
#
#   2.40913: # train for partial shape, bottle, object pts only, R, L2, dense, use confidence type 1, confidence_loss_multiplier=0.1, bigger receptive field;
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40913' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_conf=True use_confidence_R=True confidence_loss_type=1 confidence_loss_multiplier=0.1 \
# vis_frequency=500 vis=True MODEL.down_conv.nsamples='[[10], [16], [16], [31]]' \
# use_wandb=True
#
#   2.40914: (good!) # train for partial shape, bottle, object pts only, R, L2, dense, use confidence type 1, confidence_loss_multiplier=0.1, bigger receptive field;
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40914' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_conf=True use_confidence_R=True confidence_loss_type=1 confidence_loss_multiplier=0.1 \
# vis_frequency=1000 vis=True \
# use_wandb=True
#
#   2.40915: # train for partial shape, bottle, object pts only, R, L2, dense, use confidence type 1, confidence_loss_multiplier=0.1, bigger receptive field;
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40915' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_conf=True use_confidence_R=True confidence_loss_type=1 confidence_loss_multiplier=0.1 \
# vis_frequency=1000 vis=True MODEL.down_conv.nsamples='[[10], [16], [16], [31]]' \
# use_wandb=True
#
#   2.40916:  # (good!) train for partial shape, bottle, object pts only, R, L2, dense, use confidence type 1, confidence_loss_multiplier=0.1, bigger receptive field;
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40916' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_conf=True use_confidence_R=True confidence_loss_type=1 confidence_loss_multiplier=0.1 \
# vis_frequency=1000 vis=True \
# MODEL.down_conv.npoint='[256, 64, 32, 16]' \
# MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
# eval=True save=True ckpt=latest eval_mode_r=0
# use_wandb=True

  2.409161:  # encoder + decoder, modal=5, with classifyM loss
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.409161' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32*3, 16*3, pooling 1]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=5 MODEL.encoder_only=False \
pred_mode=True use_objective_M=True \
eval=True save=True ckpt=latest eval_mode_r=0
use_wandb=True

  2.409162:  # encoder + decoder, modal=2, with classifyM loss
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.409162' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=2 MODEL.encoder_only=False \
pred_mode=True use_objective_M=True \
eval=True save=True ckpt=latest eval_mode_r=0
use_wandb=True

2.4091621:  # encoder + decoder, modal=2, with classifyM loss, add mode regularization
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.4091621' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=2 MODEL.encoder_only=False \
pred_mode=True use_objective_M=True use_objective_V=True consistency_loss_multiplier=1.0 \
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_2.4091621.log
use_wandb=True

2.40916211:  # encoder + decoder, modal=2, with classifyM loss, add mode regularization, encoder + decoder, modal=2, with classifyM loss, add mode regularization, no loss added
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40916211' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=2 MODEL.encoder_only=False \
pred_mode=True use_objective_M=True check_consistency=True consistency_loss_multiplier=1.0 \
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_2.4091621.log
use_wandb=True

# overfit 100 examples
# 2.40916212:  # encoder + decoder, modal=2, with classifyM loss, add mode regularization,
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40916212' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# vis_frequency=1000 vis=True \
# MODEL.down_conv.npoint='[256, 64, 32, 16]' \
# MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
# MODEL.num_channels_R=2 MODEL.encoder_only=False \
# pred_mode=True use_objective_M=True use_objective_V=True consistency_loss_multiplier=1.0 \
# subset_samples=True \
# use_wandb=True
# eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_2.4091621.log

2.4091622:  # encoder + decoder, modal=2, with classifyM loss, add mode regularization
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.4091622' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=2 MODEL.encoder_only=False \
pred_mode=True use_objective_M=True use_objective_V=True consistency_loss_multiplier=0.1 \
use_wandb=True \
# to eval on full test set, just add below
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_2.4091622.log

#
2.4091623:  # encoder + decoder, modal=2, with classifyM loss, add mode regularization
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.4091623' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
vis_frequency=1000 vis=True \
models=se3_transformerv2 \
MODEL.num_channels_R=2 MODEL.encoder_only=False \
pred_mode=True use_objective_M=True use_objective_V=True consistency_loss_multiplier=0.1 \
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_2.4091623.log
use_wandb=True

2.4091624:  # no skip link, encoder + decoder, modal=2, with classifyM loss, add mode regularization
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.4091624' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
vis_frequency=1000 vis=True \
models=se3_transformerv2 \
MODEL.num_channels_R=2 MODEL.encoder_only=False \
pred_mode=True use_objective_M=True use_objective_V=True consistency_loss_multiplier=0.1 \
use_wandb=True
#
# 2.40917: # train for partial shape, bottle, object pts only, R, L2, dense, use confidence type 1, confidence_loss_multiplier=0.1, bigger receptive field;
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40917' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_conf=True use_confidence_R=True confidence_loss_type=1 confidence_loss_multiplier=0.1 \
# vis_frequency=1000 vis=True \
# models=se3_transformerv1 \
# use_wandb=True
#
# 2.40918: # train for partial shape, bottle, object pts only, R, L2, dense, use confidence type 1, confidence_loss_multiplier=0.1, bigger receptive field;
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40918' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_conf=True use_confidence_R=True confidence_loss_type=1 confidence_loss_multiplier=0.1 \
# vis_frequency=1000 vis=True MODEL.down_conv.nsamples='[[10], [16], [16], [31]]' \
# MODEL.num_channels_R=2 \
# use_wandb=True
#
# 2.40919: # encoder only, modal=2, with classifyM loss
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40919' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# vis_frequency=1000 vis=True \
# models=se3_transformerv3 \
# MODEL.num_channels_R=2 MODEL.encoder_only=True \
# pred_mode=True use_objective_M=True \
# use_wandb=True
# eval=True save=True
#
# 2.409191: # encoder only, modal=5, with classifyM loss
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.409191' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# vis_frequency=1000 vis=True \
# models=se3_transformerv3 \
# MODEL.num_channels_R=5 MODEL.encoder_only=True \
# pred_mode=True use_objective_M=True \
# eval=True save=True
# use_wandb=True
#
# 2.409192: # encoder only, modal=5, with classifyM loss
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.409192' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# vis_frequency=1000 vis=True \
# models=se3_transformerv3 \
# MODEL.num_channels_R=5 MODEL.encoder_only=True MODEL.down_conv.npoint='[256, 64, 16, 4]' \
# MODEL.down_conv.nsamples='[[16], [16], [15], [3]]' \
# pred_mode=True use_objective_M=True \
# eval=True save=True
# use_wandb=True
#
# 2.409193: # encoder only, modal=2, with classifyM loss, v4
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.409193' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# vis_frequency=1000 vis=True \
# models=se3_transformerv4 \
# MODEL.num_channels_R=2 MODEL.encoder_only=True \
# pred_mode=True use_objective_M=True \
# eval=True save=True ckpt=latest eval_mode_r=0
# use_wandb=True
#
# 2.409194: # encoder only, modal=2, with classifyM loss, v5
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.409194' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# vis_frequency=1000 vis=True \
# models=se3_transformerv5 \
# MODEL.num_channels_R=2 MODEL.encoder_only=True \
# pred_mode=True use_objective_M=True \
# eval=True save=True ckpt=latest eval_mode_r=0
# use_wandb=True
#
# 2.409195: # encoder only, modal=2, with classifyM loss, v6
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.409195' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# vis_frequency=1000 vis=True \
# models=se3_transformerv6 \
# MODEL.num_channels_R=2 MODEL.encoder_only=True \
# pred_mode=True use_objective_M=True \
# eval=True save=True ckpt=latest eval_mode_r=0
# use_wandb=True
#
# 2.409196: #(killed)  encoder only, modal=2, with classifyM loss, v3 with pooling;
# # another round with different pooling( normalize before averaging )
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.409196' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True \
# vis_frequency=1000 vis=True \
# models=se3_transformerv3 \
# MODEL.num_channels_R=2 MODEL.encoder_only=True \
# pred_mode=True use_objective_M=True \
# eval=True save=True ckpt=latest eval_mode_r=0
# use_wandb=True
#
# 2.409197: # (killed) encoder only, modal=2, with classifyM loss, v4 with pooling;
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.409197' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True \
# vis_frequency=1000 vis=True \
# models=se3_transformerv4 \
# MODEL.num_channels_R=2 MODEL.encoder_only=True \
# pred_mode=True use_objective_M=True \
# eval=True save=True ckpt=latest eval_mode_r=0
# use_wandb=True
#
# 2.409198: #  encoder only, modal=2, with classifyM loss, v5 with pooling;
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.409198' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True \
# vis_frequency=1000 vis=True \
# models=se3_transformerv5 \
# MODEL.num_channels_R=2 MODEL.encoder_only=True \
# pred_mode=True use_objective_M=True \
# eval=True save=True ckpt=latest eval_mode_r=0
# use_wandb=True
#
# 2.409199: #  encoder only, modal=2, with classifyM loss, v3_1 with pooling;
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.409199' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True \
# vis_frequency=1000 vis=True \
# models=se3_transformerv3_1 \
# MODEL.num_channels_R=2 MODEL.encoder_only=True \
# pred_mode=True use_objective_M=True \
# eval=True save=True ckpt=latest eval_mode_r=0
# use_wandb=True
#
# 2.4091991: # (killed) encoder only, modal=2, with classifyM loss, v3_2 with pooling;
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.4091991' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True \
# vis_frequency=1000 vis=True \
# models=se3_transformerv3_2 \
# MODEL.num_channels_R=2 MODEL.encoder_only=True \
# pred_mode=True use_objective_M=True \
# eval=True save=True ckpt=latest eval_mode_r=0
# use_wandb=True


# 2.4092: # train for partial shape, bottle, object pts only, T voting
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.4092' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# use_objective_T=True \
# eval=True save=True \
# use_wandb=True

2.4092 # T_voting training
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.4092' TRAIN.train_batch=2 TRAIN.test_batch=2 \
use_objective_T=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
use_wandb=True
eval=True save=True \

  # --->2.4093: # use type 0 to predict NOCS
  # TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_layers=12 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  # $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.4093' TRAIN.train_batch=2 TRAIN.test_batch=2 \
  # pred_nocs=True use_objective_N=True rotation_loss_type=0 use_wandb=True
  #
  # ---->2.40931: # single instance
  # TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_layers=12 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  # $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40931' TRAIN.train_batch=2 TRAIN.test_batch=2 \
  # pred_nocs=True use_objective_N=True single_instance=True rotation_loss_type=1 use_wandb=True
  #
  # ---> 2.4094: # use type 0 predict NOCS on complete shape
  # TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 name_model=ae dataset_class=HandDatasetAEGraph'
  # $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4094' TRAIN.train_batch=2 TRAIN.test_batch=2 \
  # augment=True pred_nocs=True use_objective_N=True use_wandb=True

2.40941: # use type 0 predict NOCS on complete shape, NOCS
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.40941' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True pred_nocs=True use_objective_N=True \
eval=True save=True
use_wandb=True

# 2.4095: # new best model, use partial shape for R estimation, multiple instance, dynamic graph, use confidence type 1, use hadn points, averaged R over all points
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.4095' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True \
# use_hand=True pred_seg=True use_objective_C=True \
# use_wandb=True
#
# # 2.40951
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40951' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True \
# use_wandb=True
#
# # 2.40952
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40952' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True \
# use_wandb=True

2.4096 # encoder + decoder, modal=2, with classifyM loss, add mode regularization, with hand points, with segmentation;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.4096' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=2 MODEL.encoder_only=False \
pred_mode=True use_objective_M=True use_objective_V=True consistency_loss_multiplier=0.1 \
use_hand=True pred_seg=True use_objective_C=True \
use_wandb=True \
# to eval on full test set, just add below
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_2.4096.log

2.4097: # R estimation, new best model, partial shape for, multiple instance
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.4097' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
use_hand=True pred_seg=True use_objective_C=True \
use_wandb=True

# #   2.40971: # R estimation, new best model, partial shape for, multiple instance, use confidence type 0
# # TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# # $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40971' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# # augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# # use_hand=True pred_seg=True use_objective_C=True \
# # pred_conf=True use_confidence_R=True confidence_loss_type=0 confidence_loss_multiplier=0.1 \
# # use_wandb=True
#
# 2.40972: # R estimation, new best model, partial shape for, multiple instance, use confidence type 1, confidence_loss_multiplier=0.1
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40972' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# use_hand=True pred_seg=True use_objective_C=True \
# pred_conf=True use_confidence_R=True confidence_loss_type=1 confidence_loss_multiplier=0.1 \
# use_wandb=True vis_frequency=100 vis=True
#
# 2.40973: # R estimation, new best model, partial shape for, multiple instance, use confidence type 1, confidence_loss_multiplier=0.1
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40973' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# use_hand=True pred_seg=True use_objective_C=True \
# pred_conf=True use_confidence_R=True confidence_loss_type=1 confidence_loss_multiplier=0.1 \
# use_wandb=True vis_frequency=100 vis=True MODEL.down_conv.nsamples='[[10], [16], [16], [31]]'
#
# # 2.40973: # R estimation, new best model, partial shape for, multiple instance, use confidence type 1, confidence_loss_multiplier=1
# # TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# # $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40973' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# # augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# # use_hand=True pred_seg=True use_objective_C=True \
# # pred_conf=True use_confidence_R=True confidence_loss_type=1 confidence_loss_multiplier=1.0 \
# # use_wandb=True





  2.4098: # T voting, new best model, use partial shape, use_hand
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.4098' TRAIN.train_batch=2 TRAIN.test_batch=2 \
use_objective_T=True \
use_hand=True pred_seg=True use_objective_C=True use_wandb=True

    2.40981: # T voting, new best model, use partial shape, use_hand, no segmentation
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40981' TRAIN.train_batch=2 TRAIN.test_batch=2 \
use_objective_T=True \
use_hand=True pred_seg=True use_wandb=True

  2.4099: # new best model, use partial shape for NOCS, multiple instance, dynamic graph
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.4099' TRAIN.train_batch=2 TRAIN.test_batch=2 \
pred_nocs=True use_objective_N=True \
use_hand=True pred_seg=True use_objective_C=True use_wandb=True

    2.40991 #
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40991' TRAIN.train_batch=2 TRAIN.test_batch=2 \
pred_nocs=True use_objective_N=True \
use_hand=True pred_seg=True use_wandb=True


>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2.41: jar <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  2.411 # jar, AE-Graph, synthetic complete pcloud
  TRAIN_OBJ='python train_aegan.py training=ae_gan use_wandb=True vis=True n_pts=256 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ target_category='jar' exp_num='2.411'  TRAIN.train_batch=2 TRAIN.test_batch=2

  2.412: # jar,
  TRAIN_OBJ='python train_aegan.py training=ae_gan use_wandb=True vis=True num_points=1024 n_pts=1024 MODEL.num_degrees=2 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ target_category='jar' exp_num='2.402' TRAIN.train_batch=4 TRAIN.test_batch=4

  2.413: # jar
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='jar' exp_num='2.413' TRAIN.train_batch=2 TRAIN.test_batch=2 \
  augment=True use_wandb=True

  2.414: # multiple instance, dynamic graph, with augmentation, L2, jar
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='jar' exp_num='2.414' TRAIN.train_batch=2 TRAIN.test_batch=2 \
use_objective_R=True augment=True rotation_loss_type=1 \
use_wandb=True
  eval=True save=True
#
#   2.4141: # encoder only, modal=2, with classifyM loss
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='jar' exp_num='2.4141' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# vis_frequency=1000 vis=True \
# models=se3_transformerv3 \
# MODEL.num_channels_R=2 MODEL.encoder_only=True \
# pred_mode=True use_objective_M=True \
# use_wandb=True
#
# 2.4142:
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='jar' exp_num='2.4142' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# vis_frequency=1000 vis=True \
# MODEL.down_conv.npoint='[256, 64, 32, 16]' \
# MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
# MODEL.num_channels_R=2 MODEL.encoder_only=False \
# pred_mode=True use_objective_M=True \
# use_wandb=True
#
#   2.415: # multiple instance, dynamic graph, with augmentation, T voting, jar
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='pcloud_pose' target_category='jar' exp_num='2.415' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# use_objective_T=True augment=True \
# use_wandb=True
#    eval=True save=True
#
#   2.416: # use type 0 predict NOCS on complete shape, NOCS, jar
#   TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 name_model=ae dataset_class=HandDatasetAEGraph'
#   $TRAIN_OBJ task='pcloud_pose' target_category='jar' exp_num='2.416' TRAIN.train_batch=2 TRAIN.test_batch=2 \
#   augment=True pred_nocs=True use_objective_N=True \
#   use_wandb=True
#   eval=True save=True

  2.416: # ca197 0 jar, encoder + decoder, modal=2, with classifyM loss, add mode regularization, with hand points, with segmentation;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='jar' exp_num='2.416' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=2 MODEL.encoder_only=False \
pred_mode=True use_objective_M=True use_objective_V=True consistency_loss_multiplier=0.1 \
use_hand=True pred_seg=True use_objective_C=True \
use_wandb=True \
# to eval on full test set, just add below
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_2.4096.log

  2.417:  # jar, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict full R;
  #       (try to find multiple GT, use the minimum, R loss with multiple candidates)
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='jar' exp_num='2.417' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.n_heads=4 \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.up_conv.n_heads=4 \
MODEL.n_heads=4 \
MODEL.num_channels_R=2 MODEL.encoder_only=False \
pred_mode=True use_objective_M=True use_objective_V=True consistency_loss_multiplier=0.1 \
use_wandb=True \
# to eval on full test set, just add below
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_2.4091622.log


2.4171:  # jar, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='jar' exp_num='2.4171' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True \
eval=True save=True ckpt=latest eval_mode_r=1 2>&1 | tee results/eval_2.4171_latest_upside.log
use_wandb=True \
# to eval on full test set, just add below

2.42 # all categories, AE, synthetic complete pcloud
  python train_aegan.py training=ae_gan name_model=ae dataset_class='HandDatasetAEGan' exp_num='2.42'


>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2.43: remote <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  2.431: # multiple instance, dynamic graph, with augmentation, L2, remote
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='remote' exp_num='2.431' TRAIN.train_batch=2 TRAIN.test_batch=2 \
use_objective_R=True augment=True rotation_loss_type=1 \
use_wandb=True \
eval=True save=True

  2.432: # multiple instance, dynamic graph, with augmentation, T voting, remote
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='remote' exp_num='2.432' TRAIN.train_batch=2 TRAIN.test_batch=2 \
use_objective_T=True augment=True \
use_wandb=True
  eval=True save=True

  2.433: # use type 0 predict NOCS on complete shape, NOCS, remote
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='remote' exp_num='2.433' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True pred_nocs=True use_objective_N=True \
use_wandb=True
eval=True save=True

2.434:  # remote, use R, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='remote' exp_num='2.434' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True use_adaptive_mode=True \
pred_6d=True num_modes_R=5 MODEL.num_channels_R=10 mode_features='[5, 'softmax']' \
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_2.434.log
use_wandb=True \


>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2.44: cellphone, <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
2.441:  # cellphone, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='cellphone' exp_num='2.441' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True \
use_wandb=True \
# to eval on full test set, just add below
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_2.441.log

2.442:  # ceelphone, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='cellphone' exp_num='2.442' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_adaptive_mode=True \
pred_6d=True num_modes_R=5 MODEL.num_channels_R=10 mode_features='[5, 'softmax']' \
use_wandb=True \

2.443:  # ceelphone, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='cellphone' exp_num='2.443' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True \
pred_6d=True num_modes_R=5 MODEL.num_channels_R=10 mode_features='[5, 'softmax']' \
use_wandb=True \

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2.45: camera <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  2.451: # multiple instance, dynamic graph, with augmentation, L2, camera
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='camera' exp_num='2.451' TRAIN.train_batch=2 TRAIN.test_batch=2 \
use_objective_R=True augment=True rotation_loss_type=1 \
use_wandb=True
   eval=True save=True

   2.4511: # encoder only, modal=2, with classifyM loss
 TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
 $TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='2.4511' TRAIN.train_batch=2 TRAIN.test_batch=2 \
 augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
 vis_frequency=1000 vis=True \
 models=se3_transformerv3 \
 MODEL.num_channels_R=2 MODEL.encoder_only=True \
 pred_mode=True use_objective_M=True \
 use_wandb=True

 2.4512:  # encoder + decoder, modal=2, with classifyM loss
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='2.4512' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=2 MODEL.encoder_only=False \
pred_mode=True use_objective_M=True \
use_wandb=True

#   2.452: # 2.406971, multiple instance, dynamic graph, with augmentation, T voting, camera
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='pcloud_pose' target_category='camera' exp_num='2.452' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# use_objective_T=True augment=True \
# use_wandb=True
# eval=True save=True

2.452: # ca197 1 camera, encoder + decoder, modal=2, with classifyM loss, add mode regularization, with hand points, with segmentation;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='2.452' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=2 MODEL.encoder_only=False \
pred_mode=True use_objective_M=True use_objective_V=True consistency_loss_multiplier=0.1 \
use_hand=True pred_seg=True use_objective_C=True \
use_wandb=True \
# to eval on full test set, just add below
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_2.4096.log


  2.453: # use type 0 predict NOCS on complete shape, NOCS, camera
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='camera' exp_num='2.453' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True pred_nocs=True use_objective_N=True \
use_wandb=True
eval=True save=True

# 2.454:  # camera, use R, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='2.454' TRAIN.train_batch=2 TRAIN.test_batch=2 \
# models=se3_transformer_default \
# augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
# pred_mode=True use_objective_M=True use_objective_V=True  \
# pred_6d=True num_modes_R=2 MODEL.num_channels_R=4 mode_features='[2, 'softmax']' \
# use_wandb=True \
# # to eval on full test set, just add below
# eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_2.454.log

2.4541:  # camera, use R, adaptive mode, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='2.4541' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True use_adaptive_mode=True \
pred_6d=True num_modes_R=2 MODEL.num_channels_R=4 mode_features='[2, 'softmax']' \
use_wandb=True \
# to eval on full test set, just add below
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_2.454.log

2.4542:  # camera, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='2.4542' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True \
pred_6d=True num_modes_R=2 MODEL.num_channels_R=4 mode_features='[2, 'softmax']' \
use_wandb=True \
# to eval on full test set, just add below
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_2.454.log

2.4543:  # camera, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='2.4543' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True use_adaptive_mode=True \
pred_6d=True num_modes_R=2 MODEL.num_channels_R=4 mode_features='[2, 'softmax']' \
use_wandb=True \

2.4544:  # camera, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='2.4544' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True \
use_wandb=True
# eval=True save=True eval_mode_r=1 2>&1 | tee results/eval_2.471.log
# use_wandb=True \

2.4545 # pointnet++ with pooling, knn
TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='2.4545' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
MODEL.num_channels_R=2 HEAD.R='[128, 128, 6, None]' \
pred_mode=True use_objective_M=True use_objective_V=True consistency_loss_multiplier=0.1 MODEL.use_ball_query=False \
use_wandb=True

2.4547:  # camera, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='camera' exp_num='2.4547' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_adaptive_mode=True \
pred_6d=True num_modes_R=5 MODEL.num_channels_R=10 mode_features='[5, 'softmax']' \
eval=True save=True ckpt=best eval_mode_r=0 2>&1 | tee results/eval_2.4547.log
use_wandb=True \

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2.46: knife <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
2.461:  # ca226, 0 knife, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='knife' exp_num='2.461' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True \
pred_6d=True num_modes_R=2 MODEL.num_channels_R=4 mode_features='[2, 'softmax']' \
use_wandb=True \

2.462:  # ca226, 1 knife, use R^T, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='knife' exp_num='2.462' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_adaptive_mode=True \
pred_6d=True num_modes_R=2 MODEL.num_channels_R=4 mode_features='[2, 'softmax']' \
use_wandb=True \

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2.47: bowl <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
2.47 # ca211 0 bowl, encoder + decoder, modal=2, with classifyM loss, add mode regularization, with hand points, with segmentation;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bowl' exp_num='2.47' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=2 MODEL.encoder_only=False \
pred_mode=True use_objective_M=True use_objective_V=True consistency_loss_multiplier=0.1 \
use_hand=True pred_seg=True use_objective_C=True \
use_wandb=True \
# to eval on full test set, just add below
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_2.4096.log

2.471:  # bowl, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bowl' exp_num='2.471' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True \
eval=True save=True eval_mode_r=1 2>&1 | tee results/eval_2.471.log
use_wandb=True \

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2.48: can <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
2.481:  # can, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='can' exp_num='2.481' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True use_one2many=True \
eval=True save=True
use_wandb=True \
# to eval on full test set, just add below
eval=True save=True ckpt=latest eval_mode_r=0 2>&1 | tee results/eval_2.478.log


2.482:  # can, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='can' exp_num='2.482' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True \
eval=True save=True ckpt=best eval_mode_r=0 2>&1 | tee results/eval_2.482_best_0.log
use_wandb=True \


3.1 # encoder-decoder full training, pooled prediction, 1 mode
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae datasets=modelnet40 dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='modelnet40' name_dset='modelnet40' exp_num='3.1' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=1 MODEL.encoder_only=False \
use_wandb=True
eval=True save=True 2>&1 | tee results/eval_default.log
use_wandb=True

3.11 # encoder-decoder, 2 mode, dense prediction, check visulizations and log
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae datasets=modelnet40 dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='modelnet40' name_dset='modelnet40' exp_num='3.11' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=2 MODEL.encoder_only=False \
pred_mode=True use_objective_M=True \
use_wandb=True
eval=True save=True \
use_wandb=True

3.12 # mode=1, category chair
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae datasets=modelnet40 dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='chair' exp_num='3.12' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=1 MODEL.encoder_only=False \
eval=True save=True 2>&1 | tee results/eval.2.40585.log
use_wandb=True

3.13 # mode=1, category monitor
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae datasets=modelnet40 dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='monitor' exp_num='3.13' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=1 MODEL.encoder_only=False \
eval=True save=True 2>&1 | tee results/eval.2.40585.log
use_wandb=True


3.2 # completion
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae datasets=modelnet40 dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_completion' item='modelnet40' name_dset='modelnet40' exp_num='3.2' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=1 MODEL.encoder_only=False \
use_wandb=True
eval=True save=True 2>&1 | tee results/eval_default.log
use_wandb=True

3.21 # completion for chair
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae datasets=modelnet40 dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_completion' item='modelnet40' name_dset='modelnet40' exp_num='3.21' target_category='chair' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=1 MODEL.encoder_only=False \
use_wandb=True
eval=True save=True 2>&1 | tee results/eval_default.log
use_wandb=True

# try the same thing here!!!
airplane, chair, car!!! same thing here

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> unsupervised completion <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

3.3 # unsupervised training on chair class
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae datasets=modelnet40 dataset_class=AEGraph'
$TRAIN_OBJ task='unsupervised_pcloud_pose_completion' item='modelnet40' name_dset='modelnet40' exp_num='3.31' target_category='chair' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 \
pred_6d=True num_modes_R=1 MODEL.num_channels_R=2
use_wandb=True
eval=True save=True 2>&1 | tee results/eval_default.log
use_wandb=True
# pred_mode=True use_objective_M=True use_objective_V=True \

2.471:  # bowl, encoder + decoder, modal=2, with classifyM loss, mode regularization, 4 heads, without hands, predict up axis is enough;
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True vis_frequency=1000 num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bowl' exp_num='2.471' TRAIN.train_batch=2 TRAIN.test_batch=2 \
models=se3_transformer_default \
augment=True rotation_loss_type=1 use_objective_R=True rotation_use_dense=True \
pred_mode=True use_objective_M=True use_objective_V=True \
eval=True save=True eval_mode_r=1 2>&1 | tee results/eval_2.471.log
use_wandb=True \

3.4 GAN
