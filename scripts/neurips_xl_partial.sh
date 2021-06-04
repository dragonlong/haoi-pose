#>>>>>>>>>>>>>>>>>>>>>>>>>>>> partial ModelNet40 dataset <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

# # 0.9
# 0.91: # supervision in camera space, random R, airplane, but add T estimation, dense per-point voting, R0
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.91' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_wandb=True
# eval=True save=True pre_compute_delta=True
#
# 0.911: # supervision in camera space, random R, airplane, but add T estimation, dense per-point voting, R0, project poinst into depth
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.911' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 use_objective_P=True \
# eval=True save=True pre_compute_delta=True
# use_wandb=True
#
# 0.912: # supervision in camera space, airplane, but add T estimation, random SO3 rotate points
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.912' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# augment=True MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# eval=True save=True pre_compute_delta=True \
# use_wandb=True
#
# 0.913: # supervision in camera space, airplane, but add T estimation, use fps sampling
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.913' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_fps_points=True \
# use_wandb=True
# eval=True save=True pre_compute_delta=True

0.913r: # supervision in camera space, airplane, but add T estimation, use fps sampling,
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.913r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.913r1: # ca226 0 supervision in camera space, airplane, but add T estimation, use fps sampling, add projection loss 0
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.913r1' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True p_method_type=0 use_objective_P=True \
use_wandb=True

0.913r2: # ca226 1# supervision in camera space, airplane, but add T estimation, use fps sampling, add projection loss 0
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.913r2' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True p_method_type=1 use_objective_P=True \
use_wandb=True

# 0.9131: # supervision in camera space, airplane, but add T estimation, use fps sampling, correct points prediction
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.9131' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_fps_points=True r_method_type=-1 \
# use_wandb=True
#
# 0.914: # supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.914' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_fps_points=True r_method_type=1 \
# use_wandb=True
#
# 0.9141: # supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.9141' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_fps_points=True r_method_type=1 \
# use_wandb=True

0.9141r: # corrected, supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.9141r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.9141r1: # corrected, supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.9141r1' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 use_objective_P=True p_method_type=0 \
use_wandb=True

0.9141r2: # corrected, supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.9141r2' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 use_objective_P=True p_method_type=0 \
use_wandb=True
#
# 0.915: # supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, a smaller angle constr
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.915' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_fps_points=True r_method_type=1 \
# use_wandb=True
#
# 0.916: # supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, use 0.1 * projection loss
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.916' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_fps_points=True r_method_type=1 p_method_type=0 use_objective_P=True \
# use_wandb=True
#
# 0.9161: # supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, use 0.01 * projection_loss
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.9161' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_fps_points=True r_method_type=1 p_method_type=0 use_objective_P=True \
# use_wandb=True
#
# 0.9162: # supervision in camera space, random R, airplane, but add T estimation,  and 0.01 projection loss
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.9162r' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_fps_points=True r_method_type=0 p_method_type=0 use_objective_P=True \
# use_wandb=True

0.91a: # random R, airplane, but add T estimation, dense per-point voting, R0
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.91a' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
eval=True
use_wandb=True

0.91b: # ca217, random R, airplane, but add T estimation, dense per-point voting, R0
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.91b' model.model='enc_so3net' model.pooling_method='max' model.kpconv=True \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
use_wandb=True

0.91b1: # ca217, random R, airplane, but add T estimation, dense per-point voting, R0
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.91b1' model.model='enc_so3net' model.pooling_method='max' model.kpconv=True \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
eval=True save=True
use_wandb=True

0.913d: # supervision in camera space, airplane, but add T estimation, use fps sampling, use KPConv, t_methods =1
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.913d' model.model='enc_so3net' model.pooling_method='max' model.kpconv=True \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=1 \
use_fps_points=True \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.913d1: # supervision in camera space, airplane, but add T estimation, use fps sampling, use KPConv, t_methods = 0
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.913d1' model.model='enc_so3net' model.pooling_method='max' model.kpconv=True \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.91e: # self-supervised with SE3-transformer
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
exp_num='0.91e' target_category='airplane' \
models=se3_transformer_default encoder_type=se3_transformer MODEL.num_in_channels=1 MODEL.num_channels_R=4 MODEL.num_channels_T=2 num_points=1024 model.input_num=512 num_modes_R=2 \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new dataset_class=AE \
TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True use_head_assemble=True mode_features='[2, 'softmax']' \
pred_t=True t_method_type=1 \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.911e: # self-supervised with SE3-transformer
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
exp_num='0.911e' target_category='airplane' \
models=se3_transformer_default encoder_type=se3_transformer MODEL.num_in_channels=1 MODEL.num_channels_R=4 MODEL.num_channels_T=2 num_points=1024 model.input_num=512 num_modes_R=2 \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new dataset_class=AE \
TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True use_head_assemble=True mode_features='[2, 'softmax']' \
pred_t=True t_method_type=1 \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.91f: # self-supervised with SE3-transformer
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
exp_num='0.91f' target_category='airplane' \
models=se3_transformer_default encoder_type=se3_transformer MODEL.num_in_channels=1 num_modes_R=1 MODEL.num_channels_R=2 num_points=1024 in_points=512 model.input_num=512 \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new dataset_class=AE \
TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True use_head_assemble=True mode_features='[2, 'softmax']' \
pred_t=True t_method_type=1 \
eval=True save=True pre_compute_delta=True
2>&1 | tee evaluation/logs/0.91f.txt

pre_compute_delta=True
use_wandb=True

0.91f1: # self-supervised with SE3-transformer
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
exp_num='0.91f1' target_category='airplane' \
models=se3_transformer_default encoder_type=se3_transformer MODEL.num_in_channels=1 num_modes_R=1 MODEL.num_channels_R=2 num_points=1024 in_points=512 model.input_num=512 \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new dataset_class=AE \
TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True use_head_assemble=True mode_features='[2, 'softmax']' \
pred_t=True t_method_type=1 use_fps_points=True \
eval=True save=True pre_compute_delta=True 2>&1 | tee evaluation/logs/0.91f1.txt
use_wandb=True
# >>>> 0.92, car
# 0.92: # supervision in camera space, random R, airplane, but add T estimation, R0
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.92' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='car' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# augment=True MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# eval=True save=True pre_compute_delta=True
# use_wandb=True

0.92r: # supervision in camera space, airplane, but add T estimation, use fps sampling,
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.92r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.92r1: # supervision in camera space, airplane, but add T estimation, use fps sampling, re-run
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.92r1' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
use_wandb=True


0.921r: # corrected, supervision in camera space, random R, car, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.921r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.921r1: # corrected, supervision in camera space, random R, car, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.921r1' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
use_wandb=True
# 0.921: # supervision in camera space, random R, airplane, but add T estimation, dense per-point voting, R0, ca201 0
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.921' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='car' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 use_objective_P=True \
# eval=True save=True pre_compute_delta=True
# use_wandb=True

#
0.92a: # random R, airplane, but add T estimation, dense per-point voting, R0, ca201 1
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.92a' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
eval=True
use_wandb=True

0.92b: # random R, airplane, but add T estimation, dense per-point voting, R0, ca201 1
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.92b' model.model='enc_so3net' model.pooling_method='max' model.kpconv=True \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
use_wandb=True

0.92b1: # random R, airplane, but add T estimation, dense per-point voting, R0, ca201 1
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.92b1' model.model='enc_so3net' model.pooling_method='max' model.kpconv=True \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
use_wandb=True

0.92e: # self-supervised with SE3-transformer
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
exp_num='0.92e' target_category='car' \
models=se3_transformer_default encoder_type=se3_transformer MODEL.num_in_channels=1 MODEL.num_channels_R=4 MODEL.num_channels_T=2 num_points=1024 model.input_num=512 num_modes_R=2 \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new dataset_class=AE \
TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True use_head_assemble=True mode_features='[2, 'softmax']' \
pred_t=True t_method_type=1 \
use_wandb=True

0.92f: # self-supervised with SE3-transformer
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
exp_num='0.92f' target_category='car' \
models=se3_transformer_default encoder_type=se3_transformer MODEL.num_in_channels=1 num_modes_R=1 MODEL.num_channels_R=2 num_points=1024 in_points=512 model.input_num=512 \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new dataset_class=AE \
TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True use_head_assemble=True mode_features='[2, 'softmax']' \
pred_t=True t_method_type=1 use_fps_points=True \
use_wandb=True

# # >>>> 0.93, ca203 1
# 0.93: # supervision in camera space, random R, airplane, but add T estimation, dense per-point voting, R0
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.93' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='bowl' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_wandb=True

# >>>> 0.94, ca207 1, bottle
# 0.94: # supervision in camera space, random R, airplane, but add T estimation, dense per-point voting, R0
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.94' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='bottle' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_wandb=True

0.94r: # supervision in camera space, bottle, but add T estimation, use fps sampling,
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.94r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
use_wandb=True

0.941r: # corrected, supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.941r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
use_wandb=True

0.94a: # random R, airplane, but add T estimation, dense per-point voting, R0, ca201 1
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.94a' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
use_fps_points=True use_axis=True  \
eval=True save=True \
use_wandb=True

0.94b: # random R, airplane, but add T estimation, dense per-point voting, R0, ca201 1
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.94b' model.model='enc_so3net' model.pooling_method='max' model.kpconv=True \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
use_fps_points=True use_axis=True \
eval=True save=True
use_wandb=True

0.94f: # self-supervised with SE3-transformer
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
exp_num='0.94f' target_category='bottle' \
models=se3_transformer_default encoder_type=se3_transformer MODEL.num_in_channels=1 num_modes_R=1 MODEL.num_channels_R=2 num_points=1024 in_points=512 model.input_num=512 \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new dataset_class=AE \
TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True use_head_assemble=True mode_features='[2, 'softmax']' \
pred_t=True t_method_type=1 \
use_fps_points=True use_axis=True \
use_wandb=True

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> chair <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,
0.95r: # supervision in camera space, bottle, but add T estimation, use fps sampling,
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.95r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
use_wandb=True

0.951r: # corrected, supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.951r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
use_wandb=True

0.95a: # random R, airplane, but add T estimation, dense per-point voting, R0
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.95a' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
use_wandb=True

0.95b: # random R, airplane, but add T estimation, dense per-point voting, R0
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.95b' model.model='enc_so3net' model.pooling_method='max' model.kpconv=True \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
use_wandb=True

0.95f: # self-supervised with SE3-transformer
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
exp_num='0.95f' target_category='chair' \
models=se3_transformer_default encoder_type=se3_transformer MODEL.num_in_channels=1 num_modes_R=1 MODEL.num_channels_R=2 num_points=1024 in_points=512 model.input_num=512 \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new dataset_class=AE \
TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True use_head_assemble=True mode_features='[2, 'softmax']' \
pred_t=True t_method_type=1 \
use_fps_points=True \
use_wandb=True

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> sofa <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,
0.96r: # supervision in camera space,, but add T estimation, use fps sampling,
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.96r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='sofa' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
use_wandb=True

0.961r: # corrected, supervision in camera space, random R,but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.961r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='sofa' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
use_wandb=True

0.96a: # n random R, airplane, but add T estimation, dense per-point voting, R0
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.96a' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='sofa' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
use_wandb=True

0.96b: # random R, airplane, but add T estimation, dense per-point voting, R0
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.96b' model.model='enc_so3net' model.pooling_method='max' model.kpconv=True \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='sofa' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
use_wandb=True

0.96f: # self-supervised with SE3-transformer
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
exp_num='0.96f' target_category='sofa' \
models=se3_transformer_default encoder_type=se3_transformer MODEL.num_in_channels=1 num_modes_R=1 MODEL.num_channels_R=2 num_points=1024 in_points=512 model.input_num=512 \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new dataset_class=AE \
TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True use_head_assemble=True mode_features='[2, 'softmax']' \
pred_t=True t_method_type=1 \
use_fps_points=True \
use_wandb=True
