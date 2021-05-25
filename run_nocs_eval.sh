# >>>>>>>>>>>>>>>>>>>>>>>>>> partial dataset NOCS
0.861: # supervision in camera space, random R, bowl
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.861' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.861r: # supervision in camera space, bottle, but add T estimation, use fps sampling,
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.861r' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
use_wandb=True

0.861a: # supervision in camera space, bottle, but add T estimation, use fps sampling,
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.861a' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
use_wandb=True

0.861r1: ca213 0 # supervision in camera space, bottle, but add T estimation, use fps sampling, projection_loss 0
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.861r1' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True use_objective_P=True p_method_type=0 \
use_wandb=True

0.861r2: # supervision in camera space, bottle, but add T estimation, use fps sampling, projection_loss 0
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.861r2' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True use_objective_P=True p_method_type=1 \
use_wandb=True

0.8611r: # ca217, 1corrected, supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8611r' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
use_wandb=True

0.8611r1: # supervision in camera space, bottle, but add T estimation, use fps sampling, projection_loss 0
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8611r1' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 use_objective_P=True p_method_type=0 \
use_wandb=True

0.8611r2: # supervision in camera space, bottle, but add T estimation, use fps sampling, projection_loss 0
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8611r2' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 use_objective_P=True p_method_type=1 \
use_wandb=True


0.8611: # supervision in camera space, random R, bowl, add T, s
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8611' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True \
use_wandb=True

0.8612: # supervision in camera space, random R, bowl, add T, s, Chamfer L1 + projection loss
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8612' model.model='enc_so3net' model.pooling_method='pointnet' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True use_objective_P=True t_method_type=-1 \
use_wandb=True





0.8621: # supervision in camera space, random R, mug, add T, s, Chamfer L1
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8621' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='mug' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True t_method_type=-1 \
use_wandb=True

0.8621r: # supervision in camera space, bottle, but add T estimation, use fps sampling,
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8621r' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='mug' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
use_wandb=True

0.8622r: # ca217, 1corrected, supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8622r' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='mug' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
use_wandb=True

0.863: # supervision in camera space, random R,laptop, add t, Chamfer L1
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.863' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True t_method_type=-1 \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.8631: # use R_i0
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8631' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_wandb=True

0.8631r: # supervision in camera space, bottle, but add T estimation, use fps sampling,
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8631r' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
use_wandb=True

0.8632r: # 1corrected, supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8632r' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
use_wandb=True

0.8632r1: # ca224 1corrected, supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8632r1' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 use_objective_P=True p_method_type=0 \
use_wandb=True

0.8632r2: # ca224  1corrected, supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8632r2' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 use_objective_P=True p_method_type=1 \
use_wandb=True

0.8633: # type 1, no anchors, dense
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8633' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True t_method_type=1 \
use_wandb=True

0.8634: # type 1, no anchors, dense, only projection loss
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8634' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True t_method_type=1 use_objective_P=True \
use_wandb=True

0.86341: # type -1, use chamfer L1 + projection loss
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.86341' model.model='enc_so3net' model.pooling_method='pointnet' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True t_method_type=-1 use_objective_P=True \
use_wandb=True

0.8637: # type 2, R * delta T first, only projection loss
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8637' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True t_method_type=2 use_objective_P=True \
use_wandb=True


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 0.864: bottle
0.864: # supervision in camera space, bottle, but add T estimation, use fps sampling,
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.864' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
use_wandb=True

0.8641: # 1corrected, supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8641' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
use_wandb=True

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 0.865: can
0.865: # supervision in camera space, bottle, but add T estimation, use fps sampling,
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.865' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='can' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
use_wandb=True

0.8651: # 1corrected, supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8651' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='can' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
use_wandb=True
