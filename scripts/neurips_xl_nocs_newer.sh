# >>>>>>>>>>>>>>>>>>>>>>>>>> partial dataset NOCS su training <<<<<<<<<<<<<<<<<<<<<<<<<<<<
# 'camera': 0.27, 'laptop': 0.5, 'mug': 0.21,  'bottle': 0.5, 'bowl': 0.25, 'can': 0.2, }

1.1: # su R T, scaled shape Z, camera, ca219 0
python train_aegan.py task='partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='1.1_su_nocs' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_newer item=nocs_newer name_dset=nocs_newer target_category='camera' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 r_method_type=1 \
use_objective_M=True use_objective_R=True use_objective_T=True use_objective_canon=True \
use_fps_points=True \
use_wandb=True

1.2: # su R T, scaled shape Z, laptop, ca213 1
python train_aegan.py task='partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='1.2_su_nocs' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_newer item=nocs_newer name_dset=nocs_newer target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 r_method_type=1 \
use_objective_M=True use_objective_R=True use_objective_T=True use_objective_canon=True \
use_fps_points=True \
use_wandb=True

1.3: # su R T, scaled shape Z, mug
python train_aegan.py task='partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='1.3_su_nocs' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_newer item=nocs_newer name_dset=nocs_newer target_category='mug' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 r_method_type=1 \
use_objective_M=True use_objective_R=True use_objective_T=True use_objective_canon=True \
use_fps_points=True use_pretrain=True \
use_wandb=True

1.4: # su R T, scaled shape Z, camera, ca213 0
python train_aegan.py task='partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='1.4_su_nocs' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_newer item=nocs_newer name_dset=nocs_newer target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 r_method_type=1 \
use_objective_M=True use_objective_R=True use_objective_T=True use_objective_canon=True \
use_fps_points=True use_axis=True \
use_wandb=True

1.5: # su R T, scaled shape Z, laptop
python train_aegan.py task='partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='1.5_su_nocs' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_newer item=nocs_newer name_dset=nocs_newer target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 r_method_type=1 \
use_objective_M=True use_objective_R=True use_objective_T=True use_objective_canon=True \
use_fps_points=True use_axis=True \
use_wandb=True

1.6: # su R T, scaled shape Z, mug
python train_aegan.py task='partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='1.6_su_nocs' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_newer item=nocs_newer name_dset=nocs_newer target_category='can' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 r_method_type=1 \
use_objective_M=True use_objective_R=True use_objective_T=True use_objective_canon=True \
use_fps_points=True use_axis=True \
use_wandb=True
