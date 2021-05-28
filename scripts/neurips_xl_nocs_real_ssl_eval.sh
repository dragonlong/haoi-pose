# >>>>>>>>>>>>>>>>>>>>>>>>>> partial dataset NOCS su training <<<<<<<<<<<<<<<<<<<<<<<<<<<<
# 'camera': 0.27, 'laptop': 0.5, 'mug': 0.21,  'bottle': 0.5, 'bowl': 0.25, 'can': 0.2, }

1.1_ssl: # su R T, scaled shape Z, camera, ca218 0
python train_aegan.py task='partial_pcloud_pose_completion_ssl' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='1.1_ssl' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_newer item=nocs_newer name_dset=nocs_real target_category='camera' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 r_method_type=1 \
use_fps_points=True \
eval=True save=True pre_compute_delta=True
use_wandb=True

# 1.2: # su R T, scaled shape Z, laptop, ca213 1
python train_aegan.py task='partial_pcloud_pose_completion_ssl' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='1.2_ssl' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_newer item=nocs_newer name_dset=nocs_real target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 r_method_type=1 \
use_fps_points=True \
eval=True save=True pre_compute_delta=True
use_wandb=True

1.3_ssl: # su R T, scaled shape Z, mug, ca218 1
python train_aegan.py task='partial_pcloud_pose_completion_ssl' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='1.3_ssl' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_newer item=nocs_newer name_dset=nocs_real target_category='mug' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 r_method_type=1 \
use_fps_points=True \
eval=True save=True pre_compute_delta=True
use_wandb=True

1.4_ssl: # su R T, scaled shape Z, camera, ca210 0
python train_aegan.py task='partial_pcloud_pose_completion_ssl' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='1.4_ssl' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_newer item=nocs_newer name_dset=nocs_real target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 r_method_type=1 \
use_fps_points=True use_axis=True \
eval=True save=True pre_compute_delta=True
use_wandb=True

1.5_ssl: # su R T, scaled shape Z, bowl, ca210 1
python train_aegan.py task='partial_pcloud_pose_completion_ssl' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='1.5_ssl' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_newer item=nocs_newer name_dset=nocs_real target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 r_method_type=1 \
use_fps_points=True use_axis=True \
eval=True save=True pre_compute_delta=True
use_wandb=True

1.6_ssl: # su R T, scaled shape Z, can
python train_aegan.py task='partial_pcloud_pose_completion_ssl' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='1.6_ssl' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_newer item=nocs_newer name_dset=nocs_real target_category='can' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 r_method_type=1 \
use_fps_points=True use_axis=True \
eval=True save=True pre_compute_delta=True
use_wandb=True
