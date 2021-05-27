# >>>>>>>>>>>>>>>>>>>>>>>>>> partial dataset NOCS su training <<<<<<<<<<<<<<<<<<<<<<<<<<<<
# 'camera': 0.27, 'laptop': 0.5, 'mug': 0.21,  'bottle': 0.5, 'bowl': 0.25, 'can': 0.2, }

1.1_finetune: # su R T, scaled shape Z, camera, ca218 0
python train_aegan.py task='partial_pcloud_pose_completion_finetune' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='1.1_finetune' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_newer item=nocs_newer name_dset=nocs_real target_category='camera' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 r_method_type=1 \
use_objective_M=True use_objective_R=True use_objective_T=True use_objective_canon=True \
use_fps_points=True \
use_pretrain=True pretrained_path='/groups/CESCA-CV/ICML2021/model/nocs_newer/1.1_su_nocs/checkpoints' \
use_wandb=True

# 1.2: # su R T, scaled shape Z, laptop, ca213 1
# python train_aegan.py task='partial_pcloud_pose_completion_finetune' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='1.2_finetune' model.model='enc_so3net' model.pooling_method='max' \
# datasets=nocs_newer item=nocs_newer name_dset=nocs_real target_category='laptop' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 r_method_type=1 \
# use_objective_M=True use_objective_R=True use_objective_T=True use_objective_canon=True \
# use_fps_points=True \
# use_wandb=True

1.3_finetune: # su R T, scaled shape Z, mug, ca218 1
python train_aegan.py task='partial_pcloud_pose_completion_finetune' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='1.3_finetune' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_newer item=nocs_newer name_dset=nocs_real target_category='mug' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 r_method_type=1 \
use_objective_M=True use_objective_R=True use_objective_T=True use_objective_canon=True \
use_fps_points=True \
use_pretrain=True pretrained_path='/groups/CESCA-CV/ICML2021/model/nocs_newer/1.3_su_nocs/checkpoints' \
use_wandb=True

1.4_finetune: # su R T, scaled shape Z, camera, ca210 0
python train_aegan.py task='partial_pcloud_pose_completion_finetune' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='1.4_finetune' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_newer item=nocs_newer name_dset=nocs_real target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 r_method_type=1 \
use_objective_M=True use_objective_R=True use_objective_T=True use_objective_canon=True \
use_fps_points=True use_axis=True \
use_pretrain=True pretrained_path='/groups/CESCA-CV/ICML2021/model/nocs_newer/1.4_su_nocs/checkpoints' \
use_wandb=True

1.5_finetune: # su R T, scaled shape Z, bowl, ca210 1
python train_aegan.py task='partial_pcloud_pose_completion_finetune' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='1.5_finetune' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_newer item=nocs_newer name_dset=nocs_real target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 r_method_type=1 \
use_objective_M=True use_objective_R=True use_objective_T=True use_objective_canon=True \
use_fps_points=True use_axis=True \
use_pretrain=True pretrained_path='/groups/CESCA-CV/ICML2021/model/nocs_newer/1.5_su_nocs/checkpoints' \
use_wandb=True

1.6_finetune: # su R T, scaled shape Z, can
python train_aegan.py task='partial_pcloud_pose_completion_finetune' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='1.6_finetune' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_newer item=nocs_newer name_dset=nocs_real target_category='can' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 r_method_type=1 \
use_objective_M=True use_objective_R=True use_objective_T=True use_objective_canon=True \
use_fps_points=True use_axis=True \
use_pretrain=True pretrained_path='/groups/CESCA-CV/ICML2021/model/nocs_newer/1.6_su_nocs/checkpoints' \
use_wandb=True
