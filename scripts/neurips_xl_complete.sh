# >>>>>>>>>>>>>>>>>>>>>>>>>>>>> complete modelnet40 points <<<<<<<<<<<<<<<<<<<<<<<#
# airplane,
# car
# chair
# sofa
# bottle
#
0.813: # supervision in camera space, random R, with adaptive R label cls loss
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.813' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True

0.8131: # supervision in camera space, random R, with adaptive R label cls loss
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.8131' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
r_method_type=1 \
use_wandb=True

0.813a: # supervised training
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.813a' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True use_objective_M=True \
eval=True save=True
use_wandb=True

0.813a1: # supervised training
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.813a1' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True use_objective_M=True \
eval=True save=True
use_wandb=True

0.813b: # supervised training
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.813b' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True model.kpconv=True \
eval=True save=True
use_wandb=True

##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  car >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
0.851: # car, ca202
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.851' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.8511: # supervision in camera space, random R, add quaternion limitation
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.8511' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
r_method_type=1 \
use_wandb=True

0.85a: # supervised training, ca205, 0
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.85a' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True use_objective_M=True \
use_wandb=True

0.85b: # supervised training, ca227, 0
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.85b' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True model.kpconv=True \
eval=True save=True
use_wandb=True

0.85c: # supervision in camera space, random R, pointnet++, 60 modes
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan name_model=ae exp_num='0.85c' save_frequency=10 vis=True \
models=pnet2 encoder_type=pnet2plusplus_so3 use_head_assemble=True \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=0 \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.85d: # supervision in camera space, random R, kpconv, 60 heads modes
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan name_model=ae vis=True exp_num='0.85d' save_frequency=10 vis=True \
models=epn encoder_type=enc_so3net model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=0 model.kpconv=True \
eval=True save=True pre_compute_delta=True
use_wandb=True

##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> chair >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
0.85: # supervision in camera space, random R, chair, ca227
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.85' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True \
use_wandb=True

0.8581: # supervision in camera space, random R, chair, ca227
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.8581' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
r_method_type=1 \
use_wandb=True

0.858a: # supervised training
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.858a' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True use_objective_M=True \
eval=True save=True
use_wandb=True

0.858b: # supervised training
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.858b' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True model.kpconv=True \
eval=True save=True
use_wandb=True

##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> sofa >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
#>
0.859: # supervision in camera space, random R, sofa,  ca227
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.859' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='sofa' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.8591: # supervision in camera space, random R, sofa,  ca227, quaternion limit
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.8591' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='sofa' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
r_method_type=1 \
use_wandb=True

0.859a: # supervised training
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.859a' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='sofa' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True use_objective_M=True \
eval=True save=True
use_wandb=True

0.859b: # supervised training
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.859b' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='sofa' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True model.kpconv=True \
eval=True save=True
use_wandb=True

##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  bottle >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
0.856: # bottle, ca221
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.856' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True
use_pretrain=True \
use_wandb=True

0.8561: # bottle, add quaternion limitation
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.8561' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
r_method_type=1 \
use_wandb=True

0.8562: # bottle, add quaternion limitation
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.8562' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
use_fps_points=True r_method_type=1 \
use_wandb=True

0.856a: # supervised training, ca205, 0, pred_axis, eval axis
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.856a' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 model.representation='up_axis' \
augment=True MODEL.num_in_channels=1 use_objective_R=True use_objective_M=True \
use_wandb=True

0.856a1: # supervised training, ca205, 0, pred_axis, eval axis
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.856a1' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True use_objective_M=True use_axis=True \
use_wandb=True

0.856a2: # supervised training, ca205, 0, pred_axis, eval axis, same mask
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.856a2' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True use_objective_M=True use_axis=True \
use_wandb=True

0.856b: # supervised training, ca227, 0, pred_axis, eval axis
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.856b' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True model.kpconv=True \
use_wandb=True

0.856c: # supervision in camera space, random R, pointnet++, 60 modes
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan name_model=ae exp_num='0.856c' save_frequency=10 vis=True \
models=pnet2 encoder_type=pnet2plusplus_so3 use_head_assemble=True \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=0 \
use_wandb=True

0.856d: # supervision in camera space, random R, kpconv, 60 heads modes
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan name_model=ae vis=True exp_num='0.856d' save_frequency=10 vis=True \
models=epn encoder_type=enc_so3net model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=0 model.kpconv=True \
use_wandb=True

#
# ##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> laptop >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
# 0.854:# laptop, ca234
# python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
# models=epn exp_num='0.854' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='laptop' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# augment=True MODEL.num_in_channels=1 \
# eval=True save=True pre_compute_delta=True
# use_wandb=True
#
# 0.854a: # supervised training
# python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
# models=epn exp_num='0.854a' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
# datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='laptop' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# augment=True MODEL.num_in_channels=1 use_objective_R=True use_objective_M=True \
# eval=True save=True
# use_wandb=True
#
# 0.854b: # supervised training
# python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
# models=epn exp_num='0.854b' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
# datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='laptop' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# augment=True MODEL.num_in_channels=1 use_objective_R=True model.kpconv=True \
# eval=True save=True
# use_wandb=True

#
# ##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> bowl >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>.
# 0.855:# bowl, ca230
# python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
# models=epn exp_num='0.855' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bowl' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# augment=True MODEL.num_in_channels=1 \
# eval=True save=True pre_compute_delta=True
# use_pretrain=True \
# use_wandb=True
#
# 0.855a: # supervised training, ca205, 0, pred_axis, eval axis
# python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
# models=epn exp_num='0.855a' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
# datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bowl' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 model.representation='up_axis' \
# augment=True MODEL.num_in_channels=1 use_objective_R=True use_objective_M=True \
# use_wandb=True
#
# 0.855b: # supervised training, ca227, 0, pred_axis, eval axis
# python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
# models=epn exp_num='0.855b' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
# datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bowl' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# augment=True MODEL.num_in_channels=1 use_objective_R=True model.kpconv=True \
# use_wandb=True
#
# 0.855c: # supervision in camera space, random R, pointnet++, 60 modes
# python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan name_model=ae exp_num='0.855c' save_frequency=10 vis=True \
# models=pnet2 encoder_type=pnet2plusplus_so3 use_head_assemble=True \
# datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bowl' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# augment=True MODEL.num_in_channels=0 \
# use_wandb=True
#
# 0.855d: # supervision in camera space, random R, kpconv, 60 heads modes
# python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan name_model=ae vis=True exp_num='0.855d' save_frequency=10 vis=True \
# models=epn encoder_type=enc_so3net model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bowl' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# augment=True MODEL.num_in_channels=0 model.kpconv=True \
# use_wandb=True
