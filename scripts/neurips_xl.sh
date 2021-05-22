# >>>>>>>>>>>>>>>>>>>>>>>>>>>>> complete modelnet40 points <<<<<<<<<<<<<<<<<<<<<<<#
# airplane,
# car
# bottle
# bowl
0.813(work!): # supervision in camera space, random R, with adaptive R label cls loss
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.813' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_M=True pre_compute_delta=True \
eval=True

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

0.85: # supervision in camera space, random R, chair, ca227
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.85' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True \
use_pretrain=True
use_wandb=True

torch.matmul(rot1[0], torch.Tensor([[0, 0, 1]]).permute(1, 0).cuda())

0.85a: # supervised training, ca205, 0
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.85a' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True use_objective_M=True \
eval=True save=True
use_wandb=True

0.85b: # supervised training, ca227, 0
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.85b' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True model.kpconv=True \
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


0.854:# laptop, ca234
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.854' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True
use_pretrain=True \
use_wandb=True

0.854a: # supervised training
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.854a' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True use_objective_M=True \
eval=True save=True
use_wandb=True

0.854b: # supervised training
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.854b' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True model.kpconv=True \
eval=True save=True
use_wandb=True

0.855:# bowl, ca230
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.855' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True
use_pretrain=True \
use_wandb=True

0.855a: # supervised training, ca205, 0, pred_axis, eval axis
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.855a' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 model.representation='up_axis' \
augment=True MODEL.num_in_channels=1 use_objective_R=True use_objective_M=True \
use_wandb=True

0.855b: # supervised training, ca227, 0, pred_axis, eval axis
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.855b' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True model.kpconv=True \
use_wandb=True

0.855c: # supervision in camera space, random R, pointnet++, 60 modes
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan name_model=ae exp_num='0.855c' save_frequency=10 vis=True \
models=pnet2 encoder_type=pnet2plusplus_so3 use_head_assemble=True \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=0 \
use_wandb=True

0.855d: # supervision in camera space, random R, kpconv, 60 heads modes
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan name_model=ae vis=True exp_num='0.855d' save_frequency=10 vis=True \
models=epn encoder_type=enc_so3net model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=0 model.kpconv=True \
use_wandb=True


0.856: # bottle, ca221
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.856' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True
use_pretrain=True \
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

0.858--->
0.85: # supervision in camera space, random R, chair, ca227
python train_aegan.py task='pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.85' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
eval=True save=True pre_compute_delta=True \
use_pretrain=True
use_wandb=True

0.858a: # supervised training
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.858a' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True use_objective_M=True \
use_wandb=True

0.858b: # supervised training
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.858b' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='chair' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True model.kpconv=True \
use_wandb=True

0.859: # supervision in camera space, random R, sofa,  ca227
python train_aegan.py task='ssl_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True nr_epochs=1000 save_frequency=20 \
models=epn exp_num='0.859' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='sofa' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 \
use_wandb=True

0.859a: # supervised training
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.859a' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='sofa' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True use_objective_M=True \
use_wandb=True

0.859b: # supervised training
python train_aegan.py task='pcloud_pose' training=ae_gan name_model=ae vis=True save_frequency=10 nr_epochs=500 \
models=epn exp_num='0.859b' model.model='enc_so3net' encoder_type=enc_so3net model.pooling_method='max' \
datasets=modelnet40aligned item=modelnet40aligned name_dset=modelnet40aligned target_category='sofa' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 use_objective_R=True model.kpconv=True \
use_wandb=True



#>>>>>>>>>>>>>>>>>>>>>>>>>>>> partial ModelNet40 dataset <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#
# #
# 0.847: # supervision in camera space, random R, airplane, but add T estimation, dense per-point voting, R0
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.847' model.model='enc_so3net' model.pooling_method='max' \
# datasets=shapenetaligned item=shapenetaligned name_dset=shapenetaligned target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# augment=True MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# eval=True save=True pre_compute_delta=True
# use_wandb=True

# 0.9
0.91: # supervision in camera space, random R, airplane, but add T estimation, dense per-point voting, R0
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.91' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_wandb=True
eval=True save=True pre_compute_delta=True

0.911: # supervision in camera space, random R, airplane, but add T estimation, dense per-point voting, R0, project poinst into depth
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.911' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 use_objective_P=True \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.912: # supervision in camera space, airplane, but add T estimation, random SO3 rotate points
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.912' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
eval=True save=True pre_compute_delta=True \
use_wandb=True

0.913: # supervision in camera space, airplane, but add T estimation, use fps sampling
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.913' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
use_wandb=True
eval=True save=True pre_compute_delta=True

0.913r: # supervision in camera space, airplane, but add T estimation, use fps sampling,
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.913r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
eval=True save=True pre_compute_delta=True
use_wandb=True


0.9131: # supervision in camera space, airplane, but add T estimation, use fps sampling, correct points prediction
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.9131' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=-1 \
use_wandb=True

0.914: # supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.914' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
use_wandb=True

0.9141: # supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.9141' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
use_wandb=True

0.9141r: # corrected, supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.9141r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.915: # supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, a smaller angle constr
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.915' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
use_wandb=True

0.916: # supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, use 0.1 * projection loss
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.916' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 p_method_type=0 use_objective_P=True \
use_wandb=True

0.9161: # supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, use 0.01 * projection_loss
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.9161' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 p_method_type=0 use_objective_P=True \
use_wandb=True

0.9162: # supervision in camera space, random R, airplane, but add T estimation,  and 0.01 projection loss
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.9162r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=0 p_method_type=0 use_objective_P=True \
use_wandb=True

0.91a: # random R, airplane, but add T estimation, dense per-point voting, R0
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.91a' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
eval=True
use_wandb=True

0.91b: # random R, airplane, but add T estimation, dense per-point voting, R0
python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.91b' model.model='enc_so3net' model.pooling_method='max' model.kpconv=True \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_objective_R=True use_objective_M=True use_objective_T=True \
eval=True save=True
use_wandb=True


# >>>> 0.92, car
0.92: # supervision in camera space, random R, airplane, but add T estimation, R0
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.92' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
eval=True save=True pre_compute_delta=True
use_wandb=True

0.92r: # supervision in camera space, airplane, but add T estimation, use fps sampling,
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.92r' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
use_wandb=True

0.921: # supervision in camera space, random R, airplane, but add T estimation, dense per-point voting, R0, ca201 0
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.921' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='car' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 use_objective_P=True \
eval=True save=True pre_compute_delta=True
use_wandb=True
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


# >>>> 0.93, ca203 1
0.93: # supervision in camera space, random R, airplane, but add T estimation, dense per-point voting, R0
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.93' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_wandb=True
#
# 0.911: # supervision in camera space, random R, airplane, but add T estimation, dense per-point voting, R0
# python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.911' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 use_objective_P=True \
# use_wandb=True

# 0.91a: # random R, airplane, but add T estimation, dense per-point voting, R0
# python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.91a' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_objective_R=True use_objective_M=True use_objective_T=True \
# use_wandb=True

# 0.91b: # random R, airplane, but add T estimation, dense per-point voting, R0
# python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.91b' model.model='enc_so3net' model.pooling_method='max' model.kpconv=True \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_objective_R=True use_objective_M=True use_objective_T=True \
# use_wandb=True

# >>>> 0.94, ca207 1, bottle
0.94: # supervision in camera space, random R, airplane, but add T estimation, dense per-point voting, R0
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.94' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_wandb=True

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

0.941: # supervision in camera space, random R, airplane, but add T estimation, random SO3 rotate points
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.941' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
augment=True MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_pretrain=True \
use_wandb=True

0.942: # supervision in camera space, random R, airplane, but add T estimation, random SO3 rotate points, also use fps to get regularly sampled points
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.942' model.model='enc_so3net' model.pooling_method='max' \
datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='bottle' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True \
use_wandb=True
# 0.91a: # random R, airplane, but add T estimation, dense per-point voting, R0
# python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.91a' model.model='enc_so3net' model.pooling_method='max' \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_objective_R=True use_objective_M=True use_objective_T=True \
# use_wandb=True
#
# 0.91b: # random R, airplane, but add T estimation, dense per-point voting, R0
# python train_aegan.py task='partial_pcloud_pose' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
# models=epn exp_num='0.91b' model.model='enc_so3net' model.pooling_method='max' model.kpconv=True \
# datasets=modelnet40new item=modelnet40new name_dset=modelnet40new target_category='airplane' dataset_class=AE \
# TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
# MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
# use_objective_R=True use_objective_M=True use_objective_T=True \
# use_wandb=True
#>>>>>>>>>>>>>>>>>>>>>>>>>>>> partial NOCS dataset <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<#

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

0.8611r: # ca217, 1corrected, supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8611r' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='bowl' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
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

0.8632r: # ca217, 1corrected, supervision in camera space, random R, airplane, but add T estimation, use fps sampling, use quaternion activation, 36, 0.001
python train_aegan.py task='ssl_partial_pcloud_pose_completion' training=ae_gan encoder_type=enc_so3net name_model=ae vis=True save_frequency=5 \
models=epn exp_num='0.8632r' model.model='enc_so3net' model.pooling_method='max' \
datasets=nocs_synthetic item=nocs_synthetic name_dset=nocs_synthetic target_category='laptop' dataset_class=AE \
TRAIN.train_batch=4 TRAIN.test_batch=4 num_points=1024 model.input_num=1024 \
MODEL.num_in_channels=1 pred_t=True t_method_type=0 \
use_fps_points=True r_method_type=1 \
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
