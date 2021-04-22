# 3.11 different categories
# airplane   bottle  cup      filelist.txt  lamp                        modelnet10_train.txt        night_stand  range_hood  table     wardrobe
# bathtub    bowl    curtain  flower_pot    laptop                      modelnet40_shape_names.txt  person       sink        tent      xbox
# bed        car     desk     glass_box     mantel                      modelnet40_test.txt         piano        sofa        toilet
# bench      chair   door     guitar        modelnet10_shape_names.txt  modelnet40_train.txt        plant        stairs      tv_stand
# bookshelf  cone    dresser  keyboard      modelnet10_test.txt         monitor                     radio        stool       vase

3.1 # encoder-decoder full training, pooled prediction, 1 mode
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae datasets=modelnet40 dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='modelnet40' name_dset='modelnet40' exp_num='3.1' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=1 MODEL.encoder_only=False \
eval=True save=True 2>&1 | tee results/eval_3.1.log
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
eval=True save=True 2>&1 | tee results/eval_3.11.log
use_wandb=True

3.12 # mode=1, category chair, pooled training
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 name_model=ae datasets=modelnet40 dataset_class=AEGraph'
$TRAIN_OBJ task='pcloud_pose' item='modelnet40' name_dset='modelnet40' target_category='chair' exp_num='3.12' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 use_objective_R=True \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=1 MODEL.encoder_only=False \
eval=True save=True 2>&1 | tee results/eval.3.12.log
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
$TRAIN_OBJ task='pcloud_completion' item='modelnet40' name_dset='modelnet40' exp_num='3.2' target_category='chair' TRAIN.train_batch=2 TRAIN.test_batch=2 \
augment=True rotation_loss_type=1 \
vis_frequency=1000 vis=True \
MODEL.down_conv.npoint='[256, 64, 32, 16]' \
MODEL.down_conv.nsamples='[[10], [16], [16], [15]]' \
MODEL.num_channels_R=1 MODEL.encoder_only=False \
use_wandb=True
eval=True save=True 2>&1 | tee results/eval_default.log
use_wandb=True


3.3 unsupervised training

3.4 GAN
