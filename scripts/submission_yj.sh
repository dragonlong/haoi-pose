yj_ycb_0.001 # R, t; train with all categories, no noise
python train_aegan.py \
			 task='ssl_partial_pcloud_pose_completion_ycb' \
			 training=ae_gan \
       models=epn \
       encoder_type=enc_so3net \
       name_model=ae
       model.model='enc_so3net' \
			 model.pooling_method='max' \
			 datasets=ycb \
			 item=ycb \
			 name_dset=ycb \
			 instance=None \
			 dataset_class=AE \
			 TRAIN.train_batch=4 TRAIN.test_batch=4 \
			 num_points=1024 model.input_num=1024 \
			 DATASET.add_noise=False \
       augment=True MODEL.num_in_channels=1 pred_t=True \
			 vis=True save_frequency=5 exp_num='yj_ycb_0.001' \
       use_wandb=True

yj_ycb_0.002 # R, t; train with all categories, with noise
python train_aegan.py \
			 task='ssl_partial_pcloud_pose_completion_ycb' \
			 training=ae_gan \
       models=epn \
       encoder_type=enc_so3net \
       name_model=ae
       model.model='enc_so3net' \
			 model.pooling_method='max' \
			 datasets=ycb \
			 item=ycb \
			 name_dset=ycb \
			 instance=None \
			 dataset_class=AE \
			 TRAIN.train_batch=4 TRAIN.test_batch=4 \
			 num_points=1024 model.input_num=1024 \
			 DATASET.add_noise=True \
       augment=True MODEL.num_in_channels=1 pred_t=True \
			 vis=True save_frequency=5 exp_num='yj_ycb_0.002' \
       use_wandb=True

yj_ycb_0.003 # R, t; train with all categories, no noise
python train_aegan.py \
			 task='ssl_partial_pcloud_pose_completion_ycb' \
			 training=ae_gan \
       models=epn \
       encoder_type=enc_so3net \
       name_model=ae
       model.model='enc_so3net' \
			 model.pooling_method='max' \
			 datasets=ycb \
			 item=ycb \
			 name_dset=ycb \
			 instance=1 \
			 dataset_class=AE \
			 TRAIN.train_batch=4 TRAIN.test_batch=4 \
			 num_points=1024 model.input_num=1024 \
			 DATASET.add_noise=False \
       augment=True MODEL.num_in_channels=1 pred_t=True \
			 vis=True save_frequency=5 exp_num='yj_ycb_0.003' \
       use_wandb=True

yj_ycb_0.004 # R, t; train with all categories, with noise
python train_aegan.py \
			 task='ssl_partial_pcloud_pose_completion_ycb' \
			 training=ae_gan \
       models=epn \
       encoder_type=enc_so3net \
       name_model=ae
       model.model='enc_so3net' \
			 model.pooling_method='max' \
			 datasets=ycb \
			 item=ycb \
			 name_dset=ycb \
			 instance=1 \
			 dataset_class=AE \
			 TRAIN.train_batch=4 TRAIN.test_batch=4 \
			 num_points=1024 model.input_num=1024 \
			 DATASET.add_noise=True \
       augment=True MODEL.num_in_channels=1 pred_t=True \
			 vis=True save_frequency=5 exp_num='yj_ycb_0.004' \
       use_wandb=True











