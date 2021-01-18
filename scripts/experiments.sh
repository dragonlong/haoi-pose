export PATH="/home/lxiaol9/anaconda3/bin:$PATH"
cd
. scripts/ai_power1.sh
module load cuda/10.0.130
source activate merl

screen -X -S  quit

0.1: # same model, but with new loss
  python train.py verbose=True gpu=0 name_model='pointnet2_meteornet' exp_num='0.1' TRAIN.loss='xentropy'

0.2: # new model, new loss
  python train.py verbose=True gpu=1 name_model='pointnet2_kaolin' exp_num='0.2' TRAIN.loss='xentropy' MODEL.arch_decoder='kaolin' 2>&1 | tee train_0.2.log

0.3: # new mode, but with same loss, ca197
  python train.py verbose=True gpu=0 name_model='pointnet2_kaolin' exp_num='0.3' TRAIN.loss='miou' MODEL.arch_decoder='kaolin' 2>&1 | tee train_0.3.log
  0.31: # with full unitvec loss
  python train.py verbose=True gpu=0 name_model='pointnet2_kaolin' exp_num='0.31' TRAIN.loss='miou' MODEL.arch_decoder='kaolin' 2>&1 | tee outputs/train_0.31.log
  python post_hand.py --hand --exp_num=0.31 2>&1 | tee eval_0.31.log &
  python post_hand.py --hand --exp_num=0.31 2>&1 | tee eval_0.31_regression.log &
  python post_hand.py --hand --exp_num=0.31 --domain='unseen' 2>&1 | tee eval_0.31_regression.log &

  0.32: # with normal unitvec loss
  python train.py verbose=True gpu=0 name_model='pointnet2_kaolin' exp_num='0.32' TRAIN.loss='miou' MODEL.arch_decoder='kaolin' 2>&1 | tee outputs/train_0.32.log
  python post_hand.py --hand --exp_num=0.32 2>&1 | tee eval_0.32.log
  python post_hand.py --hand --exp_num=0.32 2>&1 | tee eval_0.32_regression.log &
  python post_hand.py --hand --exp_num=0.32 --domain='unseen' 2>&1 | tee eval_0.32_unseen_regression.log &

  0.33: # with normal unitvec loss, with svd rotation and mean
  python train.py verbose=True gpu=0 name_model='pointnet2_kaolin' exp_num='0.33' TRAIN.loss='miou' MODEL.arch_decoder='kaolin' rot_align=True 2>&1 | tee outputs/train_0.33.log
  python post_hand.py --hand --exp_num=0.33 2>&1 | tee eval_0.33.log &
  python post_hand.py --hand --exp_num=0.33 2>&1 | tee eval_0.33_regression.log &
  python post_hand.py --hand --exp_num=0.33 --domain='unseen' 2>&1 | tee eval_0.33_unseen_regression.log &

0.4: # mano regression, 61 params, shape 10, articulation 45, global 6, regression is easy: but recontruction loss + mano_layer
  python train.py verbose=True gpu=1 name_model='pointnet2_kaolin' exp_num='0.4' MODEL.arch_decoder='kaolin' pred_mano=True 2>&1 | tee train_0.4.log
  python post_hand.py --hand --exp_num=0.4 --mano 2>&1 | tee eval_0.4_mano.log &

# 0.5: # mano regression, 61 params, shape 10, articulation 45, global 6, regression is easy: but recontruction loss + mano_layer
#   python train.py verbose=True gpu=0 name_model='pointnet2_kaolin' exp_num='0.5' MODEL.arch_decoder='kaolin' pred_mano=True 2>&1 | tee train_0.5.log
#   python post_hand.py --hand --exp_num=0.5 --mano 2>&1 | tee eval_0.5_mano.log &
#
# 0.6: only train with regressionR, regressionPose, partcls loss
#   python train.py verbose=True gpu=0 name_model='pointnet2_kaolin' exp_num='0.6' MODEL.arch_decoder='kaolin' pred_mano=True 2>&1 | tee train_0.6.log
#   python post_hand.py --hand --exp_num=0.6 --mano 2>&1 | tee eval_0.6_mano.log &

# 0.7: # only train with regressionR, regressionPose, partcls loss, NOCS loss--> 6D vector directly,
#   python train.py verbose=True gpu=1 name_model='pointnet2_kaolin' exp_num='0.7' MODEL.arch_decoder='kaolin' pred_mano=True 2>&1 | tee train_0.7.log
#   python post_hand.py --hand --exp_num=0.7 --mano 2>&1 | tee eval_0.7_mano.log &

0.8: # only train with regressionR, regressionPose, partcls loss, NOCS loss--> 6D vector directly, add vertices, joints loss
  python train.py verbose=True gpu=1 name_model='pointnet2_kaolin' exp_num='0.8' MODEL.arch_decoder='kaolin' pred_mano=True 2>&1 | tee train_0.8.log
  python post_hand.py --hand --exp_num=0.8 --mano 2>&1 | tee eval_0.8_mano.log &
  python pred_check.py --exp_num=0.8 --hand --mano

0.9: # retrain with larger learning rate, 0.1-0.3
python train.py verbose=True gpu=1 name_model='votenet' exp_num='0.9' MODEL.arch_decoder='votenet' pred_mano=False 2>&1 | tee outputs/train_0.9.log
python post_hand.py --hand --exp_num=0.9 --mano 2>&1 | tee eval_0.8_mano.log &

0.91: # train with normal setting, 0.3, 0.6
python train.py verbose=True gpu=1 name_model='votenet' exp_num='0.91' MODEL.arch_decoder='votenet' pred_mano=False 2>&1 | tee outputs/train_0.91.log

# 0.92: # train with normal setting, 0.2, 0.4
# python train.py verbose=True gpu=1 name_model='votenet' exp_num='0.92' MODEL.arch_decoder='votenet' pred_mano=False 2>&1 | tee outputs/train_0.92.log
#
# 0.93: # train with normal setting, 0.2, 0.3
# python train.py verbose=True gpu=1 name_model='votenet' exp_num='0.93' MODEL.arch_decoder='votenet' pred_mano=False 2>&1 | tee outputs/train_0.93.log

0.94: # retrain with larger learning rate, 0.1-0.2, add center + confidence prediction after 1st epoch
python train.py verbose=True gpu=1 name_model='votenet' exp_num='0.94' MODEL.arch_decoder='votenet' pred_mano=False 2>&1 | tee outputs/train_0.94.log
python pred_check.py --exp_num=0.94 --hand --mano --contact --domain=seen
python pred_check.py --exp_num=0.94 --hand --mano --contact --domain=unseen

#
0.95: # retrain with larger learning rate, 0.1-0.2, center + confidence prediction after 1st epoch, confidence on every prediction points, sigmoid
python train.py verbose=True gpu=1 name_model='votenet' exp_num='0.95' MODEL.arch_decoder='votenet' pred_mano=False 2>&1 | tee outputs/train_0.95.log

1.0: # regressionR(6D), regressionT, partcls loss, NOCS loss, hand vertices, joints loss
  python train.py training=hand gpu=0 name_model='pointnet2_kaolin' exp_num='1.0' MODEL.arch_decoder='kaolin' pred_mano=True hand_only=True 2>&1 | tee outputs/train_1.0.log
  python post_hand.py --hand --exp_num=0.8 --mano 2>&1 | tee eval_0.8_mano.log &
  python pred_check.py --exp_num=0.8 --hand --mano

1.1: # regressionR(6D), regressionT, partcls loss, NOCS loss, hand vertices, joints loss, using L1 loss
  python train.py training=hand gpu=0 name_model='pointnet2_kaolin' exp_num='1.1' MODEL.arch_decoder='kaolin' pred_mano=True hand_only=True 2>&1 | tee outputs/train_1.1.log
  python post_hand.py --hand --exp_num=0.8 --mano 2>&1 | tee eval_1.1_mano.log &
  python pred_check.py --exp_num=0.8 --hand --mano 2>&1 | tee

1.2: # regressionR(6D), regressionT, partcls loss, NOCS loss, hand vertices, joints loss, using L1 loss, translation put into Mano(tonight)
  python train.py training=hand gpu=0 name_model='pointnet2_kaolin' exp_num='1.2' MODEL.arch_decoder='kaolin' pred_mano=True hand_only=True 2>&1 | tee outputs/train_1.2.log
  python post_hand.py --hand --exp_num=0.8 --mano 2>&1 | tee eval_0.8_mano.log &
  python pred_check.py --exp_num=0.8 --hand --mano

#>>>>>>>>>>>>>>>>>>>>>>>>>>> on Obman Dataset
1.3: # object pose estimation on obman
  python train.py gpu=0 name_model='pointnet2_kaolin' item='obman' name_dset='obman' exp_num='1.3' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 MODEL.arch_decoder='kaolin' 2>&1 | tee outputs/train_1.3.log
  python train.py gpu=1 name_model='pointnet2_kaolin' item='obman' name_dset='obman' exp_num='1.31' MODEL.arch_decoder='kaolin' 2>&1 | tee outputs/train_1.31.log
  # bowl, 02880940
  python train.py gpu=1 name_model='pointnet2_kaolin' item='obman' name_dset='obman' exp_num='1.32' MODEL.arch_decoder='kaolin' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 2>&1 | tee outputs/train_1.32.log
  # camera: 02942699
  python train.py gpu=1 name_model='pointnet2_kaolin' item='obman' name_dset='obman' exp_num='1.33' MODEL.arch_decoder='kaolin' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 2>&1 | tee outputs/train_1.33.log
  # jar: 03593526
  python train.py gpu=1 name_model='pointnet2_kaolin' item='obman' name_dset='obman' exp_num='1.34' MODEL.arch_decoder='kaolin' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 2>&1 | tee outputs/train_1.34.log
  # preds
  python pred_check.py --exp_num=1.3 --item='obman' --viz --domain='unseen'

  # camera: 02942699
  python train.py gpu=1 name_model='pointnet++' item='obman' name_dset='obman' target_category='camera' exp_num='1.35' MODEL.arch_decoder='pointnet2_single' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 2>&1 | tee outputs/train_1.35.log

1.4: # pose on obman dataset, mixed or per-category
  python train.py gpu=0 name_model='pointnet2_kaolin' item='obman' name_dset='obman' exp_num='1.4' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 MODEL.arch_decoder='kaolin' 2>&1 | tee outputs/train_1.4.log
  python train.py gpu=1 name_model='pointnet2_kaolin' item='obman' name_dset='obman' exp_num='1.41' MODEL.arch_decoder='kaolin' target_category='camera' 2>&1 | tee outputs/train_1.41.log
  python pred_check.py --exp_num=1.4 --item='obman' --viz --domain='unseen'

  # 1.42, debug training
  python train.py gpu=0 name_model='pointnet2_kaolin' item='obman' name_dset='obman' exp_num='1.42' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 MODEL.arch_decoder='kaolin' 2>&1 | tee outputs/train_1.42.log
  python train.py gpu=0 name_model='pointnet2_kaolin' item='obman' name_dset='obman' exp_num='1.43' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.001 MODEL.arch_decoder='kaolin' 2>&1 | tee outputs/train_1.43.log
  python train.py gpu=1 name_model='pointnet2_kaolin' item='obman' name_dset='obman' exp_num='1.43' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.0=01 MODEL.arch_decoder='kaolin' target_category='camera' 2>&1 | tee outputs/train_1.44.log
1.5: # tensorflow-based training on obman dataset
  python train_obman.py item='obman' name_dset='obman' exp_num='1.5' 2>&1 | tee outputs/train_1.5.log

2.01: # use noisy nocs;
  python train_obman.py item='obman' name_dset='obman' exp_num='2.01' 2>&1 | tee outputs/occ_train_2.01.log

2.02: # no noise, oracle, with multiple category code concatenation
  python train_obman.py item='obman' name_dset='obman' exp_num='2.02' use_category_code=True 2>&1 | tee outputs/occ_train_2.02.log
  python generate_co.py generation.copy_input=True name_dset='obman' exp_num='2.02' use_category_code=True
  EXP=2.02
  python eval_objs.py eval=True item='obman' name_dset='obman' exp_num=${EXP} use_category_code=True 2>&1 | tee outputs/occ_eval_objs_${EXP}.log

2.03: # complete NOCS, done!
  python train_obman.py item='obman' name_dset='obman' exp_num='2.03' oracle_nocs=True 2>&1 | tee outputs/occ_train_2.03.log1
  python generate_co.py generation.copy_input=True name_dset='obman' exp_num='2.03' oracle_nocs=True 2>&1 | tee outputs/occ_gen_2.03.log
  EXP=2.03
  python eval_objs.py eval=True item='obman' name_dset='obman' exp_num=${EXP} oracle_nocs=True 2>&1 | tee outputs/occ_eval_objs_${EXP}.log

2.04: # complete NOCS, with multiple category code concatenation, done!
  python train_obman.py item='obman' name_dset='obman' exp_num='2.04' oracle_nocs=True use_category_code=True 2>&1 | tee outputs/occ_train_2.04.log

2.05: # partial NOCS, with multiple category code concatenation, add hand points for NOCS reconstruction training, needs additional segmentation map as normals?
  python train_obman.py item='obman' name_dset='obman' exp_num='2.05' use_transform_hand=True data.dim=4 use_category_code=True 2>&1 | tee outputs/occ_train_2.05.log
  python generate_co.py generation.copy_input=True name_dset='obman' exp_num='2.05'  use_transform_hand=True data.dim=4 use_category_code=True

2.06: # partial NOCS,but only train on one category: like bottle
  python train_obman.py item='obman' name_dset='obman' exp_num='2.06' target_category='bottle' use_category_code=False 2>&1 | tee outputs/occ_train_2.06.log
  python generate_co.py generation.copy_input=True item='obman' name_dset='obman' exp_num='2.06' target_category='bottle' use_category_code=False
  EXP=2.06
  python eval_objs.py eval=True item='obman' name_dset='obman' target_category='bottle' exp_num=${EXP} 2>&1 | tee outputs/occ_eval_objs_${EXP}.log
  # 2.07: # partial NOCS,but only train on one category: like bottle, with bigger lr 0.001
  #   python train_obman.py item='obman' name_dset='obman' exp_num='2.07' target_category='bottle' learning_rate=0.001 use_category_code=False 2>&1 | tee outputs/occ_train_2.07.log
  #   python generate_co.py generation.copy_input=True name_dset='obman' exp_num='2.07' use_category_code=True

2.08: # occupancy of both hand and objects, input hand pts which are normalized, 0.1 hand loss
python train_obman.py item='obman' name_dset='obman' exp_num='2.08' target_category='bottle' use_transform_hand=True data.dim=4 use_hand_occupancy=True use_category_code=False 2>&1 | tee outputs/occ_train_2.08.log

2.09: # occupancy of both hand and objects, input hand pts which are normalized, 0.05 hand loss
python train_obman.py item='obman' name_dset='obman' exp_num='2.09' target_category='bottle' use_wandb=False use_transform_hand=True data.dim=4 use_hand_occupancy=True use_category_code=False 2>&1 | tee outputs/occ_train_2.09.log

2.1: # conv-occupancy networks training, use transformed depth images, full occupancy;
  python train_obman.py item='obman' name_dset='obman' exp_num='2.1' use_noisy_nocs=False 2>&1 | tee outputs/occ_train_2.1.log
  EXP=2.1
  python generate_co.py generation.copy_input=True item='obman' name_dset='obman' exp_num=${EXP} use_noisy_nocs=False 2>&1 | tee outputs/occ_eval_${EXP}.log #
2.2: #
  python train_obman.py training.out_dir='outputs/pointcloud/test' 2>&1 | tee outputs/occ_train_2.2.log
# 1.1:
#   python train.py verbose=True gpu=1 name_model='pointnet2_charlesssg' ca218
