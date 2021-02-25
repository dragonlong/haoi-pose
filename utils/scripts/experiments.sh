export PATH="/home/lxiaol9/anaconda3/bin:$PATH"
cd
. scripts/ai_power1.sh
module load cuda/10.1.168
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

  # # 1.42, add shared layers + dp, 0.01, mixed training
  # python train.py gpu=0 name_model='pointnet2_kaolin' item='obman' name_dset='obman' exp_num='1.42' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 MODEL.arch_decoder='kaolin' 2>&1 | tee outputs/train_1.42.log
  # # 1.43, add shared layers + dp 0.001, mixed training
  # python train.py gpu=0 name_model='pointnet2_kaolin' item='obman' name_dset='obman' exp_num='1.43' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.001 MODEL.arch_decoder='kaolin' 2>&1 | tee outputs/train_1.43.log
  # # 1.44, add shared layers + dp 0.01, no symmetry, only on categories without symmetry
  # python train.py gpu=1 name_model='pointnet2_kaolin' item='obman' name_dset='obman' exp_num='1.44' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 MODEL.arch_decoder='kaolin' target_category='camera' 2>&1 | tee outputs/train_1.44.log
  # 1.45 on bottle/can/bowl/jar? choose one category, then train with aligned symmetry/M=36 symmetry;
  # bottle

  # 1.45, ca212
  python train.py gpu=1 exp_num='1.45' use_pretrain=True pretrained_path='/groups/CESCA-CV/ICML2021/model/obman/1.45' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 target_category='bottle' use_1vN_nocs=True 2>&1 | tee outputs/train_1.45.log1
  python pred_check.py --exp_num=1.45 --item='obman' --domain='unseen' --target_category='bottle' --is_special --save

  # can 1.451, ca201, 0
  python train.py gpu=0 exp_num='1.451' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 target_category='can' use_1vN_nocs=True 2>&1 | tee outputs/train_1.451.log
  python train.py gpu=0 exp_num='1.451' use_pretrain=True pretrained_path='/groups/CESCA-CV/ICML2021/model/obman/1.451' TRAIN.lr_encoder=0.005 TRAIN.lr_decoder=0.005 target_category='can' use_1vN_nocs=True 2>&1 | tee outputs/train_1.451.log1
  python pred_check.py --exp_num=1.451 --item='obman' --domain='unseen' --target_category='can' --is_special --save

  # bowl 1.452
  python train.py gpu=1 exp_num='1.452' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 target_category='bowl' use_1vN_nocs=True 2>&1 | tee outputs/train_1.452.log
  python train.py gpu=1 exp_num='1.452' use_pretrain=True pretrained_path='/groups/CESCA-CV/ICML2021/model/obman/1.452' TRAIN.lr_encoder=0.005 TRAIN.lr_decoder=0.005 target_category='bowl' use_1vN_nocs=True 2>&1 | tee outputs/train_1.452.log1
  python pred_check.py --exp_num=1.452 --item='obman' --domain='unseen' --target_category='bowl' --is_special --save

  # jar 1.453 is wrong on unseen
  python train.py gpu=0 exp_num='1.453' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 target_category='jar' use_1vN_nocs=True 2>&1 | tee outputs/train_1.453.log
  python pred_check.py --exp_num=1.453 --item='obman' --domain='unseen' --target_category='jar' --save
  #  ca233, gpu 1
    python train.py gpu=0 exp_num='1.4531' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 target_category='jar' use_1vN_nocs=True 2>&1 | tee outputs/train_1.4531.log

  # 1.47, train with cellphone, M=4, symmetry, with reflection symmetry
  python train.py gpu=1 exp_num='1.47' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 target_category='cellphone' use_1vN_nocs=True 2>&1 | tee outputs/train_1.47.log
  python pred_check.py --exp_num=1.47 --item='obman' --domain='unseen'
  python pred_check.py --exp_num=1.47 --item='obman' --domain='unseen' --target_category='cellphone' --save

  # 1.471 remote is wrong on unseen
  python train.py gpu=1 exp_num='1.471' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 target_category='remote' use_1vN_nocs=True 2>&1 | tee outputs/train_1.471.log
  python pred_check.py --exp_num=1.471 --item='obman' --domain='unseen' --target_category='remote' --save

  # 1.48: knife/camera: simply train it!!! -->  knife
  # domain gap between train/val
  python train.py gpu=1 exp_num='1.48' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 target_category='knife' 2>&1 | tee outputs/train_1.48.log
  python pred_check.py --exp_num=1.48 --item='obman' --domain='unseen' # about
  python pred_check.py --exp_num=1.48 --item='obman' --domain='unseen' --target_category='knife' --save

  # 1.481, knife with 180 symmetry, ca207, gpu 1,
  python train.py gpu=1 exp_num='1.481' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 target_category='knife' 2>&1 | tee outputs/train_1.481.log

  # 1.49 camera
  python train.py gpu=1 exp_num='1.49' TRAIN.lr_encoder=0.01 TRAIN.lr_decoder=0.01 target_category='camera' 2>&1 | tee outputs/train_1.49.log
  python pred_check.py --exp_num=1.49 --item='obman' --domain='unseen' --target_category='camera' --save

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

2.06: # partial NOCS, train on one category: like bottle
  python train_obman.py item='obman' name_dset='obman' exp_num='2.06' target_category='bottle' use_category_code=False 2>&1 | tee outputs/occ_train_2.06.log
  python generate_co.py generation.copy_input=True item='obman' name_dset='obman' exp_num='2.06' target_category='bottle' use_category_code=False
  EXP=2.06
  python eval_objs.py eval=True item='obman' name_dset='obman' target_category='bottle' exp_num=${EXP} 2>&1 | tee outputs/occ_eval_objs_${EXP}.log

2.07: # partial NOCS, train on one category: like bottle, use occupancy network
  python train_obman.py item='obman' name_dset='obman' pointcloud='onet' exp_num='2.07' target_category='bottle' use_category_code=False 2>&1 | tee outputs/occ_train_2.07.log
  python generate_co.py generation.copy_input=True item='obman' name_dset='obman' pointcloud='onet' exp_num='2.07' target_category='bottle' use_category_code=False
  python eval_objs.py eval=True test.category_id=0 item='obman' name_dset='obman' exp_num=2.07 target_category='bottle' use_category_code=False 2>&1 | tee outputs/occ_eval_objs_2.07.log

# 2.071 jar
  python train_obman.py item='obman' name_dset='obman' pointcloud='onet' exp_num='2.071' target_category='jar' use_category_code=False 2>&1 | tee outputs/occ_train_2.071.log
  python generate_co.py generation.copy_input=True item='obman' name_dset='obman' pointcloud='onet' exp_num='2.071' target_category='jar' use_category_code=False
  python eval_objs.py eval=True test.category_id=4 item='obman' name_dset='obman' exp_num=2.071 target_category='jar' use_category_code=False 2>&1 | tee outputs/occ_eval_objs_2.071.log

# 2.072 cellphone
  python train_obman.py item='obman' name_dset='obman' pointcloud='onet' exp_num='2.072' target_category='cellphone' use_category_code=False 2>&1 | tee outputs/occ_train_2.072.log
  python generate_co.py generation.copy_input=True item='obman' name_dset='obman' pointcloud='onet' exp_num='2.072' target_category='cellphone' use_category_code=False
  python eval_objs.py eval=True test.category_id=6 item='obman' name_dset='obman' exp_num=2.072 target_category='cellphone' use_category_code=False 2>&1 | tee outputs/occ_eval_objs_2.072.log

# 2.073 bowl
  python train_obman.py item='obman' name_dset='obman' pointcloud='onet' exp_num='2.073' target_category='bowl' use_category_code=False 2>&1 | tee outputs/occ_train_2.073.log
  # python generate_co.py generation.copy_input=True item='obman' name_dset='obman' pointcloud='onet' exp_num='2.072' target_category='cellphone' use_category_code=False
  # python eval_objs.py eval=True test.category_id=6 item='obman' name_dset='obman' exp_num=2.072 target_category='cellphone' use_category_code=False 2>&1 | tee outputs/occ_eval_objs_2.072.log

# 2.073 hand +
#>>>>>>>>>>>>>>>>>>> joint trainings over hand + object together;

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

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>. point cloud completion
2.3: # GR-Net, see how it goes under same same setting compared to Onet;, check the Object reconstruction, think about the probability training loss when we have 1 to N relationship;
  python train_grnet.py dataset_class='HandDatasetComplete' exp_num='2.3' 2>&1 | tee outputs/complete_train_2.3.log
- Auto-encoder for hand pose estimation with objects;(grasping pose, infer grasping pose, object pose from observation)
- Object pose estimation using adversarial training;(real dataset, checking data quality)
- Contacts modeling as heatmap representations;
-
# cellphone, camera is wrong:
2.31 # bottle
  python train_grnet.py dataset_class='HandDatasetComplete' target_category='bottle' exp_num='2.31' TRAIN.N_EPOCHS=100 2>&1 | tee outputs/complete_train_2.31.log
  python train_grnet.py eval=True CONST.WEIGHTS='/groups/CESCA-CV/ICML2021/model/obman/2.31/checkpoints/ckpt-best.pth' dataset_class='HandDatasetComplete' target_category='bottle' exp_num='2.31' 2>&1 | tee outputs/complete_eval_2.31.log
2.32 # jar
  python train_grnet.py dataset_class='HandDatasetComplete' target_category='jar' exp_num='2.32' TRAIN.N_EPOCHS=100 2>&1 | tee outputs/complete_train_2.32.log
2.33 # bowl
  python train_grnet.py dataset_class='HandDatasetComplete' target_category='bowl' exp_num='2.33' TRAIN.N_EPOCHS=100 2>&1 | tee outputs/complete_train_2.33.log
2.34 # camera
  python train_grnet.py dataset_class='HandDatasetComplete' target_category='camera' exp_num='2.34' TRAIN.N_EPOCHS=100 2>&1 | tee outputs/complete_train_2.34.log

#>>>>>>>>>>>>>>>>>>>>>>>>>>> AE, VAE training
2.4 # bottle, AE, synthetic complete pcloud
TRAIN_OBJ='python train_aegan.py training=ae_gan name_model=ae dataset_class='HandDatasetAEGan' use_wandb=True vis=True num_points=1024 n_pts=1024'
$TRAIN_OBJ target_category='bottle' exp_num='2.4'  MODEL.num_degrees=4 \
eval=True ckpt='latest' \
set='train'

2.401 # bottle, AE-Graph, synthetic complete pcloud
TRAIN_OBJ='python train_aegan.py training=ae_gan use_wandb=True vis=True n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ target_category='bottle' exp_num='2.401' DATASET.train_batch=2 DATASET.test_batch=2 \
eval=True ckpt='latest' augment=True

2.402: # bottle, AE-Graph, synthetic complete pcloud, degree=2
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 MODEL.num_degrees=2 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ target_category='bottle' exp_num='2.402' DATASET.train_batch=4 DATASET.test_batch=4 \
use_wandb=True eval=True ckpt='latest' augment=True use_preprocess=True

# test 1, shows using knn to place ball query would result in big difference, also that type0 + type1 is not invariant;
# test 2, use type 0 only; also bad, but seems to be invariant;
# test 3, use type 0 only, fixed ball_query, and only the points rotate, quite amazing; invariant
# test 4, use type0 + type 1 for rotation estimation,

2.403: # bottle, AE-Graph, synthetic complete pcloud, degree=1
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 MODEL.num_degrees=1 MODEL.num_channels=128 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ target_category='bottle' exp_num='2.403' DATASET.train_batch=4 DATASET.test_batch=4 \
eval=True ckpt='latest' augment=True use_wandb=True

# 2.404: # x, z bottle, use AE-Graph for pose estimation, degree=2, use rot_mat
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.404' DATASET.train_batch=2 DATASET.test_batch=2 \
# augment=True num_channels_R=1 use_wandb=True fetch_cache=True
#
# 2.4041: # use bigger recep_field by adding one layer? 4, ampled different points, and different, 180 rotation only;
# TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 MODEL.num_layers=4 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
# $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4041' DATASET.train_batch=2 DATASET.test_batch=2 \
# augment=True eval=True use_wandb=True
# # add herachical structure like pointNet++; adopt a equivalent network;
# # noisy

2.404 # bottle, multiple instance, dynamic graph, random rotation, layer 12, channels into 32, 512 pts # 10G
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_completion' target_category='bottle' exp_num='2.404' DATASET.train_batch=2 DATASET.test_batch=2 \
MODEL.num_channels_R=1 MODEL.num_layers=12 augment=True use_wandb=True

# only use one channel, rotation only
2.405: # x, z bottle, use AE-Graph for pose estimation, degree=2, use up axis
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.405' DATASET.train_batch=2 DATASET.test_batch=2 \
num_channels_R=1 single_instance=True fetch_cache=True augment=True use_wandb=True

# 2.4051 multiple instance, dynamic graph, random rotation, layer 12, channels into 32, 512 pts # 10G
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4051' DATASET.train_batch=2 DATASET.test_batch=2 \
MODEL.num_channels_R=1 MODEL.num_layers=12 augment=True use_wandb=True

# 2.4052 use original angle loss
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4052' DATASET.train_batch=2 DATASET.test_batch=2 \
MODEL.num_channels_R=1 MODEL.num_layers=12 augment=True use_wandb=True

# 2.4053 per-point loss, use angle loss
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4053' DATASET.train_batch=2 DATASET.test_batch=2 \
MODEL.num_channels_R=1 MODEL.num_layers=12 augment=True rotation_use_dense=True rotation_loss_type=0 use_wandb=True

# 2.4054 per-point loss use L2 vector loss
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4054' DATASET.train_batch=2 DATASET.test_batch=2 \
MODEL.num_channels_R=1 MODEL.num_layers=12 augment=True rotation_use_dense=True rotation_loss_type=1 use_wandb=True

2.406: # predict T only!!!
# single instance, fixed graph, random rotation, work!!!
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.406' DATASET.train_batch=2 DATASET.test_batch=2 \
MODEL.num_channels_R=1 single_instance=True fetch_cache=True augment=True use_wandb=True

  # 2.4061 single instance, fixed graph, fixed rotation, work!!!
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4061' DATASET.train_batch=2 DATASET.test_batch=2 \
  MODEL.num_channels_R=1 single_instance=True fetch_cache=True use_wandb=True

  # 2.4062 single instance, dynamic graph, random rotation,
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4062' DATASET.train_batch=2 DATASET.test_batch=2 \
  MODEL.num_channels_R=1 single_instance=True augment=True use_wandb=True

  # 2.4063 multiple instance, fixed graph, random rotation
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4063' DATASET.train_batch=2 DATASET.test_batch=2 \
  MODEL.num_channels_R=1 fetch_cache=True augment=True use_wandb=True

  # 2.4064 single instance, dynamic graph, fixed rotation
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4064' DATASET.train_batch=2 DATASET.test_batch=2 \
  MODEL.num_channels_R=1 single_instance=True use_wandb=True

  # 2.4065 single instance, dynamic graph, fixed rotation, layer 3, 512 pts
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4065' DATASET.train_batch=2 DATASET.test_batch=2 \
  MODEL.num_channels_R=1 MODEL.num_layers=3 single_instance=True use_wandb=True

  # fastest twice than 2.067
  # 2.4066 single instance, dynamic graph, fixed rotation, layer 6, 512 pts = 8G --> # what if we use random rotation
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4066' DATASET.train_batch=2 DATASET.test_batch=2 \
  MODEL.num_channels_R=1 MODEL.num_layers=6 single_instance=True use_wandb=True

  current best!!!
  ----> # 2.4067 single instance, dynamic graph, fixed rotation, layer 12, channels into 16, 512 pts # 4.5G--> # what if we use random rotation
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=16 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4067' DATASET.train_batch=2 DATASET.test_batch=2 \
  MODEL.num_channels_R=1 MODEL.num_layers=12 single_instance=True use_wandb=True

  ----># 2.40671 single instance, dynamic graph, random rotation, layer 12, channels into 16, 512 pts # 4.5G, not well!!
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=16 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.40671' DATASET.train_batch=2 DATASET.test_batch=2 \
  MODEL.num_channels_R=1 MODEL.num_layers=12 single_instance=True augment=True use_wandb=True

  -----># 2.40672 single instance, dynamic graph, random rotation, layer 24, channels into 16, 512 pts # 6G, reasonablely well!!!
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=16 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.40672' DATASET.train_batch=2 DATASET.test_batch=2 \
  MODEL.num_channels_R=1 MODEL.num_layers=24 single_instance=True augment=True use_wandb=True

  #>>>>>>>>>>>> multiple instances
  # 2.4068 multiple instance, fixed graph, random rotation, layer 6, 512 pts = 8G, not working very well!!!
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4068' DATASET.train_batch=2 DATASET.test_batch=2 \
  MODEL.num_channels_R=1 MODEL.num_layers=6 fetch_cache=True augment=True use_wandb=True

  # 2.4069 multiple instance, fixed graph, random rotation, layer 12, channels into 16, 512 pts # 4.5G, work!!!
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=16 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4069' DATASET.train_batch=2 DATASET.test_batch=2 \
  MODEL.num_channels_R=1 MODEL.num_layers=12 fetch_cache=True augment=True use_wandb=True

  # 2.40691 multiple instance, dynamic graph, random rotation, layer 12, channels into 32, 512 pts # 10G
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.40691' DATASET.train_batch=2 DATASET.test_batch=2 \
  MODEL.num_channels_R=1 MODEL.num_layers=12 augment=True use_wandb=True

  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.40692' DATASET.train_batch=2 DATASET.test_batch=2 \
  MODEL.num_channels_R=1 MODEL.num_layers=12
  # TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  # $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.40691' DATASET.train_batch=2 DATASET.test_batch=2 \
  # MODEL.num_channels_R=1 MODEL.num_layers=12 augment=True use_wandb=True
  #
  # ----> 2.407: # 6d rotation with different input setting
  # TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  # $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.407' DATASET.train_batch=2 DATASET.test_batch=2 \
  # augment=True use_wandb=True

2.407: # pointnet++, T only,
  TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.407' DATASET.train_batch=2 DATASET.test_batch=2 \
  augment=True use_objective_T=True use_wandb=True

    # pointnet++, T only, add mean substraction
  TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4071' DATASET.train_batch=2 DATASET.test_batch=2 \
  augment=True use_objective_T=True use_wandb=True

  2.4072: # pointnet++, R only, type 0
  TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4072' DATASET.train_batch=2 DATASET.test_batch=2 \
  augment=True rotation_loss_type=0 use_objective_R=True use_wandb=True

  2.4073: # pointnet++, R only, type 1
  TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4073' DATASET.train_batch=2 DATASET.test_batch=2 \
  augment=True rotation_loss_type=1 use_objective_R=True use_wandb=True

  2.4074: # pointnet++ on complete data for NOCS
  TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose' target_category='bottle' exp_num='2.4074' DATASET.train_batch=2 DATASET.test_batch=2 \
  augment=True rotation_loss_type=1 pred_nocs=True use_objective_N=True use_wandb=True

  2.4075: # pointnet++ on partial data for NOCS
  TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.4075' DATASET.train_batch=2 DATASET.test_batch=2 \
  augment=True rotation_loss_type=1 pred_nocs=True use_objective_N=True use_wandb=True

  TRAIN_OBJ='python train_aegan.py training=ae_gan models=pnet2 encoder_type='pnetplusplus' vis=True num_points=512 n_pts=512 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40751' DATASET.train_batch=2 DATASET.test_batch=2 \
  augment=True rotation_loss_type=1 pred_nocs=True use_objective_N=True single_instance=True use_wandb=True

2.408: # S+R+T at the same time,  type=0, ca203, 1
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose_completion' target_category='bottle' exp_num='2.408' DATASET.train_batch=2 DATASET.test_batch=2 \
MODEL.num_channels_R=1 MODEL.num_layers=12 augment=True rotation_loss_type=0 use_wandb=True

  2.4081: # S+R+T at the same time, type=1, ca197. 0
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_pose_completion' target_category='bottle' exp_num='2.4081' DATASET.train_batch=2 DATASET.test_batch=2 \
  MODEL.num_channels_R=1 MODEL.num_layers=12 augment=True rotation_loss_type=1 use_wandb=True

  # 2.408: # use partial rotated input, predict GT pose; how to make it work!!!
  # TRAIN_OBJ='python train_aegan.py training=ae_gan name_model=ae module=ae num_points=1024 n_pts=1024 dataset_class=HandDatasetAEGraph'
  # $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.408' DATASET.train_batch=2 DATASET.test_batch=2 \
  # vis=True augment=True use_wandb=True single_instance=True fetch_cache=True

# >>>>>>>>>>>>>>>>>>> partial point clouds >>>>>>>>>>>>>>>>
2.409: #  train for partial shape, bottle, R + T, type 0, ca197 1
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.409' DATASET.train_batch=2 DATASET.test_batch=2 \
MODEL.num_channels_R=1 MODEL.num_layers=12 augment=True rotation_loss_type=0 use_wandb=True

  2.4091: # train for partial shape, bottle, R + T, type 1, ca203, 0
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.4091' DATASET.train_batch=2 DATASET.test_batch=2 \
  MODEL.num_channels_R=1 MODEL.num_layers=12 augment=True rotation_loss_type=1 use_wandb=True

  2.4092: # train for partial shape, certain category, R + T + S
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='partial_pcloud_pose_completion' target_category='bottle' exp_num='2.4092' DATASET.train_batch=2 DATASET.test_batch=2 \
  MODEL.num_channels_R=1 MODEL.num_layers=12 augment=True rotation_loss_type=0 use_wandb=True

  2.4093: # use type 0 to predict NOCS
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_layers=12 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.4093' DATASET.train_batch=2 DATASET.test_batch=2 \
  pred_nocs=True use_objective_N=True rotation_loss_type=0 use_wandb=True

TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_layers=12 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40931' DATASET.train_batch=2 DATASET.test_batch=2 \
pred_nocs=True use_objective_N=True single_instance=True rotation_loss_type=1 use_wandb=True

  2.4094: # use type 0 predict NOCS on complete shape
  TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_layers=12 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ task='pcloud_
  pose' target_category='bottle' exp_num='2.4094' DATASET.train_batch=2 DATASET.test_batch=2 \
  augment=True pred_nocs=True use_objective_N=True rotation_loss_type=0 use_wandb=True

  2.4095: # use partial shape for R estimation, type 1 loss-> single instance, 10-->32-->5, 10+original
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=512 n_pts=512 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='partial_pcloud_pose' target_category='bottle' exp_num='2.40955' DATASET.train_batch=2 DATASET.test_batch=2 \
MODEL.num_channels_R=1 MODEL.num_layers=12 augment=True rotation_loss_type=1 single_instance=True use_objective_R=True use_wandb=True

2.41 # jar, AE, synthetic complete pcloud
  python train_aegan.py training=ae_gan name_model=ae dataset_class='HandDatasetAEGan' target_category='jar' exp_num='2.41' \
  eval=True ckpt='latest' split='train'

2.411 # jar, AE-Graph, synthetic complete pcloud
  TRAIN_OBJ='python train_aegan.py training=ae_gan use_wandb=True vis=True n_pts=256 name_model=ae dataset_class=HandDatasetAEGraph'
  $TRAIN_OBJ target_category='jar' exp_num='2.411'  DATASET.train_batch=2 DATASET.test_batch=2

2.412: # jar,
TRAIN_OBJ='python train_aegan.py training=ae_gan use_wandb=True vis=True num_points=1024 n_pts=1024 MODEL.num_degrees=2 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ target_category='jar' exp_num='2.402' DATASET.train_batch=4 DATASET.test_batch=4

2.413: # jar
TRAIN_OBJ='python train_aegan.py training=ae_gan vis=True num_points=1024 n_pts=1024 MODEL.num_degrees=2 MODEL.num_channels=32 name_model=ae dataset_class=HandDatasetAEGraph'
$TRAIN_OBJ task='pcloud_pose' target_category='jar' exp_num='2.413' DATASET.train_batch=2 DATASET.test_batch=2 \
augment=True use_wandb=True

2.42 # all categories, AE, synthetic complete pcloud
  python train_aegan.py training=ae_gan name_model=ae dataset_class='HandDatasetAEGan' exp_num='2.42'

2.421 #
  python train_aegan.py training=ae_gan use_wandb=True vis=True n_pts=512 name_model=ae dataset_class='HandDatasetAEGraph' target_category='bottle' exp_num='2.421'  DATASET.train_batch=2 DATASET.test_batch=2

2.5 # all categories, VAE
  python train_aegan.py training=ae_gan module=vae dataset_class='HandDatasetAEGan' exp_num='2.5' lr=5e-4 nr_epochs=2000

2.51 # jar, VAE
  python train_aegan.py training=ae_gan module=vae dataset_class='HandDatasetAEGan' exp_num='2.51' lr=5e-4 nr_epochs=2000 target_category='jar'

2.52 # bottle, VAE, ca233 0, 800
  python train_aegan.py training=ae_gan module=vae dataset_class='HandDatasetAEGan' exp_num='2.52' lr=5e-4 nr_epochs=2000 target_category='bottle'

#>>>>>>>>>>>>>>>>>> GAN training
2.6 # bottle, with VAE, training GAN, ca207 0, first synthetic data, then with real data, but use it as an domian shift;
  python train_aegan.py training=ae_gan module=gan dataset_class='HandDatasetAEGan' exp_num='2.6' lr=5e-4 nr_epochs=500 batch_size=100 save_frequency=100 task='adversarial_adaptation' target_category='bottle' num_points=1024 vis=True \
  pretrain_ae_path=/groups/CESCA-CV/ICML2021/model/obman/2.4/checkpoints/ckpt_epoch500.pth \
  pretrain_vae_path=/groups/CESCA-CV/ICML2021/model/obman/2.52/checkpoints/ckpt_epoch500.pth \
  eval=True ckpt='100' split='val'

2.61 # jar, with VAE;
  python train_aegan.py training=ae_gan module=gan gpu_ids=1 dataset_class='HandDatasetAEGan' exp_num='2.61' lr=5e-4 nr_epochs=500 batch_size=100 save_frequency=100 task='adversarial_adaptation' target_category='jar' num_points=1024 vis=True \
  pretrain_ae_path=/groups/CESCA-CV/ICML2021/model/obman/2.41/checkpoints/latest.pth \
  pretrain_vae_path=/groups/CESCA-CV/ICML2021/model/obman/2.51/checkpoints/ckpt_epoch500.pth \

2.62 # all categories, with VAE
  python train_aegan.py training=ae_gan module=gan dataset_class='HandDatasetAEGan' exp_num='2.62' lr=5e-4 nr_epochs=500 batch_size=100 save_frequency=100 task='adversarial_adaptation' num_points=1024 vis=True \
  pretrain_ae_path=/groups/CESCA-CV/ICML2021/model/obman/2.42/checkpoints/latest.pth \
  pretrain_vae_path=/groups/CESCA-CV/ICML2021/model /obman/2.5/checkpoints/latest.pth

2.7 # training GAN without VAE, bottle, ca233 0
  python train_aegan.py training=ae_gan multimodal=False module=gan dataset_class='HandDatasetAEGan' exp_num='2.7' lr=5e-4 nr_epochs=500 batch_size=100 save_frequency=100 task='adversarial_adaptation' target_category='bottle' num_points=1024 vis=True \
  pretrain_ae_path=/groups/CESCA-CV/ICML2021/model/obman/2.4/checkpoints/ckpt_epoch500.pth \
  pretrain_vae_path=/groups/CESCA-CV/ICML2021/model/obman/2.52/checkpoints/ckpt_epoch500.pth \
  eval=True ckpt='100' split='val'

  2.701 # bottle, GAN without VAE, W-GAN
  python train_aegan.py training=ae_gan multimodal=False module=gan dataset_class='HandDatasetAEGan' exp_num='2.701' lr=5e-4 nr_epochs=500 batch_size=100 save_frequency=100 task='adversarial_adaptation' target_category='bottle' num_points=1024 vis=True \
  pretrain_ae_path=/groups/CESCA-CV/ICML2021/model/obman/2.4/checkpoints/ckpt_epoch500.pth \
  pretrain_vae_path=/groups/CESCA-CV/ICML2021/model/obman/2.52/checkpoints/ckpt_epoch500.pth \
  use_wgan=True use_wandb=True

  2.702  # bottle, GAN without VAE, plain GAN
  python train_aegan.py training=ae_gan multimodal=False module=gan dataset_class='HandDatasetAEGan' exp_num='2.702' lr=5e-4 nr_epochs=500 batch_size=100 save_frequency=100 task='adversarial_adaptation' target_category='bottle' num_points=1024 vis=True \
  pretrain_ae_path=/groups/CESCA-CV/ICML2021/model/obman/2.4/checkpoints/ckpt_epoch500.pth \
  pretrain_vae_path=/groups/CESCA-CV/ICML2021/model/obman/2.52/checkpoints/ckpt_epoch500.pth \
  use_wandb=True

  2.71 # training GAN without multimodal, jar ca233 1
  python train_aegan.py training=ae_gan gpu_ids=1 multimodal=False module=gan dataset_class='HandDatasetAEGan' exp_num='2.71' lr=5e-4 nr_epochs=500 batch_size=100 save_frequency=100 task='adversarial_adaptation' target_category='jar' num_points=1024 vis=True \
  pretrain_ae_path=/groups/CESCA-CV/ICML2021/model/obman/2.41/checkpoints/latest.pth \
  pretrain_vae_path=/groups/CESCA-CV/ICML2021/model/obman/2.51/checkpoints/ckpt_epoch500.pth \
  eval=True ckpt='100' split='val'

2.8 # training without VAE, and use raw partial depth point cloud, together with
python train_aegan.py training=ae_gan task='adversarial_adaptation' module=gan dataset_class='HandDatasetAEGraph' target_category='bottle' num_points=1024 n_pts=1024 exp_num='2.8' MODEL.num_degrees=2 latent_dim=64 \
lr=5e-4 nr_epochs=500 save_frequency=100 DATASET.train_batch=12 DATASET.test_batch=12 \
pretrain_ae_path=/groups/CESCA-CV/ICML2021/model/obman/2.402/checkpoints/latest.pth \
pretrain_vae_path=/groups/CESCA-CV/ICML2021/model/obman/2.52/checkpoints/ckpt_epoch500.pth \
use_wandb=True multimodal=False vis=True
# 1. type 0 equivariant;
# 2. 256--> 1024;
# 3. SDF;
