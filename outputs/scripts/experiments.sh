0.1: # same model, but with new loss
  python train.py verbose=True gpu=0 name_model='pointnet2_meteornet' exp_num='0.1' TRAIN.loss='xentropy'

0.2: # new model, new loss
  python train.py verbose=True gpu=1 name_model='pointnet2_kaolin' exp_num='0.2' TRAIN.loss='xentropy' MODEL.arch_decoder='kaolin' 2>&1 | tee train_0.2.log

0.3: # new mode, but with same loss
  python train.py verbose=True gpu=0 name_model='pointnet2_kaolin' exp_num='0.3' TRAIN.loss='miou' MODEL.arch_decoder='kaolin' 2>&1 | tee train_0.3.log

0.4: # mano regression, 61 params, shape 10, articulation 45, global 6, regression is easy: but recontruction loss + mano_layer
  python train.py verbose=True gpu=1 name_model='pointnet2_kaolin' exp_num='0.4' MODEL.arch_decoder='kaolin' pred_mano=True 2>&1 | tee train_0.4.log

1.0:
  python train.py verbose=True gpu=1 name_model='pointnet2_meteornet' exp_num=1
  python train.py verbose=True gpu=1 name_model='pointnet2_meteornet' exp_num=1 eval=True

1.1:
  python train.py verbose=True gpu=1 name_model='pointnet2_charlesssg' ca218
