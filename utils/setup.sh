GRNET_HOME=$PWD

# Chamfer Distance
cd $GRNET_HOME/extensions/chamfer_dist
python setup.py install --user

# Cubic Feature Sampling
cd $GRNET_HOME/extensions/cubic_feature_sampling
python setup.py install --user

# Gridding & Gridding Reverse
cd $GRNET_HOME/extensions/gridding
python setup.py install --user

# Gridding Loss
cd $GRNET_HOME/extensions/gridding_loss
python setup.py install --user

pip install lmdb h5py requests gdown easydict tqdm termcolor
