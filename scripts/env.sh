pip install hydra_colorlog
pip install hydra-core --upgrade
pip install GPUtil h5py opencv-python numba pytransform3d
pip install wandb plyfile
pip install transforms3d pyquaternion
pip install contextlib2
sudo apt-get install liboctomap-dev
pip install python-fcl descartes pyrender vispy scikit-learn
conda install -c conda-forge igl
conda install meshplot

pip install future pytest requests scipy pybind11
projectq 0.3.6 requires pybind11!=2.2.0,>=1.7, which is not installed.
projectq 0.3.6 requires pytest>=3.1, which is not installed.
projectq 0.3.6 requires requests, which is not installed.
projectq 0.3.6 requires scipy, which is not installed.
pip install neural_renderer_pytorch
pip install mmcv
conda install -c conda-forge scikit-image matplotlib opencv pyyaml tensorboardX
#
http://www.codeplastic.com/2019/03/12/how-to-install-python-modules-in-blender/
python -m ensurepip
bpy & ./pip3 install scipy imageio

pip install git+https://github.com/hassony2/chumpy.git
cd manopth
pip install .
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

pip install openmesh
pip install https://github.com/fredrikaverpil/oiio-python/releases/download/2.0.5%2B20190203/oiio-2.0.5-cp36-none-linux_x86_64.whl

export INCLUDE=/home/lxiaol9/3DGenNet2019/votenet/pointnet2/_ext_src/include/:$INCLUDE
export CPLUS_INCLUDE_PATH=/home/lxiaol9/3DGenNet2019/votenet/pointnet2/_ext_src/include/:$CPLUS_INCLUDE_PATH
pip install gin-config==0.1.1
tensorboardX 2.1

conda install -c conda-forge igl

pip install dgl-cu101==0.4.3.post2

# add glic for open3d
curl -O http://ftp.gnu.org/gnu/glibc/glibc-2.18.tar.gz
tar -xzvf glibc-2.18.tar.gz
cd glibc-2.18/
mkdir build
cd build/
../configure --prefix=/home/lxiaol9/pkg/open3d
make -j2
make install


cmake -DCMAKE_INSTALL_PREFIX=/home/lxiaol9/pkg/open3d ..

# make sure you have cuda & pytorch
cd models/pointnet_lib/ && python setup.py install
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
conda create --name ENVNAME python=3.6
pip install future pybind11 pytest requests scipy
pip install dgl-cu110==0.5.3 open3d==0.8.0
# kaolin building 0.1.0
>>> import kaolin as kal
>>> print(kal.__version__)

source activate ptrans36
module load cuda/11.0.1

cp -r vgtk/ ../haoi-pose/utils/
cp SPConvNets/models/*so3* ../haoi-pose/models/encoder/
cp SPConvNets/utils/*so3* ../haoi-pose/models/encoder/
cp SPConvNets/options.py ../haoi-pose/
