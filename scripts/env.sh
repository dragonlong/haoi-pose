pip install hydra_colorlog
pip install hydra-core --upgrade
pip install GPUtil
sudo apt-get install liboctomap-dev
pip install python-fcl

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
pip install dgl-cu101==0.5.3
