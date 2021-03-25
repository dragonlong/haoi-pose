-----pointAEPose architecture-----
/home/lxiaol9/anaconda3/envs/merl/lib/python3.6/site-packages/torch/include/c10/core/TensorTypeSet.h(44): warning: integer conversion resulted in a change of sign

creating build/lib.linux-x86_64-3.6
g++ -pthread -shared -B /home/lxiaol9/anaconda3/envs/merl/compiler_compat -L/home/lxiaol9/anaconda3/envs/merl/lib -Wl,-rpath=/home/lxiaol9/anaconda3/envs/merl/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-3.6/src/pointnet2_api.o build/temp.linux-x86_64-3.6/src/ball_query.o build/temp.linux-x86_64-3.6/src/ball_query_gpu.o build/temp.linux-x86_64-3.6/src/group_points.o build/temp.linux-x86_64-3.6/src/group_points_gpu.o build/temp.linux-x86_64-3.6/src/interpolate.o build/temp.linux-x86_64-3.6/src/interpolate_gpu.o build/temp.linux-x86_64-3.6/src/sampling.o build/temp.linux-x86_64-3.6/src/sampling_gpu.o -L/opt/apps/cuda/10.1.168/lib64 -lcudart -o build/lib.linux-x86_64-3.6/pointnet2_cuda.cpython-36m-x86_64-linux-gnu.so
creating build/bdist.linux-x86_64
creating build/bdist.linux-x86_64/egg
copying build/lib.linux-x86_64-3.6/pointnet2_cuda.cpython-36m-x86_64-linux-gnu.so -> build/bdist.linux-x86_64/egg
creating stub loader for pointnet2_cuda.cpython-36m-x86_64-linux-gnu.so
byte-compiling build/bdist.linux-x86_64/egg/pointnet2_cuda.py to pointnet2_cuda.cpython-36.pyc
creating build/bdist.linux-x86_64/egg/EGG-INFO
copying pointnet2.egg-info/PKG-INFO -> build/bdist.linux-x86_64/egg/EGG-INFO
copying pointnet2.egg-info/SOURCES.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying pointnet2.egg-info/dependency_links.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying pointnet2.egg-info/top_level.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
writing build/bdist.linux-x86_64/egg/EGG-INFO/native_libs.txt
zip_safe flag not set; analyzing archive contents...
__pycache__.pointnet2_cuda.cpython-36: module references __file__
creating dist
creating 'dist/pointnet2-0.0.0-py3.6-linux-x86_64.egg' and adding 'build/bdist.linux-x86_64/egg' to it
removing 'build/bdist.linux-x86_64/egg' (and everything under it)
Processing pointnet2-0.0.0-py3.6-linux-x86_64.egg
removing '/home/lxiaol9/anaconda3/envs/merl/lib/python3.6/site-packages/pointnet2-0.0.0-py3.6-linux-x86_64.egg' (and everything under it)
creating /home/lxiaol9/anaconda3/envs/merl/lib/python3.6/site-packages/pointnet2-0.0.0-py3.6-linux-x86_64.egg
Extracting pointnet2-0.0.0-py3.6-linux-x86_64.egg to /home/lxiaol9/anaconda3/envs/merl/lib/python3.6/site-packages
pointnet2 0.0.0 is already the active version in easy-install.pth

Installed /home/lxiaol9/anaconda3/envs/merl/lib/python3.6/site-packages/pointnet2-0.0.0-py3.6-linux-x86_64.egg
Processing dependencies for pointnet2==0.0.0
Finished processing dependencies for pointnet2==0.0.0


# >>>>>>>>>>>>>>>>>>>>>>>>>> another pointnet2
(merl) [lxiaol9@ca215 pointnet2]$ ls
build  dist  _ext_src  pointnet2.egg-info  pointnet2_modules.py  pointnet2_test.py  pointnet2_utils.py  __pycache__  pytorch_utils.py  setup.py
(merl) [lxiaol9@ca215 pointnet2]$ python setup.py install
running install
running bdist_egg
running egg_info
writing pointnet2.egg-info/PKG-INFO
writing dependency_links to pointnet2.egg-info/dependency_links.txt
writing top-level names to pointnet2.egg-info/top_level.txt
reading manifest file 'pointnet2.egg-info/SOURCES.txt'
writing manifest file 'pointnet2.egg-info/SOURCES.txt'
installing library code to build/bdist.linux-x86_64/egg
running install_lib
running build_ext
creating build/bdist.linux-x86_64/egg
creating build/bdist.linux-x86_64/egg/pointnet2
copying build/lib.linux-x86_64-3.6/pointnet2/_ext.cpython-36m-x86_64-linux-gnu.so -> build/bdist.linux-x86_64/egg/pointnet2
creating stub loader for pointnet2/_ext.cpython-36m-x86_64-linux-gnu.so
byte-compiling build/bdist.linux-x86_64/egg/pointnet2/_ext.py to _ext.cpython-36.pyc
creating build/bdist.linux-x86_64/egg/EGG-INFO
copying pointnet2.egg-info/PKG-INFO -> build/bdist.linux-x86_64/egg/EGG-INFO
copying pointnet2.egg-info/SOURCES.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying pointnet2.egg-info/dependency_links.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
copying pointnet2.egg-info/top_level.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
writing build/bdist.linux-x86_64/egg/EGG-INFO/native_libs.txt
zip_safe flag not set; analyzing archive contents...
pointnet2.__pycache__._ext.cpython-36: module references __file__
creating 'dist/pointnet2-0.0.0-py3.6-linux-x86_64.egg' and adding 'build/bdist.linux-x86_64/egg' to it
removing 'build/bdist.linux-x86_64/egg' (and everything under it)
Processing pointnet2-0.0.0-py3.6-linux-x86_64.egg
removing '/home/lxiaol9/anaconda3/envs/merl/lib/python3.6/site-packages/pointnet2-0.0.0-py3.6-linux-x86_64.egg' (and everything under it)
creating /home/lxiaol9/anaconda3/envs/merl/lib/python3.6/site-packages/pointnet2-0.0.0-py3.6-linux-x86_64.egg
Extracting pointnet2-0.0.0-py3.6-linux-x86_64.egg to /home/lxiaol9/anaconda3/envs/merl/lib/python3.6/site-packages
pointnet2 0.0.0 is already the active version in easy-install.pth

Installed /home/lxiaol9/anaconda3/envs/merl/lib/python3.6/site-packages/pointnet2-0.0.0-py3.6-linux-x86_64.egg
Processing dependencies for pointnet2==0.0.0
Finished processing dependencies for pointnet2==0.0.0
