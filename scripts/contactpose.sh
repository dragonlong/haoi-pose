
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh

./configure --prefix=${CONDA_PREFIX}          \
            --enable-opengl --disable-gles1 --disable-gles2   \
            --disable-va --disable-xvmc --disable-vdpau       \
            --enable-shared-glapi                             \
            --disable-texture-float                           \
            --enable-gallium-llvm --enable-llvm-shared-libs   \
            --with-gallium-drivers=swrast,swr                 \
            --disable-dri --with-dri-drivers=                 \
            --disable-egl --with-egl-platforms= --disable-gbm \
            --disable-glx                                     \
            --disable-osmesa --enable-gallium-osmesa          \
            ac_cv_path_LLVM_CONFIG=llvm-config-6.0
#
python startup.py
make[5]: Nothing to be done for 'install-exec-am'.
make[5]: Nothing to be done for 'install-data-am'.
make[5]: Leaving directory '/home/dragon/Documents/software/mesa-18.3.3/src/gallium/state_trackers/osmesa'
make[4]: Leaving directory '/home/dragon/Documents/software/mesa-18.3.3/src/gallium/state_trackers/osmesa'
Making install in targets/osmesa
make[4]: Entering directory '/home/dragon/Documents/software/mesa-18.3.3/src/gallium/targets/osmesa'
make[5]: Entering directory '/home/dragon/Documents/software/mesa-18.3.3/src/gallium/targets/osmesa'
 /bin/mkdir -p '/home/dragon/miniconda3/envs/contactpose/lib'
 /bin/bash ../../../../libtool   --mode=install /usr/bin/install -c   libOSMesa.la '/home/dragon/miniconda3/envs/contactpose/lib'
libtool: warning: relinking 'libOSMesa.la'
libtool: install: (cd /home/dragon/Documents/software/mesa-18.3.3/src/gallium/targets/osmesa; /bin/bash "/home/dragon/Documents/software/mesa-18.3.3/libtool"  --silent --tag CXX --mode=relink g++ -g -O2 -Wall -fno-math-errno -fno-trapping-math -Wno-missing-field-initializers -no-undefined -version-number 8 -Wl,--gc-sections -Wl,--no-undefined -Wl,--version-script=../../../../src/gallium/targets/osmesa/osmesa.sym -L/usr/lib/llvm-6.0/lib -L/usr/lib/llvm-6.0/lib -o libOSMesa.la -rpath /home/dragon/miniconda3/envs/contactpose/lib target.lo ../../../../src/mesa/libmesagallium.la ../../../../src/gallium/auxiliary/libgallium.la ../../../../src/gallium/winsys/sw/null/libws_null.la ../../../../src/gallium/drivers/softpipe/libsoftpipe.la ../../../../src/gallium/state_trackers/osmesa/libosmesa.la ../../../../src/mapi/glapi/libglapi.la ../../../../src/mapi/shared-glapi/libglapi.la -lm -lpthread -pthread -ldl ../../../../src/gallium/drivers/llvmpipe/libllvmpipe.la -lLLVM-6.0 ../../../../src/gallium/drivers/swr/libmesaswr.la -lLLVM-6.0 )
libtool: install: /usr/bin/install -c .libs/libOSMesa.so.8.0.0T /home/dragon/miniconda3/envs/contactpose/lib/libOSMesa.so.8.0.0
libtool: install: (cd /home/dragon/miniconda3/envs/contactpose/lib && { ln -s -f libOSMesa.so.8.0.0 libOSMesa.so.8 || { rm -f libOSMesa.so.8 && ln -s libOSMesa.so.8.0.0 libOSMesa.so.8; }; })
libtool: install: (cd /home/dragon/miniconda3/envs/contactpose/lib && { ln -s -f libOSMesa.so.8.0.0 libOSMesa.so || { rm -f libOSMesa.so && ln -s libOSMesa.so.8.0.0 libOSMesa.so; }; })
libtool: install: /usr/bin/install -c .libs/libOSMesa.lai /home/dragon/miniconda3/envs/contactpose/lib/libOSMesa.la
libtool: finish: PATH="/home/dragon/miniconda3/envs/contactpose/bin:/home/dragon/.nvm/versions/node/v14.8.0/bin:/opt/ros/melodic/bin:/home/dragon/gems/bin:/home/dragon/.cargo/bin:/home/dragon/miniconda3/condabin:/home/dragon/.cargo/bin:/home/dragon/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/sbin" ldconfig -n /home/dragon/miniconda3/envs/contactpose/lib
----------------------------------------------------------------------
