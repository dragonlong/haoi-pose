try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions
# pykdtree (kd tree)
pykdtree = Extension(
    'utils.external.libkdtree.pykdtree.kdtree',
    sources=[
        'utils/external/libkdtree/pykdtree/kdtree.c',
        'utils/external/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
    include_dirs=[numpy_include_dir]
)

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'utils.external.libmcubes.mcubes',
    sources=[
        'utils/external/libmcubes/mcubes.pyx',
        'utils/external/libmcubes/pywrapper.cpp',
        'utils/external/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'utils.external.libmesh.triangle_hash',
    sources=[
        'utils/external/libmesh/triangle_hash.pyx'
    ],
    libraries=['m'],  # Unix-like specific
    include_dirs=[numpy_include_dir]
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'utils.external.libmise.mise',
    sources=[
        'utils/external/libmise/mise.pyx'
    ],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'utils.external.libsimplify.simplify_mesh',
    sources=[
        'utils/external/libsimplify/simplify_mesh.pyx'
    ],
    include_dirs=[numpy_include_dir]
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    'utils.external.libvoxelize.voxelize',
    sources=[
        'utils/external/libvoxelize/voxelize.pyx'
    ],
    libraries=['m']  # Unix-like specific
)

# Gather all extension modules
ext_modules = [
    pykdtree,
    mcubes_module,
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)
