from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(["distance_utils.pyx", "pairwise_kernel.pyx"]))
