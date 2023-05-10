from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(["_lloyd_iter.pyx", "pairwise_kernel.pyx"]))
