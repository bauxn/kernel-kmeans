from distutils.core import setup
from Cython.Build import cythonize


extensions = ["lloyd_utils.pyx", "kernel_utils.pyx"]
compiler_directives = {"language_level": "3"}

setup(ext_modules=cythonize(extensions, compiler_directives=compiler_directives))