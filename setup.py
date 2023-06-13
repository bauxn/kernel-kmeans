from distutils.core import setup
from Cython.Build import cythonize


extensions = ["utils.pyx", "lloyd.pyx", "kernels.pyx", "quality_utils.pyx", "elkan.pyx", ]
compiler_directives = {"language_level": "3"}

setup(ext_modules=cythonize(extensions, compiler_directives=compiler_directives))