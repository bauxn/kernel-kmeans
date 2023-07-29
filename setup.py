from setuptools import setup, Extension
from Cython.Build import cythonize
# from distutils.ccompiler import new_compiler  # used to determine command line argument for openmMP 


# print("HELLO")
# # gcc
# compile_args = ["-O3", "-ffast-math", "-DCYTHON_WITHOUT_ASSERTIONS", "-fopenmp"]

# ccompiler = new_compiler()
# if ccompiler.compiler_type == "msvc":
#     compile_args = ["/O2", "-DCYTHON_WITHOUT_ASSERTIONS", "/openmp"]

# if ccompiler.compiler_type not in ("gcc", "g++", "msvc"):
#     print("unknown compiler used! Please provide compile args with\
#           -C or --cargs. At least option to enable openmp heavily \
#           recommended.")

compile_args = ["-DCYTHON_WITHOUT_ASSERTIONS"]

compiler_directives = {
    "language_level": 3, 
    "boundscheck": False, 
    "wraparound": False, 
    "cdivision": True,
}

extensions = [
    Extension("KKMeans.utils", ["src/KKMeans/utils.pyx"], extra_compile_args=compile_args),
    Extension("KKMeans.lloyd", ["src/KKMeans/lloyd.pyx"], extra_compile_args=compile_args),
    Extension("KKMeans.elkan", ["src/KKMeans/elkan.pyx"], extra_compile_args=compile_args),
    Extension("KKMeans.kernels", ["src/KKMeans/kernels.pyx"], extra_compile_args=compile_args),
    Extension("KKMeans.quality", ["src/KKMeans/quality.pyx"], extra_compile_args=compile_args),
    Extension("KKMeans", ["src/KKMeans/KKMeans.py"])
]

              
setup(ext_modules=cythonize(extensions, compiler_directives=compiler_directives))