from setuptools import setup, Extension
from Cython.Build import cythonize


# # default
compile_args = ["-DCYTHON_WITHOUT_ASSERTIONS"]

# # msvc
# compile_args = ["/O2", "-DCYTHON_WITHOUT_ASSERTIONS", "/openmp"]

# # gcc
# compile_args = ["-O3", "-ffast-math", "-DCYTHON_WITHOUT_ASSERTIONS", "-fopenmp"]

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

# This is used for building the distribution for pipy
# https://stackoverflow.com/questions/73766918/creating-python-package-which-uses-cython-valueerror
package_data = {
    "KKMeans" : ["utils.pyx", "lloyd.pyx", "elkan.pyx", "kernels.pyx", "quality.pyx", "KKMeans.py"]
}
        
setup(ext_modules=cythonize(extensions, compiler_directives=compiler_directives),package_dir={"KKMeans":"src/KKMeans"}, package_data=package_data, include_package_data=True)