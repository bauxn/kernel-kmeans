from distutils.core import setup
from Cython.Build import cythonize

from distutils.extension import Extension
from distutils.ccompiler import new_compiler  # used to determine command line argument for openmMP 

compile_args = ["-O3", "-ffast-math", "-DCYTHON_WITHOUT_ASSERTIONS", "-fopenmp"]

ccompiler = new_compiler()
if ccompiler.compiler_type == "msvc":
    compile_args = ["/O2", "-DCYTHON_WITHOUT_ASSERTIONS", "/openmp"]

# TODO: command line arg, dann mach compile_args = ["-DCYTHON_WITHOUT_ASSERTIONS"] + [command line args]

compiler_directives = {
    "language_level": 3, 
    "boundscheck": False, 
    "wraparound": False, 
    "cdivision": True,
}

extensions = [
    Extension("utils", ["utils.pyx"], extra_compile_args=compile_args),
    Extension("lloyd", ["lloyd.pyx"], extra_compile_args=compile_args),
    Extension("elkan", ["elkan.pyx"], extra_compile_args=compile_args),
    Extension("kernels", ["kernels.pyx"], extra_compile_args=compile_args),
    Extension("quality_utils", ["quality_utils.pyx"], extra_compile_args=compile_args),
]
              
setup(ext_modules=cythonize(extensions, compiler_directives=compiler_directives))