from setuptools import setup, Extension
from Cython.Build import cythonize
from distutils.ccompiler import new_compiler  # used to determine command line argument for openmMP 
from argparse import ArgumentParser

# gcc
compile_args = ["-O3", "-ffast-math", "-DCYTHON_WITHOUT_ASSERTIONS", "-fopenmp"]

ccompiler = new_compiler()
if ccompiler.compiler_type == "msvc":
    compile_args = ["/O2", "-DCYTHON_WITHOUT_ASSERTIONS", "/openmp"]

if ccompiler.compiler_type not in ("gcc", "g++", "msvc"):
    print("unknown compiler used! Please provide compile args with\
          -C or --cargs. At least option to enable openmp heavily \
          recommended.")

parser = ArgumentParser()
parser.add_argument("-C", "--cargs", nargs="*", required=False)
args = parser.parse_args()

if args.cargs is not None:
    #TODO test if this happens when no args 
    # print(args.cargs)
    compile_args = args.cargs


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
    Extension("quality", ["quality.pyx"], extra_compile_args=compile_args),
]
              
setup(ext_modules=cythonize(extensions, compiler_directives=compiler_directives), script_args=['build_ext'], options={'build_ext':{'inplace':True}})