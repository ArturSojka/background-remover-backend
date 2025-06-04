from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy
import sys

if sys.platform.startswith('win'):
    extensions = [
        Extension(
            "*",
            ["foreground_estimation/*.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=['/openmp'],
        )
    ]
else:
    extensions = [
        Extension(
            "*",
            ["foreground_estimation/*.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'],
        ),
    ]
setup(
    name="foreground_estimation",
    ext_modules=cythonize(extensions, compiler_directives={'language_level' : "3"}),
)