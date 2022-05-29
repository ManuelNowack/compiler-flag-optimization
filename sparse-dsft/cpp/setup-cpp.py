# Install with command:
# > python3 setup.py build_ext -i

import setuptools
from setuptools import setup
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext

cpp_args = ['-std=c++11', '-march=native', "-O3", "-L.", "-ldsft", "-pthread", "-fPIC"]
__version__ = "0.0.1"


ext_modules = [
    Pybind11Extension("_fit",
        ["pybindings.cpp"],
        #~ include_dirs=["src"],
        define_macros = [('VERSION_INFO', __version__)],
        language='c++',
        extra_compile_args = cpp_args,
        extra_link_args = cpp_args
    ),
]

setup(
    name="fit-test",
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    packages=setuptools.find_packages()
)
