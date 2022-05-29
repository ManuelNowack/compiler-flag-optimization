import setuptools
from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import sys

__version__ = "0.0.1"

cpp_args = ['-std=c++11', '-march=native', "-O3", "-L.", "-ldsft", "-pthread", "-fPIC", "-L./cpp", "-ldsft"]

ext_modules = [
    Pybind11Extension("_powerset_enum",
        ["ssftapprox/powerset_enumeration.cpp", "ssftapprox/multiset_enumeration.cpp", "ssftapprox/binding.cpp"],
        define_macros = [('VERSION_INFO', __version__)],
		extra_compile_args= ['-fopenmp'],
		extra_link_args=['-fopenmp']
        ),
    Pybind11Extension("_fit",
        ["cpp/pybindings.cpp"],
        define_macros = [('VERSION_INFO', __version__)],
        language='c++',
        extra_compile_args = cpp_args,
        extra_link_args = cpp_args
    ),
]

setup(
    name="ssftapprox-wendler-wszola",
    version=__version__,
    author="Chris Wendler & Eliza Wszola",
    author_email="chris.wendler@inf.ethz.ch",
    description="Fourier-sparse approximations of set functions.",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    install_requires=['pybind11','numpy', 
                      'cvxpy==1.1.11',
                      'scikit-learn',
                      'scipy',
                      'matplotlib',
                      'tqdm'],
    packages=setuptools.find_packages()
)
