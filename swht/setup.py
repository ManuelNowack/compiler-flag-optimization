from setuptools import setup, Extension
from pathlib import Path
from os import chdir

# Set paths
root = Path(__file__).resolve().parent
src_folder = root / "src" / "python_module"
build_folder = root / "build"
build_src_folder = build_folder / "src"

# Move to project root
start_dir = Path.cwd()
chdir(root)

# Fetch kernel lib name
with open(build_src_folder / "libname.log", 'r') as name_log:
    lib_files = name_log.read().splitlines()
lib_dir = lib_files.pop(0)

# Build package
setup(
    name = "swht",
    version = "1.3.1",
    description = "A fast sparse Walsh-Hadamard transform.",
    url = "https://github.com/kaikoveritch/swht",
    author = "Kaïko Bonstein",
    packages = ["swht"],
    ext_modules = [
        Extension(
            name = "swht.swht",
            sources = ["src/python_module/swhtmodule.cpp"],
            include_dirs = ["src", "src/utils", "build/include"],
            library_dirs = ["build/src"],
            libraries = ["swht_cpp"],
            runtime_library_dirs = [lib_dir],
            extra_compile_args = ['-std=c++17', '-O3', '-march=native']
        )
    ],
    package_dir = {
        'swht': src_folder
    },
    package_data = {
        'swht': ['__init__.pyi', '__init__.py']
    }
)

# Return to original dir
chdir(start_dir)
