#!/usr/bin/env python3
"""
Build script for Monad Core pybind11 module.
Usage: python setup_pybind.py build_ext --inplace
"""

import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class get_pybind_include:
    """Helper class to determine the pybind11 include path."""
    def __str__(self):
        import pybind11
        return pybind11.get_include()

class get_eigen_include:
    """Helper class to determine the Eigen include path."""
    def __str__(self):
        # Try common locations
        candidates = [
            "C:/eigen",              # Windows common
            "C:/eigen3",
            "3rdparty/eigen",        # Local download
            os.path.expanduser("~/eigen"),
            "/usr/include/eigen3",   # Linux
            "/usr/local/include/eigen3",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        # Fallback: try to use conda/pip installed eigen
        import sysconfig
        site_packages = sysconfig.get_path('purelib')
        eigen_conda = os.path.join(site_packages, "..", "Library", "include", "eigen3")
        if os.path.exists(eigen_conda):
            return eigen_conda
        raise RuntimeError("Eigen3 not found. Install with: pip install eigen or download to C:/eigen")

# Source files
ext_modules = [
    Extension(
        'monad.monad_core',  # Will be importable as monad.monad_core
        sources=['src/python_bindings.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_eigen_include(),
            'src',
            'src/grid',
            'src/kernel',
            'src/solver',
            'src/aggregator',
            'src/ssj',
            'src/blocks',
            'src/analysis',
            'src/experiments',
            'src/io',
        ],
        language='c++',
        extra_compile_args=['/std:c++17', '/utf-8', '/O2', '/EHsc'] if sys.platform == 'win32' 
                      else ['-std=c++17', '-O3', '-fPIC'],
    ),
]

setup(
    name='monad_core',
    version='2.0.0',
    author='Monad Team',
    description='Monad Engine Core - C++ SSJ Solvers for HANK/RANK Models',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=['pybind11>=2.6', 'numpy'],
)
