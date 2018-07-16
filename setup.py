#!/usr/bin/env python

import os.path
from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup

# ----------------------  Cython compilation
# Build cython extensions as part of setup. Based on
# https://stackoverflow.com/questions/4505747/how-should-i-structure-a-python-package-that-contains-cython-code
USE_CYTHON = True
from distutils.core import setup
from distutils.extension import Extension

if USE_CYTHON:
    try:
        # Trick from
        # https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
        from Cython.Distutils import build_ext as _build_ext

        class build_ext(_build_ext):
            def finalize_options(self):
                _build_ext.finalize_options(self)
                # Prevent numpy from thinking it is still in its setup process:
                __builtins__.__NUMPY_SETUP__ = False
                import numpy

                self.include_dirs.append(numpy.get_include())

    except:
        # Cython is apparently not available on the system. Make do without it
        USE_CYTHON = False

cmdclass = {}
ext_modules = []

# New Cython modules must be registered here
if USE_CYTHON:
    ext_modules += [
        Extension(
            "porepy.numerics.fv.cythoninvert",
            ["src/porepy/numerics/fv/invert_diagonal_blocks.pyx"],
        )
    ]
    cmdclass.update({"build_ext": build_ext})

# --------------------- End of cython part


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open("requirements.txt") as f:
    required = f.read().splitlines()


long_description = read("Readme.rst")

setup(
    name="porepy",
    version="0.4.0",
    license="GPL",
    keywords=["porous media simulation fractures deformable"],
    author="Runar Berge, Alessio Fumagalli, Eirik Keilegavlen and Ivar Stefansson",
    install_requires=required,
    description="Simulation tool for fractured and deformable porous media",
    long_description=long_description,
    maintainer="Eirik Keilegavlen",
    maintainer_email="Eirik.Keilegavlen@uib.no",
    platforms=["Linux", "Windows"],
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[
        os.path.splitext(os.path.basename(path))[0] for path in glob("src/*.py")
    ],
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
