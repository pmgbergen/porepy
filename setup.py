#!/usr/bin/env python


from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup


def read(fname):
        return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='porepy',
    version='0.0.1',
    licence='GPL',
    keywords=['porous media simulation fractures deformable'],
    author='Runar Berge, Alessio Fumagalli, Eirik Keilegavlen and Ivar Stefansson',
    maintainer='Eirik Keilegavlen',
    maintainer_email='Eirik.Keilegavlen@uib.no',
    platforms=['Linux', 'Windows'],
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')]
)


