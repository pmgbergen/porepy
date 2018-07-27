[![Build Status](https://travis-ci.org/pmgbergen/porepy.svg?branch=develop)](https://travis-ci.org/pmgbergen/porepy) [![Coverage Status](https://coveralls.io/repos/github/pmgbergen/porepy/badge.svg?branch=develop)](https://coveralls.io/github/pmgbergen/porepy?branch=develop)
[![DOI](https://zenodo.org/badge/89228838.svg)](https://zenodo.org/badge/latestdoi/89228838)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# PorePy: A Simulation Tool for Fractured and Deformable Porous Media written in Python.
PorePy currently has the following distinguishing features:
- General grids in 2d and 3d, as well as mixed-dimensional grids defined by intersecting fracture networks.
- Support for analysis, visualization and gridding of fractured domains.
- Discretization of flow and transport, using finite volume methods and virtual finite elements.
- Discretization of elasticity and poro-elasticity, using finite volume methods.

PorePy is developed by the [Porous Media Group](http://pmg.b.uib.no/) at the University of Bergen, Norway. The software is developed under projects funded by the Research Council of Norway and Statoil.

# Reproduce results from papers and preprints
Runscripts for most, if not all, papers that uses porepy is available at [here](https://github.com/pmgbergen/porepy/tree/develop/examples/papers).
Note that you may have to revert to an older version of PorePy to run the examples (we try to keep the runscripts updated, but sometime fail to do so, for various reasons).

# Citing
If you use PorePy in your research, we ask you to cite the following publication

E. Keilegavlen, A. Fumagalli, R. Berge, I. Stefansson, I. Berre: PorePy: An Open-Source Simulation Tool for Flow and Transport in Deformable Fractured Rocks. [arXiv:1712.00460](https://arxiv.org/abs/1712.00460)

# Installation
PorePy depends on `numpy`, `scipy` and `networkx`, and (for the moment) also on `meshio`, `sympy` and `matplotlib`. The latter packages may be droped / changed later. To install (on Linux, probably also OSX), use

    pip install porepy

We recommend installing from source (see below), rather than pulling from pypi. Installation by pip on Windows may cause problems with buliding the requirements (`numpy` etc) unless conda is used.

For more detailed install instructions, including how to access GMSH (for meshing), see
[Install](https://github.com/pmgbergen/porepy/blob/develop/Install.md).

PorePy is developed under Python 3. It should also be compatible with Python 2.7, however, apart from unit testing, it is not being used with Python 2, so be cautious.

# From source
To get the most current version, install from github:

    git clone https://github.com/pmgbergen/porepy.git

    cd porepy

    pip install -r requirements.txt

Finally to install PorePy

    pip install .

or for editable installs into the user directory:

    pip install --user -e .


# Using Docker
A docker image is available, courtesy of Davide Baroli. To pull the docker.io/pmgbergen/porepylib with tag py27 image from cloud infrastructure:
```bash
>  docker pull docker.io/pmgbergen/porepylib:py27
```
Docker will pull the py27 tag of the image pmgbergen/porepylib from docker.io based on python 2.7. The download is around 4.085 GB. The  image is a great place to start experimenting with porepy and includes all dependencies already compiled for you.
Once the download is complete you can start porepy for the first time. Just run:
```bash
>  docker run -ti  docker.io/pmgbergen/porepy:py27
```
The image is auto-deployed at each push in the branch "develop".

For more details see the [docker md](./Docker.md).

# (Semi-) Optional packages
To function optimally, PorePy should have access to the pypi packages:
*  `pymetis` (for mesh partitioning). Will be installed on Linux (not so on Windows, to avoid installation issues for the core package in the case where no C compiler is available).
* Some computationally expensive methods can be accelerated with `Cython` or `Numba`. Cython is automatically installed on many Linux systems, if not, use pip or conda. Numba can be installed using `conda`.
* Visualization by either matplotlib or (preferrable for larger problems) vtk/paraview. To dump data to paraview, a vtk filter must be available; the only solution we have found is from the 'conda' repositories, e.g. 'conda install -c clinicalgraphics vtk=7.1.0' (note that vtk should be version 7.0.0 or later, hence not the official channel)
* Meshing: currently by [gmsh](http://gmsh.info/doc/texinfo/gmsh.html). For its configuration see [Install](https://github.com/pmgbergen/porepy/blob/develop/Install.md).

# Testing
To test build locally, the second command requires gmsh (see [Install](https://github.com/pmgbergen/porepy/blob/develop/Install.md))

    pip install -r requirements-dev.txt

    python setup.py test

# Getting started
Confer the [tutorials](https://github.com/pmgbergen/porepy/tree/develop/tutorials). Also see unit tests.

# Problems
Create an [issue](https://github.com/pmgbergen/porepy/issues)

# License
See [license md](./LICENSE.md).


