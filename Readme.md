![build](https://github.com/pmgbergen/porepy/workflows/Build%20Test/badge.svg)
[![DOI](https://zenodo.org/badge/89228838.svg)](https://zenodo.org/badge/latestdoi/89228838)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# PorePy: A Simulation Tool for Fractured and Deformable Porous Media written in Python.
PorePy currently has the following distinguishing features:
- General grids in 2d and 3d, as well as mixed-dimensional grids defined by intersecting fracture networks.
- Automatic gridding for complex fracture networks in 2d and 3d.
- Discretization of mixed-dimensional multi-physics processes:
    - Finite volume and mixed and virtual finite element methods for flow
    - Finite volume methods for transport and thermo-poroelasticity
    - Deformation of existing fractures treated as a frictional contact problem
    - Some functionality for fracture propagation along existing grid lines


For more information, see the [tutorials](tutorials) and the [Wiki](https://github.com/pmgbergen/porepy/wiki).

PorePy is developed by the [Porous Media Group](http://pmg.b.uib.no/) at the University of Bergen, Norway. The software is developed under projects funded by the Research Council of Norway, the European Research Council and Equinor.

# Citing
If you use PorePy in your research, we ask you to cite the following publication

Keilegavlen, E., Berge, R., Fumagalli, A., Starnoni, M., Stefansson, I., Varela, J., & Berre, I. PorePy: an open-source software for simulation of multiphysics processes in fractured porous media. Computational Geosciences,  25, 243–265 (2021), [doi:10.1007/s10596-020-10002-5](https://doi.org/10.1007/s10596-020-10002-5)

Runscripts for most, if not all, papers that uses porepy is available at [here](./Papers.md).
Note that you may have to revert to an older version of PorePy to run the examples (we try to keep the runscripts updated, but sometime fail to do so, for various reasons).

# Installation from source
Install instructions can be found here [Install](https://github.com/pmgbergen/porepy/blob/develop/Install.md).
Note that there are a few simple but non-obvious steps in the installation, so please read the entire document before sending questions.

PorePy is developed under Python >=3.8.

To get the most current version, install from github:

    git clone https://github.com/pmgbergen/porepy.git

    cd porepy

To get the stable (though not very frequently updated) version:
    git checkout master

Install
    pip install -r requirements.txt

Finally to install PorePy

    pip install .

or for editable installs into the user directory:

    pip install --user -e .



# (Semi-) Optional packages
To function optimally, PorePy should have access to some more packages:
*  `pymetis` (for mesh partitioning).
* Some computationally expensive methods can be accelerated with `Numba`.
* We use `shapely` for certain geometry-operations.
* Meshing: currently by [gmsh](http://gmsh.info/doc/texinfo/gmsh.html).

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


