![Pytest](https://github.com/pmgbergen/porepy/actions/workflows/run-pytest.yml/badge.svg)
![Pytest including slow](https://github.com/pmgbergen/porepy/actions/workflows/run-pytest-all.yml/badge.svg)
![Mypy, black, isort, flake8](https://github.com/pmgbergen/porepy/actions/workflows/run-static-checks.yml/badge.svg)
[![DOI](https://zenodo.org/badge/89228838.svg)](https://zenodo.org/badge/latestdoi/89228838)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


# PorePy: A Simulation Tool for Fractured and Deformable Porous Media written in Python.
PorePy currently has the following distinguishing features:
- General grids in 2d and 3d, as well as mixed-dimensional grids defined by, possibly intersecting, fracture networks.
- Automatic gridding for complex fracture networks in 2d and 3d.
- Discretization of mixed-dimensional multi-physics processes:
    - Finite volume and mixed and virtual finite element methods for flow
    - Finite volume methods for transport and thermo-poroelasticity
    - Deformation of existing fractures treated as a frictional contact problem



PorePy is developed by the Porous Media Group at the University of Bergen, Norway. The software is developed under projects funded by the Research Council of Norway, the European Research Council and Equinor.


# Installation
We recommend using PorePy through a Docker image. The development and stable versions can be obtained, respectively, by 

    docker pull porepy/dev

and

    docker pull porepy/stable

Instructions to install from source can be found [here](https://github.com/pmgbergen/porepy/blob/develop/Install.md).

PorePy is developed under Python >=3.11.

# Getting started
Confer the [tutorials](https://github.com/pmgbergen/porepy/tree/develop/tutorials). 
Documentation can be found [here](https://pmgbergen.github.io/porepy/html/docsrc/porepy/porepy.html) (still under construction).

# Citing
If you use PorePy in your research, we ask you to cite the following publication

Keilegavlen, E., Berge, R., Fumagalli, A., Starnoni, M., Stefansson, I., Varela, J., & Berre, I. PorePy: an open-source software for simulation of multiphysics processes in fractured porous media. Computational Geosciences,  25, 243–265 (2021), [doi:10.1007/s10596-020-10002-5](https://doi.org/10.1007/s10596-020-10002-5)


# Contributions
For guidelines on how to contribute to PorePy, see [here](./CONTRIBUTING.md)

# License
See [license md](./LICENSE.md).
