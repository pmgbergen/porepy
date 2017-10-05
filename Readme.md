[![Build Status](https://travis-ci.org/pmgbergen/porepy.svg?branch=develop)](https://travis-ci.org/pmgbergen/porepy) [![Coverage Status](https://coveralls.io/repos/github/pmgbergen/porepy/badge.svg?branch=develop)](https://coveralls.io/github/pmgbergen/porepy?branch=develop)

# PorePy: A Simulation Tool for Fractured and Deformable Porous Media written in Python.
PorePy currently has the following distinguishing features:
- General grids in 2d and 3d, as well as mixed-dimensional grids defined by intersecting fracture networks.
- Support for analysis, visualization and gridding of fractured domains.
- Discretization of flow and transport, using finite volume methods and virtual finite elements.
- Discretization of elasticity and poro-elasticity, using finite volume methods.

PorePy is developed by the [Porous Media Group](http://pmg.b.uib.no/) at the University of Bergen, Norway. The software is developed under projects funded by the Reserach Council of Norway and Statoil.

# Installation
PorePy depends on `numpy`, `scipy` and `networkx`, and (for the moment) also on `meshio`, `sympy` and `matplotlib`. The latter packages may be droped / changed later. To install (on Linux, probably also OSX), use

	pip install porepy

Installation by pip on Windows may cause problems with buliding the requirements (`numpy` etc). Intallation with conda is recommended; a Conda distribution will be added shortly.

For more detailed install instructions, including how to access GMSH (for meshing), see Install.md.

PorePy is developed under Python 3, but should also be compatible with Python 2.7.

# From source
To get the most current version, install from github:

	git clone https://github.com/pmgbergen/porepy.git

	cd porepy

	pip install -r requirements.txt

	pip install .

# (Semi-) Optional packages
To function optimally, PorePy should have access to the pypi packages:
*  `triangle` (for meshing of fractured domains) and `pymetis` (for mesh partitioning). These will be installed on Linux (not so on Windows, to avoid installation issues for the core package in the case where no C compiler is available).
* Some computationally expensive methods can be accelerated with `Cython` or `Numba`. Cython is automatically installed on many Linux systems, if not, use pip or conda. Numba can be installed using `conda`.
* Visualization by either matplotlib or (preferrable for larger problems) vtk/paraview. To dump data to paraview, a vtk filter must be available; the only solution we have found is from the 'conda' repositories, e.g. 'conda install -c clinicalgraphics vtk=7.1.0' (note that vtk should be version 7.0.0 or later, hence not the official channel)
* Meshing currently by [gmsh](http://gmsh.info/doc/texinfo/gmsh.html) .`triangle` and `tetgen` should be added in the not too distant future.

# Testing
To test build locally

	pip install -r requirements-dev.txt
	
	python setup.py test

# Getting started
Confer the [tutorials](https://github.com/pmgbergen/porepy/tree/develop/tutorials). Also see unit tests.

# Problems
Create an [issue](https://github.com/pmgbergen/porepy/issues)


