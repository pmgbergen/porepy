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

For more detailed install instructions, including how to access GMSH (for meshing), see [Install](https://github.com/pmgbergen/porepy/blob/develop/LICENSE.md) .

PorePy is developed under Python 3, but should also be compatible with Python 2.7.

# From source
To get the most current version, install from github:

	git clone https://github.com/pmgbergen/porepy.git

	cd porepy

	pip install -r requirements.txt

	pip install .
	
### Docker
Alternatively, a way to run the porepy library is to use our prebuilt and high-performance Docker images.
Docker containers are extremely lightweight, secure, and are based on open standards that run on all major Linux distributions, macOS and Microsoft Windows platforms.

Install Docker for your platform by following [these instructions](https://docs.docker.com/engine/getstarted/step_one/).
If using the Docker Toolbox (macOS versions < 10.10 or Windows versions < 10), make sure you run all commands inside the Docker Quickstart Terminal.

Now we will pull the docker.io/pmgbergen/porepy image from cloud infrastructure:
```bash
>  docker pull docker.io/pmgbergen/porepy:latest
```
Docker will pull the latest tag of the image pmgbergen/porepy from docker.io. The download is around nb GB. The  image is a great place to start experimenting with porepy and includes all dependencies already compiled for you.
Once the download is complete you can start porepy for the first time. Just run:
```bash
>  docker run -ti  docker.io/pmgbergen/porepy:latest
```
To facilitate the devoloping, using the text editor,version control and other tools already installed on your computers,
it is possible to share files from the host into the container:

```bash
>  docker run -ti -v $(pwd):/home/porepy/shared  pmgbergen/porepy:latest
```
To allow the X11 forwarding in the container, on Linux system just run:

```bash
>  docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  -v $(pwd):/home/porepy/shared  pmgbergen/porepy:latest
```

For Windows system, you need to install Cygwin/X version and running the command in Cygwin terminal. While for mac system, you need to install xquartz. 
# For Developing/ enhance Docker
If you would like to compile Docker for developing porpose. You could associate this github repo with docker cloud service for deployment. Alternatively, on you own machine on terminal (Linux) or on Docker terminal (Mac/Win) you just run:
```bash
> cd  dockerfiles && docker build . --tag porepy:develop
```
The tag of your container will be "porepy" and the version "develop".

# (Semi-) Optional packages
To function optimally, PorePy should have access to the pypi packages:
*  `pymetis` (for mesh partitioning). Will be installed on Linux (not so on Windows, to avoid installation issues for the core package in the case where no C compiler is available).
* Some computationally expensive methods can be accelerated with `Cython` or `Numba`. Cython is automatically installed on many Linux systems, if not, use pip or conda. Numba can be installed using `conda`.
* Visualization by either matplotlib or (preferrable for larger problems) vtk/paraview. To dump data to paraview, a vtk filter must be available; the only solution we have found is from the 'conda' repositories, e.g. 'conda install -c clinicalgraphics vtk=7.1.0' (note that vtk should be version 7.0.0 or later, hence not the official channel)
* Meshing: currently by [gmsh](http://gmsh.info/doc/texinfo/gmsh.html) .

# Testing
To test build locally

	pip install -r requirements-dev.txt
	
	python setup.py test

# Getting started
Confer the [tutorials](https://github.com/pmgbergen/porepy/tree/develop/tutorials). Also see unit tests.

# Problems
Create an [issue](https://github.com/pmgbergen/porepy/issues)


