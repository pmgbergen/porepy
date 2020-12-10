
# Install miniconda with:
# - intepython3_core=2020.1
# - do: apt-get update && apt-get install g++
# - use intel as priority 1. channel
FROM intelpython/intelpython3_core

# Add conda-forge to dependencies
# 1. intel, 2. conda-forge, 3. defaults
RUN conda config --append channels conda-forge
RUN conda config --append channels defaults

# Install porepy dependencies
# Second line is testing dependencies. Can be removed.
RUN conda install meshio networkx sympy matplotlib cython future shapely \
	pytest pytest-cov pytest-runner black mypy flake8 \

# Install vtk by pip (vtk=9.0.1 not supported on conda for unknown reason)
RUN pip install vtk

# Install gmsh by pip (python doesn't find gmsh from conda for unknown reason)
# Install gmsh system dependencies
RUN apt-get install libgl1 libglu1 libxcursor1 libxft2 libxinerama1 -y
RUN pip install gmsh

# matplotlib dependency
RUN conda install tornado

# Install conda-build to use 'conda develop'
RUN conda install conda-build

# PyPardiso
RUN conda install -c haasad pypardiso

# -- Prepare installation of PorePy --
WORKDIR /home

# Install robust-point-in-polyhedron
RUN wget https://raw.githubusercontent.com/keileg/polyhedron/master/polyhedron.py
RUN mkdir poly
RUN mv polyhedron.py poly/robust_point_in_polyhedron.py
RUN conda develop poly

# porepy
RUN git clone https://github.com/pmgbergen/porepy.git
RUN conda develop porepy/src
