# Setting up a PorePy environment
Installation of PorePy itself should be straightforward, following the instructions in Readme.md.

To get the code fully working requires a few more steps, as described below.

## Installation on Linux
Instructions are found on the GitHub webpage.

## Intall on Windows
The recommended solution for Windows is to use VirtualBox with a Linux image, or equivalent options.

If you do not want to use VirtualBox, we recommended 
to install the dependencies using `conda`, and then `pip install porepy`, preferrably installing from source.
Most likely, parts of PorePy will not work on Windows due to missing libraries etc. This is not fully clear.

## Installation on Mac
Install on Mac is possible, but may be a bit complicated. We have little experience with this.


## Docker
A Docker image is available, but should be considered experimental for now.

# Setting up GMSH
PorePy depends on `GMSH` for meshing of fractured domains. 
Our exprience is that version 4 of Gmsh is much improved compared to earlier versions, in particular for complex geometries.

To make this work, you need gmsh installed on your system, and PorePy needs to know where to look for it.
For Linux users: Gmsh is available through apt-get, but be sure that the version available is >=4.0. If this is not possible, do the manual install below.

Manual install:
First, visit the [Gmsh webpage](http://gmsh.info) and download a suitable version. 
Extract, and move the binary (probably located in the subfolder gmsh-x.x.x-Linux/bin or similar) to whereever you prefer.


The location of the gmsh file is specific for each user's setup, and is therefore not included in the library. 
Instead, to get the path to the gmsh executable, PorePy assumes there is a file called `porepy_config.py` somewhere in `$PYTHONPATH`. 
So, open a file called `porepy_config.py`, and place the line
```python
config = {'gmsh_path': 'path/to/gmsh/executable'} # example config = {'gmsh_path': '/usr/bin/gmsh'}
```
Note that the path should be set as a string. To read more about the config system, see `porepy.utils.read_config.py`.

# Point-in-polyhedron test
Some functionality depends on a point-in-polyhedron test. The PorePy function that provides this is located in pp.utils.comp_geom.is_inside_polyhedron(). 
The only robust test, with reasonable installation, we are aware of is available [here](https://github.com/mdickinson/polyhedron/blob/master/polyhedron.py). Unfortunately, the file is not available through pip or conda. Instead, download the file, and place it somewhere in PYTHONPATH with the name robust_point_in_polyhedron.py. The PorePy function is_inside_polyhedron() then acts as a wrapper around this external package.

# Other packages
Others libraries that should be installed found in the file requirements.txt

