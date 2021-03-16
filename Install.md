# Setting up a PorePy environment
Installation of PorePy itself should be straightforward, following the instructions in Readme.md.

To get the code fully working requires a few more steps, as described below. Please read the full instructions (including the part on point_in_polyhedron) before reporting an issue on installation.

## Installation on Linux
Instructions are found on the GitHub webpage. Others libraries that should be installed found in the file `pyproject.toml`.

## Intall on Windows
Installation on Windows is currently (Spring 2021) rather easy, following thes Linux instructions. The dependencies should be installed  using either `conda` or the Intel python distribution and then `pip install porepy` from source. 

Please note that running PorePy on Windows is not officially supported, in the sense that we may introduce updates to the code or new dependencies which may break Windows compatibility. 

## Installation on Mac
Install on Mac should also be straightforward if using `conda`. Similar to Windows, compatibility with Mac is not officially supported.

# Setting up ancillary requirements

Some of these requirements aren't strictly necessary to run porepy, however, many workflows require these ancillary components.

## Optional python packages for faster runtimes

Several computationally expensive methods can be accelerated with Cython or Numba. Shapely is used for certain geometry-operations. To install these packages, an administrative level pip install should be sufficient, i.e. through `sudo pip install cython numba shapely`

### Metis & pymetis
The metis package is used to partition meshes. In order to use this package, you must install metis from George Karypis
http://glaros.dtc.umn.edu/gkhome/metis/metis/overview. In linux, this should be available through apt-get or apt install:
`sudo apt install metis`. Windows and Mac OS users should try using the docker container. Once metis is installed, pymetis can be installed through PiPy `pip install pymetis`.

If the apt install doesn't work, try the followig:
- Download the Metis library source files from: http://glaros.dtc.umn.edu/gkhome/metis/metis/download
- Make sure you have cmake and build-essentials installed via `sudo apt install cmake build-essentials`
- Run the following from the Metis source file directory:
    make config
    make
- You should now have Metis installed on your machine. Test this by running mpmetis -help.
- Make sure you have pybind11 installed via `pip install pybind11`
- Install pymetis via `pip install pymetis`


## Point-in-polyhedron test
Some functionality depends on a point-in-polyhedron test. The PorePy function that provides this is located in pp.geometry.geometry_property_checks.point_in_polyhedron(). The only robust test, with reasonable installation, we are aware of is available [here](https://github.com/mdickinson/polyhedron/blob/master/polyhedron.py). Unfortunately, the file is not available through pip or conda. Instead, download the file (polyhedron.py), and place it somewhere in the PYTHONPATH with the name 'robust_point_in_polyhedron.py'. The PorePy function point_in_polyhedron() then acts as a wrapper around this external package, which provides a `Polyhedron` class.

## Paraview
The bulk of the visualization in 3D relies on the visualization toolkit [(VTK)](https://github.com/Kitware/VTK) and a visualization client, which [Paraview](https://www.paraview.org/) is likely the most widely used.
