# Install PorePy from source
If you do not want to use the Docker image as described in the Readme, instructions for installation on Linux are given below.
Get the most current version from GitHub:

    git clone https://github.com/pmgbergen/porepy.git

    cd porepy

The default branch is `develop`. To get the stable version:

    git checkout main

Install PorePy and the needed requirements

    pip install .[development,testing]

or for editable installs into the user directory:

    pip install --user -e .[development,testing]

## Installation on Windows
Installation on Windows is currently (Spring 2021) rather easy, following the above instructions. The dependencies should be installed using either `conda` or the Intel python distribution and then `pip install porepy` from source. 

Please note that running PorePy on Windows is not officially supported, in the sense that we may introduce updates to the code or new dependencies which may break Windows compatibility. 

## Installation on Mac
Install on Mac should also be straightforward if using `conda`. Similar to Windows, compatibility with Mac is not officially supported.

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

## Paraview
The bulk of the visualization in 3D relies on the visualization toolkit [(VTK)](https://github.com/Kitware/VTK) and a visualization client, which [Paraview](https://www.paraview.org/) is likely the most widely used.
