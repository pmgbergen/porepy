# Setting up a PorePy environment
All developers use the conda distribution of python, and we recommend that conda functionality is applied whenever possible.

Installation of PorePy itself should be straightforward, using pip.
In practice, installing from source is the preferred option to get the newest version of the code - the code hosted on pypi is not always up to date. 

To get the code fully working requires a few more steps, as described below.

## Installation on Linux
Instructions are found on the GitHub webpage. The best option is to download the source code from github, and install by `pip install porepy`.

## Intall on Windows
This is a bit more tricky, since installing the dependencies (e.g. `numpy`, `scipy`) using pip requires access to a compiler.
The recommended solution (for working with Python on Windows in general, it seems)
is to install the dependencies using `conda`, and then `pip install porepy`, preferrably installing from source.
Most likely, parts of PorePy will not work on Windows due to missing libraries etc. This is not fully clear.

## Docker
There is also a third-party option using Docker containers. For now this should be considered an experimental option.

## How about Mac?
Frankly, we are not sure. None of the devopers use Mac, so testing this has not been a priority.
The expectation is that it should work out nicely. If you try this, please let us know.

# Setting up GMSH
PorePy currently depends on `GMSH` for meshing of fractured domains. 
To make this work, you need gmsh installed on your system, and PorePy needs to know where to look for it.
First, visit the [Gmsh webpage](http://gmsh.info) and download a suitable version. 
Extract, and move the binary (probably located in the subfolder gmsh-x.x.x-Linux/bin or similar) to whereever you prefer.

NOTE: For complex fracture geometries, our experience is that Gmsh sometimes fails, but the result may depend on which gmsh version is applied. To cope with this, we have ended up switching between Gmsh versions when relevant, always trying to apply an as updated version as possible. The situation is not ideal, but for now this appears to be the best option.

Note to Linux users: Although Gmsh is available through the standard packaging tools, it tends to be hopelessly outdated, 
and resulted in severe issues for the fracture meshing last time we checked. Use the GMSH web page instead.

The location of the gmsh file is specific for each user's setup, and is therefore not included in the library. 
Instead, to get the path to the gmsh executable, PorePy assumes there is a file called `porepy_config.py` somewhere in `$PYTHONPATH`. 
So, open a file called `porepy_config.py`, and place the line
```python
config = {'gmsh_path': 'path/of/gmsh/executable'} # example config = {'gmsh_path': '/usr/bin/gmsh'}
```
Note that the path should be set as a string. To read more about the config system, see `porepy.utils.read_config.py`.

# Other packages
Others libraries that should be installed are numpy (available on both conda and pip), scipy (conda and pip), networkx (conda and pip, NOTE: version number 1.10, NOT 2 - will be updated), meshio (pip only), sympy (conda and pip). In addition libraries like cython, numba, vtk, pymetis and pyamg should be installed to get full functionality.

# Fast unique arrays
Improvements in Numpy's unique function, introduced in numpy version 1.13, can in certain cases speed up PorePy's performance immensely
(we have observed runtimes dropping by orders of magnitude, which of course also shows that the homebrewed code used when numpy is too old is less than optimal). 
If you have to use an older version of numpy, you can still use the relevant function by an ugly hack:

1. Download the relevant numpy function from [GitHub](https://github.com/numpy/numpy) (locating it can be somewhat difficult, 
try to search for 'unique' and look for a file called `numpy/lib/arraysetopts.py`. Locate the function `unique` within the file,
and copy it.
2. Open a file called `numpy_113_unique.py` somewhere in your `$PYTHONPATH`. Paste the copied unique file, and modify the method name, so that the first few lines looks like this:

	import numpy as np

	def unique_np1130(ar, return_index=False, return_inverse=False,
        	   return_counts=False, axis=None):

PorePy (specifically `porepy.utils.setmembership.unique_tol_columns`) will now access the copied function when possible.

Needless to say, using a newer version of numpy is preferable.
