# Setting up a PorePy environment
Installation of PorePy itself should be straightforward, using pip.
In practice, installing from source is the preferred option to get the newest version of the code.

To get the code fully working requires a few more steps, as described below.
We also strongly recommend using a virtual environment for your PorePy install.

## Installation on Linux
Instructions are found on the GitHub webpage. Simply type `pip install porepy`.

## Intall on Windows
This is a bit more tricky, since installing the dependencies (e.g. `numpy`, `scipy`) using pip requires access to a compiler.
The recommended solution (for working with Python on Windows in general, it seems)
is to install the dependencies using `conda`, and then `pip install porepy`.
We plan to provide conda install for PorePy as well, but have not come that far yet.

## How about Mac?
Frankly, we are not sure. None of the devopers use Mac, so testing this has not been a priority.
The expectation is that it should work out nicely. If you try this, please let us know.

# Setting up GMSH
PorePy currently depends on `GMSH` for meshing of fractured domains. 
To make this work, you need gmsh installed on your system, and PorePy needs to know where to look for it.
First, visit the [Gmsh webpage](http://gmsh.info) and download a suitable version. 
Extract, and move the binary (probably located in the subfolder gmsh-x.x.x-Linux/bin or similar) to whereever you prefer.

NOTE: We have experienced that some fracture geometries are best handled by somewhat older versions of Gmsh (2.11 seems to be a good version) - newer versions are often result in error messages. Until this is resolved, we therefore recommend to use Gmsh 2.11 with PorePy.

Note to Linux users: Although Gmsh is available through the standard packaging tools, it tends to be hopelessly outdated, 
and resulted in severe issues for the fracture meshing last time we checked. Use the GMSH web page instead.


The location of the gmsh file is specific for each user's setup, and is therefore not included in the library. 
Instead, to get the path to the gmsh executable, PorePy assumes there is a file called `porepy_config.py` somewhere in `$PYTHONPATH`. 
So, open a file called `porepy_config.py`, and place the line

	config = {'gmsh_path': 'path/to/where/you/put/the/gmsh/executable/'}

Note that the path should be set as a string. To read more about the config system, see `porepy.utils.read_config.py`.

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
