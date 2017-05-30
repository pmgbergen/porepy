# Virtual element methods for Darcy problems

In this part the basic tools to create the matrix and right-had side for a Darcy problem.

We have:
* mixed virtual element method [dual.py](dual.py)
* hybridized mixed virtual element method [hybrid.py](hybrid.py)
* hybrid-dimensional coupler for the mixed virtual element method [dual_coupling.py](dual_coupling.py)

Dimensions available: 0d, 1d, 2d, 3d. In 1d and 2d also not on the (x,y) plane.

Examples are reported in [examples](examples) folder.
