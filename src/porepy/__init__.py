"""   PorePy.

Root directory for the PorePy package. Contains the following sub-packages:

fracs: Meshing, analysis, manipulations of fracture networks.

grids: Grid class, constructors, partitioning, etc.

numerics: Discretization schemes.

params: Physical parameters, constitutive laws, boundary conditions etc.

utils: Utility functions, array manipulation, computational geometry etc.

viz: Visualization; paraview, matplotlib.

"""

__version__ = '0.2.2'

#------------------------------------
# Simplified namespaces. The rue of thumb is that classes and modules that a
# user can be exposed to should have a shortcut here. Borderline cases will be
# decided as needed

# Numerics
# Control volume, elliptic
from porepy.numerics.fv.mpsa import Mpsa
from porepy.numerics.fv.tpfa import Tpfa, TpfaMixedDim
from porepy.numerics.fv.mpfa import Mpfa, MpfaMixedDim
from porepy.numerics.fv.biot import Biot

# Virtual elements, elliptic
from porepy.numerics.vem.vem_dual import DualVEM, DualVEMMixedDim
from porepy.numerics.vem.hybrid import HybridDualVEM

# Transport related
from porepy.numerics.fv.transport.upwind import Upwind, UpwindMixedDim
from porepy.numerics.fv.mass_matrix import MassMatrix, InvMassMatrix

# Grids
from porepy.grids.structured import CartGrid, TensorGrid
from porepy.grids.simplex import TriangleGrid, TetrahedralGrid
from porepy.grids.simplex import StructuredTriangleGrid, StructuredTetrahedralGrid

# Fractures
from porepy.fracs.fractures import Fracture, EllipticFracture, FractureNetwork
from porepy.fracs import meshing, simplex, structured

# Parameters
from porepy.params import units
from porepy.params.bc import BoundaryCondition
from porepy.params import bc
#from porepy.params.data import Parameters
