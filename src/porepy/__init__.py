"""   PorePy.

Root directory for the PorePy package. Contains the following sub-packages:

fracs: Meshing, analysis, manipulations of fracture networks.

grids: Grid class, constructors, partitioning, etc.

numerics: Discretization schemes.

params: Physical parameters, constitutive laws, boundary conditions etc.

utils: Utility functions, array manipulation, computational geometry etc.

viz: Visualization; paraview, matplotlib.

"""

__version__ = '0.1.0'

#------------------------------------
# Simplified namespaces. The rue of thumb is that classes and modules that a
# user can be exposed to should have a shortcut here. Borderline cases will be
# decided as needed

# Numerics
# Control volume, elliptic
from porepy.numerics.fv.mpsa import Mpsa
from porepy.numerics.fv.tpfa import Tpfa, TpfaMixDim
from porepy.numerics.fv.mpfa import Mpfa, MpfaMixDim
from porepy.numerics.fv.biot import Biot

# Virtual elements, elliptic
from porepy.numerics.vem.dual import DualVEM, DualVEMMixDim
from porepy.numerics.vem.hybrid import HybridDualVEM

# Transport related
from porepy.numerics.fv.transport.upwind import Upwind, UpwindMixDim
from porepy.numerics.fv.mass_matrix import MassMatrix, InvMassMatrix

# Model related
from porepy.numerics.pde_solver import AbstractSolver, Implicit, BDF2, Explicit
from porepy.numerics.pde_solver import CrankNicolson
from porepy.numerics.darcy_and_transport import DarcyAndTransport
from porepy.numerics.elliptic import Elliptic, EllipticData
from porepy.numerics.parabolic import ParabolicProblem, ParabolicData

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
