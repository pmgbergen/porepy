"""   PorePy.

Root directory for the PorePy package. Contains the following sub-packages:

fracs: Meshing, analysis, manipulations of fracture networks.

grids: Grid class, constructors, partitioning, etc.

numerics: Discretization schemes.

params: Physical parameters, constitutive laws, boundary conditions etc.

utils: Utility functions, array manipulation, computational geometry etc.

viz: Visualization; paraview, matplotlib.

"""

__version__ = '0.2.3'

#------------------------------------
# Simplified namespaces. The rue of thumb is that classes and modules that a
# user can be exposed to should have a shortcut here. Borderline cases will be
# decided as needed

__all__ = []

# Numerics
from porepy.numerics.mixed_dim.coupler import Coupler

# Control volume, elliptic
from porepy.numerics.fv.mpsa import Mpsa
from porepy.numerics.fv.tpfa import Tpfa, TpfaMixedDim
from porepy.numerics.fv.mpfa import Mpfa, MpfaMixedDim
from porepy.numerics.fv.biot import Biot
from porepy.numerics.fv.source import Integral, IntegralMixedDim

# Virtual elements, elliptic
from porepy.numerics.vem.vem_dual import DualVEM, DualVEMMixedDim
from porepy.numerics.vem.vem_source import DualSource, DualSourceMixedDim
from porepy.numerics.elliptic import DualEllipticModel

# Transport related
from porepy.numerics.fv.transport.upwind import Upwind, UpwindMixedDim
from porepy.numerics.fv.mass_matrix import MassMatrix, InvMassMatrix

# Physical models
from porepy.numerics.elliptic import EllipticModel, EllipticDataAssigner
from porepy.numerics.parabolic import ParabolicModel, ParabolicDataAssigner
from porepy.numerics.compressible import SlightlyCompressibleModel, SlightlyCompressibleDataAssigner
from porepy.numerics.mechanics import StaticModel, StaticDataAssigner
# Time steppers
from porepy.numerics.time_stepper  import Implicit, BDF2, CrankNicolson, Explicit

# Linear solver library
from porepy.numerics.linalg.linsolve import Factory as LSFactory

#Grid
from porepy.grids.grid import Grid
from porepy.grids.grid_bucket import GridBucket
from porepy.grids.structured import CartGrid, TensorGrid
from porepy.grids.simplex import TriangleGrid, TetrahedralGrid
from porepy.grids.simplex import StructuredTriangleGrid, StructuredTetrahedralGrid
from porepy.grids.point_grid import PointGrid

# Fractures
from porepy.fracs.fractures import Fracture, EllipticFracture, FractureNetwork

# Parameters
from porepy.params.bc import BoundaryCondition
from porepy.params.tensor import SecondOrderTensor, FourthOrderTensor
from porepy.params.data import Parameters

# Visualization
from porepy.viz.exporter import Exporter
from porepy.viz.plot_grid import plot_grid
from porepy.viz.fracture_visualization import plot_fractures

# Modules
from porepy.utils import comp_geom as cg
from porepy.fracs import meshing, importer, extrusion
from porepy.grids import structured, simplex, coarsening
from porepy.params import units
from porepy.numerics.fv import fvutils
from porepy.utils import error


