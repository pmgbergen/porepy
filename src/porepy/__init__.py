"""   PorePy.

Root directory for the PorePy package. Contains the following sub-packages:

fracs: Meshing, analysis, manipulations of fracture networks.

grids: Grid class, constructors, partitioning, etc.

numerics: Discretization schemes.

params: Physical parameters, constitutive laws, boundary conditions etc.

utils: Utility functions, array manipulation, computational geometry etc.

viz: Visualization; paraview, matplotlib.

"""

__version__ = "0.4"

# ------------------------------------
# Simplified namespaces. The rue of thumb is that classes and modules that a
# user can be exposed to should have a shortcut here. Borderline cases will be
# decided as needed

__all__ = []

# Numerics
# Control volume, elliptic
from porepy.numerics.fv.mpsa import Mpsa, FracturedMpsa
from porepy.numerics.fv.tpfa import Tpfa, TpfaMixedDim
from porepy.numerics.fv.mpfa import Mpfa, MpfaMixedDim
from porepy.numerics.fv.biot import Biot
from porepy.numerics.fv.source import Integral, IntegralMixedDim

# Virtual elements, elliptic
from porepy.numerics.vem.vem_dual import DualVEM, DualVEMMixedDim
from porepy.numerics.vem.vem_source import DualSource, DualSourceMixedDim
from porepy.numerics.elliptic import DualEllipticModel

# Finite elements, elliptic
from porepy.numerics.fem.p1 import P1, P1MixedDim
from porepy.numerics.fem.source import P1Source, P1SourceMixedDim
from porepy.numerics.fem.mass_matrix import P1MassMatrix, P1MassMatrixMixedDim
from porepy.numerics.fem.rt0 import RT0, RT0MixedDim


# Transport related
from porepy.numerics.fv.transport.upwind import Upwind, UpwindMixedDim, UpwindCoupling
from porepy.numerics.fv.mass_matrix import MassMatrix, MassMatrixMixedDim
from porepy.numerics.fv.mass_matrix import InvMassMatrix, InvMassMatrixMixedDim

# Physical models
from porepy.numerics.elliptic import EllipticModel, EllipticDataAssigner
from porepy.numerics.parabolic import ParabolicModel, ParabolicDataAssigner
from porepy.numerics.compressible import (
    SlightlyCompressibleModel,
    SlightlyCompressibleDataAssigner,
)
from porepy.numerics.mechanics import StaticModel, StaticDataAssigner
from porepy.numerics.fracture_deformation import (
    FrictionSlipModel,
    FrictionSlipDataAssigner,
)

# Time steppers
from porepy.numerics.time_stepper import Implicit, Explicit

# Grids
from porepy.grids.grid import Grid
from porepy.grids.grid_bucket import GridBucket
from porepy.grids.structured import CartGrid, TensorGrid
from porepy.grids.simplex import TriangleGrid, TetrahedralGrid
from porepy.grids.simplex import StructuredTriangleGrid, StructuredTetrahedralGrid
from porepy.grids.point_grid import PointGrid
from porepy.grids.mortar_grid import MortarGrid, BoundaryMortar

# Fractures
from porepy.fracs.fractures import Fracture, EllipticFracture, FractureNetwork

# Parameters
from porepy.params.bc import (
    BoundaryCondition,
    BoundaryConditionVectorial,
    BoundaryConditionNode,
)
from porepy.params.tensor import SecondOrderTensor, FourthOrderTensor
from porepy.params.data import Parameters
from porepy.params.rock import UnitRock, Shale, SandStone, Granite

# Visualization
from porepy.viz.exporter import Exporter
from porepy.viz.plot_grid import plot_grid
from porepy.viz.fracture_visualization import plot_fractures

# Modules
from porepy.utils import permutations
from porepy.utils import comp_geom as cg
from porepy.fracs import utils as frac_utils
from porepy.fracs import meshing, importer, extrusion
from porepy.grids import structured, simplex, coarsening, partition
from porepy.params.units import *
from porepy.numerics.fv import fvutils
from porepy.utils import error
