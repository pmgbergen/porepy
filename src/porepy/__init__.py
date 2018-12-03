"""   PorePy.

Root directory for the PorePy package. Contains the following sub-packages:

fracs: Meshing, analysis, manipulations of fracture networks.

grids: Grid class, constructors, partitioning, etc.

numerics: Discretization schemes.

params: Physical parameters, constitutive laws, boundary conditions etc.

utils: Utility functions, array manipulation, computational geometry etc.

viz: Visualization; paraview, matplotlib.

"""

__version__ = "0.5.0"

# ------------------------------------
# Simplified namespaces. The rue of thumb is that classes and modules that a
# user can be exposed to should have a shortcut here. Borderline cases will be
# decided as needed

__all__ = []

# Numerics
# Control volume, elliptic
from porepy.numerics.fv.mpsa import Mpsa, FracturedMpsa
from porepy.numerics.fv.tpfa import Tpfa
from porepy.numerics.fv.mpfa import Mpfa
from porepy.numerics.fv.biot import Biot
from porepy.numerics.fv.source import Integral

# Virtual elements, elliptic
from porepy.numerics.vem.mvem import MVEM
from porepy.numerics.vem.vem_source import DualIntegral
from porepy.numerics.elliptic import DualEllipticModel

# Finite elements, elliptic
from porepy.numerics.fem.p1 import P1
from porepy.numerics.fem.source import P1Source
from porepy.numerics.fem.mass_matrix import P1MassMatrix
from porepy.numerics.fem.rt0 import RT0

# Mixed-dimensional discretizations and assemblers
from porepy.numerics.mixed_dim.elliptic_assembler import EllipticAssembler
from porepy.numerics.interface_laws.elliptic_interface_laws import (
    RobinCoupling,
    FluxPressureContinuity,
    RobinContact,
    StressDisplacementContinuity,
)
from porepy.numerics.mixed_dim.assembler import Assembler

# Transport related
from porepy.numerics.fv.upwind import Upwind
from porepy.numerics.interface_laws.hyperbolic_interface_laws import UpwindCoupling
from porepy.numerics.fv.mass_matrix import MassMatrix
from porepy.numerics.fv.mass_matrix import InvMassMatrix

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
from porepy.grids.fv_sub_grid import FvSubGrid
from porepy.grids.grid_bucket import GridBucket
from porepy.grids.structured import CartGrid, TensorGrid
from porepy.grids.simplex import TriangleGrid, TetrahedralGrid
from porepy.grids.simplex import StructuredTriangleGrid, StructuredTetrahedralGrid
from porepy.grids.point_grid import PointGrid
from porepy.grids.mortar_grid import MortarGrid, BoundaryMortar

# Fractures
from porepy.fracs.fractures import Fracture, EllipticFracture, FractureNetwork
from porepy.fracs.meshing import simplex_grid

# Parameters
from porepy.params.bc import (
    BoundaryCondition,
    BoundaryConditionVectorial,
    BoundaryConditionNode,
    face_on_side,
)
from porepy.params.tensor import SecondOrderTensor, FourthOrderTensor
from porepy.params.data import Parameters, initialize_data
from porepy.params.rock import UnitRock, Shale, SandStone, Granite
from porepy.params.water import Water

# Visualization
from porepy.viz.exporter import Exporter
from porepy.viz.plot_grid import plot_grid, save_img
from porepy.viz.fracture_visualization import plot_fractures, plot_wells

# Modules
from porepy.utils import permutations
from porepy.utils import comp_geom as cg
from porepy.fracs import utils as frac_utils
from porepy.fracs import meshing, importer, extrusion, mortars
from porepy.grids import structured, simplex, coarsening, partition, refinement
from porepy.numerics.fv import fvutils
from porepy.utils import error

# Constants, units and keywords
from porepy.utils.common_constants import *
