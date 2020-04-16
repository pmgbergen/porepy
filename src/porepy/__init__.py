"""   PorePy.

Root directory for the PorePy package. Contains the following sub-packages:

fracs: Meshing, analysis, manipulations of fracture networks.

grids: Grid class, constructors, partitioning, etc.

numerics: Discretization schemes.

params: Physical parameters, constitutive laws, boundary conditions etc.

utils: Utility functions, array manipulation, computational geometry etc.

viz: Visualization; paraview, matplotlib.

"""

__version__ = "1.1.0"

# ------------------------------------
# Simplified namespaces. The rue of thumb is that classes and modules that a
# user can be exposed to should have a shortcut here. Borderline cases will be
# decided as needed

__all__ = []


# Parameters
from porepy.params.bc import (
    BoundaryCondition,
    BoundaryConditionVectorial,
    face_on_side,
)
from porepy.params.tensor import SecondOrderTensor, FourthOrderTensor
from porepy.params.data import (
    Parameters,
    initialize_data,
    initialize_default_data,
    set_state,
)
from porepy.params.rock import UnitRock, Shale, SandStone, Granite
from porepy.params.fluid import Water, UnitFluid

# Grids
from porepy.grids.grid import Grid
from porepy.grids.fv_sub_grid import FvSubGrid
from porepy.grids.grid_bucket import GridBucket
from porepy.grids.structured import CartGrid, TensorGrid
from porepy.grids.simplex import TriangleGrid, TetrahedralGrid
from porepy.grids.simplex import StructuredTriangleGrid, StructuredTetrahedralGrid
from porepy.grids.point_grid import PointGrid
from porepy.grids.mortar_grid import MortarGrid, BoundaryMortar
from porepy.grids.standard_grids import grid_buckets_2d
from porepy.grids import grid_extrusion

# Fractures
from porepy.fracs.fractures_3d import Fracture, EllipticFracture, FractureNetwork3d
from porepy.fracs.fractures_2d import FractureNetwork2d

# Numerics
from porepy.numerics.discretization import VoidDiscretization
from porepy.numerics.interface_laws.elliptic_discretization import (
    EllipticDiscretization,
)

# Control volume, elliptic
from porepy.numerics.fv import fvutils
from porepy.numerics.fv.mpsa import Mpsa
from porepy.numerics.fv.fv_elliptic import FVElliptic
from porepy.numerics.fv.tpfa import Tpfa
from porepy.numerics.fv.mpfa import Mpfa
from porepy.numerics.fv.biot import Biot, GradP, DivU, BiotStabilization
from porepy.numerics.fv.source import ScalarSource

# Virtual elements, elliptic
from porepy.numerics.vem.dual_elliptic import project_flux
from porepy.numerics.vem.mvem import MVEM
from porepy.numerics.vem.mass_matrix import MixedMassMatrix, MixedInvMassMatrix
from porepy.numerics.vem.vem_source import DualScalarSource

# Finite elements, elliptic
from porepy.numerics.fem.rt0 import RT0

# Mixed-dimensional discretizations and assemblers
from porepy.numerics.interface_laws.elliptic_interface_laws import (
    RobinCoupling,
    FluxPressureContinuity,
)

from porepy.numerics.interface_laws.cell_dof_face_dof_map import CellDofFaceDofMap
from porepy.numerics.mixed_dim.assembler import Assembler

import porepy.numerics

# Transport related
from porepy.numerics.fv.upwind import Upwind
from porepy.numerics.interface_laws.hyperbolic_interface_laws import UpwindCoupling
from porepy.numerics.fv.mass_matrix import MassMatrix
from porepy.numerics.fv.mass_matrix import InvMassMatrix

# Contact mechanics
from porepy.numerics.interface_laws.contact_mechanics_interface_laws import (
    PrimalContactCoupling,
    DivUCoupling,
    MatrixScalarToForceBalance,
    FractureScalarToForceBalance,
)
from porepy.numerics.contact_mechanics.contact_conditions import ColoumbContact
from porepy.numerics.contact_mechanics import contact_conditions

# Related to models and solvers
from porepy.numerics.nonlinear.nonlinear_solvers import NewtonSolver
from porepy.numerics.linear_solvers import LinearSolver
from porepy.models.run_models import run_stationary_model, run_time_dependent_model

# Visualization
from porepy.viz.exporter import Exporter
from porepy.viz.plot_grid import plot_grid, save_img
from porepy.viz.fracture_visualization import plot_fractures, plot_wells

# Modules
from porepy.utils import permutations

from porepy.geometry import (
    intersections,
    distances,
    constrain_geometry,
    map_geometry,
    geometry_property_checks,
    bounding_box,
)
from porepy.fracs import utils as frac_utils
from porepy.fracs import meshing, fracture_importer, mortars
from porepy.grids import structured, simplex, coarsening, partition, refinement
from porepy.utils import error, grid_utils
from porepy.utils.tangential_normal_projection import TangentialNormalProjection
import porepy.utils.derived_discretizations

# Constants, units and keywords
from porepy.utils.common_constants import *
