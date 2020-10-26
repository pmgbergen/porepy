"""   PorePy.

Root directory for the PorePy package. Contains the following sub-packages:

fracs: Meshing, analysis, manipulations of fracture networks.

grids: Grid class, constructors, partitioning, etc.

numerics: Discretization schemes.

params: Physical parameters, constitutive laws, boundary conditions etc.

utils: Utility functions, array manipulation, computational geometry etc.

viz: Visualization; paraview, matplotlib.

"""

__version__ = "1.2.6"

# ------------------------------------
# Simplified namespaces. The rue of thumb is that classes and modules that a
# user can be exposed to should have a shortcut here. Borderline cases will be
# decided as needed

import porepy.numerics
import porepy.utils.derived_discretizations
from porepy.fracs import fracture_importer, meshing
from porepy.fracs import utils as frac_utils
from porepy.fracs.fractures_2d import FractureNetwork2d
from porepy.fracs.fractures_3d import EllipticFracture, Fracture, FractureNetwork3d
from porepy.geometry import (
    bounding_box,
    constrain_geometry,
    distances,
    geometry_property_checks,
    intersections,
    map_geometry,
)
from porepy.grids import coarsening, grid_extrusion, match_grids, partition, refinement
from porepy.grids.fv_sub_grid import FvSubGrid
from porepy.grids.grid import Grid
from porepy.grids.grid_bucket import GridBucket
from porepy.grids.mortar_grid import BoundaryMortar, MortarGrid
from porepy.grids.point_grid import PointGrid
from porepy.grids.simplex import (
    StructuredTetrahedralGrid,
    StructuredTriangleGrid,
    TetrahedralGrid,
    TriangleGrid,
)
from porepy.grids.standard_grids import grid_buckets_2d
from porepy.grids.structured import CartGrid, TensorGrid
from porepy.models.contact_mechanics_biot_model import ContactMechanicsBiot
from porepy.models.contact_mechanics_model import ContactMechanics
from porepy.models.run_models import run_stationary_model, run_time_dependent_model
from porepy.numerics.contact_mechanics import contact_conditions
from porepy.numerics.contact_mechanics.contact_conditions import ColoumbContact
from porepy.numerics.discretization import VoidDiscretization
from porepy.numerics.fem.rt0 import RT0
from porepy.numerics.fv import fvutils
from porepy.numerics.fv.biot import Biot, BiotStabilization, DivU, GradP
from porepy.numerics.fv.fv_elliptic import (
    EllipticDiscretizationZeroPermeability,
    FVElliptic,
)
from porepy.numerics.fv.mass_matrix import InvMassMatrix, MassMatrix
from porepy.numerics.fv.mpfa import Mpfa
from porepy.numerics.fv.mpsa import Mpsa
from porepy.numerics.fv.source import ScalarSource
from porepy.numerics.fv.tpfa import Tpfa
from porepy.numerics.fv.upwind import Upwind
from porepy.numerics.interface_laws.cell_dof_face_dof_map import CellDofFaceDofMap
from porepy.numerics.interface_laws.contact_mechanics_interface_laws import (
    DivUCoupling,
    FractureScalarToForceBalance,
    MatrixScalarToForceBalance,
    PrimalContactCoupling,
)
from porepy.numerics.interface_laws.elliptic_discretization import (
    EllipticDiscretization,
)
from porepy.numerics.interface_laws.elliptic_interface_laws import (
    FluxPressureContinuity,
    RobinCoupling,
)
from porepy.numerics.interface_laws.hyperbolic_interface_laws import UpwindCoupling
from porepy.numerics.linear_solvers import LinearSolver
from porepy.numerics.mixed_dim import assembler_filters
from porepy.numerics.mixed_dim.assembler import Assembler
from porepy.numerics.nonlinear.nonlinear_solvers import NewtonSolver
from porepy.numerics.vem.dual_elliptic import project_flux
from porepy.numerics.vem.mass_matrix import MixedInvMassMatrix, MixedMassMatrix
from porepy.numerics.vem.mvem import MVEM
from porepy.numerics.vem.vem_source import DualScalarSource
from porepy.params.bc import BoundaryCondition, BoundaryConditionVectorial, face_on_side
from porepy.params.data import (
    Parameters,
    initialize_data,
    initialize_default_data,
    set_iterate,
    set_state,
)
from porepy.params.fluid import UnitFluid, Water
from porepy.params.rock import Granite, SandStone, Shale, UnitRock
from porepy.params.tensor import FourthOrderTensor, SecondOrderTensor
from porepy.utils import error, grid_utils, permutations
from porepy.utils.common_constants import *
from porepy.utils.tangential_normal_projection import TangentialNormalProjection
from porepy.viz.exporter import Exporter
from porepy.viz.fracture_visualization import plot_fractures, plot_wells
from porepy.viz.plot_grid import plot_grid, save_img
