"""   PorePy.

Root directory for the PorePy package. Contains the following sub-packages:

fracs: Meshing, analysis, manipulations of fracture networks.

grids: Grid class, constructors, partitioning, etc.

numerics: Discretization schemes.

params: Physical parameters, constitutive laws, boundary conditions etc.

utils: Utility functions, array manipulation, computational geometry etc.

viz: Visualization; paraview, matplotlib.


isort:skip_file

"""
import os, sys
from pathlib import Path
import configparser
import warnings

__version__ = "1.6.0"

# Give a deprecation warning if the user is using python 3.8 or older
if sys.version_info.major <= 3 and sys.version_info.minor <= 8:
    warnings.warn(
        "Python 3.8 or older will soon be deprecated."
        " Please upgrade to Python 3.9 or newer.",
        DeprecationWarning,
    )

# Try to read the config file from the directory where python process was launched
try:
    cwd = Path(os.getcwd())
    pth = cwd / Path("porepy.cfg")
    cfg = configparser.ConfigParser()
    cfg.read(pth)
    config = dict(cfg)
except:
    # the assumption is that no configurations are given
    config = {}

# ------------------------------------
# Simplified namespaces. The rule of thumb is that classes and modules that a
# user can be exposed to should have a shortcut here. Borderline cases will be
# decided as needed

from porepy.utils.common_constants import *
from porepy.utils.porepy_types import *

from porepy.utils.tangential_normal_projection import TangentialNormalProjection

from porepy.utils import permutations
from porepy.utils.interpolation_tables import (
    InterpolationTable,
    AdaptiveInterpolationTable,
)
from porepy.utils import array_operations
from porepy.numerics.linalg import matrix_operations

# Geometry
from porepy.geometry import (
    intersections,
    distances,
    constrain_geometry,
    map_geometry,
    geometry_property_checks,
    point_in_polyhedron_test,
    half_space,
    domain,
)
from porepy.geometry.domain import Domain

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
    set_iterate,
)
from porepy.params.rock import UnitRock, Shale, SandStone, Granite
from porepy.params.fluid import Water, UnitFluid

# Grids
from porepy.grids.grid import Grid
from porepy.grids.mortar_grid import MortarGrid
from porepy.grids.md_grid import MixedDimensionalGrid
from porepy.grids.structured import CartGrid, TensorGrid
from porepy.grids.simplex import TriangleGrid, TetrahedralGrid
from porepy.grids.simplex import StructuredTriangleGrid, StructuredTetrahedralGrid
from porepy.grids.point_grid import PointGrid
from porepy.grids.boundary_grid import BoundaryGrid
from porepy.grids import match_grids
from porepy.grids.standard_grids import md_grids_2d, md_grids_3d
from porepy.grids import grid_extrusion
from porepy.utils import grid_utils
from porepy.utils import adtree

# Fractures
from porepy.fracs.plane_fracture import PlaneFracture, create_elliptic_fracture
from porepy.fracs.line_fracture import LineFracture
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network import create_fracture_network

# Wells
from porepy.fracs.wells_3d import (
    Well,
    WellNetwork3d,
    compute_well_fracture_intersections,
)


# Numerics
from porepy.numerics.discretization import VoidDiscretization
from porepy.numerics.interface_laws.elliptic_discretization import (
    EllipticDiscretization,
)

# Control volume, elliptic
from porepy.numerics.fv import fvutils
from porepy.numerics.fv.mpsa import Mpsa
from porepy.numerics.fv.fv_elliptic import (
    FVElliptic,
    EllipticDiscretizationZeroPermeability,
)
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
    WellCoupling,
)

from porepy.numerics.interface_laws.cell_dof_face_dof_map import CellDofFaceDofMap
from porepy.numerics.mixed_dim import assembler_filters
from porepy.numerics.mixed_dim.dof_manager import DofManager
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
from porepy.numerics.fracture_deformation.contact_conditions import (
    ColoumbContact,
    ContactTraction,
)
from porepy.numerics.fracture_deformation import contact_conditions, propagate_fracture
from porepy.numerics.fracture_deformation.conforming_propagation import (
    ConformingFracturePropagation,
)

# Related to models and solvers
from porepy.numerics.nonlinear.nonlinear_solvers import NewtonSolver
from porepy.numerics.linear_solvers import LinearSolver
from porepy.models.run_models import (
    run_stationary_model,
    run_time_dependent_model,
)


from porepy.numerics import ad
from porepy.numerics.ad.operators import wrap_as_ad_array, wrap_as_ad_matrix
from porepy.numerics.ad.equation_system import EquationSystem

# Time stepping control
from porepy.numerics.time_step_control import TimeManager

from porepy import models
from porepy.models.abstract_equations import (
    BalanceEquation,
    VariableMixin,
)
from porepy.models.geometry import ModelGeometry
from porepy.models.units import Units
from porepy.models.material_constants import (
    FluidConstants,
    SolidConstants,
    MaterialConstants,
)


from porepy.viz.data_saving_model_mixin import DataSavingMixin
from porepy.models.solution_strategy import SolutionStrategy
from porepy.models import constitutive_laws

# "Primary" models
from porepy.models import fluid_mass_balance, momentum_balance

# "Secondary" models inheriting from primary models
from porepy.models import (
    poromechanics,
    energy_balance,
    mass_and_energy_balance,
    thermoporomechanics,
)


# Visualization
from porepy.viz.exporter import Exporter
from porepy.viz.plot_grid import plot_grid, save_img
from porepy.viz.fracture_visualization import plot_fractures, plot_wells

from porepy.utils import error


# Modules
from porepy.fracs import utils as frac_utils
from porepy.fracs import meshing, fracture_importer
from porepy.grids import coarsening, partition, refinement
from porepy.numerics import displacement_correlation
from porepy.utils.default_domains import (
    CubeDomain,
    SquareDomain,
    UnitSquareDomain,
    UnitCubeDomain,
)
