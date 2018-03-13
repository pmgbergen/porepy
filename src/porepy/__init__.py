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
# Control volume, elliptic
from porepy.numerics.fv.mpsa import Mpsa
from porepy.numerics.fv.tpfa import Tpfa, TpfaMixedDim
from porepy.numerics.fv.mpfa import Mpfa, MpfaMixedDim
from porepy.numerics.fv.biot import Biot
from porepy.numerics.fv.source import Integral, IntegralMixedDim

# Virtual elements, elliptic
from porepy.numerics.vem.vem_dual import DualVEM, DualVEMMixedDim
from porepy.numerics.vem.vem_source import DualSource, DualSourceMixedDim

# Transport related
from porepy.numerics.fv.transport.upwind import Upwind, UpwindMixedDim
from porepy.numerics.fv.mass_matrix import MassMatrix, InvMassMatrix

#Grid
from porepy.grids.grid import Grid
from porepy.grids.grid_bucket import GridBucket

# Fractures
from porepy.fracs.fractures import Fracture, FractureNetwork

# Parameters
from porepy.params.bc import BoundaryCondition
from porepy.params.tensor import SecondOrderTensor, FourthOrderTensor
from porepy.params.data import Parameters

# Visualization
from porepy.viz.exporter import Exporter

# Modules
from porepy.utils import comp_geom as cg
from porepy.fracs import meshing, importer
from porepy.grids import structured, simplex, coarsening
from porepy.params import units
