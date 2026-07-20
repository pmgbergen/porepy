"""
Geothermal flow simulation with H2O and low NaCl content (Figure 8).

This script includes a backtracking line search algorithm to improve Newton
convergence. The line search can be enabled/disabled via the 'use_line_search'
parameter in the params dictionary.

Line search parameters (in DriesnerModelConfiguration.backtracking_line_search):
- alpha_init: Initial step length (default: 1.0)
- rho: Step reduction factor (default: 0.5)
- c: Armijo parameter (default: 1e-4)
- max_iterations: Maximum backtracking steps (default: 10)
"""
from __future__ import annotations

import os

import time
from typing import cast, Sequence

import numpy as np

import porepy as pp

# geometry description 2D case
from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import (  # noqa: E501
    Figure8Geometry2D as ModelGeometryFigure8,
)

from porepy.examples.geothermal_flow.model_configuration.DriesnerModelConfiguration import (  # noqa: E501
    DriesnerBrineFlowModel,               # HU / PPU (standard primary equations)
    DriesnerBrineFractionalFlowModel,     # HU-mw   (fractional-flow primary equations)
)
from porepy.examples.geothermal_flow.model_configuration.flow_model_base import (  # noqa: E501
    geothermal_nonlinear_solver,
)
from porepy.examples.geothermal_flow.model_configuration.geothermal_export import (  # noqa: E501
    DriesnerPhaseExport,
)

from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (  # noqa: E501
    BC_two_phase_Figure_8_left_panel as BC,
)

from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (  # noqa: E501
    IC_two_phase_Figure_8_left_panel as IC,
)
from porepy.examples.geothermal_flow.obl_sampler import NSplineSampler, VTKSampler

# Main directives
case_name = "condition_1"
final_times = {
    "condition_1": [18250000.0],  # final time [50000 years]
}

day_to_second = 86400
to_Mega = 1.0e-6
# Configuration dictionary mapping cases to their specific classes
simulation_cases = {
    "condition_1": {
        "tf": final_times[case_name][0] * day_to_second,  # final time [years]
        "dt": 12.5 *  365.0 * day_to_second,  # final time [1 years]
        "bc": BC,
        "ic": IC,
        "geometry": ModelGeometryFigure8,
    }
}

tf = cast(float, simulation_cases[case_name]["tf"])
dt = cast(float, simulation_cases[case_name]["dt"])
BoundaryConditions: type = cast(type, simulation_cases[case_name]["bc"])
InitialConditions: type = cast(type, simulation_cases[case_name]["ic"])
ModelGeometry: type = cast(type, simulation_cases[case_name]["geometry"])

# Export configuration: number of time steps between consecutive VTK/PVD exports.
export_every_n_steps = 8

# Build times_to_export as multiples of dt. Include t=0 and final time tf.
times = list(np.arange(0.0, tf, dt * export_every_n_steps))
times.append(tf)
times_to_export = times
# now times_to_export can be overridden later by params if desired

time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

# time_manager = pp.TimeManager(
#     schedule=[0.0, tf],
#     dt_init=dt,
#     constant_dt=False,
#     dt_min_max=((1.0/365.0) * dt, 1.0 * dt),
#     iter_relax_factors=(0.5, 1.5),
#     iter_optimal_range=(3, 8),
#     recomp_factor=0.3,
#     print_info=True,
# )



solid_constants = pp.SolidConstants(
    permeability=1e-15,
    porosity=0.1,
    thermal_conductivity=2.0 * to_Mega,
    density=2700.0,
    specific_heat_capacity=880.0 * to_Mega,
)
material_constants = {"solid": solid_constants}
# Scheme switch (= porepy_3d_solver._SCHEME_CONFIG): the fractional_flow flag pairs with
# the base template -- False -> DriesnerBrineFlowModel, True -> the fractional-flow one.
_SCHEME_CONFIG = {
    "hu":    dict(fractional_flow=False, buoyancy_upwinding="hybrid"),
    "hu-mw": dict(fractional_flow=True,  buoyancy_upwinding="hybrid"),
}
import argparse
_ap = argparse.ArgumentParser(
    description="Weis et al. (2014) Fig. 8(A-C) heat-flux plume (9 km x 3 km, "
                "5 W/m^2 over the central 1 km of the bottom boundary).")
_ap.add_argument("--consistent", action="store_true",
                 help="consistent flux discretization (MPFA); default TPFA")
_ap.add_argument("--grid-type", default=None, choices=["cartesian", "simplex"],
                 help="mesh type; default: the geometry class's choice")
_ap.add_argument("--cell-size", type=float, default=None, metavar="M",
                 help="target cell size [m]; default: the geometry class's value")
_ap.add_argument("--scheme", default="hu", choices=list(_SCHEME_CONFIG),
                 help="HU (standard template, hybrid), HU-mw (fractional-flow template), "
                      "PPU (standard template, phase-potential); default HU")
_args = _ap.parse_args()

params = {
    "folder_name": os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"visualization_{_args.scheme.lower().replace('-', '_')}"),
    "enable_buoyancy_effects": True,
    "material_constants": material_constants,
    "time_manager": time_manager,
    "times_to_export": times_to_export,
    # Schur-reduced CPR linear solver -- exactly porepy_3d_solver's "cpr" mode.
    "use_petsc": False,
    "petsc_preconditioner": "cpr",
    "cpr_rtol": 1.0e-5,           # CPR GMRES relative tolerance
    "cpr_maxit": 400,             # CPR GMRES iteration cap
    "cpr_accuracy_tol": 1.0e-3,   # post-solve gate -> direct fallback above this
    "step_control_method": "None",
}
params["consistent_discretization"] = _args.consistent
if _args.grid_type is not None:
    params["grid_type"] = _args.grid_type            # Figure8Geometry2D reads this key
params.update(_SCHEME_CONFIG[_args.scheme])
FlowModel = (DriesnerBrineFractionalFlowModel if params["fractional_flow"]
             else DriesnerBrineFlowModel)


class GeothermalBrineFlowModel(
    DriesnerPhaseExport, ModelGeometry, BoundaryConditions, InitialConditions, FlowModel
):
    # flux discretization comes from the base TPFA/MPFA switch (--consistent)

    def meshing_arguments(self) -> dict:
        mesh_args = super().meshing_arguments()
        if _args.cell_size is not None:              # default: the geometry class's value
            mesh_args = {**mesh_args,
                         "cell_size": self.units.convert_units(_args.cell_size, "m")}
        return mesh_args


# Instance of the computational model
model = GeothermalBrineFlowModel(params)

HERE = os.path.dirname(os.path.abspath(__file__))
# Constitutive approach shared by every subsection_4_2 solver: Driesner opensowat OBL
# tables sampled with the C2 tensor-spline backend (consistent value/Jacobian).
TABLE_LEVEL = 2                           # opensowat .vtr level (0..4 available)
USE_SPLINE = True                         # True -> NSplineSampler; False -> VTKSampler probe
_TABLE_DIR = os.path.join(
    HERE, os.pardir, os.pardir, "model_configuration", "constitutive_description",
    "driesner_vtk_files")


def _attach_samplers(model) -> None:
    """Attach the level-``TABLE_LEVEL`` Driesner OBL samplers (phz + ptz), exactly as
    porepy_1d_solver / porepy_3d_solver do."""
    Sampler = NSplineSampler if USE_SPLINE else VTKSampler
    phz = Sampler(os.path.join(_TABLE_DIR, f"opensowat_xph_l_{TABLE_LEVEL}_grads.vtr"))
    phz.conversion_factors = (1.0, 1.0, 1.0)                 # (z, h, p)
    model.obl_sampler = phz
    ptz = Sampler(os.path.join(_TABLE_DIR, f"opensowat_xpt_l_{TABLE_LEVEL}_grads.vtr"))
    ptz.conversion_factors = (1.0, 1.0, 1.0)                 # (z, t, p)
    ptz.translation_factors = (0.0, -273.15, 0.0)            # T in degC -> K in the sampler
    model.obl_sampler_ptz = ptz


_attach_samplers(model)


tb = time.time()
solver_params = {
    "nl_convergence_criteria": {
        "res_abs": pp.solvers.ResidualBasedAbsoluteCriterion(
            tol=1.0e-3, metric=pp.EquationBasedLebesgueMetric(model)),
    },
    "nl_divergence_criteria": {
        "max_iter": pp.solvers.MaxIterationsCriterion(max_iterations=50),
    },
}
runner = pp.ModelRunner(model, solver_params,
                        nonlinear_solver=geothermal_nonlinear_solver(solver_params))
te = time.time()
print("Elapsed time prepare simulation: ", te - tb)
print("Simulation prepared for total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid employed: ", model.mdg)
model.schur_complement_primary_equations = (
    pp.compositional_flow.get_primary_equations_cf(model)
)
model.schur_complement_primary_variables = (
    pp.compositional_flow.get_primary_variables_cf(model)
)

# print geometry
model.exporter.write_vtu()
tb = time.time()
runner.run()
te = time.time()
print("Elapsed time run_time_dependent_model: ", te - tb)
print("Total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid information: ", model.mdg)

# Retrieve the grid and boundary information
grid = model.mdg.subdomains()[0]
bc_sides = model.domain_boundary_sides(grid)

# Integrated overall mass flux on all facets
mn = model.equation_system.evaluate(model.darcy_flux(model.mdg.subdomains()))
mn = cast(np.ndarray, mn)

inlet_idx, outlet_idx = model.get_inlet_outlet_sides(model.mdg.subdomains()[0])
print("Inflow values : ", mn[inlet_idx])
print("Outflow values : ", mn[outlet_idx])
