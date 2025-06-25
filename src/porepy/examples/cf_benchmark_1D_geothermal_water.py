"""Implements 1D benchmark examples using water and a 2D pipe acting as a 1D domain.

Simulations at endtime of 500 days should yield Figure 2 (horizontal case) and Figures
7 and 8 (vertical case) in section 4.1 of the reference [1].

Note:
    Reference [1] uses the IAPWS-97 formulation for the water properties, while this
    example relies on an in-house implementation of the Peng-Robinson equation of state.

References:
    [1] Yang Wang, Denis Voskov, Mark Khait, David Bruhn,
    An efficient numerical simulator for geothermal simulation: A benchmark study,
    Applied Energy,
    Volume 264,
    2020,
    114693,
    ISSN 0306-2619,
    https://doi.org/10.1016/j.apenergy.2020.114693.

"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from typing import Any, Callable, Literal, Sequence, Optional

# Benchmark is small, no need for JIT compilation.
os.environ["NUMBA_DISABLE_JIT"] = "1"

import numba as nb
import numpy as np

import porepy as pp
import porepy.models.compositional_flow_with_equilibrium as cfle
from porepy.examples.co2_injection import NewtonArmijoSolver
import porepy.compositional.peng_robinson as pr
import porepy.compositional.compiled_flash.eos_compiler as eosc

# Select the case to run.
CASE: Literal["horizontal", "vertical"] = "horizontal"
"""Both cases mimic a pipe. Horizontal case is advection-driven, vertical case is
gravity-driven."""

EXPORT_SCHEDULED_TIME_ONLY: bool = False
"""If True, only export the solution at the scheduled time points. If False, export the
solution at every time step."""

Z_EPS: float = 1e-6
"""A small overall fraction for a pseudo-component to overcome limitations of the used
flash procedure (which requires at least two components).

The second component is also modelled after water, but with a very small mass fraction
and no interaction between the components.

"""


# region Model configurations found in the paper.
# Liquid water at 50 C, vapor at 350 C.
CONFIG_HORIZONTAL_CASE = {
    # Inlet, outlet and initial pressure and temperature in Pa and Kelvin respectively.
    "p_in": 9e6,
    "p_out": 1e6,
    "T_in": 323.15,
    "T_out": 623.15,
    "p_initial": 1e6,
    "T_initial": 623.15,
    "time_schedule": [0.0, 500 * pp.DAY],
    # Solid properties.
    "solid": pp.SolidConstants(
        porosity=0.2,
        permeability=10 * pp.MILLIDARCY,
        thermal_conductivity=180e3 / (pp.DAY),  # 180 kJ / m / day / K
        # NOTE specific heat capacity and density are chosen such that the volumetric
        # heat capacity is 2200 kJ / m^3 / K.
        specific_heat_capacity=2200.0,
        density=1e3,
    ),
}
"""Model configuration for the horizontal case."""


CONFIG_VERTICAL_CASE = {
    "p_bottom": 10e6,
    "T_bottom": 323.15,
    # The vertical case has initial values such that water is initially in liquid form
    # in the top-most cell and in vapor form below.
    "p_initial": 1e6,
    "T_initial_top": 323.15,
    "T_initial_bottom": 623.15,
    "time_schedule": [0.0, 100 * pp.DAY, 200 * pp.DAY, 500 * pp.DAY],
    "solid": pp.SolidConstants(
        porosity=0.2,
        permeability=50 * pp.MILLIDARCY,
        thermal_conductivity=180e3 / (pp.DAY),
        specific_heat_capacity=2200.0,
        density=1e3,
    ),
}
"""Model configuration for the horizontal case."""

CONFIG: dict[str, dict[str, Any]] = {
    "horizontal": CONFIG_HORIZONTAL_CASE,
    "vertical": CONFIG_VERTICAL_CASE,
}


class Pipe2D(pp.PorePyModel):
    """Benchmark geometry, which is a 2D domain mimicking a 1D pipe.

    Covers both horizontal and vertical cases.

    """

    @property
    def dx(self) -> float:
        """Returns the cell size of the unit square cells for the different cases."""
        if CASE == "horizontal":
            return self.units.convert_units(10.0, "m")
        elif CASE == "vertical":
            return self.units.convert_units(20.0, "m")

    def grid_type(self) -> Literal["cartesian"]:
        return "cartesian"

    def meshing_arguments(self) -> dict:
        return {"cell_size": float(self.dx)}

    def set_domain(self) -> None:
        """Domain is 2D mimicing 1D since only one cell in y-direction."""
        if CASE == "horizontal":
            self._domain = pp.Domain(
                bounding_box={
                    "xmin": 0.0,
                    "xmax": 50 * self.dx,
                    "ymin": 0.0,
                    "ymax": self.dx,
                }
            )
        elif CASE == "vertical":
            self._domain = pp.Domain(
                bounding_box={
                    "xmin": 0.0,
                    "xmax": self.dx,
                    "ymin": 0.0,
                    "ymax": 10 * self.dx,
                }
            )


# endregion


class WaterEoS(pr.PengRobinsonCompiler):
    """An equation of state for water based on Peng-Robinson, but using IAPWS-97
    correlations to compute the transport properties viscosity and thermal conductivity.
    """

    def get_viscosity_function(self) -> eosc.ScalarFunction:
        @nb.njit(nb.f8(nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def mu_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> float:
            return 1e-3
            # # Liquidlike
            # if prearg[3] < 1.0:
            #     mu = 8e-4
            # # Gaslike
            # elif prearg[3] < 2.0:
            #     mu = 3e-5
            # else:
            #     assert False, "Uncovered physical state in viscosity function."
            # return mu

        return mu_c

    def get_viscosity_derivative_function(self) -> eosc.VectorFunction:
        @nb.njit(nb.f8[:](nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def dmu_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            return np.zeros(2 + xn.shape[0], dtype=np.float64)

        return dmu_c

    def get_conductivity_function(self) -> eosc.ScalarFunction:
        @nb.njit(nb.f8(nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def kappa_c(prearg: np.ndarray, p: float, T: float, xn: np.ndarray) -> float:
            return 1.0
            # # Liquidlike
            # if prearg[3] < 1.0:
            #     kappa = 0.1
            # # Gaslike
            # elif prearg[3] < 2.0:
            #     kappa = 0.08
            # else:
            #     assert False, "Uncovered physical state in viscosity function."
            # return kappa

        return kappa_c

    def get_conductivity_derivative_function(self) -> eosc.VectorFunction:
        @nb.njit(nb.f8[:](nb.f8[:], nb.f8[:], nb.f8, nb.f8, nb.f8[:]))
        def dkappa_c(
            prearg_val: np.ndarray,
            prearg_jac: np.ndarray,
            p: float,
            T: float,
            xn: np.ndarray,
        ) -> np.ndarray:
            return np.zeros(2 + xn.shape[0], dtype=np.float64)

        return dkappa_c


class TwoPhaseWaterFluid(pp.PorePyModel):
    """2-component, 2-phase fluid with H2O and another component mimicking water, and a
    liquid and gas phase.

    This is done to overcome the limitations of the flash procedure, which requires at
    least two components. The second component is modelled after water, and the
    interaction between the components is neglected.

    The Peng-Robinson equation of state is used to model the fluid properties.

    """

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    def get_components(self) -> Sequence[pp.FluidComponent]:
        h2o = pp.compositional.load_fluid_constants(["H2O", "H2O"], "chemicals")[0]
        ph2o_ = asdict(h2o)
        ph2o_["name"] = "ph2o"
        del ph2o_["constants_in_SI"]
        del ph2o_["_initialized"]
        ph2o = pp.FluidComponent(**ph2o_)
        return [h2o, ph2o]

    def get_phase_configuration(
        self, components: Sequence[pp.FluidComponent]
    ) -> Sequence[
        tuple[pp.compositional.PhysicalState, str, pp.compositional.EquationOfState]
    ]:
        import porepy.compositional.peng_robinson as pr

        eos = WaterEoS(components, [pr.h_ideal_H2O, pr.h_ideal_H2O], np.eye(2))
        return [
            (pp.compositional.PhysicalState.liquid, "L", eos),
            (pp.compositional.PhysicalState.gas, "G", eos),
        ]

    def dependencies_of_phase_properties(
        self, phase: pp.Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        return [self.pressure, self.temperature] + [  # type:ignore[return-value]
            phase.extended_fraction_of[comp] for comp in phase
        ]


class BoundaryConditions(pp.PorePyModel):
    """BC for vertical and horizontal cases of the geothermal water benchmark."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        if CASE == "horizontal":
            return pp.BoundaryCondition(sd, sides.east | sides.west, "dir")
        elif CASE == "vertical":
            return pp.BoundaryCondition(sd, sides.bottom, "dir")

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_darcy_flux(sd)

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_darcy_flux(sd)

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_darcy_flux(sd)

    def bc_type_equilibrium(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_darcy_flux(sd)

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        vals = np.zeros(boundary_grid.num_cells)
        sides = self.domain_boundary_sides(boundary_grid)
        if CASE == "horizontal":
            vals[sides.west] = CONFIG[CASE]["p_in"]
            vals[sides.east] = CONFIG[CASE]["p_out"]
        elif CASE == "vertical":
            vals[sides.bottom] = CONFIG[CASE]["p_bottom"]

        return vals

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        vals = np.zeros(boundary_grid.num_cells)
        sides = self.domain_boundary_sides(boundary_grid)
        if CASE == "horizontal":
            vals[sides.west] = CONFIG[CASE]["T_in"]
            vals[sides.east] = CONFIG[CASE]["T_out"]
        elif CASE == "vertical":
            vals[sides.bottom] = CONFIG[CASE]["T_bottom"]

        return vals

    def bc_values_overall_fraction(
        self, component: pp.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        assert component.name == "ph2o"
        return np.ones(boundary_grid.num_cells) * Z_EPS


class InitialConditions(pp.PorePyModel):
    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        if CASE == "horizontal":
            p = np.ones(sd.num_cells) * CONFIG[CASE]["p_initial"]
        elif CASE == "vertical":
            p = np.ones(sd.num_cells) * CONFIG[CASE]["p_initial"]
        return p

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        if CASE == "horizontal":
            T = np.ones(sd.num_cells) * CONFIG[CASE]["T_initial"]
        elif CASE == "vertical":
            T = np.ones(sd.num_cells) * CONFIG[CASE]["T_initial_bottom"]
            # Find top cell and set temperature to target value for liquid.
            idx = sd.closest_cell(
                np.array(
                    [sd.cell_centers[0].max(), sd.cell_centers[1].max(), 1.0]
                ).reshape((3, 1))
            )[0]
            T[idx] = CONFIG[CASE]["T_initial_top"]
        return T

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        assert component.name == "ph2o"
        return np.ones(sd.num_cells) * Z_EPS


class GeothermalWaterModel(  # type:ignore[misc]
    BoundaryConditions,
    InitialConditions,
    TwoPhaseWaterFluid,
    Pipe2D,
    cfle.EnthalpyBasedCFFLETemplate,
):
    """Model assembly for the geothermal water benchmark."""

    def after_nonlinear_convergence(self) -> None:
        super().after_nonlinear_convergence()
        print("Number of iterations: ", self.nonlinear_solver_statistics.num_iteration)
        print("Time value (day): ", self.time_manager.time / pp.DAY)
        print("Time index: ", self.time_manager.time_index)
        print("")


if __name__ == "__main__":
    max_iterations = 40
    newton_tol = 1e-6
    newton_tol_increment = 5e-6

    time_manager = pp.TimeManager(
        schedule=np.array(CONFIG[CASE]["time_schedule"]),
        dt_init=100 * pp.DAY,
        dt_min_max=(pp.DAY, 100 * pp.DAY),
        iter_max=max_iterations,
        iter_optimal_range=(9, 15),
        iter_relax_factors=(0.8, 2.0),
        recomp_factor=0.6,
        recomp_max=15,
        print_info=True,
        rtol=0.0,
    )

    phase_property_params = {"phase_property_params": [0.0]}

    material_params = {"solid": CONFIG[CASE]["solid"]}

    flash_params = {
        "mode": "sequential",
        "solver": "npipm",
        "solver_params": {
            "tolerance": 1e-8,
            "max_iterations": 80,
            "armijo_rho": 0.99,
            "armijo_kappa": 0.4,
            "armijo_max_iterations": 30,
            "npipm_u1": 10,
            "npipm_u2": 10,
            "npipm_eta": 0.5,
        },
        "global_iteration_stride": 1,
        "fallback_to_iterate": False,
    }
    flash_params.update(phase_property_params)

    solver_params = {
        "max_iterations": max_iterations,
        "nl_convergence_tol": newton_tol_increment,
        "nl_convergence_tol_res": newton_tol,
        "apply_schur_complement_reduction": True,
        "linear_solver": "scipy_sparse",
        "nonlinear_solver": NewtonArmijoSolver,
        "armijo_line_search": True,
        "armijo_line_search_weight": 0.95,
        "armijo_line_search_incline": 0.2,
        "armijo_line_search_max_iterations": 10,
        "solver_statistics_file_name": "solver_statistics.json",
    }

    model_params = {
        "equilibrium_condition": "unified-p-h",
        "flash_params": flash_params,
        "fractional_flow": True,
        "material_constants": material_params,
        "time_manager": time_manager,
        "prepare_simulation": False,
        "buoyancy_on": True if CASE == "vertical" else False,
        "compile": True,
        "flash_compiler_args": ("p-T", "p-h"),
    }

    if EXPORT_SCHEDULED_TIME_ONLY:
        model_params["times_to_export"] = CONFIG[CASE]["time_schedule"]

    model_params.update(phase_property_params)
    model_params.update(solver_params)

    assert CASE in CONFIG, f"Invalid case: {CASE}. Choose 'horizontal' or 'vertical'."

    model = GeothermalWaterModel(model_params)  # type:ignore[abstract]

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("porepy").setLevel(logging.DEBUG)
    t_0 = time.time()
    model.prepare_simulation()
    prep_sim_time = time.time() - t_0
    logging.getLogger("porepy").setLevel(logging.INFO)

    print(
        "Initial vapor saturation: ",
        model.equation_system.evaluate(
            model.fluid.phases[1].saturation(model.mdg.subdomains())
        ),
    )

    model.schur_complement_primary_equations = cfle.cf.get_primary_equations_cf(model)
    model.schur_complement_primary_variables = cfle.cf.get_primary_variables_cf(model)
    model.nonlinear_solver_statistics.num_iteration_armijo = 0  # type:ignore[attr-defined]

    t_0 = time.time()
    pp.run_time_dependent_model(model, model_params)
    sim_time = time.time() - t_0
