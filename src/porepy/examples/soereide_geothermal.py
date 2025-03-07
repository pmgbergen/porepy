"""This module implements a multiphase-multicomponent flow model with phase change
using the Soereide model to define the fluid mixture.

Note:
    It uses the numba-compiled version of the Peng-Robinson EoS and a numba-compiled
    unified flash.

    Import and compilation take some time.

References:
    [1] Ingolf Søreide, Curtis H. Whitson,
        Peng-Robinson predictions for hydrocarbons, CO2, N2, and H2 S with pure water
        and NaCI brine,
        Fluid Phase Equilibria,
        Volume 77,
        1992,
        https://doi.org/10.1016/0378-3812(92)85105-H

"""

from __future__ import annotations

import logging
import os
import time
import warnings

# os.environ["NUMBA_DISABLE_JIT"] = "1"

compile_time = 0.0
logging.basicConfig(level=logging.INFO)
logging.getLogger("porepy").setLevel(logging.DEBUG)

from typing import Any, Callable, Optional, Sequence, no_type_check

import numpy as np

import porepy as pp

t_0 = time.time()
import porepy.compositional.peng_robinson as pr

compile_time += time.time() - t_0

import porepy.models.compositional_flow_with_equilibrium as cfle

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=RuntimeWarning)


class SoereideMixture:
    """Model fluid using the Soereide mixture, a Peng-Robinson based EoS for
    NaCl brine with CO2, H2S and N2."""

    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    def get_components(self) -> Sequence[pp.FluidComponent]:
        return pp.compositional.load_fluid_constants(["H2O", "CO2"], "chemicals")

    def get_phase_configuration(
        self, components: Sequence[pp.FluidComponent]
    ) -> Sequence[
        tuple[pp.compositional.EoSCompiler, pp.compositional.PhysicalState, str]
    ]:
        eos = pr.PengRobinsonCompiler(
            components, [pr.h_ideal_H2O, pr.h_ideal_CO2], pr.get_bip_matrix(components)
        )
        return [
            (eos, pp.compositional.PhysicalState.liquid, "L"),
            (eos, pp.compositional.PhysicalState.gas, "G"),
        ]

    def dependencies_of_phase_properties(
        self, phase: pp.Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Variable]]:
        return [self.pressure, self.temperature] + [  # type:ignore[return-value]
            phase.extended_fraction_of[comp] for comp in phase
        ]


class SolutionStrategy(pp.PorePyModel):
    """Provides some pre- and post-processing for flash methods."""

    flash: pp.compositional.Flash

    pressure_variable: str
    temperature_variable: str
    enthalpy_variable: str
    pressure: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    enthalpy: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]
    temperature: Callable[[pp.SubdomainsOrBoundaries], pp.ad.Operator]

    def after_nonlinear_failure(self):
        self.exporter.write_pvd()
        super().after_nonlinear_failure()  # type:ignore

    def get_fluid_state(
        self, subdomains: Sequence[pp.Grid], state: Optional[np.ndarray] = None
    ) -> pp.compositional.FluidProperties:
        """Method to pre-process the evaluated fractions. Normalizes the extended
        fractions where they violate the unity constraint."""
        fluid_state: pp.compositional.FluidProperties = super().get_fluid_state(  # type:ignore
            subdomains, state
        )

        # sanity checks
        # p = self.equation_system.get_variable_values(
        #     [self.pressure_variable], iterate_index=0
        # )
        # T = self.equation_system.get_variable_values(
        #     [self.temperature_variable], iterate_index=0
        # )
        # if np.any(p <= 0.0):
        #     raise ValueError("Pressure diverged to negative values.")
        # if np.any(T <= 0):
        #     raise ValueError("Temperature diverged to negative values.")

        unity_tolerance = 0.05

        sat_sum = np.sum(fluid_state.sat, axis=0)
        if np.any(sat_sum > 1 + unity_tolerance):
            raise ValueError("Saturations violate unity constraint")
        y_sum = np.sum(fluid_state.y, axis=0)
        idx = fluid_state.y < 0
        if np.any(idx):
            fluid_state.y[idx] = 0.0
        idx = fluid_state.y > 1.0
        if np.any(idx):
            fluid_state.y[idx] = 1.0
        y_sum = np.sum(fluid_state.y, axis=0)
        idx = y_sum > 1.0
        if np.any(idx):
            fluid_state.y[:, idx] = pp.compositional.normalize_rows(
                fluid_state.y[:, idx].T
            ).T

        # if np.any(y_sum > 1 + unity_tolerance):
        #     raise ValueError("Phase fractions violate unity constraint")
        # if np.any(fluid_state.y < 0 - unity_tolerance) or np.any(
        #     fluid_state.y > 1 + unity_tolerance
        # ):
        #     raise ValueError("Phase fractions out of bound.")

        for phase in fluid_state.phases:
            x_sum = phase.x.sum(axis=0)
            idx = x_sum > 1.0
            if np.any(idx):
                phase.x[:, idx] = pp.compositional.normalize_rows(phase.x[:, idx].T).T
            idx = phase.x < 0
            if np.any(idx):
                phase.x[idx] = 0.0
            idx = phase.x > 1.0
            if np.any(idx):
                phase.x[idx] = 1.0
            # if np.any(x_sum > 1 + unity_tolerance):
            #     raise ValueError(
            #         f"Extended fractions in phase {j} violate unity constraint."
            #     )
            # if np.any(phase.x < 0 - unity_tolerance) or np.any(
            #     phase.x > 1 + unity_tolerance
            # ):
            #     raise ValueError(f"Extended fractions in phase {j} out of bound.")

        return fluid_state

    def postprocess_flash(
        self,
        subdomain: pp.Grid,
        fluid_state: pp.compositional.FluidProperties,
        success: np.ndarray,
    ) -> pp.compositional.FluidProperties:
        """A post-processing where the flash is again attempted where not succesful.

        1. Where not successful, it re-attempts the flash from scratch, with
           initialization procedure.
           If the new attempt did not reach the desired precision (max iter), it
           treats it as success.
        2. Where still not successful, it falls back to the values of the previous
           iterate.

        """
        failure = success > 0
        equilibrium_type = str(pp.compositional.get_equilibrium_type(self))
        if np.any(failure):
            logger.warning(
                f"Flash from iterate state failed in {failure.sum()} cases."
                + " Re-starting with computation of initial guess."
            )
            z = [
                comp.fraction([subdomain]).value(self.equation_system)[failure]  # type:ignore[index]
                for comp in self.fluid.components
            ]
            p = self.equation_system.evaluate(self.pressure([subdomain]))[failure]  # type:ignore[index]
            h = self.equation_system.evaluate(self.enthalpy([subdomain]))[failure]  # type:ignore[index]
            T = self.equation_system.evaluate(self.temperature([subdomain]))[failure]  # type:ignore[index]

            logger.info(f"Failed at\nz: {z}\np: {p}\nT: {T}\nh: {h}")
            # no initial guess, and this model uses only p-h flash.
            flash_kwargs: dict[str, Any] = {
                "z": z,
                "p": p,
                "params": self.params.get("flash_params"),
            }

            if "p-h" in equilibrium_type:
                flash_kwargs["h"] = h
            elif "p-T" in equilibrium_type:
                flash_kwargs["T"] = T

            sub_state, sub_success, _ = self.flash.flash(**flash_kwargs)

            fallback = sub_success > 1
            max_iter_reached = sub_success == 1
            # treat max iter reached as success, and hope for the best in the PDE iter
            sub_success[max_iter_reached] = 0
            # fall back to previous iter values
            if np.any(fallback):
                logger.warning(
                    f"Falling back to previous iteration values in {fallback.sum()}"
                    + f" cells on grid {subdomain}"
                )
                sub_state = self.fall_back(subdomain, sub_state, failure, fallback)
                sub_success[fallback] = 0
            # update parent state with sub state values
            success[failure] = sub_success
            fluid_state.T[failure] = sub_state.T
            fluid_state.h[failure] = sub_state.h
            fluid_state.rho[failure] = sub_state.rho

            for j in range(len(fluid_state.phases)):
                fluid_state.sat[j][failure] = sub_state.sat[j]
                fluid_state.y[j][failure] = sub_state.y[j]

                fluid_state.phases[j].x[:, failure] = sub_state.phases[j].x

                fluid_state.phases[j].rho[failure] = sub_state.phases[j].rho
                fluid_state.phases[j].h[failure] = sub_state.phases[j].h
                fluid_state.phases[j].mu[failure] = sub_state.phases[j].mu
                fluid_state.phases[j].kappa[failure] = sub_state.phases[j].kappa

                fluid_state.phases[j].drho[:, failure] = sub_state.phases[j].drho
                fluid_state.phases[j].dh[:, failure] = sub_state.phases[j].dh
                fluid_state.phases[j].dmu[:, failure] = sub_state.phases[j].dmu
                fluid_state.phases[j].dkappa[:, failure] = sub_state.phases[j].dkappa

                fluid_state.phases[j].phis[:, failure] = sub_state.phases[j].phis
                fluid_state.phases[j].dphis[:, :, failure] = sub_state.phases[j].dphis

        # Parent method performs a check that everything is successful.
        return super().postprocess_flash(subdomain, fluid_state, success)  # type:ignore

    @no_type_check
    def fall_back(
        self,
        grid: pp.Grid,
        sub_state: pp.compositional.FluidProperties,
        failed_idx: np.ndarray,
        fallback_idx: np.ndarray,
    ) -> pp.compositional.FluidProperties:
        """Falls back to previous iterate values for the fluid state, on given
        grid."""
        sds = [grid]
        data = self.mdg.subdomain_data(grid)
        equilibrium_type = str(pp.compositional.get_equilibrium_type(self))
        if "T" not in equilibrium_type:
            sub_state.T[fallback_idx] = pp.get_solution_values(
                self.temperature_variable, data, iterate_index=0
            )[failed_idx][fallback_idx]
        if "h" not in equilibrium_type:
            sub_state.h[fallback_idx] = pp.get_solution_values(
                self.pressure_variable, data, iterate_index=0
            )[failed_idx][fallback_idx]
        if "p" not in equilibrium_type:
            sub_state.p[fallback_idx] = pp.get_solution_values(
                self.enthalpy_variable, data, iterate_index=0
            )[failed_idx][fallback_idx]

        for j, phase in enumerate(self.fluid.phases):
            if j > 0:
                sub_state.sat[j][fallback_idx] = pp.get_solution_values(
                    phase.saturation(sds).name, data, iterate_index=0
                )[failed_idx][fallback_idx]
                sub_state.y[j][fallback_idx] = pp.get_solution_values(
                    phase.fraction(sds).name, data, iterate_index=0
                )[failed_idx][fallback_idx]

            # mypy: ignore[union-attr]
            sub_state.phases[j].rho[fallback_idx] = pp.get_solution_values(
                phase.density.name, data, iterate_index=0
            )[failed_idx][fallback_idx]
            sub_state.phases[j].h[fallback_idx] = pp.get_solution_values(
                phase.specific_enthalpy.name, data, iterate_index=0
            )[failed_idx][fallback_idx]
            sub_state.phases[j].mu[fallback_idx] = pp.get_solution_values(
                phase.viscosity.name, data, iterate_index=0
            )[failed_idx][fallback_idx]
            sub_state.phases[j].kappa[fallback_idx] = pp.get_solution_values(
                phase.thermal_conductivity.name, data, iterate_index=0
            )[failed_idx][fallback_idx]

            sub_state.phases[j].drho[:, fallback_idx] = pp.get_solution_values(
                phase.density._name_derivatives, data, iterate_index=0
            )[:, failed_idx][:, fallback_idx]
            sub_state.phases[j].dh[:, fallback_idx] = pp.get_solution_values(
                phase.specific_enthalpy._name_derivatives, data, iterate_index=0
            )[:, failed_idx][:, fallback_idx]
            sub_state.phases[j].dmu[:, fallback_idx] = pp.get_solution_values(
                phase.viscosity._name_derivatives, data, iterate_index=0
            )[:, failed_idx][:, fallback_idx]
            sub_state.phases[j].dkappa[:, fallback_idx] = pp.get_solution_values(
                phase.thermal_conductivity._name_derivatives, data, iterate_index=0
            )[:, failed_idx][:, fallback_idx]

            for i, comp in enumerate(phase):
                sub_state.phases[j].x[i, fallback_idx] = pp.get_solution_values(
                    phase.extended_fraction_of[comp](sds).name, data, iterate_index=0
                )[failed_idx][fallback_idx]

                sub_state.phases[j].phis[i, fallback_idx] = pp.get_solution_values(
                    phase.fugacity_coefficient_of[comp].name, data, iterate_index=0
                )[failed_idx][fallback_idx]
                # NOTE numpy does some weird transpositions when dealing with 3D arrays
                dphi = sub_state.phases[j].dphis[i]
                dphi[:, fallback_idx] = pp.get_solution_values(
                    phase.fugacity_coefficient_of[comp]._name_derivatives,
                    data,
                    iterate_index=0,
                )[:, failed_idx][:, fallback_idx]
                sub_state.phases[j].dphis[i, :, :] = dphi

        # reference phase fractions and saturations must be computed, since not stored
        sub_state.y[self.fluid.reference_phase_index, :] = 1 - np.sum(
            sub_state.y[1:, :], axis=0
        )
        sub_state.sat[self.fluid.reference_phase_index, :] = 1 - np.sum(
            sub_state.sat[1:, :], axis=0
        )

        return sub_state


class Geometry(pp.PorePyModel):
    def set_domain(self) -> None:
        self._domain = pp.Domain(
            {
                "xmin": 0.0,
                "xmax": self.units.convert_units(10.0, "m"),
                "ymin": 0.0,
                "ymax": self.units.convert_units(5.0, "m"),
            }
        )

    # def set_fractures(self) -> None:
    #     """Setting a diagonal fracture"""
    #     frac_1_points = self.solid.convert_units(
    #         np.array([[1., 9.], [1., 4.]]), "m"
    #     )
    #     frac_1 = pp.LineFracture(frac_1_points)
    #     self._fractures = [frac_1]


class InitialConditions(pp.PorePyModel):
    """Define initial pressure, temperature and compositions."""

    _p_IN: float
    _p_OUT: float

    _p_INIT: float = 12e6
    _T_INIT: float = 540.0
    _z_INIT: dict[str, float] = {"H2O": 0.995, "CO2": 0.005}

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        # f = lambda x: self._p_IN + x /10 * (self._p_OUT - self._p_IN)
        # vals = np.array(list(map(f, sd.cell_centers[0])))
        # return vals
        return np.ones(sd.num_cells) * self._p_INIT

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        return np.ones(sd.num_cells) * self._T_INIT

    def ic_values_overall_fraction(
        self, component: pp.Component, sd: pp.Grid
    ) -> np.ndarray:
        return np.ones(sd.num_cells) * self._z_INIT[component.name]


class BoundaryConditions(pp.PorePyModel):
    """Boundary conditions defining a ``left to right`` flow in the matrix (2D)

    Mass flux:

    - No flux conditions on top and bottom
    - Dirichlet data (pressure, temperatyre and composition) on left and right faces

    Heat flux:

    - No flux on left, top, right
    - heated bottom side (given by temperature as a Dirichlet-type BC)

    Trivial Neumann conditions for fractures.

    """

    _p_INIT: float
    _T_INIT: float
    _z_INIT: dict[str, float]

    _p_IN: float = 15e6
    _p_OUT: float = InitialConditions._p_INIT

    _T_IN: float = InitialConditions._T_INIT
    _T_OUT: float = InitialConditions._T_INIT
    _T_HEATED: float = InitialConditions._T_INIT

    _z_IN: dict[str, float] = {
        "H2O": InitialConditions._z_INIT["H2O"] - 0.095,
        "CO2": InitialConditions._z_INIT["CO2"] + 0.095,
    }
    _z_OUT: dict[str, float] = {
        "H2O": InitialConditions._z_INIT["H2O"],
        "CO2": InitialConditions._z_INIT["CO2"],
    }

    def _inlet_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define inlet."""
        sides = self.domain_boundary_sides(sd)

        inlet = np.zeros(sd.num_faces, dtype=bool)
        inlet[sides.west] = True
        inlet &= sd.face_centers[1] > 1
        inlet &= sd.face_centers[1] < 4

        return inlet

    def _outlet_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define outlet."""

        sides = self.domain_boundary_sides(sd)

        outlet = np.zeros(sd.num_faces, dtype=bool)
        outlet[sides.east] = True
        outlet &= sd.face_centers[1] > 1
        outlet &= sd.face_centers[1] < 4

        return outlet

    def _heated_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define heated boundary with D-type conditions for conductive flux."""
        sides = self.domain_boundary_sides(sd)

        heated = np.zeros(sd.num_faces, dtype=bool)
        heated[sides.south] = True
        heated &= sd.face_centers[0] > 3.5
        heated &= sd.face_centers[0] < 6.5

        return heated

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # Setting only conditions on matrix
        if sd.dim == 2:
            inout = self._inlet_faces(sd) | self._outlet_faces(sd)
            # Define boundary condition on all boundary faces.
            return pp.BoundaryCondition(sd, inout, "dir")
        # In fractures we set trivial NBC
        else:
            return pp.BoundaryCondition(sd)

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if sd.dim == 2:
            inout = self._inlet_faces(sd) | self._outlet_faces(sd)
            heated = self._heated_faces(sd) | inout
            return pp.BoundaryCondition(sd, heated, "dir")
        # In fractures we set trivial NBC
        else:
            return pp.BoundaryCondition(sd)

    def bc_type_enthalpy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # return self.bc_type_darcy_flux(sd)
        if sd.dim == 2:
            return pp.BoundaryCondition(sd, self._inlet_faces(sd), "dir")
        else:
            return pp.BoundaryCondition(sd)

    def bc_type_fluid_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        return self.bc_type_enthalpy_flux(sd)

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sd = boundary_grid.parent
        sides = self.domain_boundary_sides(sd)

        if sd.dim == 2:
            vals = np.ones(sd.num_faces) * self._p_INIT

            inlet = self._inlet_faces(sd)
            outlet = self._outlet_faces(sd)

            vals[inlet] = self._p_IN
            vals[outlet] = self._p_OUT

            vals = vals[sides.all_bf]
        else:
            vals = np.zeros(boundary_grid.num_cells)

        return vals

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sd = boundary_grid.parent
        sides = self.domain_boundary_sides(sd)

        if sd.dim == 2:
            vals = np.ones(sd.num_faces) * self._T_INIT

            inlet = self._inlet_faces(sd)
            outlet = self._outlet_faces(sd)
            heated = self._heated_faces(sd)

            vals[inlet] = self._T_IN
            vals[outlet] = self._T_OUT
            vals[heated] = self._T_HEATED

            vals = vals[sides.all_bf]
        else:
            vals = np.zeros(boundary_grid.num_cells)

        return vals

    def bc_values_overall_fraction(
        self, component: pp.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        sd = boundary_grid.parent
        sides = self.domain_boundary_sides(sd)

        if sd.dim == 2:
            vals = np.ones(sd.num_faces) * self._z_INIT[component.name]

            inlet = self._inlet_faces(sd)
            outlet = self._outlet_faces(sd)

            vals[inlet] = self._z_IN[component.name]
            vals[outlet] = self._z_OUT[component.name]

            vals = vals[sides.all_bf]
        else:
            vals = np.zeros(boundary_grid.num_cells)

        return vals


fractional_flow: bool = False


if fractional_flow:

    class GeothermalFlow(  # type:ignore[misc]
        pp.constitutive_laws.DarcysLawAd,
        pp.constitutive_laws.FouriersLawAd,
        Geometry,
        SoereideMixture,
        SolutionStrategy,
        InitialConditions,
        BoundaryConditions,
        cfle.EnthalpyBasedCFFLETemplate,
    ): ...
else:

    class GeothermalFlow(  # type:ignore[misc,no-redef]
        Geometry,
        SoereideMixture,
        SolutionStrategy,
        InitialConditions,
        BoundaryConditions,
        cfle.EnthalpyBasedCFLETemplate,
    ): ...


# Model parametrization
t_scale = 1e-2

max_iterations = 25
newton_tol = 1e-5
newton_tol_increment = 1e-1

equilibrium_type: str = "unified-p-h"

flash_params = {
    "mode": "parallel",
    "solver": "npipm",
    "solver_params": {
        "tolerance": 1e-8,
        "max_iterations": 150,
        "armijo_rho": 0.99,
        "armijo_kappa": 0.4,
        "armijo_max_iterations": 50 if "p-T" in equilibrium_type else 30,
        "npipm_u1": 10,
        "npipm_u2": 10,
        "npipm_eta": 0.5,
    },
    "initial_guess_params": {
        "N1": 3,
        "N2": 1,
        "N3": 5,
        "tolerance": 1e-8,
    },
}

time_manager = pp.TimeManager(
    schedule=[0, 1 * pp.DAY * t_scale],
    dt_init=pp.HOUR * t_scale,
    dt_min_max=(pp.MINUTE * t_scale, 3 * pp.HOUR),
    iter_max=max_iterations,
    iter_optimal_range=(4, 7),
    iter_relax_factors=(0.9, 1.3),
    recomp_factor=0.5,
    recomp_max=5,
    print_info=True,
)

material_constants = {
    "solid": pp.SolidConstants(
        permeability=1e-16,
        porosity=0.2,
        thermal_conductivity=1.6736,
        specific_heat_capacity=603.0,
    )
}

params = {
    "equilibrium_type": equilibrium_type,
    "has_time_dependent_boundary_equilibrium": False,
    "eliminate_reference_phase": True,
    "eliminate_reference_component": True,
    "normalize_state_constraints": True,
    "use_semismooth_complementarity": True,
    "reduce_linear_system": False,
    "flash_params": flash_params,
    "fractional_flow": fractional_flow,
    "rediscretize_fourier_flux": fractional_flow,
    "rediscretize_darcy_flux": fractional_flow,
    "material_constants": material_constants,
    "grid_type": "simplex",
    "meshing_arguments": {
        "cell_size": 1.0,
        "cell_size_fracture": 0.5,
    },
    "time_manager": time_manager,
    "max_iterations": max_iterations,
    "nl_convergence_tol": newton_tol_increment,
    "nl_convergence_tol_res": newton_tol,
    "prepare_simulation": False,
    "progressbars": True,
}

model = GeothermalFlow(params)

t_0 = time.time()
model.prepare_simulation()
prep_sim_time = time.time() - t_0
compile_time += prep_sim_time

model.primary_equations = model.get_primary_equations_cf()
model.primary_variables = model.get_primary_variables_cf()

t_0 = time.time()
pp.run_time_dependent_model(model, params)
sim_time = time.time() - t_0

print(f"Finished prepare_simulation in {prep_sim_time} seconds.")
print(f"Finished simulation in {sim_time} seconds.")
