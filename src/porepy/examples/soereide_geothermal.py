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
import pathlib
import time

# os.environ["NUMBA_DISABLE_JIT"] = "1"
compile_time = 0.0
logging.basicConfig(level=logging.INFO)
logging.getLogger("porepy").setLevel(logging.INFO)

from typing import Sequence, cast

import numpy as np

import porepy as pp

t_0 = time.time()
import porepy.compositional as ppc
import porepy.compositional.peng_robinson as ppcpr
from porepy.compositional.peng_robinson.pr_utils import (
    get_bip_matrix,
    h_ideal_CO2,
    h_ideal_H2O,
)

compile_time += time.time() - t_0

import porepy.models.compositional_flow_with_equilibrium as cfle
from porepy.applications.md_grids.domains import nd_cube_domain

logger = logging.getLogger(__name__)


class SoereideMixture:
    """Model fluid using the Soereide mixture, a Peng-Robinson based EoS for
    NaCl brine with CO2, H2S and N2."""

    def get_components(self) -> Sequence[ppc.Component]:
        components = [
            ppc.Component.from_species(s) for s in ppc.load_species(["H2O", "CO2"])
        ]
        return components

    def get_phase_configuration(
        self, components: Sequence[ppc.Component]
    ) -> Sequence[tuple[ppc.EoSCompiler, int, str]]:

        eos = ppcpr.PengRobinsonCompiler(
            components, [h_ideal_H2O, h_ideal_CO2], get_bip_matrix(components)
        )

        return [(eos, ppc.PhysicalState.liquid, "L"), (eos, ppc.PhysicalState.gas, "G")]


class CompiledFlash(ppc.FlashMixin):
    """Sets the compiled flash as the flash class, consistent with the EoS."""

    flash_params = {
        "mode": "parallel",
    }

    def set_up_flasher(self) -> None:
        eos = self.fluid_mixture.reference_phase.eos
        eos = cast(ppcpr.PengRobinsonCompiler, eos)

        flash = ppc.CompiledUnifiedFlash(self.fluid_mixture, eos)

        # Compiling the flash and the EoS
        eos.compile()
        flash.compile()

        # Configuring solver for the requested equilibrium type
        flash.tolerance = 1e-8
        flash.max_iter = 150
        flash.heavy_ball_momentum = False
        flash.armijo_parameters["rho"] = 0.99
        flash.armijo_parameters["kappa"] = 0.4
        flash.armijo_parameters["max_iter"] = (
            50 if "p-T" in self.equilibrium_type else 30
        )
        flash.npipm_parameters["u1"] = 10.0
        flash.npipm_parameters["u2"] = (
            10.0  # if 'p-T' in self.equilibrium_type else 1.0
        )
        flash.npipm_parameters["eta"] = 0.5
        flash.initialization_parameters["N1"] = 3
        flash.initialization_parameters["N2"] = 1
        flash.initialization_parameters["N3"] = 5
        flash.initialization_parameters["eps"] = flash.tolerance

        # Setting the attribute of the mixin
        self.flash = flash

    def get_fluid_state(
        self, subdomains: Sequence[pp.Grid], state: np.ndarray | None = None
    ) -> ppc.FluidProperties:
        """Method to pre-process the evaluated fractions. Normalizes the extended
        fractions where they violate the unity constraint."""
        fluid_state = super().get_fluid_state(subdomains, state)

        # sanity checks
        p = self.equation_system.get_variable_values(
            [self.pressure_variable], iterate_index=0
        )
        T = self.equation_system.get_variable_values(
            [self.temperature_variable], iterate_index=0
        )
        if np.any(p <= 0.0):
            raise ValueError("Pressure diverged to negative values.")
        if np.any(T <= 0):
            raise ValueError("Temperature diverged to negative values.")

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
            fluid_state.y[:, idx] = ppc.normalize_rows(fluid_state.y[:, idx].T).T

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
                phase.x[:, idx] = ppc.normalize_rows(phase.x[:, idx].T).T
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
        self, subdomain: pp.Grid, fluid_state: ppc.FluidProperties, success: np.ndarray
    ) -> ppc.FluidProperties:
        """A post-processing where the flash is again attempted where not succesful.

        1. Where not successful, it re-attempts the flash from scratch, with
           initialization procedure.
           If the new attempt did not reach the desired precision (max iter), it
           treats it as success.
        2. Where still not successful, it falls back to the values of the previous
           iterate.

        """
        failure = success > 0
        if np.any(failure):
            logger.warning(
                f"Flash from iterate state failed in {failure.sum()} cases."
                + " Re-starting with computation of initial guess."
            )
            z = [
                comp.fraction([subdomain]).value(self.equation_system)[failure]
                for comp in self.fluid_mixture.components
            ]
            p = self.pressure([subdomain]).value(self.equation_system)[failure]
            h = self.enthalpy([subdomain]).value(self.equation_system)[failure]
            T = self.temperature([subdomain]).value(self.equation_system)[failure]

            logger.info(f"Failed at\nz: {z}\np: {p}\nT: {T}\nh: {h}")
            # no initial guess, and this model uses only p-h flash.
            flash_kwargs = {
                "z": z,
                "p": p,
                "parameters": self.flash_params,
            }

            if "p-h" in self.equilibrium_type:
                flash_kwargs["h"] = h
            elif "p-T" in self.equilibrium_type:
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
        return super().postprocess_flash(subdomain, fluid_state, success)

    def fall_back(
        self,
        grid: pp.Grid,
        sub_state: ppc.FluidProperties,
        failed_idx: np.ndarray,
        fallback_idx: np.ndarray,
    ) -> ppc.FluidProperties:
        """Falls back to previous iterate values for the fluid state, on given
        grid."""
        sds = [grid]
        data = self.mdg.subdomain_data(grid)
        if "T" not in self.equilibrium_type:
            sub_state.T[fallback_idx] = pp.get_solution_values(
                self.temperature_variable, data, iterate_index=0
            )[failed_idx][fallback_idx]
        if "h" not in self.equilibrium_type:
            sub_state.h[fallback_idx] = pp.get_solution_values(
                self.pressure_variable, data, iterate_index=0
            )[failed_idx][fallback_idx]
        if "p" not in self.equilibrium_type:
            sub_state.p[fallback_idx] = pp.get_solution_values(
                self.enthalpy_variable, data, iterate_index=0
            )[failed_idx][fallback_idx]

        for j, phase in enumerate(self.fluid_mixture.phases):
            if j > 0:
                sub_state.sat[j][fallback_idx] = pp.get_solution_values(
                    phase.saturation(sds).name, data, iterate_index=0
                )[failed_idx][fallback_idx]
                sub_state.y[j][fallback_idx] = pp.get_solution_values(
                    phase.fraction(sds).name, data, iterate_index=0
                )[failed_idx][fallback_idx]

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
                phase.conductivity.name, data, iterate_index=0
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
                phase.conductivity._name_derivatives, data, iterate_index=0
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
        sub_state.y[0, :] = 1 - np.sum(sub_state.y[1:, :], axis=0)
        sub_state.sat[0, :] = 1 - np.sum(sub_state.sat[1:, :], axis=0)

        return sub_state


class ModelGeometry:
    def set_domain(self) -> None:
        size = self.solid.convert_units(1, "m")
        self._domain = nd_cube_domain(2, size)

    # def set_fractures(self) -> None:
    #     """Setting a diagonal fracture"""
    #     frac_1_points = self.solid.convert_units(
    #         np.array([[0.2, 1.8], [0.2, 1.8]]), "m"
    #     )
    #     frac_1 = pp.LineFracture(frac_1_points)
    #     self._fractures = [frac_1]

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.05, "m")
        cell_size_fracture = self.solid.convert_units(0.05, "m")
        mesh_args: dict[str, float] = {
            "cell_size": cell_size,
            "cell_size_fracture": cell_size_fracture,
        }
        return mesh_args


class InitialConditions:
    """Define initial pressure, temperature and compositions."""

    _p_IN: float
    _p_OUT: float

    _p_INIT: float = 12e6
    _T_INIT: float = 540.0
    _z_INIT: dict[str, float] = {"H2O": 0.995, "CO2": 0.005}

    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        # f = lambda x: self._p_IN + x * (self._p_OUT - self._p_IN)
        # vals = np.array(list(map(f, sd.cell_centers[0])))
        # return vals
        return np.ones(sd.num_cells) * self._p_INIT

    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        return np.ones(sd.num_cells) * self._T_INIT

    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        return np.ones(sd.num_cells) * self._z_INIT[component.name]


class BoundaryConditions:
    """Boundary conditions defining a ``left to right`` flow in the matrix (2D)

    Mass flux:

    - No flux conditions on top and bottom
    - Dirichlet data (pressure, temperatyre and composition) on left and right faces

    Heat flux:

    - No flux on left, top, right
    - heated bottom side (given by temperature as a Dirichlet-type BC)

    Trivial Neumann conditions for fractures.

    """

    mdg: pp.MixedDimensionalGrid

    _p_INIT: float
    _T_INIT: float
    _z_INIT: dict[str, float]

    _p_IN: float = 15e6
    _p_OUT: float = InitialConditions._p_INIT

    _T_IN: float = InitialConditions._T_INIT
    _T_OUT: float = InitialConditions._T_INIT
    _T_HEATED: float = InitialConditions._T_INIT

    _z_IN: dict[str, float] = {
        "H2O": InitialConditions._z_INIT["H2O"] - 0.105,
        "CO2": InitialConditions._z_INIT["CO2"] + 0.105,
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
        inlet &= sd.face_centers[1] > 0.2
        inlet &= sd.face_centers[1] < 0.5

        return inlet

    def _outlet_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define outlet."""

        sides = self.domain_boundary_sides(sd)

        outlet = np.zeros(sd.num_faces, dtype=bool)
        outlet[sides.east] = True
        outlet &= sd.face_centers[1] > 0.75
        outlet &= sd.face_centers[1] < 0.99

        return outlet

    def _heated_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define heated boundary with D-type conditions for conductive flux."""
        sides = self.domain_boundary_sides(sd)

        heated = np.zeros(sd.num_faces, dtype=bool)
        heated[sides.south] = True
        heated &= sd.face_centers[0] > 0.2
        heated &= sd.face_centers[0] < 0.8

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

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:

        sd = boundary_grid.parent
        sides = self.domain_boundary_sides(sd)

        if sd.dim == 2:
            vals = np.ones(sd.num_faces) * self._p_INIT

            inlet = self._inlet_faces(sd)
            outlet = self._outlet_faces(sd)

            # vals[sides.west] = 10e6
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
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
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


class GeothermalFlow(
    pp.constitutive_laws.DarcysLawAd,
    pp.constitutive_laws.FouriersLawAd,
    ModelGeometry,
    SoereideMixture,
    CompiledFlash,
    InitialConditions,
    BoundaryConditions,
    cfle.CFLEModelMixin_ph,
):
    """Geothermal flow using a fluid defined by the Soereide model and the compiled
    flash."""

    def after_nonlinear_failure(self):
        self.exporter.write_pvd()
        super().after_nonlinear_failure()


days = 365
t_scale = 1e-5
T_end = 40 * days * t_scale
dt_init = 1 * days * t_scale
max_iterations = 80
newton_tol = 1e-6
newton_tol_increment = newton_tol

time_manager = pp.TimeManager(
    schedule=[0, T_end],
    dt_init=dt_init,
    dt_min_max=(0.1 * dt_init, 2 * dt_init),
    iter_max=max_iterations,
    iter_optimal_range=(2, 10),
    iter_relax_factors=(0.9, 1.1),
    recomp_factor=0.1,
    recomp_max=5,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    {
        "permeability": 1e-16,
        "porosity": 0.2,
        "thermal_conductivity": 1.6736,
        "specific_heat_capacity": 603.0,
    }
)
material_constants = {"solid": solid_constants}

restart_options = {
    "restart": True,
    "is_mdg_pvd": True,
    "pvd_file": pathlib.Path("./visualization/data.pvd"),
}

# Model setup:
# eliminate reference phase fractions  and reference component.
params = {
    "material_constants": material_constants,
    "equilibrium_type": "unified-p-h",
    "has_time_dependent_boundary_equilibrium": False,
    "eliminate_reference_phase": True,
    "eliminate_reference_component": True,
    "normalize_state_constraints": True,
    "use_semismooth_complementarity": True,
    "reduce_linear_system_q": False,
    "time_manager": time_manager,
    "max_iterations": max_iterations,
    "nl_convergence_tol": newton_tol_increment,
    "nl_convergence_tol_res": newton_tol,
    "prepare_simulation": False,
    "progressbars": True,
    # 'restart_options': restart_options,  # NOTE no yet fully integrated in porepy
}
model = GeothermalFlow(params)

t_0 = time.time()
model.prepare_simulation()
prep_sim_time = time.time() - t_0
compile_time += prep_sim_time
t_0 = time.time()
pp.run_time_dependent_model(model, params)
sim_time = time.time() - t_0
print(f"Finished prepare_simulation in {prep_sim_time} seconds.")
print(f"Finished simulation in {sim_time} seconds.")
