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

import os

os.environ["NUMBA_DISABLE_JIT"] = "1"

import logging
import pathlib
import time

logging.basicConfig(level=logging.WARNING)
numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

from typing import Sequence, cast

import numpy as np

import porepy as pp
import porepy.composite as ppc
import porepy.composite.peng_robinson as ppcpr
import porepy.models.compositional_flow_with_equilibrium as cfle
from porepy.applications.md_grids.domains import nd_cube_domain


class SoereideMixture:
    """Model fluid using the Soereide mixture, a Peng-Robinson based EoS for
    NaCl brine with CO2, H2S and N2."""

    def get_components(self) -> Sequence[ppc.Component]:
        chems = ["H2O", "CO2"]
        species = ppc.load_species(chems)
        components = [
            ppcpr.H2O.from_species(species[0]),
            ppcpr.CO2.from_species(species[1]),
        ]
        return components

    def get_phase_configuration(
        self, components: Sequence[ppc.Component]
    ) -> Sequence[tuple[ppc.EoSCompiler, int, str]]:
        eos = ppcpr.PengRobinsonCompiler(components)
        return [(eos, 0, "liq"), (eos, 1, "gas")]


class CompiledFlash(ppc.FlashMixin):
    """Sets the compiled flash as the flash class, consistent with the EoS."""

    flash_params = {
        "mode": "parallel",
    }

    def set_up_flasher(self) -> None:
        eos = self.fluid_mixture.reference_phase.eos
        eos = cast(ppcpr.PengRobinsonCompiler, eos)  # cast type

        flash = ppc.CompiledUnifiedFlash(self.fluid_mixture, eos)

        # Compiling the flash and the EoS
        eos.compile()
        flash.compile()

        # NOTE There is place to configure the solver here
        flash.armijo_parameters["max_iter"] = 30
        flash.tolerance = 1e-8
        flash.max_iter = 150
        flash.heavy_ball_momentum = False

        # Setting the attribute of the mixin
        self.flash = flash

    def get_fluid_state(
        self, subdomains: Sequence[pp.Grid], state: np.ndarray | None = None
    ) -> ppc.FluidState:
        """Method to pre-process the evaluated fractions. Normalizes the extended
        fractions where they violate a certain threshold."""
        fluid_state = super().get_fluid_state(subdomains, state)

        unity_tolerance = 1.05

        for j, phase in enumerate(fluid_state.phases):
            x_sum = phase.x.sum(axis=0)
            if np.any(x_sum > unity_tolerance):
                raise ValueError(
                    f"Extended fractions in phase {j} violate unity constraint."
                )
            idx = x_sum > 1.0 + 1e-10
            phase.x[:, idx] = ppc.normalize_rows(phase.x[:, idx].T).T

        return fluid_state

    def postprocess_failures(
        self, fluid_state: ppc.FluidState, success: np.ndarray
    ) -> ppc.FluidState:
        """A post-processing where the flash is again attempted where not succesful.

        But the new attempt does not use iterate values as initial guesses, but computes
        the flash from scratch.

        """
        failure = success > 0
        if np.any(failure):
            sds = self.mdg.subdomains()

            print("failure at")
            print(
                "z: ",
                [
                    comp.fraction(sds).value(self.equation_system)[failure]
                    for comp in self.fluid_mixture.components
                ],
            )
            print("p: ", self.pressure(sds).value(self.equation_system)[failure])
            print("h: ", self.enthalpy(sds).value(self.equation_system)[failure])
            # no initial guess, and this model uses only p-h flash.
            flash_kwargs = {
                "z": [
                    comp.fraction(sds).value(self.equation_system)[failure]
                    for comp in self.fluid_mixture.components
                ],
                "p": self.pressure(sds).value(self.equation_system)[failure],
                "h": self.enthalpy(sds).value(self.equation_system)[failure],
                "parameters": self.flash_params,
            }

            sub_state, sub_success, _ = self.flash.flash(**flash_kwargs)

            # update parent state with sub state values
            success[failure] = sub_success
            fluid_state.T[failure] = sub_state.T

            for j in range(len(fluid_state.phases)):
                fluid_state.sat[j][failure] = sub_state.sat[j]
                fluid_state.y[j][failure] = sub_state.y[j]

                fluid_state.phases[j].x[:, failure] = sub_state.phases[j].x

        # Parent method performs a check that everything is successful.
        return super().postprocess_failures(fluid_state, success)


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
        cell_size = self.solid.convert_units(0.1, "m")
        cell_size_fracture = self.solid.convert_units(0.05, "m")
        mesh_args: dict[str, float] = {
            "cell_size": cell_size,
            "cell_size_fracture": cell_size_fracture,
        }
        return mesh_args


class InitialConditions:
    """Define initial pressure, temperature and compositions."""

    _p_INIT: float = 15e6
    _T_INIT: float = 550.0
    _z_INIT: dict[str, float] = {"H2O": 0.995, "CO2": 0.005}

    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        # Initial pressure of 10 MPa
        return np.ones(sd.num_cells) * self._p_INIT

    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        # Initial temperature of 550 K
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

    has_time_dependent_boundary_equilibrium = False
    """Constant BC for primary variables, hence constant BC for all other."""

    _p_INIT: float
    _T_INIT: float
    _z_INIT: dict[str, float]

    _p_IN: float = 16e6
    _p_OUT: float = 15e6

    _T_IN: float = 550.0
    _T_OUT: float = 550.0
    _T_HEATED: float = 550.0

    _z_IN: dict[str, float] = {"H2O": 0.99, "CO2": 0.01}
    _z_OUT: dict[str, float] = {"H2O": 0.995, "CO2": 0.005}

    def _inlet_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define inlet."""
        sides = self.domain_boundary_sides(sd)

        inlet = np.zeros(sd.num_faces, dtype=bool)
        inlet[sides.west] = True
        inlet &= sd.face_centers[1] > 0.2
        inlet &= sd.face_centers[1] < 0.8

        return inlet

    def _outlet_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define outlet."""

        sides = self.domain_boundary_sides(sd)

        outlet = np.zeros(sd.num_faces, dtype=bool)
        outlet[sides.east] = True
        outlet &= sd.face_centers[1] > 0.2
        outlet &= sd.face_centers[1] < 0.8

        return outlet

    def _heated_faces(self, sd: pp.Grid) -> np.ndarray:
        """Define heated boundary with D-type conditions for conductive flux."""
        sides = self.domain_boundary_sides(sd)

        heated = np.zeros(sd.num_faces, dtype=bool)
        heated[sides.bottom] = True
        heated &= sd.face_centers[1] > 0.5
        heated &= sd.face_centers[1] < 1.5

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
            heated = self._heated_faces(sd)
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
T_end = 100 * days * t_scale
dt_init = 1 * days * t_scale
max_iterations = 80
newton_tol = 1e-5

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
    {"permeability": 1e-11, "porosity": 0.2, "thermal_conductivity": 1.92}
)
material_constants = {"solid": solid_constants}

restart_options = {
    "restart": True,
    "is_mdg_pvd": True,
    "pdv_file": pathlib.Path("./visualization/data.pvd"),
}

# Model setup:
# eliminate reference phase fractions  and reference component.
params = {
    "material_constants": material_constants,
    "eliminate_reference_phase": True,
    "eliminate_reference_component": True,
    "normalize_state_constraints": True,
    "use_semismooth_complementarity": True,
    "reduce_linear_system_q": False,
    "time_manager": time_manager,
    "max_iterations": max_iterations,
    "nl_convergence_tol": newton_tol,
    "prepare_simulation": False,
    "progressbars": True,
    # 'restart_options': restart_options,  # NOTE no yet fully integrated in porepy
}
model = GeothermalFlow(params)

model.equilibrium_type = "p-T"

start = time.time()
model.prepare_simulation()
print(f"Finished prepare_simulation in {time.time() - start} seconds")


result, success, num_iter = model.flash.flash(
    z=np.array([0.995, 0.005]),
    p=np.array(15e6),
    T=np.array([550.0]),
)

pp.run_time_dependent_model(model, params)
pp.plot_grid(model.mdg, "pressure", figsize=(10, 8), plot_2d=True)
