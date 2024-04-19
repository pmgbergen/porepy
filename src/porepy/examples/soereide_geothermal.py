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
from typing import Sequence, cast

import numpy as np

import porepy as pp
import porepy.composite as ppc
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.models.compositional_flow_with_equilibrium import CFLEModelMixin_ph

ppc.COMPOSITE_LOGGER.setLevel(logging.DEBUG)  # prints informatin about compile progress
from porepy.composite.peng_robinson import PengRobinsonCompiler

ppc.COMPOSITE_LOGGER.setLevel(logging.INFO)


class SoereideMixture:
    """Model fluid using the Soereide mixture, a Peng-Robinson based EoS for
    NaCl brine with CO2, H2S and N2."""

    def get_components(self) -> Sequence[ppc.Component]:
        chems = ["H2O", "CO2"]
        species = ppc.load_species(chems)
        components = [
            ppc.peng_robinson.H2O.from_species(species[0]),
            ppc.peng_robinson.CO2.from_species(species[1]),
        ]
        return components

    def get_phase_configuration(
        self, components: Sequence[ppc.Component]
    ) -> Sequence[tuple[ppc.EoSCompiler, int, str]]:
        eos = PengRobinsonCompiler(components)
        return [(eos, 0, "liq"), (eos, 1, "gas")]


class CompiledFlash(ppc.FlashMixin):
    """Sets the compiled flash as the flash class, consistent with the EoS."""

    flash_params = {
        "mode": "parallel",
        "verbosity": 1,
    }

    def set_up_flasher(self) -> None:
        eos = self.fluid_mixture.reference_phase.eos
        eos = cast(PengRobinsonCompiler, eos)  # cast type

        flash = ppc.CompiledUnifiedFlash(self.fluid_mixture, eos)

        # Compiling the flash and the EoS
        eos.compile(verbosity=2)
        # pre-compile solvers for given mixture to avoid waiting times in
        # prepare simulation and the first iteration
        flash.compile(verbosity=2, precompile_solvers=False)

        # NOTE There is place to configure the solver here

        # Setting the attribute of the mixin
        self.flash = flash


class ModelGeometry:
    def set_domain(self) -> None:
        size = self.solid.convert_units(2, "m")
        self._domain = nd_cube_domain(2, size)

    def set_fractures(self) -> None:
        """Setting a diagonal fracture"""
        frac_1_points = self.solid.convert_units(
            np.array([[0.2, 1.8], [0.2, 1.8]]), "m"
        )
        frac_1 = pp.LineFracture(frac_1_points)
        self._fractures = [frac_1]

    def grid_type(self) -> str:
        return self.params.get("grid_type", "simplex")

    def meshing_arguments(self) -> dict:
        cell_size = self.solid.convert_units(0.25, "m")
        mesh_args: dict[str, float] = {"cell_size": cell_size}
        return mesh_args


class InitialConditions:
    """Define initial pressure, temperature and compositions."""

    def initial_pressure(self, sd: pp.Grid) -> np.ndarray:
        # Initial pressure of 10 MPa
        return np.ones(sd.num_cells) * 10e6

    def initial_temperature(self, sd: pp.Grid) -> np.ndarray:
        # Initial temperature of 550 K
        return np.ones(sd.num_cells) * 550.0

    def initial_overall_fraction(
        self, component: ppc.Component, sd: pp.Grid
    ) -> np.ndarray:
        # Homogenous initial composition, with 0.5 % CO2
        if component.name == "H2O":
            return np.ones(sd.num_cells) * 0.995
        elif component.name == "CO2":
            return np.ones(sd.num_cells) * 0.005
        else:
            raise NotImplementedError(
                f"Initial overlal fraction not implemented for component {component.name}"
            )


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

    has_time_dependent_boundary_equilibrium = False
    """Constant BC for primary variables, hence constant BC for all other."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        # Setting only conditions on matrix
        if sd.dim == 2:
            # Define boundary condition on all boundary faces.
            return pp.BoundaryCondition(sd, sides.east | sides.west, "dir")
        # In fractures we set trivial NBC
        else:
            return pp.BoundaryCondition(sd)

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        sides = self.domain_boundary_sides(sd)
        if sd.dim == 2:
            # Temperature at inlet and outlet, as well as heated bottom
            boundary_faces = sides.east | sides.west | sides.bottom
            return pp.BoundaryCondition(sd, boundary_faces, "dir")
        # In fractures we set trivial NBC
        else:
            return pp.BoundaryCondition(sd)

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        # need to define pressure on east and west side of matrix
        p_init = 10e6
        sd = boundary_grid.parent
        sides = self.domain_boundary_sides(sd)

        # non-trivial BC on matrix
        if sd.dim == 2:
            vals = np.ones(sd.num_faces) * p_init

            vals[sides.west] = 15e6
            vals[sides.east] = 10e6

            vals = vals[sides.all_bf]
        else:
            vals = np.zeros(boundary_grid.num_cells)

        return vals

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        T_init = 550.0
        sd = boundary_grid.parent
        sides = self.domain_boundary_sides(sd)

        # non-trivial BC on matrix
        if sd.dim == 2:
            vals = np.ones(sd.num_faces) * T_init

            vals[sides.west] = 460.0
            vals[sides.east] = 550.0
            # T values on heated bottom
            vals[sides.bottom] = 600.0

            vals = vals[sides.all_bf]
        else:
            vals = np.zeros(boundary_grid.num_cells)

        return vals

    def bc_values_overall_fraction(
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        sd = boundary_grid.parent
        sides = self.domain_boundary_sides(sd)

        if component.name == "H2O":
            z_init = 0.995
            z_inlet = 0.99
            z_outlet = z_init
        elif component.name == "CO2":
            z_init = 0.005
            z_inlet = 0.01
            z_outlet = z_init
        else:
            NotImplementedError(
                f"Initial overlal fraction not implemented for component {component.name}"
            )

        if sd.dim == 2:
            vals = np.ones(sd.num_faces) * z_init

            vals[sides.west] = z_inlet
            vals[sides.east] = z_outlet

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
    CFLEModelMixin_ph,
):
    """Geothermal flow using a fluid defined by the Soereide model and the compiled
    flash."""


time_manager = pp.TimeManager(
    schedule=[0, 0.3, 0.6],
    dt_init=1e-4,
    dt_min_max=[1e-4, 0.1],
    constant_dt=False,
    iter_max=50,
    print_info=True,
)

# Model setup:
# eliminate reference phase fractions  and reference component.
params = {
    "eliminate_reference_phase": True,
    "eliminate_reference_component": True,
    "normalize_state_constraints": True,
    "use_semismooth_complementarity": True,
    "reduce_linear_system_q": True,
    "time_manager": time_manager,
}
model = GeothermalFlow(params)
pp.run_time_dependent_model(model, params)
pp.plot_grid(model.mdg, "pressure", figsize=(10, 8), plot_2d=True)
