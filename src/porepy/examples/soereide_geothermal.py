"""This module implements a multiphase-multicomponent flow model with phase change
using the Soereide model to define the fluid mixture

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

from typing import Sequence, cast

import numpy as np

import porepy as pp
import porepy.composite as ppc
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.composite.peng_robinson.eos_c import PengRobinsonCompiler
from porepy.models.compositional_balance import (
    BoundaryConditionsCompositionalFlow,
    CompositionalFlow,
    InitialConditionsCompositionalFlow,
)


class SoereideMixture(ppc.FluidMixtureMixin):
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
        # This takes some time
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


class InitialConditions(InitialConditionsCompositionalFlow):
    """Define initial pressure, temperature and compositions."""

    def intial_pressure(self, sd: pp.Grid) -> np.ndarray:
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


class BoundaryConditions(BoundaryConditionsCompositionalFlow):
    """Boundary conditions defining a ``left to right`` flow in the matrix (2D)

    Mass flux:

    - No flux conditions on top and bottom
    - Dirichlet data (pressure, temperatyre and composition) on left and right faces

    Heat flux:

    - No flux on left, top, right
    - heated bottom side

    Trivial Neumann conditions for fractures.

    """

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        # Setting only conditions on matrix
        if sd.dim == 2:
            # Define boundary faces, east west as dirichlet
            boundary_faces = (
                self.domain_boundary_sides(sd).east
                | self.domain_boundary_sides(sd).west
            )
            # Define boundary condition on all boundary faces.
            return pp.BoundaryCondition(sd, boundary_faces, "dir")
        # In fractures we set trivial NBC
        else:
            return pp.BoundaryCondition(sd)

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        if sd.dim == 2:
            # Temperature at inlet and outlet, as well as heated bottom
            boundary_faces = (
                self.domain_boundary_sides(sd).east
                | self.domain_boundary_sides(sd).west
                | self.domain_boundary_sides(sd).bottom
            )
            return pp.BoundaryCondition(sd, boundary_faces, "dir")
        # In fractures we set trivial NBC
        else:
            return pp.BoundaryCondition(sd)

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        # need to define pressure on east and west side of matrix
        sd = boundary_grid.parent
        bs = self.domain_boundary_sides(sd)
        vals = np.zeros(sd.num_faces)
        if sd.dim == 2:
            inlet_faces = bs.west
            outlet_faces = bs.east
            vals[inlet_faces] = 15e6
            vals[outlet_faces] = 10e6

        vals = vals[bs.all_bf]
        assert vals.shape == (
            boundary_grid.num_cells,
        ), "Mismatch in shape of BC values."
        return vals

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        sd = boundary_grid.parent
        bs = self.domain_boundary_sides(sd)
        vals = np.zeros(sd.num_faces)
        # non-trivial BC on matrix
        if sd.dim == 2:
            # T values on inlet and outlet faces to compute boundary equilibrium
            inlet_faces = bs.west
            outlet_faces = bs.east
            vals[inlet_faces] = 460.0
            # outlet values correspond to initial conditions
            vals[outlet_faces] = 550.0

            # T values on heated bottom
            heated_faces = bs.bottom
            vals[heated_faces] = 600.0

        vals = vals[bs.all_bf]
        assert vals.shape == (
            boundary_grid.num_cells,
        ), "Mismatch in shape of BC values."
        return vals

    def bc_values_overall_fraction(
        self, component: ppc.Component, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        sd = boundary_grid.parent
        bs = self.domain_boundary_sides(sd)
        vals = np.zeros(sd.num_faces)
        if sd.dim == 2:
            inlet_faces = bs.west
            outlet_faces = bs.east

            # on inlet, more CO2 enters the system
            if component.name == "H2O":
                vals[inlet_faces] = 0.99
            elif component.name == "CO2":
                vals[inlet_faces] = 0.01
            else:
                raise NotImplementedError(
                    f"Initial overlal fraction not implemented for component {component.name}"
                )

            # on outlet we define something corresponding to initial conditions
            if component.name == "H2O":
                vals[outlet_faces] = 0.995
            elif component.name == "CO2":
                vals[outlet_faces] = 0.005
            else:
                raise NotImplementedError(
                    f"Initial overlal fraction not implemented for component {component.name}"
                )

        vals = vals[bs.all_bf]
        assert vals.shape == (
            boundary_grid.num_cells,
        ), "Mismatch in shape of BC values."
        return vals


class GeothermalFlow(
    ModelGeometry,
    SoereideMixture,
    CompiledFlash,
    InitialConditions,
    BoundaryConditions,
    CompositionalFlow,
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
# Target equilibrium is in terms of pressure and enthalpy.
# extended fractions are create because of equilibrium definition.
# BC are constant and the boundary equilibrium needs to be computed only once.
params = {
    "eliminate_reference_phase": True,
    "eliminate_reference_component": True,
    "normalize_state_constraints": True,
    "use_semismooth_complementarity": True,
    "has_time_dependent_boundary_equilibrium": False,
    "equilibrium_type": "p-h",
    "has_extended_fractions": True,
    "time_manager": time_manager,
}
model = GeothermalFlow(params)
pp.run_time_dependent_model(model, params)
pp.plot_grid(model.mdg, "pressure", figsize=(10, 8), plot_2d=True)
