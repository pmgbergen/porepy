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
from porepy.models.compositional_balance import CompositionalFlow
from porepy.models.fluid_mixture_equilibrium import MixtureMixin, EquilibriumEquationsMixin, FlashMixin


class SoereideMixture(MixtureMixin):
    """Model fluid using the Soereide mixture, a Peng-Robinson based EoS for
    NaCl brine with CO2, H2S and N2.

    """

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
    
class EquilibriumMixin(EquilibriumEquationsMixin):
    """Defines the p-h flash as the target equilibrium system."""

    equilibrium_type = 'p-h'


class CompiledFlash(FlashMixin):
    """Sets the compiled flash as the flash class, consistent with the EoS."""

    def set_up_flasher(self) -> None:

        eos = self.fluid_mixture.reference_phase.eos
        eos = cast(PengRobinsonCompiler, eos)  # cast type

        flash = ppc.CompiledUnifiedFlash(self.fluid_mixture, eos)

        # Compiling the flash and the EoS
        eos.compile(verbosity=2)
        flash.compile(verbosity=2, precompile_solvers=True)

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


class GeothermalFlow(
    ModelGeometry,
    SoereideMixture,
    EquilibriumMixin,
    CompiledFlash,
    CompositionalFlow,
):
    """Geothermal flow using a fluid defined by the Soereide model and the compiled
    flash."""


time_manager = pp.TimeManager(
    schedule=[0, 0.3, 0.6],
    dt_init=1e-4,
    constant_dt=False,
    iter_max=50,
    print_info=True,
)

params = {
    "eliminate_reference_phase": True,
    "eliminate_reference_component": True,
    "normalize_state_constraints": True,
    "use_semismooth_complementarity": True,
    "time_manager": time_manager,
}
model = GeothermalFlow(params)
pp.run_time_dependent_model(model, params)
pp.plot_grid(model.mdg, "pressure", figsize=(10, 8), plot_2d=True)
