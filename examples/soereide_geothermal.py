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
from typing import Sequence

import porepy as pp
import porepy.composite as ppc

from porepy.composite.peng_robinson.eos_c import PengRobinsonCompiler
from porepy.models.fluid_mixture_equilibrium import MixtureMixin
from porepy.models.compositional_balance import CompositionalFlow


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
        return [(eos, 0, 'liq'), (eos, 1, 'gas')]


class GeothermalFlow(
    SoereideMixture,
    CompositionalFlow,
):
    """Geothermal flow using a fluid defined by the Soereide model."""

time_manager = pp.TimeManager(
    schedule=[0, 0.3, 0.6],
    dt_init=1e-4,
    constant_dt=False,
    iter_max=50,
    print_info=True,
)

params = {
    'eliminate_reference_phase' : True,
    'eliminate_reference_component' : True,
    'normalize_state_constraints' : True,
    'use_semismooth_complementarity' : True,
    'time_manager' : time_manager,
}
model = GeothermalFlow(params)
pp.run_time_dependent_model(model, params)
pp.plot_grid(model.mdg, "pressure", figsize=(10, 8), plot_2d=True)