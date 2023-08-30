import numpy as np

import porepy as pp

chems = ["H2O", "CO2"]
species = pp.composite.load_species(chems)
comps = [
    pp.composite.peng_robinson.H2O.from_species(species[0]),
    pp.composite.peng_robinson.CO2.from_species(species[1]),
]

phases = [
    pp.composite.Phase(
        pp.composite.peng_robinson.PengRobinsonEoS(gaslike=False), name="L"
    ),
    pp.composite.Phase(
        pp.composite.peng_robinson.PengRobinsonEoS(gaslike=True), name="G"
    ),
]

mix = pp.composite.NonReactiveMixture(comps, phases)

mix.set_up()

PRC = pp.composite.peng_robinson.PR_Compiler(mix)