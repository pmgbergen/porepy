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

t = np.array([0.01, 1e6, 300, 0.9, 0.1, 0.2, 0.3, 0.4])

print(PRC.equations['p-T'](t))
print(PRC.jacobians['p-T'](t))