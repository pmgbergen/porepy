import numpy as np
import porepy as pp

chems = ["H2O", "CO2"]

z1 = np.ones(1) * 1e-5
p = np.ones(1) * 15
T = np.ones(1) * 643.8775510204082
X = [z1, 1 - z1]

species = pp.composite.load_fluid_species(chems)

comps = [
    pp.composite.peng_robinson.H2O.from_species(species[0]),
    pp.composite.peng_robinson.CO2.from_species(species[1]),
]

eos_l = pp.composite.peng_robinson.PengRobinsonEoS(False)
eos_l.components = comps

prop_l = eos_l.compute(p, T, X)
G_l = eos_l._g_ideal(T, X) + eos_l._g_dep(prop_l.A, prop_l.B, prop_l.Z)
