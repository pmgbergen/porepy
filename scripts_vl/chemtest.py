import porepy as pp
import numpy as np

chems = ["H2O"]

species  = pp.composite.load_fluid_species(chems)

comps = [
    pp.composite.peng_robinson.H2O.from_species(species[0]),
    # pp.composite.peng_robinson.N2.from_species(species[1]),
    # pp.composite.peng_robinson.CO2.from_species(species[2]),
    # pp.composite.peng_robinson.H2S.from_species(species[3]),
]

eos = pp.composite.peng_robinson.PengRobinsonEoS(True)

p = np.array([0.101325])
T = np.array([380])
# X = [np.array([0.25]) for _ in comps]
X = [np.array([1])]

eos.components = comps

props = eos.compute(p, T, X, True)
# Z = eos._Z(np.array([0.]), np.array([0.]))

print("---")