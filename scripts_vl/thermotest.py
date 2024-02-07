import numpy as np
from thermo import (  # PRMIX,; FlashVL,
    PR78MIX,
    CEOSGas,
    CEOSLiquid,
    ChemicalConstantsPackage,
    FlashVLN,
)
from thermo.interaction_parameters import IPDB

import porepy as pp

COMPONENTS = ["H2O", "CO2"]
MAX_LIQ_PHASES = 1
z_co2 = 0.01
p = 7e6
T = 400

constants, properties = ChemicalConstantsPackage.from_IDs(COMPONENTS)
kijs = IPDB.get_ip_asymmetric_matrix("ChemSep PR", constants.CASs, "kij")
eos_kwargs = {
    "Pcs": constants.Pcs,
    "Tcs": constants.Tcs,
    "omegas": constants.omegas,
    "kijs": kijs,
}

GAS = CEOSGas(
    PR78MIX, eos_kwargs=eos_kwargs, HeatCapacityGases=properties.HeatCapacityGases
)
LIQs = [
    CEOSLiquid(
        PR78MIX,
        eos_kwargs=eos_kwargs,
        HeatCapacityGases=properties.HeatCapacityGases,
    )
    for _ in range(MAX_LIQ_PHASES)
]

flasher = FlashVLN(constants, properties, liquids=LIQs, gas=GAS)
results = flasher.flash(P=p, T=T, zs=[1 - z_co2, z_co2])

eos_g = pp.composite.peng_robinson.PengRobinson(True)
eos_l = pp.composite.peng_robinson.PengRobinson(False)
species = pp.composite.load_species(["H2O", "CO2"])
comps = [
    pp.composite.peng_robinson.H2O.from_species(species[0]),
    pp.composite.peng_robinson.CO2.from_species(species[1]),
]
eos_g.components = comps
eos_l.components = comps

p_ = np.array([p]) * 1e-6
T_ = np.array([T])
z_ = np.array([z_co2])

print("Gas fraction: ", results.VF)
if results.liquids:
    print("Liquid Comp: ", results.liquid0.zs)
    phis_l = eos_l.compute(p_, T_, [1 - z_, z_]).phis
    print("thermo phis: ", results.liquid0.phis())
    print("my phis: ", phis_l)
if results.gas:
    print("Gas Comp: ", results.gas.zs)
    print("thermo phis: ", results.gas.phis())
    phis_g = eos_g.compute(p_, T_, [1 - z_, z_]).phis
    print("my phis: ", phis_g)

print("")
