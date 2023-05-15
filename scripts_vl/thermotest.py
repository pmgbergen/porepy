import numpy as np
from thermo import (  # PRMIX,; FlashVL,
    PR78MIX,
    CEOSGas,
    CEOSLiquid,
    ChemicalConstantsPackage,
    FlashVLN,
)
from thermo.interaction_parameters import IPDB

COMPONENTS = ['H2O', 'CO2']
MAX_LIQ_PHASES = 1
z_co2 = 0.01
p = 2e6
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
print("Gas fraction: ", results.VF)
if results.liquids:
    print("Liquid Comp: ", results.liquid0.zs)
if results.gas:
    print("Gas Comp: ", results.gas.zs)

print('')