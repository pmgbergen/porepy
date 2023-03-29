import matplotlib.pyplot as plt
import numpy as np

from thermo import (  # PRMIX,; FlashVL,
    PR78MIX,
    CEOSGas,
    CEOSLiquid,
    ChemicalConstantsPackage,
    FlashVLN,
)
from thermo.interaction_parameters import IPDB

COMPONENTS = ['water', 'CO2']
MAX_LIQ_PHASES = 2


def _init_thermo() -> FlashVLN:
    """Helper function to initiate the thermo flasher and the results data structure."""
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

    return flasher


flasher = _init_thermo()
state= flasher.flash(V=0.5, U=-300, zs=[0.99, 0.1])
print('done')