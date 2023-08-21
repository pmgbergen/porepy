"""This script is an example of how to run a flash using the PorePy package.

It is devised for the publication
``Unified flash calculations with isenthalpic and isochoric constraints``
and does not necessarily work with other versions of PorePy.

Use only in combination with the version contained in the publication-accompanying
docker image.

"""

import numpy as np

import porepy as pp

chems = ["H2O", "CO2"]

vec = np.ones(1)
z = [vec * 0.01]  # only co2 fraction is enough
p = vec * 13000000.0
T = vec * 550.0
verbosity = 2

### Setting up the fluid mixture
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

### Setting feed fraction values using PorePy's AD framework
[
    mix.system.set_variable_values(val, [comp.fraction.name], 0, 0)
    for val, comp in zip(z, comps)
]

### Flash object and solver settings
flash = pp.composite.FlashNR(mix)
flash.use_armijo = True
flash.armijo_parameters["rho"] = 0.99
flash.armijo_parameters["j_max"] = 150
flash.armijo_parameters["return_max"] = True
flash.newton_update_chop = 1.0
flash.tolerance = 1e-5
flash.max_iter = 150

### p-T flash
### Other flash types are performed analogously.
success, results_pT = flash.flash(
    state={"p": p, "T": T},
    eos_kwargs={"apply_smoother": True},
    feed=z,
    verbosity=verbosity,
)
print("Results p-T:\n" + "------------")
print(str(results_pT))
print("------------")
