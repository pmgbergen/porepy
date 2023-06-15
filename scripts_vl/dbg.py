import numpy as np

import porepy as pp

chems = ["H2O", "CO2"]

vec = np.ones(1)
z = [vec * 0.01]  # only co2 fraction is enough
salt = vec * 0.0141289
p = vec * 9.5911e6
T = vec * 550.0
h = vec * 5.6709e3
verbosity = 2

species = pp.composite.load_species(chems)

comps = [
    pp.composite.peng_robinson.NaClBrine.from_species(species[0]),
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

[
    mix.system.set_variable_values(val, [comp.fraction.name], 0, 0)
    for val, comp in zip(z, comps)
]

comps[0].compute_molalities(salt, store=True)

flash = pp.composite.FlashNR(mix)
flash.use_armijo = True
flash.armijo_parameters["rho"] = 0.99
flash.armijo_parameters["j_max"] = 200
flash.armijo_parameters["return_max"] = True
flash.newton_update_chop = 1.0
flash.tolerance = 1e-5
flash.max_iter = 120

success, results_pT = flash.flash(
    state={"p": p, "h": h},
    eos_kwargs={"apply_smoother": True},
    feed=z,
    verbosity=verbosity,
)
print(results_pT)
