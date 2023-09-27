import numpy as np

import porepy as pp

chems = ["H2O", "CO2"]

vec = np.ones(1)
z = [vec * 0.99,  vec * 0.01]
p = vec * 27000000.0
T = vec * 664.2857142857142
h = vec * -13185.069163143246
v = vec * 5.110938535645555e-05
verbosity = 2

species = pp.composite.load_species(chems)

comps = [
    pp.composite.peng_robinson.H2O.from_species(species[0]),
    pp.composite.peng_robinson.CO2.from_species(species[1]),
]

phases = [
    pp.composite.Phase(
        pp.composite.peng_robinson.PengRobinson(gaslike=False), name="L"
    ),
    pp.composite.Phase(
        pp.composite.peng_robinson.PengRobinson(gaslike=True), name="G"
    ),
]

mix = pp.composite.NonReactiveMixture(comps, phases)

mix.set_up()

[
    mix.system.set_variable_values(val, [comp.fraction.name], 0, 0)
    for val, comp in zip(z, comps)
]

flash = pp.composite.FlashNR(mix)
flash.use_armijo = True
flash.armijo_parameters["rho"] = 0.9
flash.armijo_parameters["j_max"] = 150
flash.armijo_parameters["return_max"] = True
flash.newton_update_chop = 1.0
flash.tolerance = 1e-5
flash.max_iter = 150

success, results_ = flash.flash(
    state={"h": h, "v": v},
    eos_kwargs={"apply_smoother": True},
    feed=z,
    verbosity=verbosity,
)
print(results_)

# p-T flash
success, results_pT = flash.flash(
    state={"p": p, "T": T},
    eos_kwargs={"apply_smoother": True},
    feed=z,
    verbosity=verbosity,
)
print("Results p-T:\n" + "------------")
print(str(results_pT))
print("------------")

# p-h flash
success, results_ph = flash.flash(
    state={"p": p, "h": results_pT.h},
    eos_kwargs={"apply_smoother": True},
    feed=z,
    verbosity=verbosity,
)

print("Difference between p-T and p-h:\n" + "-------------------------------")
print(str(results_pT.diff(results_ph)))
print("-------------------------------")

# h-v- flash
flash.armijo_parameters["j_max"] = 80
flash.armijo_parameters["rho"] = 0.99
flash.use_armijo = True
flash.max_iter = 150
success, results_hv = flash.flash(
    state={"h": results_pT.h, "v": results_pT.v},
    eos_kwargs={"apply_smoother": True},
    feed=z,
    verbosity=verbosity,
)
print("Difference between p-T and h-v:\n" + "-------------------------------")
print(str(results_pT.diff(results_hv)))
print("-------------------------------")
print("")