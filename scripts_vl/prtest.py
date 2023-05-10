import numpy as np
import porepy as pp

chems = ["H2O", "CO2"]

z = [np.array([0.1])]  # only co2 fraction is enough
p = np.array([1.]) * 1e6
T = np.array([350.])

species = pp.composite.load_fluid_species(chems)

comps = [
    pp.composite.peng_robinson.H2O.from_species(species[0]),
    pp.composite.peng_robinson.CO2.from_species(species[1]),
]

phases = [
    pp.composite.Phase(pp.composite.peng_robinson.PengRobinsonEoS(gaslike=False), name='L'),
    pp.composite.Phase(pp.composite.peng_robinson.PengRobinsonEoS(gaslike=True), name='G'),
]

mix = pp.composite.NonReactiveMixture(comps, phases)

mix.set_up()

[
    mix.system.set_variable_values(val, [comp.fraction.name], 0, 0)
    for val, comp in zip(z, comps)
]

yr = mix.reference_phase.fraction.evaluate(mix.system)

flash = pp.composite.FlashNR(mix)
flash.use_armijo = True
flash.armijo_parameters["rho"] = 0.99
flash.armijo_parameters["j_max"] = 80
flash.armijo_parameters["return_max"] = True
flash.newton_update_chop = 1.0
flash.tolerance = 1e-8
flash.max_iter = 140

# p-T flash
success, results_pT = flash.flash(
    state={'p': p, 'T': T}, eos_kwargs={'apply_smoother': True},
    feed = z,
    verbosity=1,
)
print("Results p-T:")
print(str(results_pT))

# p-h flash
success, results_ph = flash.flash(
    state={'p': p, 'h': results_pT.h}, eos_kwargs={'apply_smoother': True},
    feed = z,
    verbosity=1,
)

print("Difference between p-T and p-h:")
print(str(results_pT.diff(results_ph)))

# h-v- flash
success, results_hv = flash.flash(
    state={'h': results_ph.h, 'v': results_ph.v}, eos_kwargs={'apply_smoother': True},
    feed = z,
    verbosity=1,
)
print("Difference between p-T and h-v:")
print(str(results_pT.diff(results_hv)))

print("Done")