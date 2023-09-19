import numpy as np

import porepy as pp

vec = np.ones(1)
z = [vec * 0.99,  vec * 0.01]
p = vec * 5e6
T = vec * 500
verbosity = 2

t = np.array([0.01, 5e6, 500, 0.9, 0.1, 0.2, 0.3, 0.4])

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
[
    mix.system.set_variable_values(val, [comp.fraction.name], 0, 0)
    for val, comp in zip(z, comps)
]

PRC = pp.composite.peng_robinson.PR_Compiler(mix)

PRC.equations['p-T'](t)
PRC.jacobians['p-T'](t)

flash = pp.composite.FlashNR(mix)
flash.use_armijo = True
flash.armijo_parameters["rho"] = 0.9
flash.armijo_parameters["j_max"] = 150
flash.armijo_parameters["return_max"] = True
flash.newton_update_chop = 1.0
flash.tolerance = 1e-5
flash.max_iter = 150

# success, results_pT = flash.flash(
_, oldsys = flash.flash(
    state={"p": p, "T": T},
    eos_kwargs={"apply_smoother": True},
    feed=z,
    verbosity=verbosity,
    quickshot=True,
    return_system=True,
)
# print("Results p-T:\n" + "------------")
# print(str(results_pT))
# print("------------")

initstate = oldsys.state
oldsys_eval = oldsys(initstate, True)
print("old value", oldsys_eval.val)
print("old matrix")
print(oldsys_eval.jac.todense()[:-1, :-1])

t[3:] = initstate[:-1]

newval = PRC.equations['p-T'](t)
newjac = PRC.jacobians['p-T'](t)
print('new val')
print(newval)
print('new matrix')
print(newjac)

print("diff vals", np.linalg.norm(newval - oldsys_eval.val))
print("diff jac", np.linalg.norm(newjac - oldsys_eval.jac.todense()[:-1, :-1]))
