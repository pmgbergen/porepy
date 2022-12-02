import porepy as pp
import numpy as np
import math

from matplotlib import pyplot as plt


M = pp.composite.PR_Composition()
sys = M.ad_system
dm = M.ad_system.dof_manager
nc = dm.mdg.num_subdomain_cells()
vec = np.ones(nc)
H2O = pp.composite.H2O(M.ad_system)

M.add_component(H2O)

temperature = 373.16
pressure = 0.10132

sys.set_var_values(H2O.fraction_name, 1 * vec, True)
sys.set_var_values(M.T_name, temperature * vec, True)
sys.set_var_values(M.p_name, pressure * vec, True)
sys.set_var_values(M.h_name, 0 * vec, True)

M.initialize()

c2 = M.c2.evaluate(dm).val[0]
c1 = M.c1.evaluate(dm).val[0]
c0 = M.c0.evaluate(dm).val[0]
# c2 = -7.5
# c1 = 17
# c0 = -12

def FZ(x):
    return x**3 + c2 * x**2 + c1 * x + c0

p = (3*c1 - c2**2) / 3
q =  (2*c2**3 - 9 * c2*c1 + 27*c0) / 27

discr3 = 18*c2*c1*c0 - 4 * c2**2 * c0**2 + c2**2 * c1**2 - 4 * c1**3 - 27* c0**2
discr2 = - (4 * p**3 + 27 * q**2)
delta2 = q**2 / 4 + p**3 / 27

t1 = - q/2 + delta2 ** (1/2)


if t1 < 0:
    u = 1j * (-t1)**(1/3)
else:
    u = t1 ** (1/3)

z1 = u - p / (3*u) - c2/3
z2 = -(u-p/(3*u)) / 2 + 1j/2*math.sqrt(3) * (u + p / (3*u)) - c2/3
z3 = -(u-p/(3*u)) / 2 - 1j/2*math.sqrt(3) * (u + p / (3*u)) - c2/3

print("t1: ", t1)
print("delta: ", delta2)
print("p: ", p)
print("u: ", u)
print("z1: ", z1)
print("z2: ", z2)
print("z3: ", z3)

X = np.linspace(-5, 5, 200)
Z = FZ(X)
plt.plot(X, Z)
plt.grid(True)
plt.ylim(-2, 2.)
plt.show()