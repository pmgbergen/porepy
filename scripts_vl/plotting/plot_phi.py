import porepy as pp
import numpy as np
import matplotlib.pyplot as plt

species = pp.composite.load_fluid_species(['H2O', 'CO2'])

comps = [
    pp.composite.peng_robinson.H2O.from_species(species[0]),
    pp.composite.peng_robinson.CO2.from_species(species[1]),
]


eosl = pp.composite.peng_robinson.PengRobinsonEoS(False)
eosl.components = comps
eosg = pp.composite.peng_robinson.PengRobinsonEoS(True)
eosg.components = comps

p = 1
T = 350
z = 0.99

def phi(x):

    prop_l = eosl.compute(p, T, [x, 1 - x])
    prop_g = eosg.compute(p, T, [x, 1 - x])

    return prop_l.phis, prop_g.phis


X = np.linspace(0, 1, 100, endpoint=True)

x_l = [0.9996656628027017, 0.00033433719729830555]
x_g = [0.04177388808030789, 0.9582261119196921]

nc = len(X)
vec = np.ones(nc)
p = vec * p
T = vec * T
z1 = vec * z

phi_r_l, phi_r_g = phi(vec * x_l[0])

def f_l(x):

    phi_l, phi_g = phi(x)

    k1 = phi_r_l[0] / phi_g[0]
    k2 = phi_r_l[1] / phi_g[1]

    return x - x_l[0] * k1, x - x_l[1] * k2

def f_g(x):

    phi_l, phi_g = phi(x)

    return z1 * phi_r_g[0] - x * phi_l[0], (1-z1) * phi_r_g[1] - (1 - x) * phi_l[1]

phis_l, phis_g = phi(X)

plt.subplot(3,2,1)
plt.plot(X, phis_l[0])
plt.xlabel("x1")
plt.ylabel("phi L 1")
plt.subplot(3,2,2)
plt.plot(X, phis_l[1])
plt.xlabel("x1")
plt.ylabel("phi L 2")
plt.subplot(3,2,3)
plt.plot(X, phis_g[0])
plt.xlabel("x1")
plt.ylabel("phi G 1")
plt.subplot(3,2,4)
plt.plot(X, phis_g[1])
plt.xlabel("x1")
plt.ylabel("phi G 2")

plt.subplot(3,2,5)
plt.plot(X, phis_l[0] / phis_g[0])
plt.xlabel("x1")
plt.ylabel("K 1")

plt.subplot(3,2,6)
plt.plot(X, phis_l[1] / phis_g[1])
plt.xlabel("x1")
plt.ylabel("K 2")

plt.show()

plt.subplot(1,2,1)
f_1, f_2 = f_l(X)
plt.plot(X, f_1)
plt.axhline(y=0)
plt.axvline(x=x_g[0])
plt.subplot(1,2,2)
plt.plot(X, f_2)
plt.axhline(y=0)
plt.axvline(x=x_g[1])
plt.show()
print('')
