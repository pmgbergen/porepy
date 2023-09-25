import matplotlib.pyplot as plt
import numpy as np

import porepy as pp

species = pp.composite.load_species(["H2O", "CO2"])

comps = [
    pp.composite.peng_robinson.H2O.from_species(species[0]),
    pp.composite.peng_robinson.CO2.from_species(species[1]),
]


eosl = pp.composite.peng_robinson.PengRobinson(False)
eosl.components = comps
eosg = pp.composite.peng_robinson.PengRobinson(True)
eosg.components = comps

p = 7
T = 600
z = 0.99
eps = 1e-12


def phi(x1, x2):

    x1_ = x1 / (x1 + x2)
    x2_ = x2 / (x1 + x2)

    p_ = p * np.ones(len(x1_))
    T_ = T * np.ones(len(x1_))

    prop_l = eosl.compute(p_, T_, [x1_, x2_])
    prop_g = eosg.compute(p_, T_, [x1_, x2_])

    return prop_l.phis, prop_g.phis


X1 = np.linspace(eps, 1, 100, endpoint=True)
X2 = np.linspace(eps, 1, 100, endpoint=True)

x_l = [0.9996656628027017, 0.00033433719729830555]
x_g = [0.04177388808030789, 0.9582261119196921]


def f_1(x):

    vec = np.ones(len(x))

    phi_r_l, phi_r_g = phi(vec * x_l[0], vec * x_l[1])
    phi_l, phi_g = phi(x, 1 - x)

    k1 = phi_r_l[0] / phi_g[0]
    k2 = phi_r_l[1] / phi_g[1]

    return x - x_l[0] * k1, (1 - x) - x_l[1] * k2


def f_2(x):

    z1_ = z * np.ones(len(x))

    phi_r_l, phi_r_g = phi(z1_, 1 - z1_)

    phi_l, phi_g = phi(x, 1 - x)

    # return z1 * phi_r_g[0] - x * phi_l[0], (1-z1) * phi_r_g[1] - x * phi_l[1]
    return x * phi_g[0] - z1_ * phi_r_l[0], x * phi_g[1] - (1 - z1_) * phi_r_l[1]


def f_3(x, y):

    z1_ = z * np.ones(len(x))

    phi_r_l, phi_r_g = phi(z1_, 1 - z1_)

    phi_l, phi_g = phi(x, y)

    return x * phi_g[0] - y * phi_l[0], (1 - x) * phi_g[1] - (1 - y) * phi_l[1]
    # return z1_ * phi_r_g[0] - x * phi_l[0], (1-z1_) * phi_r_g[1] - y * phi_l[1]
    # return x * phi_g[0] - z1_ * phi_r_l[0], y * phi_g[1] - (1-z1_) * phi_r_l[1]


# f1, f2 = f_2(X1)
# plt.subplot(1,2,1)
# plt.plot(X1, f1)
# plt.axhline(y=0, color='red', linestyle='--')
# # plt.yscale('symlog')
# plt.title("Isofugacity constraint H2O")
# plt.xlabel("X H2O")
# plt.subplot(1,2,2)
# plt.plot(X1, f2)
# plt.yscale('symlog')
# plt.axhline(y=0, color='red', linestyle='--')
# plt.title("Isofugacity constraint CO2")
# plt.xlabel("X CO2")

# plt.show()

# plt.subplot(1,2,1)
# f1, f2 = f_1(X1)
# plt.plot(X1, f1)
# plt.axhline(y=0, color='green')
# plt.axvline(x=x_g[0], color='red')
# plt.subplot(1,2,2)
# plt.plot(X1, f2)
# plt.axhline(y=0, color='green')
# plt.axvline(x=x_g[1], color='red')
# plt.show()

X, Y = np.meshgrid(X1, X2)
z1, z2 = f_3(np.ravel(X), np.ravel(Y))
z1 = np.array(z1)
z2 = np.array(z2)
Z1 = z1.reshape(X.shape)
Z2 = z2.reshape(X.shape)
zero = Z1 * 0.0

fig = plt.figure()
alpha = 0.6
ax = fig.add_subplot(111, projection="3d")
res1 = ax.plot_surface(X, Y, Z1, color="blue", alpha=alpha)
res1.set_label("Isofugacity res H20")
res2 = ax.plot_surface(X, Y, Z2, color="red", alpha=alpha)
res2.set_label("Isofugacity res CO2")
res0 = ax.plot_surface(X, Y, zero, color="green", alpha=0.3)
res0.set_label("0")

l = 2
simplex_d = ax.plot(X1, 1 - X1, np.zeros(len(X1)), linewidth=l, color="black")
simplex_1 = ax.plot(
    X1, np.zeros(len(X1)), np.zeros(len(X1)), linewidth=l, color="black"
)
simplex_2 = ax.plot(
    np.zeros(len(X1)), X1, np.zeros(len(X1)), linewidth=l, color="black"
)

ax.set_zscale("symlog")

ax.set_xlabel("XL")
ax.set_ylabel("XG")
ax.set_zlabel("res")
ax.set_zlim(-1, 1)
ax.set_title("Isofugacity residuals")

plt.show()

print("")
