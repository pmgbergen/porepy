import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.forward_mode import AdArray
import porepy.numerics.ad.functions as af

import pdb
import os

os.system("clear")

THIS IS A TRY 


def myprint(var):
    print("\n" + var + " = ", eval(var))


x = AdArray(np.array([2]), sps.eye(1, 1))


def func(x):
    u = np.ones(x.val.shape)
    u = AdArray(np.array([2]), sps.eye(1, 1))
    return u * x
    # return x**2+3


myprint("func(x)")


x = AdArray(np.array([2, 3]), sps.eye(2))
y = np.array([2, 3])

myprint("x*y")
myprint("y*x")
myprint("x.val*y")
myprint("y*x.val")

x = AdArray(np.array([2, 3]), sps.eye(2))  # is this conceptually wrong?
y = AdArray(np.array([1, 1]), sps.eye(2))

z = np.dot(x, y)  # this is element-wise. ?
myprint("np.dot(x,y)")
myprint("x*y")

# x = AdArray(np.array([3, 2]), sps.eye(2))
# z = x.val[0]*x.val[1] # no...

variables = pp.ad.initAdArrays([np.ones(3), 0.5 * np.ones(3)])
pressure = variables[0]
saturation = variables[1]


def density_ad(pressure):
    C = 2
    return C * pressure


density = density_ad(pressure)
s_rho = saturation * density

tmp = np.zeros(7)
tmp = AdArray(np.zeros(7), sps.eye(7))
tmp.val[0:3] = pressure.val
tmp.jac = pressure.jac


# # you can't:
# tmp = AdArray(np.array([2, 3]), sps.eye(2))
# tmp = pp.ad.functions.maximum(-np.real(tmp), -1e6) + np.imag(tmp) * 1j
# beta_faces = 0.5 + 1 / np.pi * np.arctan(tmp)


# heaviside jacobian: -----------------------------
tmp = AdArray(np.zeros(7), sps.eye(7))
out = pp.ad.functions.heaviside(tmp)


pdb.set_trace()
