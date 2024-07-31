import numpy as np


def __below_x_predicate(x, y, z, bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x_pre = np.logical_or(x < xmin, np.isclose(x, xmin))
    return x_pre


def __above_x_predicate(x, y, z, bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x_pre = np.logical_or(x > xmax, np.isclose(x, xmax))
    return x_pre


def __below_y_predicate(x, y, z, bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    y_pre = np.logical_or(y < ymin, np.isclose(y, ymin))
    return y_pre


def __above_y_predicate(x, y, z, bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    y_pre = np.logical_or(y > ymax, np.isclose(y, ymax))
    return y_pre


def __below_z_predicate(x, y, z, bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    z_pre = np.logical_or(z < zmin, np.isclose(z, zmin))
    return z_pre


def __above_z_predicate(x, y, z, bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    z_pre = np.logical_or(z > zmax, np.isclose(z, zmax))
    return z_pre


def x_range_predicate(x, y, z, bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x_range_pre = np.logical_and(
        np.logical_or(x > xmin, np.isclose(x, xmin)),
        np.logical_or(x < xmax, np.isclose(x, xmax)),
    )
    return x_range_pre


def y_range_predicate(x, y, z, bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    y_range_pre = np.logical_and(
        np.logical_or(y > ymin, np.isclose(y, ymin)),
        np.logical_or(y < ymax, np.isclose(y, ymax)),
    )
    return y_range_pre


def z_range_predicate(x, y, z, bounds):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    z_range_pre = np.logical_and(
        np.logical_or(z > zmin, np.isclose(z, zmin)),
        np.logical_or(z < zmax, np.isclose(z, zmax)),
    )
    return z_range_pre


def e_predicate(x, y, z, bounds):
    x_pre = __above_x_predicate(x, y, z, bounds)
    y_range_pre = y_range_predicate(x, y, z, bounds)
    z_range_pre = z_range_predicate(x, y, z, bounds)
    yz_pre = np.logical_and(y_range_pre, z_range_pre)
    return np.logical_and(x_pre, yz_pre)


def w_predicate(x, y, z, bounds):
    x_pre = __below_x_predicate(x, y, z, bounds)
    y_range_pre = y_range_predicate(x, y, z, bounds)
    z_range_pre = z_range_predicate(x, y, z, bounds)
    yz_pre = np.logical_and(y_range_pre, z_range_pre)
    return np.logical_and(x_pre, yz_pre)


def s_predicate(x, y, z, bounds):
    y_pre = __below_y_predicate(x, y, z, bounds)
    x_range_pre = x_range_predicate(x, y, z, bounds)
    z_range_pre = z_range_predicate(x, y, z, bounds)
    xz_pre = np.logical_and(x_range_pre, z_range_pre)
    return np.logical_and(y_pre, xz_pre)


def n_predicate(x, y, z, bounds):
    y_pre = __above_y_predicate(x, y, z, bounds)
    x_range_pre = x_range_predicate(x, y, z, bounds)
    z_range_pre = z_range_predicate(x, y, z, bounds)
    xz_pre = np.logical_and(x_range_pre, z_range_pre)
    return np.logical_and(y_pre, xz_pre)


def b_predicate(x, y, z, bounds):
    z_pre = __below_z_predicate(x, y, z, bounds)
    x_range_pre = x_range_predicate(x, y, z, bounds)
    y_range_pre = y_range_predicate(x, y, z, bounds)
    xy_pre = np.logical_and(x_range_pre, y_range_pre)
    return np.logical_and(z_pre, xy_pre)


def t_predicate(x, y, z, bounds):
    z_pre = __above_z_predicate(x, y, z, bounds)
    x_range_pre = x_range_predicate(x, y, z, bounds)
    y_range_pre = y_range_predicate(x, y, z, bounds)
    xy_pre = np.logical_and(x_range_pre, y_range_pre)
    return np.logical_and(z_pre, xy_pre)


# x range members
def sb_predicate(x, y, z, bounds):
    y_pre = __below_y_predicate(x, y, z, bounds)
    z_pre = __below_z_predicate(x, y, z, bounds)
    x_range_pre = x_range_predicate(x, y, z, bounds)
    y_and_z_pre = np.logical_and(y_pre, z_pre)
    return np.logical_and(x_range_pre, y_and_z_pre)


def nb_predicate(x, y, z, bounds):
    y_pre = __above_y_predicate(x, y, z, bounds)
    z_pre = __below_z_predicate(x, y, z, bounds)
    x_range_pre = x_range_predicate(x, y, z, bounds)
    y_and_z_pre = np.logical_and(y_pre, z_pre)
    return np.logical_and(x_range_pre, y_and_z_pre)


def st_predicate(x, y, z, bounds):
    y_pre = __below_y_predicate(x, y, z, bounds)
    z_pre = __above_z_predicate(x, y, z, bounds)
    x_range_pre = x_range_predicate(x, y, z, bounds)
    y_and_z_pre = np.logical_and(y_pre, z_pre)
    return np.logical_and(x_range_pre, y_and_z_pre)


def nt_predicate(x, y, z, bounds):
    y_pre = __above_y_predicate(x, y, z, bounds)
    z_pre = __above_z_predicate(x, y, z, bounds)
    x_range_pre = x_range_predicate(x, y, z, bounds)
    y_and_z_pre = np.logical_and(y_pre, z_pre)
    return np.logical_and(x_range_pre, y_and_z_pre)


# y range members
def wb_predicate(x, y, z, bounds):
    x_pre = __below_x_predicate(x, y, z, bounds)
    z_pre = __below_z_predicate(x, y, z, bounds)
    y_range_pre = y_range_predicate(x, y, z, bounds)
    x_and_z_pre = np.logical_and(x_pre, z_pre)
    return np.logical_and(y_range_pre, x_and_z_pre)


def eb_predicate(x, y, z, bounds):
    x_pre = __above_x_predicate(x, y, z, bounds)
    z_pre = __below_z_predicate(x, y, z, bounds)
    y_range_pre = y_range_predicate(x, y, z, bounds)
    x_and_z_pre = np.logical_and(x_pre, z_pre)
    return np.logical_and(y_range_pre, x_and_z_pre)


def wt_predicate(x, y, z, bounds):
    x_pre = __below_x_predicate(x, y, z, bounds)
    z_pre = __above_z_predicate(x, y, z, bounds)
    y_range_pre = y_range_predicate(x, y, z, bounds)
    x_and_z_pre = np.logical_and(x_pre, z_pre)
    return np.logical_and(y_range_pre, x_and_z_pre)


def et_predicate(x, y, z, bounds):
    x_pre = __above_x_predicate(x, y, z, bounds)
    z_pre = __above_z_predicate(x, y, z, bounds)
    y_range_pre = y_range_predicate(x, y, z, bounds)
    x_and_z_pre = np.logical_and(x_pre, z_pre)
    return np.logical_and(y_range_pre, x_and_z_pre)


# z range members
def ws_predicate(x, y, z, bounds):
    x_pre = __below_x_predicate(x, y, z, bounds)
    y_pre = __below_y_predicate(x, y, z, bounds)
    z_range_pre = z_range_predicate(x, y, z, bounds)
    x_and_y_pre = np.logical_and(x_pre, y_pre)
    return np.logical_and(z_range_pre, x_and_y_pre)


def es_predicate(x, y, z, bounds):
    x_pre = __above_x_predicate(x, y, z, bounds)
    y_pre = __below_y_predicate(x, y, z, bounds)
    z_range_pre = z_range_predicate(x, y, z, bounds)
    x_and_y_pre = np.logical_and(x_pre, y_pre)
    return np.logical_and(z_range_pre, x_and_y_pre)


def wn_predicate(x, y, z, bounds):
    x_pre = __below_x_predicate(x, y, z, bounds)
    y_pre = __above_y_predicate(x, y, z, bounds)
    z_range_pre = z_range_predicate(x, y, z, bounds)
    x_and_y_pre = np.logical_and(x_pre, y_pre)
    return np.logical_and(z_range_pre, x_and_y_pre)


def en_predicate(x, y, z, bounds):
    x_pre = __above_x_predicate(x, y, z, bounds)
    y_pre = __above_y_predicate(x, y, z, bounds)
    z_range_pre = z_range_predicate(x, y, z, bounds)
    x_and_y_pre = np.logical_and(x_pre, y_pre)
    return np.logical_and(z_range_pre, x_and_y_pre)


def wsb_predicate(x, y, z, bounds):
    x_pre = __below_x_predicate(x, y, z, bounds)
    y_pre = __below_y_predicate(x, y, z, bounds)
    z_pre = __below_z_predicate(x, y, z, bounds)
    x_and_y_pre = np.logical_and(x_pre, y_pre)
    return np.logical_and(z_pre, x_and_y_pre)


def esb_predicate(x, y, z, bounds):
    x_pre = __above_x_predicate(x, y, z, bounds)
    y_pre = __below_y_predicate(x, y, z, bounds)
    z_pre = __below_z_predicate(x, y, z, bounds)
    x_and_y_pre = np.logical_and(x_pre, y_pre)
    return np.logical_and(z_pre, x_and_y_pre)


def wnb_predicate(x, y, z, bounds):
    x_pre = __below_x_predicate(x, y, z, bounds)
    y_pre = __above_y_predicate(x, y, z, bounds)
    z_pre = __below_z_predicate(x, y, z, bounds)
    x_and_y_pre = np.logical_and(x_pre, y_pre)
    return np.logical_and(z_pre, x_and_y_pre)


def enb_predicate(x, y, z, bounds):
    x_pre = __above_x_predicate(x, y, z, bounds)
    y_pre = __above_y_predicate(x, y, z, bounds)
    z_pre = __below_z_predicate(x, y, z, bounds)
    x_and_y_pre = np.logical_and(x_pre, y_pre)
    return np.logical_and(z_pre, x_and_y_pre)


def wst_predicate(x, y, z, bounds):
    x_pre = __below_x_predicate(x, y, z, bounds)
    y_pre = __below_y_predicate(x, y, z, bounds)
    z_pre = __above_z_predicate(x, y, z, bounds)
    x_and_y_pre = np.logical_and(x_pre, y_pre)
    return np.logical_and(z_pre, x_and_y_pre)


def est_predicate(x, y, z, bounds):
    x_pre = __above_x_predicate(x, y, z, bounds)
    y_pre = __below_y_predicate(x, y, z, bounds)
    z_pre = __above_z_predicate(x, y, z, bounds)
    x_and_y_pre = np.logical_and(x_pre, y_pre)
    return np.logical_and(z_pre, x_and_y_pre)


def wnt_predicate(x, y, z, bounds):
    x_pre = __below_x_predicate(x, y, z, bounds)
    y_pre = __above_y_predicate(x, y, z, bounds)
    z_pre = __above_z_predicate(x, y, z, bounds)
    x_and_y_pre = np.logical_and(x_pre, y_pre)
    return np.logical_and(z_pre, x_and_y_pre)


def ent_predicate(x, y, z, bounds):
    x_pre = __above_x_predicate(x, y, z, bounds)
    y_pre = __above_y_predicate(x, y, z, bounds)
    z_pre = __above_z_predicate(x, y, z, bounds)
    x_and_y_pre = np.logical_and(x_pre, y_pre)
    return np.logical_and(z_pre, x_and_y_pre)
