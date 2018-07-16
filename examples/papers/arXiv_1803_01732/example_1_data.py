import numpy as np

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.params.data import Parameters

# ------------------------------------------------------------------------------#


def add_data(gb, tol):
    """
    Define the permeability, apertures, boundary conditions
    """
    gb.add_node_props(["param"])

    # Aavatsmark_transmissibilities only for tpfa intra-dimensional coupling

    for g, d in gb:
        d["Aavatsmark_transmissibilities"] = True

        if g.dim < 2:
            continue

        param = Parameters(g)

        # Permeability
        kxx = np.array([perm(*pt) for pt in g.cell_centers.T])
        param.set_tensor("flow", tensor.SecondOrderTensor(3, kxx))

        # Source term
        frac_id = d["frac_id"][0]
        source = np.array([source_f[frac_id](*pt) for pt in g.cell_centers.T])
        param.set_source("flow", g.cell_volumes * source)

        # Boundaries
        bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size != 0:
            bound_face_centers = g.face_centers[:, bound_faces]

            labels = np.array(["dir"] * bound_faces.size)

            bc_val = np.zeros(g.num_faces)
            bc = [sol_f[frac_id](*pt) for pt in bound_face_centers.T]
            bc_val[bound_faces] = np.array(bc)

            param.set_bc("flow", BoundaryCondition(g, bound_faces, labels))
            param.set_bc_val("flow", bc_val)
        else:
            param.set_bc("flow", BoundaryCondition(g, np.empty(0), np.empty(0)))

        d["param"] = param

    gb.add_edge_prop("Aavatsmark_transmissibilities")
    for _, d in gb.edges_props():
        d["Aavatsmark_transmissibilities"] = True


# ------------------------------------------------------------------------------#


def assign_frac_id(gb):

    gb.add_node_props(["frac_id"])

    for g, d in gb:
        if g.dim < 2:
            d["frac_id"] = -1 * np.ones(g.num_cells, dtype=np.int)

        if np.isclose(np.amin(np.abs(g.nodes[0, :])), 0.5) and np.isclose(
            np.amax(np.abs(g.nodes[0, :])), 0.5
        ):
            d["frac_id"] = 2 * np.ones(g.num_cells, dtype=np.int)

        if np.isclose(np.amin(np.abs(g.nodes[1, :])), 0) and np.isclose(
            np.amax(np.abs(g.nodes[1, :])), 0
        ):
            d["frac_id"] = 1 * np.ones(g.num_cells, dtype=np.int)

        if np.isclose(np.amin(np.abs(g.nodes[2, :])), 0) and np.isclose(
            np.amax(np.abs(g.nodes[2, :])), 0
        ):
            d["frac_id"] = 0 * np.ones(g.num_cells, dtype=np.int)


# ------------------------------------------------------------------------------#


def source_f0(x, y, z):

    val = (
        0.1
        * (-0.5 - x)
        * (
            6 * x
            - 16 * y * y
            - (16 * x * x * y * y) / (x * x + y * y)
            + 48 * x * y * np.arctan2(y, x)
        )
        + 0.1
        * (-0.5 - x)
        * (
            16 * x * x
            + (16 * x * x * y * y) / (x * x + y * y)
            + 48 * x * y * np.arctan2(y, x)
        )
        + (1 / 5.)
        * (
            -3 * x * x
            + 8 * x * y * y
            - 16 * x * x * y * np.arctan2(y, x)
            - 8 * y * (x * x + y * y) * np.arctan2(y, x)
        )
    )

    return -val


# ------------------------------------------------------------------------------#


def source_f1(x, y, z):

    val = (
        (3 / 5.) * (-0.5 - x) * x
        - (3 * x * x) / 5.
        - (24 / 5.) * np.pi * (-0.5 - x) * x * np.abs(z)
        + (24 / 5.) * np.pi * x * x * np.abs(z)
    )

    return -val


# ------------------------------------------------------------------------------#


def source_f2(x, y, z):

    val = (
        2 * (-1 + y) * y * (1 + y)
        + 2 * (-1 + y) * (-1 + z) * z
        + 2 * y * (-1 + z) * z
        + 2 * (1 + y) * (-1 + z) * z
    )

    return -val


# ------------------------------------------------------------------------------#


source_f = [source_f0, source_f1, source_f2]

# ------------------------------------------------------------------------------#


def sol_f0(x, y, z):
    return ((-0.1) * (0.5 + x)) * (
        x * x * x + 8 * x * y * (x * x + y * y) * np.arctan2(y, x)
    )


# ------------------------------------------------------------------------------#


def sol_f1(x, y, z):
    return (1 / 10.) * (-0.5 - x) * x * x * x - np.abs(z) * np.pi * (4 / 5.) * (
        -0.5 - x
    ) * x * x * x


# ------------------------------------------------------------------------------#


def sol_f2(x, y, z):
    return y * (y - 1) * (y + 1) * z * (z - 1)


# ------------------------------------------------------------------------------#


sol_f = [sol_f0, sol_f1, sol_f2]

# ------------------------------------------------------------------------------#


def vel_f0(x, y, z):
    u = np.array(
        [
            0.1
            * (-0.5 - x)
            * (
                3 * x ** 2
                - 8 * x * y ** 2
                + 16 * x ** 2 * y * np.arctan2(y, x)
                + 8 * y * (x ** 2 + y ** 2) * np.arctan2(y, x)
            )
            + 0.1 * (-x ** 3 - 8 * x * y * (x ** 2 + y ** 2) * np.arctan2(y, x)),
            0.1
            * (-0.5 - x)
            * (
                8 * x ** 2 * y
                + 16 * x * y ** 2 * np.arctan2(y, x)
                + 8 * x * (x ** 2 + y ** 2) * np.arctan2(y, x)
            ),
            0,
        ]
    )
    return -u


# ------------------------------------------------------------------------------#


def vel_f1(x, y, z):
    u = np.array(
        [
            (3 / 10.) * (-0.5 - x) * x ** 2
            - x ** 3 / 10.
            - (12 / 5.) * np.pi * (-0.5 - x) * x ** 2 * np.abs(z)
            + (4 / 5.) * np.pi * x ** 3 * np.abs(z),
            0,
            -np.sign(z) * (4 / 5.) * np.pi * (-0.5 - x) * x ** 3,
        ]
    )
    return -u


# ------------------------------------------------------------------------------#


def vel_f2(x, y, z):
    u = np.array(
        [
            0,
            (-1 + y) * y * (-1 + z) * z
            + (-1 + y) * (1 + y) * (-1 + z) * z
            + y * (1 + y) * (-1 + z) * z,
            (-1 + y) * y * (1 + y) * (-1 + z) + (-1 + y) * y * (1 + y) * z,
        ]
    )
    return -u


# ------------------------------------------------------------------------------#

vel_f = [vel_f0, vel_f1, vel_f2]

# ------------------------------------------------------------------------------#


def perm(x, y, z):
    return 1


# ------------------------------------------------------------------------------#


def error_pressure(gb, p_name):

    error = 0
    for g, d in gb:
        if g.dim < 2:
            d["err"] = np.zeros(g.num_cells)
            continue
        frac_id = d["frac_id"][0]
        sol = np.array([sol_f[frac_id](*pt) for pt in g.cell_centers.T])
        d["err"] = np.abs(d[p_name] - sol)
        error += np.sum(np.power(d["err"], 2) * g.cell_volumes)

    return np.sqrt(error)


# ------------------------------------------------------------------------------#


def error_discharge(gb, P0u_name):

    error = np.zeros(3)
    for g, d in gb:
        if g.dim < 2:
            continue
        frac_id = d["frac_id"][0]
        sol = np.array([vel_f[frac_id](*pt) for pt in g.cell_centers.T]).T
        err = np.abs(d[P0u_name] - sol)
        for i in np.arange(3):
            error[i] += np.sum(np.power(err[i, :], 2) * g.cell_volumes)
    return np.sqrt(np.sum(error))


# ------------------------------------------------------------------------------#
