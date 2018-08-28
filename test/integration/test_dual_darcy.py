import numpy as np
import unittest

from porepy.numerics import elliptic
from porepy.grids.structured import CartGrid
from porepy.fracs import meshing
from porepy.params.data import Parameters
from porepy.params import tensor, bc
from porepy.utils import tags


class BasicsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------#

    def test_mono_equals_multi(self):
        """
        test that the mono_dimensional elliptic solver gives the same answer as
        the grid bucket elliptic
        """
        g = CartGrid([10, 10])
        g.compute_geometry()
        gb = meshing.cart_grid([], [10, 10])
        param_g = Parameters(g)

        def bc_val(g):
            left = g.face_centers[0] < 1e-6
            right = g.face_centers[0] > 10 - 1e-6

            bc_val = np.zeros(g.num_faces)
            bc_val[left] = -1
            bc_val[right] = 1

            return bc_val

        def bc_labels(g):
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            bound_face_centers = g.face_centers[:, bound_faces]
            left = bound_face_centers[0] < 1e-6
            right = bound_face_centers[0] > 10 - 1e-6

            labels = np.array(["neu"] * bound_faces.size)
            labels[np.logical_or(right, left)] = "dir"
            bc_labels = bc.BoundaryCondition(g, bound_faces, labels)

            return bc_labels

        param_g.set_bc_val("flow", bc_val(g))
        param_g.set_bc("flow", bc_labels(g))

        gb.add_node_props(["param"])
        for sub_g, d in gb:
            d["param"] = Parameters(sub_g)
            d["param"].set_bc_val("flow", bc_val(sub_g))
            d["param"].set_bc("flow", bc_labels(sub_g))

        problem_mono = elliptic.DualEllipticModel(g, {"param": param_g})
        problem_mult = elliptic.DualEllipticModel(gb)

        up_mono = problem_mono.solve()
        up_mult = problem_mult.solve()

        assert np.allclose(up_mono, up_mult)

        g_gb = next(problem_mult.grid().nodes())

        problem_mono.pressure("pressure")
        problem_mult.split()
        problem_mult.pressure("pressure")

        assert np.allclose(problem_mono.data()["pressure"], g_gb[1]["pressure"])

        problem_mono.discharge("u")
        problem_mult.discharge("u")

        assert np.allclose(problem_mono.data()["u"], g_gb[1]["u"])

        problem_mono.project_discharge("P0u")
        problem_mult.project_discharge("P0u")

        problem_mono.save(["pressure", "P0u"])
        problem_mult.save(["pressure", "P0u"])

        assert np.allclose(problem_mono.data()["P0u"], g_gb[1]["P0u"])

    # ------------------------------------------------------------------------------#

    def test_elliptic_uniform_flow_cart(self):
        gb = setup_2d_1d([10, 10])
        problem = elliptic.DualEllipticModel(gb)
        problem.solve()
        problem.split()
        problem.pressure("pressure")

        for g, d in gb:
            pressure = d["pressure"]
            p_analytic = g.cell_centers[1]
            p_diff = pressure - p_analytic
            assert np.max(np.abs(p_diff)) < 0.0004

    # ------------------------------------------------------------------------------#

    def test_elliptic_uniform_flow_simplex(self):
        """
        Unstructured simplex grid. Note that the solution depends
        on the grid quality. Also sensitive to the way in which
        the tpfa half transmissibilities are computed.
        """
        gb = setup_2d_1d(np.array([10, 10]), simplex_grid=True)
        problem = elliptic.DualEllipticModel(gb)
        problem.solve()
        problem.split()
        problem.pressure("pressure")

        for g, d in gb:
            pressure = d["pressure"]
            p_analytic = g.cell_centers[1]
            p_diff = pressure - p_analytic
            assert np.max(np.abs(p_diff)) < 0.0004

    # ------------------------------------------------------------------------------#

    def test_elliptic_dirich_neumann_source_sink_cart(self):
        gb = setup_3d(np.array([4, 4, 4]), simplex_grid=False)
        problem = elliptic.DualEllipticModel(gb)
        problem.solve()
        problem.split()
        problem.pressure("pressure")

        for g, d in gb:
            if g.dim == 3:
                p_ref = elliptic_dirich_neumann_source_sink_cart_ref_3d()
                assert np.allclose(d["pressure"], p_ref)
            if g.dim == 0:
                p_ref = [-260.13394502]
                assert np.allclose(d["pressure"], p_ref)
        return gb


# ------------------------------------------------------------------------------#


def setup_3d(nx, simplex_grid=False):
    f1 = np.array([[0.2, 0.2, 0.8, 0.8], [0.2, 0.8, 0.8, 0.2], [0.5, 0.5, 0.5, 0.5]])
    f2 = np.array([[0.2, 0.8, 0.8, 0.2], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]])
    f3 = np.array([[0.5, 0.5, 0.5, 0.5], [0.2, 0.8, 0.8, 0.2], [0.2, 0.2, 0.8, 0.8]])
    fracs = [f1, f2, f3]
    if not simplex_grid:
        gb = meshing.cart_grid(fracs, nx, physdims=[1, 1, 1])
    else:
        mesh_kwargs = {}
        mesh_size = .3
        mesh_kwargs = {
            "mesh_size_frac": mesh_size,
            "mesh_size_bound": 2 * mesh_size,
            "mesh_size_min": mesh_size / 20,
        }
        domain = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}
        gb = meshing.simplex_grid(fracs, domain, **mesh_kwargs)

    gb.add_node_props(["param"])
    for g, d in gb:
        a = 0.01 / np.max(nx)
        a = np.power(a, gb.dim_max() - g.dim)
        param = Parameters(g)
        param.set_aperture(a)

        # BoundaryCondition
        left = g.face_centers[0] < 1e-6
        top = g.face_centers[2] > 1 - 1e-6
        dir_faces = np.argwhere(left)
        bc_cond = bc.BoundaryCondition(g, dir_faces, ["dir"] * dir_faces.size)
        bc_val = np.zeros(g.num_faces)
        bc_val[dir_faces] = 3
        bc_val[top] = 2.4
        param.set_bc("flow", bc_cond)
        param.set_bc_val("flow", bc_val)

        # Source and sink
        src = np.zeros(g.num_cells)
        src[0] = np.pi
        src[-1] = -np.pi
        param.set_source("flow", src)
        d["param"] = param

    gb.add_edge_props("kn")
    for e, d in gb.edges():
        g = gb.nodes_of_edge(e)[0]
        mg = d["mortar_grid"]
        check_P = mg.low_to_mortar_avg()
        d["kn"] = 1 / (check_P * gb.node_props(g, "param").get_aperture())

    return gb


# ------------------------------------------------------------------------------#


def setup_2d_1d(nx, simplex_grid=False):
    frac1 = np.array([[0.2, 0.8], [0.5, 0.5]])
    frac2 = np.array([[0.5, 0.5], [0.8, 0.2]])
    fracs = [frac1, frac2]
    if not simplex_grid:
        gb = meshing.cart_grid(fracs, nx, physdims=[1, 1])
    else:
        mesh_kwargs = {"mesh_size_frac": .2, "mesh_size_min": .02}
        domain = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}
        gb = meshing.simplex_grid(fracs, domain, **mesh_kwargs)

    gb.compute_geometry()
    gb.assign_node_ordering()

    gb.add_node_props(["param"])
    for g, d in gb:
        kxx = np.ones(g.num_cells)
        perm = tensor.SecondOrderTensor(3, kxx)
        a = 0.01 / np.max(nx)
        a = np.power(a, gb.dim_max() - g.dim)
        param = Parameters(g)
        param.set_tensor("flow", perm)
        param.set_aperture(a)
        if g.dim == 2:
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            bound = bc.BoundaryCondition(
                g, bound_faces.ravel("F"), ["dir"] * bound_faces.size
            )
            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces] = g.face_centers[1, bound_faces]
            param.set_bc("flow", bound)
            param.set_bc_val("flow", bc_val)
        d["param"] = param

    gb.add_edge_props("kn")
    for e, d in gb.edges():
        g = gb.nodes_of_edge(e)[0]
        mg = d["mortar_grid"]
        check_P = mg.low_to_mortar_avg()
        d["kn"] = 1 / (check_P * gb.node_props(g, "param").get_aperture())

    return gb


# ------------------------------------------------------------------------------#


def elliptic_dirich_neumann_source_sink_cart_ref_3d():
    p_ref = np.array(
        [
            1.12612408,
            -10.44575,
            -18.21089071,
            -21.7313788,
            -1.64733896,
            -11.68522994,
            -19.65108784,
            -22.70209994,
            -2.46050104,
            -13.07136257,
            -21.06707504,
            -23.66942273,
            -2.49465073,
            -12.70131007,
            -20.25343208,
            -23.5515183,
            -2.10125633,
            -12.72123638,
            -21.06957633,
            -24.37000057,
            -2.83315597,
            -13.77682234,
            -24.53350194,
            -25.88570125,
            -3.64680989,
            -17.38985286,
            -28.34027307,
            -27.46735567,
            -3.35449112,
            -15.09334279,
            -23.79378233,
            -26.83132116,
            -4.07091283,
            -16.21653586,
            -25.30091244,
            -28.87240343,
            -4.44070969,
            -17.12486388,
            -28.75574562,
            -30.46663996,
            -5.16346912,
            -20.7431062,
            -32.72835189,
            -32.4244607,
            -4.89893689,
            -18.56531854,
            -28.50788954,
            -32.60763571,
            -7.49402606,
            -21.6895137,
            -31.15162219,
            -35.43782529,
            -7.74799483,
            -22.45132443,
            -32.65315461,
            -36.77044405,
            -8.13594619,
            -23.81413333,
            -34.55191292,
            -39.01167853,
            -8.13415535,
            -23.6740419,
            -34.7528843,
            -43.04584746,
        ]
    )
    return p_ref


# ------------------------------------------------------------------------------#
if __name__ == "__main__":
    unittest.main()
