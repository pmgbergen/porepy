import numpy as np
import unittest

import porepy as pp


class BasicsTest(unittest.TestCase):

    # ------------------------------------------------------------------------------#

    def test_mono_equals_multi(self):
        """
        test that the mono_dimensional elliptic solver gives the same answer as
        the grid bucket elliptic
        """
        g = pp.CartGrid([10, 10])
        g.compute_geometry()
        gb = pp.meshing.cart_grid([], [10, 10])
        param_g = pp.Parameters(g)

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
            bc_labels = pp.BoundaryCondition(g, bound_faces, labels)

            return bc_labels

        param_g.set_bc_val("flow", bc_val(g))
        param_g.set_bc("flow", bc_labels(g))

        gb.add_node_props(["param"])
        for sub_g, d in gb:
            d["param"] = pp.Parameters(sub_g)
            d["param"].set_bc_val("flow", bc_val(g))
            d["param"].set_bc("flow", bc_labels(sub_g))

        for e, d in gb.edges():
            gl, _ = gb.nodes_of_edge(e)
            d_l = gb.node_props(gl)
            d["kn"] = 1. / np.mean(d_l["param"].get_aperture())

        problem_mono = pp.EllipticModel(g, {"param": param_g})
        problem_mult = pp.EllipticModel(gb)

        p_mono = problem_mono.solve()
        p_mult = problem_mult.solve()

        assert np.allclose(p_mono, p_mult)

    # ------------------------------------------------------------------------------#

    def test_elliptic_uniform_flow_cart(self):
        gb = setup_2d_1d([10, 10])
        problem = pp.EllipticModel(gb)
        p = problem.solve()
        problem.split("pressure")

        for g, d in gb:
            pressure = d["pressure"]
            p_analytic = g.cell_centers[1]
            p_diff = pressure - p_analytic
            assert np.max(np.abs(p_diff)) < 2e-2

    # ------------------------------------------------------------------------------#

    def test_elliptic_uniform_flow_simplex(self):
        """
        Unstructured simplex grid. Note that the solution depends
        on the grid quality. Also sensitive to the way in which
        the tpfa half transmissibilities are computed.
        """
        gb = setup_2d_1d(np.array([10, 10]), simplex_grid=True)
        problem = pp.EllipticModel(gb)
        p = problem.solve()
        problem.split("pressure")

        for g, d in gb:
            pressure = d["pressure"]
            p_analytic = g.cell_centers[1]
            p_diff = pressure - p_analytic
            assert np.max(np.abs(p_diff)) < 0.033

    def test_elliptic_dirich_neumann_source_sink_cart(self):
        gb = setup_3d(np.array([4, 4, 4]), simplex_grid=False)
        problem = pp.EllipticModel(gb)
        p = problem.solve()
        problem.split("pressure")

        for g, d in gb:
            if g.dim == 3:
                p_ref = elliptic_dirich_neumann_source_sink_cart_ref_3d()
                assert np.allclose(d["pressure"], p_ref)
            if g.dim == 0:
                p_ref = [-10681.52153285]
                assert np.allclose(d["pressure"], p_ref)
        return gb


def setup_3d(nx, simplex_grid=False):
    f1 = np.array([[0.2, 0.2, 0.8, 0.8], [0.2, 0.8, 0.8, 0.2], [0.5, 0.5, 0.5, 0.5]])
    f2 = np.array([[0.2, 0.8, 0.8, 0.2], [0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.8, 0.8]])
    f3 = np.array([[0.5, 0.5, 0.5, 0.5], [0.2, 0.8, 0.8, 0.2], [0.2, 0.2, 0.8, 0.8]])
    fracs = [f1, f2, f3]
    if not simplex_grid:
        gb = pp.meshing.cart_grid(fracs, nx, physdims=[1, 1, 1])
    else:
        mesh_size = .3
        mesh_kwargs = {
            "mesh_size_frac": mesh_size,
            "mesh_size_bound": 2 * mesh_size,
            "mesh_size_min": mesh_size / 20,
        }
        domain = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}
        gb = pp.meshing.simplex_grid(fracs, domain, **mesh_kwargs)

    gb.add_node_props(["param"])
    for g, d in gb:
        a = 0.01 / np.max(nx)
        a = np.power(a, gb.dim_max() - g.dim)
        param = pp.Parameters(g)
        param.set_aperture(a)

        # BoundaryCondition
        left = g.face_centers[0] < 1e-6
        top = g.face_centers[2] > 1 - 1e-6
        dir_faces = np.argwhere(left)
        bc_cond = pp.BoundaryCondition(g, dir_faces, ["dir"] * dir_faces.size)
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

    for e, d in gb.edges():
        gl, _ = gb.nodes_of_edge(e)
        d_l = gb.node_props(gl)
        d["kn"] = 1. / np.mean(d_l["param"].get_aperture())

    return gb


def setup_2d_1d(nx, simplex_grid=False):
    frac1 = np.array([[0.2, 0.8], [0.5, 0.5]])
    frac2 = np.array([[0.5, 0.5], [0.8, 0.2]])
    fracs = [frac1, frac2]
    if not simplex_grid:
        gb = pp.meshing.cart_grid(fracs, nx, physdims=[1, 1])
    else:
        mesh_kwargs = {}
        mesh_size = .08
        mesh_kwargs = {
            "mesh_size_frac": mesh_size,
            "mesh_size_bound": 2 * mesh_size,
            "mesh_size_min": mesh_size / 20,
        }
        domain = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}
        gb = pp.meshing.simplex_grid(fracs, domain, **mesh_kwargs)

    gb.compute_geometry()
    gb.assign_node_ordering()
    gb.add_node_props(["param"])
    for g, d in gb:
        kxx = np.ones(g.num_cells)
        perm = pp.SecondOrderTensor(3, kxx)
        a = 0.01 / np.max(nx)
        a = np.power(a, gb.dim_max() - g.dim)
        param = pp.Parameters(g)
        param.set_tensor("flow", perm)
        param.set_aperture(a)
        if g.dim == 2:
            bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            bound = pp.BoundaryCondition(
                g, bound_faces.ravel("F"), ["dir"] * bound_faces.size
            )
            bc_val = np.zeros(g.num_faces)
            bc_val[bound_faces] = g.face_centers[1, bound_faces]
            param.set_bc("flow", bound)
            param.set_bc_val("flow", bc_val)
        d["param"] = param

    for e, d in gb.edges():
        gl, _ = gb.nodes_of_edge(e)
        d_l = gb.node_props(gl)
        d["kn"] = 1. / np.mean(d_l["param"].get_aperture())

    return gb


def elliptic_dirich_neumann_source_sink_cart_ref_3d():
    p_ref = np.array(
        [
            0.54569807,
            -11.33847701,
            -19.44469667,
            -23.13270824,
            -2.0383034,
            -12.7325353,
            -20.96170085,
            -24.14598827,
            -2.81411413,
            -14.03099116,
            -22.26965618,
            -25.0599583,
            -2.82347796,
            -13.61559858,
            -21.47525449,
            -24.92530184,
            -2.46109987,
            -13.72237413,
            -22.34590059,
            -25.80743978,
            -3.22886902,
            -15.2932041,
            -26.21562784,
            -27.42958569,
            -3.99191228,
            -18.7230516,
            -29.82037611,
            -28.89888689,
            -3.68767708,
            -16.13267072,
            -25.09046136,
            -28.24069273,
            -4.36105414,
            -17.1731891,
            -26.53936451,
            -30.32156462,
            -4.81756349,
            -18.63759603,
            -30.4521418,
            -32.07998565,
            -5.48967413,
            -22.06830969,
            -34.23202695,
            -33.94382163,
            -5.17800149,
            -19.54656499,
            -29.78331274,
            -34.04812085,
            -7.71447238,
            -22.60555667,
            -32.40402645,
            -36.85946853,
            -8.0057511,
            -23.53053876,
            -34.01171608,
            -38.25281452,
            -8.37193199,
            -24.79207392,
            -35.81905886,
            -40.46008782,
            -8.34409276,
            -24.5705303,
            -35.99938955,
            -44.22465628,
        ]
    )
    return p_ref


if __name__ == "__main__":
    unittest.main()
