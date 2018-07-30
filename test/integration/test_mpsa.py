import numpy as np
import scipy.sparse as sps

from porepy.params import tensor, bc
from porepy.grids import structured, simplex
from porepy.numerics.fv import mpsa, fvutils
from test.integration import setup_grids_mpfa_mpsa_tests as setup_grids


def setup_stiffness(g, mu=1, l=1):
    mu = np.ones(g.num_cells) * mu
    l = np.ones(g.num_cells) * l
    return tensor.FourthOrderTensor(g.dim, mu, l)


def test_uniform_strain():
    g_list = setup_grids.setup_2d()

    for g in g_list:
        bound_faces = np.argwhere(np.abs(g.cell_faces).sum(axis=1).A.ravel("F") == 1)
        bound = bc.BoundaryCondition(
            g, bound_faces.ravel("F"), ["dir"] * bound_faces.size
        )
        mu = 1
        l = 1
        constit = setup_stiffness(g, mu, l)

        # Python inverter is most efficient for small problems
        stress, bound_stress = mpsa.mpsa(g, constit, bound, inverter="python")

        div = fvutils.vector_divergence(g)
        a = div * stress

        xc = g.cell_centers
        xf = g.face_centers

        gx = np.random.rand(1)
        gy = np.random.rand(1)

        dc_x = np.sum(xc * gx, axis=0)
        dc_y = np.sum(xc * gy, axis=0)
        df_x = np.sum(xf * gx, axis=0)
        df_y = np.sum(xf * gy, axis=0)

        d_bound = np.zeros((g.dim, g.num_faces))
        d_bound[0, bound.is_dir] = df_x[bound.is_dir]
        d_bound[1, bound.is_dir] = df_y[bound.is_dir]

        rhs = div * bound_stress * d_bound.ravel("F")

        d = np.linalg.solve(a.todense(), -rhs)

        traction = stress * d + bound_stress * d_bound.ravel("F")

        s_xx = (2 * mu + l) * gx + l * gy
        s_xy = mu * (gx + gy)
        s_yx = mu * (gx + gy)
        s_yy = (2 * mu + l) * gy + l * gx

        n = g.face_normals
        traction_ex_x = s_xx * n[0] + s_xy * n[1]
        traction_ex_y = s_yx * n[0] + s_yy * n[1]

        assert np.max(np.abs(d[::2] - dc_x)) < 1e-8
        assert np.max(np.abs(d[1::2] - dc_y)) < 1e-8
        assert np.max(np.abs(traction[::2] - traction_ex_x)) < 1e-8
        assert np.max(np.abs(traction[1::2] - traction_ex_y)) < 1e-8


def test_uniform_displacement():

    g_list = setup_grids.setup_2d()

    for g in g_list:
        bound_faces = np.argwhere(np.abs(g.cell_faces).sum(axis=1).A.ravel("F") == 1)
        bound = bc.BoundaryCondition(
            g, bound_faces.ravel("F"), ["dir"] * bound_faces.size
        )
        constit = setup_stiffness(g)

        # Python inverter is most efficient for small problems
        stress, bound_stress = mpsa.mpsa(g, constit, bound, inverter="python")

        div = fvutils.vector_divergence(g)
        a = div * stress

        d_x = np.random.rand(1)
        d_y = np.random.rand(1)
        d_bound = np.zeros((g.dim, g.num_faces))
        d_bound[0, bound.is_dir] = d_x
        d_bound[1, bound.is_dir] = d_y

        rhs = div * bound_stress * d_bound.ravel("F")

        d = np.linalg.solve(a.todense(), -rhs)

        traction = stress * d + bound_stress * d_bound.ravel("F")

        assert np.max(np.abs(d[::2] - d_x)) < 1e-8
        assert np.max(np.abs(d[1::2] - d_y)) < 1e-8
        assert np.max(np.abs(traction)) < 1e-8


def test_uniform_displacement_neumann():
    physdims = [1, 1]
    g_size = [4, 8]
    g_list = [structured.CartGrid([n, n], physdims=physdims) for n in g_size]
    [g.compute_geometry() for g in g_list]
    error = []
    for g in g_list:
        bot = np.ravel(np.argwhere(g.face_centers[1, :] < 1e-10))
        left = np.ravel(np.argwhere(g.face_centers[0, :] < 1e-10))
        dir_faces = np.hstack((left, bot))
        bound = bc.BoundaryCondition(g, dir_faces.ravel("F"), ["dir"] * dir_faces.size)
        constit = setup_stiffness(g)

        # Python inverter is most efficient for small problems
        stress, bound_stress = mpsa.mpsa(g, constit, bound, inverter="python")

        div = fvutils.vector_divergence(g)
        a = div * stress

        d_x = np.random.rand(1)
        d_y = np.random.rand(1)
        d_bound = np.zeros((g.dim, g.num_faces))
        d_bound[0, bound.is_dir] = d_x
        d_bound[1, bound.is_dir] = d_y

        rhs = div * bound_stress * d_bound.ravel("F")

        d = np.linalg.solve(a.todense(), -rhs)

        traction = stress * d + bound_stress * d_bound.ravel("F")
        assert np.max(np.abs(d[::2] - d_x)) < 1e-8
        assert np.max(np.abs(d[1::2] - d_y)) < 1e-8
        assert np.max(np.abs(traction)) < 1e-8


def test_conservation_of_momentum():
    pts = np.random.rand(3, 9)
    corners = [
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1],
    ]
    pts = np.hstack((corners, pts))
    gt = simplex.TetrahedralGrid(pts)
    gc = structured.CartGrid([3, 3, 3], physdims=[1, 1, 1])
    g_list = [gt, gc]
    [g.compute_geometry() for g in g_list]
    for g in g_list:
        g.compute_geometry()
        bot = np.ravel(np.argwhere(g.face_centers[1, :] < 1e-10))
        left = np.ravel(np.argwhere(g.face_centers[0, :] < 1e-10))
        dir_faces = np.hstack((left, bot))
        bound = bc.BoundaryCondition(g, dir_faces.ravel("F"), ["dir"] * dir_faces.size)
        constit = setup_stiffness(g)

        # Python inverter is most efficient for small problems
        stress, bound_stress = mpsa.mpsa(g, constit, bound, inverter="python")

        div = fvutils.vector_divergence(g)
        a = div * stress

        bndr = g.get_all_boundary_faces()
        d_x = np.random.rand(bndr.size)
        d_y = np.random.rand(bndr.size)
        d_bound = np.zeros((g.dim, g.num_faces))
        d_bound[0, bndr] = d_x
        d_bound[1, bndr] = d_y

        rhs = div * bound_stress * d_bound.ravel("F")

        d = np.linalg.solve(a.todense(), -rhs)

        traction = stress * d + bound_stress * d_bound.ravel("F")
        traction_2d = traction.reshape((g.dim, -1), order="F")
        for cell in range(g.num_cells):
            fid, _, sgn = sps.find(g.cell_faces[:, cell])
            assert np.all(np.sum(traction_2d[:, fid] * sgn, axis=1) < 1e-10)


if __name__ == "__main__":
    test_uniform_displacement()
    test_uniform_strain()
