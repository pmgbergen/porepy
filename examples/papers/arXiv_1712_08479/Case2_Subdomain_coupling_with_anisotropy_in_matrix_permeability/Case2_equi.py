"""
Case 2, equidimensional, MPFA only discretization. Coarse and reference
solutions computed for both flow and transport.
"""
import numpy as np

from porepy.fracs import meshing
from porepy.numerics.elliptic import EllipticModel, EllipticDataAssigner
from porepy.params import bc, tensor
from porepy.utils import comp_geom as cg
from porepy.numerics.fv import mpfa
from porepy.numerics.parabolic import ParabolicModel, ParabolicDataAssigner

from examples.papers.arXiv_1712_08479.utils import edge_params, assign_data
from Case2_reduced import BothProblems


def define_grid(nx, ny_red):
    """
    Make cartesian grids and a bucket. One horizontal 1d fracture in
    a 2d matrix domain.
    """

    mesh_kwargs = {"physdims": np.array([1, 1])}
    fracs = []
    h = 1e-3
    ny = ny_red + 1
    gb = meshing.cart_grid(fracs, [nx, ny], **mesh_kwargs)
    gb.assign_node_ordering()
    g = gb.grids_of_dimension(2)[0]
    n = g.nodes
    ymax = 1
    n_layer = nx + 1
    ycoords = np.linspace(0, ymax, ny)
    y1 = ycoords[0 : int(ny_red / 2)]
    y2 = ycoords[int(ny_red / 2 + 1) : ny]
    ycoords = np.concatenate((y1, np.array([ymax / 2 - h / 2, ymax / 2 + h / 2]), y2))
    for i in range(ny):
        ind = (i) * n_layer + np.arange(1, n_layer + 1) - 1
        n[1, ind] = ycoords[i]
    gb.compute_geometry()

    eps = 1e-5
    horizontal = np.logical_and(
        g.cell_centers[1, :] < ymax / 2 + eps / 10,
        g.cell_centers[1, :] > ymax / 2 - eps / 10,
    )
    return gb, horizontal


def boundary_face_type(g):
    """
    Extract the faces where Dirichlet conditions are to be set.
    """
    bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    bound_face_centers = g.face_centers[:, bound_faces]
    onev = np.ones(bound_face_centers.shape[1])
    dirface1 = np.where(
        np.array(bound_face_centers[0, :] < 0.25 * onev)
        & np.array(bound_face_centers[1, :] < 1e-5 * onev)
    )
    dirface2 = np.where(
        np.array(bound_face_centers[0, :] > 0.75 * onev)
        & np.array(bound_face_centers[1, :] > (1 - 1e-5) * onev)
    )
    return bound_faces, dirface1, dirface2


def bc_values(g):
    bc_val = np.zeros(g.num_faces)
    if g.dim == 1:
        return bc_val
    bound_faces, dirface1, _ = boundary_face_type(g)
    bc_val[bound_faces[dirface1]] = 1
    return bc_val


class FlowData(EllipticDataAssigner):
    """
    Assign flow problem data to a given grid.
    """

    def __init__(self, g, d):
        EllipticDataAssigner.__init__(self, g, d)

    def aperture(self):
        a = 1
        return np.ones(self.grid().num_cells) * a

    def permeability(self):
        kxx = np.ones(self.grid().num_cells) * np.power(1e4, 2 - self.grid().dim)
        return tensor.SecondOrderTensor(3, kxx)

    def bc(self):
        if self.grid().dim == 1:
            return bc.BoundaryCondition(self.grid())

        bound_faces, dirface1, dirface2 = boundary_face_type(self.grid())
        labels = np.array(["neu"] * bound_faces.size)

        dirfaces = np.concatenate((dirface1, dirface2))
        labels[dirfaces] = "dir"
        return bc.BoundaryCondition(self.grid(), bound_faces, labels)

    def bc_val(self):
        return bc_values(self.grid())


def anisotropy(gb, deg, yfactor):
    """
    Set anisotropic permeability in the 2d matrix.
    """
    for g, d in gb:
        # Get rotational tensor R
        perm_x = 1
        perm_y = 1 / yfactor
        perm_z = 1
        rad = deg * np.pi / 180
        v = np.array([0, 0, 1])
        R = cg.rot(rad, v)
        # Set up orthogonal tensor and rotate it
        k_orth = np.array([[perm_x, 0, 0], [0, perm_y, 0], [0, 0, perm_z]])
        k = np.dot(np.dot(R, k_orth), R.T)
        kf = np.ones(g.num_cells)
        kxx = kf * k[0, 0]
        kyy = kf * k[1, 1]
        kxy = kf * k[0, 1]
        kxz = kf * k[0, 2]
        kyz = kf * k[1, 2]
        kzz = kf * k[2, 2]

        k_frac = 1e4
        kxx[fracture_cells] = k_frac
        kyy[fracture_cells] = k_frac
        kzz[fracture_cells] = k_frac
        kxy[fracture_cells] = 0
        kxz[fracture_cells] = 0
        kyz[fracture_cells] = 0
        perm = tensor.SecondOrderTensor(
            3, kxx=kxx, kyy=kyy, kzz=kzz, kxy=kxy, kxz=kxz, kyz=kyz
        )
        d["param"].set_tensor("flow", perm)


class TransportData(ParabolicDataAssigner):
    """
    Assign transport problem data to given grid.
    """

    def __init__(self, g, d):
        ParabolicDataAssigner.__init__(self, g, d)

    def bc(self):
        if self.grid().dim == 1:
            return bc.BoundaryCondition(self.grid())

        bound_faces, dirface1, dirface2 = boundary_face_type(self.grid())
        labels = np.array(["neu"] * bound_faces.size)

        dirfaces = np.concatenate((dirface1, dirface2))
        labels[dirfaces] = "dir"
        return bc.BoundaryCondition(self.grid(), bound_faces, labels)

    def bc_val(self, t):
        return bc_values(self.grid()) * 0

    def initial_condition(self):
        return np.ones(self.grid().num_cells)


class DarcySolver(EllipticModel):
    """
    Set up Darcy solver with MPFA.
    """

    def __init__(self, gb):
        EllipticModel.__init__(self, gb, **kw)

    def flux_disc(self):
        return mpfa.MpfaMixedDim(physics=self.physics)


class TransportSolver(ParabolicModel):
    """
    Make a ParabolicModel for the transport problem with specified parameters.
    """

    def __init__(self, gb):
        self._g = gb
        ParabolicModel.__init__(self, gb, **kw)
        self._solver.parameters["store_results"] = True

    def grid(self):
        return self._g

    def space_disc(self):
        return self.advective_disc()

    def time_step(self):
        return 0.5

    def end_time(self):
        return 1


if __name__ == "__main__":
    ny = [4, 8, 16, 32, 256]
    yfactor = [1, 2, 4, 6]
    deg = [10, 30, 60]
    main_folder = "results_equi"
    for n in ny:
        gb, fracture_cells = define_grid(n, n)
        assign_data(gb, FlowData, "problem")
        edge_params(gb)
        assign_data(gb, TransportData, "transport_data")

        for y in yfactor:
            for d in deg:
                appendix = "{}cells_{}degrees_{}factor".format(n, d, y)
                kw = {"folder_name": main_folder}
                anisotropy(gb, d, y)
                darcy_problem = DarcySolver(gb)
                transport_problem = TransportSolver(gb)
                Full = BothProblems(darcy_problem, transport_problem)
                p, t = Full.solve()
                Full.save_text(t, n, main_folder, appendix)

        np.savetxt(
            "results_equi" + "/cell_volumes_nx_{}.csv".format(n),
            gb.grids_of_dimension(2)[0].cell_volumes,
            delimiter=",",
        )
