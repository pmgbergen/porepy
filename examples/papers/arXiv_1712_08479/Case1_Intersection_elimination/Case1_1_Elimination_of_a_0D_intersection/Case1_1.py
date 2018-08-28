"""
Case 1.1. Flow and transport with and without Schur complement elimination
on a case consisting of two intersecting 1D fractures. Simulations for a range
of permeability ratios.
"""
import numpy as np
import scipy.sparse as sps

from porepy.params import bc, tensor
from porepy.numerics.elliptic import EllipticModel, EllipticDataAssigner
from porepy.fracs import meshing

import matplotlib.pyplot as plt
from examples.papers.arXiv_1712_08479.utils import (
    perform_condensation,
    assign_data,
    compute_errors,
)


def define_grid(nx, ny):
    """
    Make cartesian grids and a bucket. One horizontal and one vertical 1D
    fracture in a 2D matrix domain.
    """
    mesh_kwargs = {"physdims": np.array([1, 1])}
    f_1 = np.array([[.5, .5], [0, 1]])
    f_2 = np.array([[0, 1], [.5, .5]])
    fracs = [f_1, f_2]
    gb = meshing.cart_grid(fracs, [nx, ny], **mesh_kwargs)
    gb.assign_node_ordering()
    return gb


class FlowData(EllipticDataAssigner):
    """
    Assign flow problem data to a given grid.
    """

    def __init__(self, g, d):
        EllipticDataAssigner.__init__(self, g, d)

    def aperture(self):
        a = np.power(1e-2, 2 - self.grid().dim)
        return np.ones(self.grid().num_cells) * a

    def permeability(self):
        kxx = np.ones(self.grid().num_cells) * np.power(1e4, self.grid().dim < 2)
        return tensor.SecondOrderTensor(3, kxx)

    def bc(self):
        # Default values (Neumann) on the vertical fracture and
        # 0D grid
        if self.grid().dim < 1 or abs(self.grid().face_centers[0, 0] - .5) < 1e-3:
            return bc.BoundaryCondition(self.grid())
        # Dirichlet on the two other
        dirfaces = bc.face_on_side(self.grid(), ["xmin", "xmax"])
        dirfaces = np.asarray(dirfaces).flatten()
        labels = np.array(["dir"] * dirfaces.size)

        return bc.BoundaryCondition(self.grid(), dirfaces, labels)

    def bc_val(self):
        bc_values = np.zeros(self.grid().num_faces)
        if self.grid().dim > 0 and abs(self.grid().face_centers[0, 0] - .5) > 1e-3:
            # p_D = 1-x for the two Dirichlet grids
            bc_values[bc.face_on_side(self.grid(), "xmin")[0]] = 1
        return bc_values


def update_perm(gb, k_hor, k_ver, k_intersection):
    """
    Reassign permeabilities in the fractures.
    """
    for g, d in gb:
        if g.dim > 1:
            continue
        if np.isclose(g.cell_centers[0, 0], .5):
            perm = tensor.SecondOrderTensor(3, k_ver * np.ones(g.num_cells))
        if np.isclose(g.cell_centers[1, 0], .5) and g.dim == 1:
            perm = tensor.SecondOrderTensor(3, k_hor * np.ones(g.num_cells))
        if g.dim == 0:
            perm = tensor.SecondOrderTensor(3, k_intersection * np.ones(g.num_cells))
        d["param"].set_tensor("flow", perm)


class BothProblems:
    """
    Wrapper for the full and eliminated elliptic (flow) problem.
    """

    def __init__(self, full, reduced):
        self.full = full
        self.el = reduced

    def solve(self, h, v, i):
        # Update and solve full problem
        update_perm(self.full.grid(), h, v, i)
        p = self.full.solve()
        # Obtain the reduced solution matrix and solve
        perform_condensation(self.full, self.el, 0)
        self.el.x = sps.linalg.spsolve(self.el.lhs, self.el.rhs)
        # Evaluate condition numbers
        self.full.cond = np.linalg.cond(self.full.lhs.todense())
        self.el.cond = np.linalg.cond(self.el.lhs.todense())

        return p, self.el.x, self.full.cond, self.el.cond


def import_mrst_data(k_h, k_v):
    p = np.loadtxt("MRST/pressures_{}h_{}v.csv".format(k_h, k_v), delimiter=",")
    c = np.loadtxt("MRST/condition_number_{}h_{}v.csv".format(k_h, k_v))
    return p, c


def barplot(values, title="Schur complement", plot_type="error_"):
    if plot_type == "cond_":
        values = np.log10(values)
    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")
    x, y = np.random.rand(2, 100) * 4
    hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])
    hist = values
    nx = values.shape[0]
    ny = values.shape[1]
    xedges = np.linspace(0, nx, nx + 1)
    yedges = np.linspace(0, ny, ny + 1)
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    xpos = xpos.flatten("F")
    ypos = ypos.flatten("F")
    zpos = np.zeros_like(xpos)

    # Construct arrays with the dimensions for the 16 bars.
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = hist.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color="b", zsort="average")

    if plot_type == "cond_":
        z_label = "$R_C$"
        es = np.array([1, 10, 100])
        zedges = np.log10(es)
        ax.set_zlim([0, 2.5])
    else:
        z_label = "$E$"
        es = np.array([.05, .1, .15])
        zedges = es
        ax.set_zlim([0, .155])
    ax.view_init(27, 125)
    ax.set_zticks(zedges)
    sz_small = 10
    ax.set_zticklabels(["    %.1e" % e for e in es], size=sz_small)
    plt.xticks(
        xedges,
        ["1e%.0d" % np.log10(k) for k in horizontal_permeabilities],
        size=sz_small,
    )
    plt.yticks(
        yedges, ["1e%.0d" % np.log10(k) for k in vertical_permeabilities], size=sz_small
    )
    sz = 14
    ax.text2D(1, 0.54, z_label, transform=ax.transAxes, size=sz)
    ax.text2D(0.25, 0, "$K_h$", transform=ax.transAxes, size=sz)
    ax.text2D(.85, 0.08, "$K_v$", transform=ax.transAxes, size=sz)
    #    plt.title(t1 + ', ' + title, position=(.5, 1.06), size=sz)
    plt.savefig("figures/" + plot_type + title + ".png")


def split_mrst_to_gb(gb, u):
    Both.el.flux_disc().split(gb, "pressure", u)


if __name__ == "__main__":
    # Set up grid
    nc = 4
    gb = define_grid(nc, nc)
    assign_data(gb, FlowData, "problem")
    # Copy it for the elimination
    gb_el, _ = gb.duplicate_without_dimension(0)
    # Initialize model and solver class
    problem = EllipticModel(gb)
    problem_el = EllipticModel(gb_el)
    # Merge for convenience
    Both = BothProblems(problem, problem_el)
    # Define permeability ranges
    horizontal_permeabilities = [10 ** i for i in range(-3, 4)]
    vertical_permeabilities = horizontal_permeabilities.copy()

    global_errors_porepy = np.zeros(
        (len(horizontal_permeabilities), len(vertical_permeabilities))
    )
    global_errors_mrst = np.zeros(
        (len(horizontal_permeabilities), len(vertical_permeabilities))
    )
    improvement_porepy = np.zeros(
        (len(horizontal_permeabilities), len(vertical_permeabilities))
    )
    improvement_mrst = np.zeros(
        (len(horizontal_permeabilities), len(vertical_permeabilities))
    )
    # Solve and compare solutions for entire permeability range
    for i, k_h in enumerate(horizontal_permeabilities):
        for j, k_v in enumerate(vertical_permeabilities):
            p, p_el, cond, cond_el = Both.solve(k_h, k_v, k_v)
            # Obtain stored Star-Delta solution computed using MRST
            p_mrst, cond_mrst = import_mrst_data(k_h, k_v)

            # Compare:
            es_porepy, es_mrst, E_porepy, E_mrst = compute_errors(
                gb_el, p, p_el, p_mrst
            )
            global_errors_porepy[i, j] = E_porepy
            global_errors_mrst[i, j] = E_mrst
            improvement_porepy[i, j] = cond / cond_el
            improvement_mrst[i, j] = cond / cond_mrst

            if k_h == 1e3 and k_v == 1e-3:
                # Save for vizualization

                Both.full.exporter.change_name("pressure_no_elimination")
                Both.full.pressure()
                Both.full.save(["pressure"])
                Both.el.exporter.change_name("pressure_Schur_complement")
                Both.el.pressure()
                Both.el.save(["pressure"])
                # Solve with high intersection permeability for comparison to
                # Star-Delta solution.
                Both.el.exporter.change_name("pressure_high_intersection_perm")
                p, p_el, cond, cond_el = Both.solve(k_h, k_v, 1e10)
                Both.el.pressure()
                Both.el.save(["pressure"])
                Both.el.exporter.change_name("pressure_mrst")
                split_mrst_to_gb(gb_el, p_mrst)
                Both.el.save(["pressure"])

    # Plot errors and condition numbers
    barplot(global_errors_mrst, "Star-Delta")
    barplot(global_errors_porepy, "Schur complement")
    barplot(improvement_mrst, "Star-Delta", "cond_")
    barplot(improvement_porepy, "Schur complement", "cond_")
    plt.show()
