import numpy as np
import porepy as pp
import time
# Create grid
n = 20
g = pp.CartGrid([n, n, n])
g.compute_geometry()


# Create stiffness matrix
lam = np.ones(g.num_cells)
mu = np.ones(g.num_cells)
C = pp.FourthOrderTensor(mu, lam)


dirich = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))
bound = pp.BoundaryConditionVectorial(g, dirich, ["dir"] * dirich.size)


top_faces = np.ravel(np.argwhere(g.face_centers[1] > n - 1e-10))
bot_faces = np.ravel(np.argwhere(g.face_centers[1] < 1e-10))

u_b = np.zeros((g.dim, g.num_faces))
u_b[1, top_faces] = -1 * g.face_areas[top_faces]
u_b[:, bot_faces] = 0

u_b = u_b.ravel("F")

parameter_keyword = "mechanics"

mpsa_class = pp.Mpsa(parameter_keyword)
f = np.zeros(g.dim * g.num_cells)

specified_parameters = {
    "fourth_order_tensor": C,
    "source": f,
    "bc": bound,
    "bc_values": u_b,
}
data = pp.initialize_default_data(g, {}, parameter_keyword, specified_parameters)
tb = time.time()
mpsa_class.discretize(g, data)
te = time.time()
print("Elapsed time: ",te-tb)
# A, b = mpsa_class.assemble_matrix_rhs(g, data)
#
# u = np.linalg.solve(A.A, b)