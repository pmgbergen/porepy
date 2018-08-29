import numpy as np
import porepy as pp

class Data(object):

    def __init__(self, data, tol=1e-8):

        self.gb = None
        self.domain = None
        self.data = data

        self.tol = self.data.get("tol", 1e-8)

        self.make_gb()

    # ------------------------------------------------------------------------------#

    def eff_kf_n(self):
        return self.data["kf_n"]/self.data["aperture"]

    # ------------------------------------------------------------------------------#

    def eff_kf_t(self):
        return self.data["aperture"]*self.data["kf_t"]

    # ------------------------------------------------------------------------------#

    def make_gb(self):
        mesh_kwargs = {}
        mesh_kwargs = {"mesh_size_frac": self.data["mesh_size"],
                       "mesh_size_min": self.data["mesh_size"] / 20}

        file_name = "network.csv"
        self.write_network(file_name)
        self.gb, self.domain = pp.importer.dfm_3d_from_csv(file_name, **mesh_kwargs)
        self.gb.compute_geometry()

    # ------------------------------------------------------------------------------#


    def add_to_gb(self):
        """
        Define the permeability, apertures, boundary conditions
        """
        self.gb.add_node_props(["param", "is_tangential"])

        for g, d in self.gb:
            d["is_tangential"] = True
            param = pp.Parameters(g)

            unity = np.ones(g.num_cells)

            # Permeability
            if g.dim == 3:
                kxx = unity
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=kxx)
            elif g.dim == 2:
                kxx = self.data["kf_t"] * unity
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=1)
            else:
                kxx = self.data["kf_t"] * unity
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=1, kzz=1)

            param.set_tensor("flow", perm)

            # Source term
            param.set_source("flow", np.zeros(g.num_cells))

            # Assign apertures
            aperture = np.power(self.data["aperture"], 3 - g.dim)
            param.set_aperture(aperture * unity)

            # Boundaries
            bound_faces = g.get_boundary_faces()
            if bound_faces.size != 0:
                b_dir, b_in, b_out = self.bound_faces_dir(g)

                labels = np.array(["neu"] * bound_faces.size)
                labels[b_dir] = "dir"

                bc_val = np.zeros(g.num_faces)
                bc_val[bound_faces[b_in]] = 1
                bc_val[bound_faces[b_out]] = -1

                param.set_bc("flow", pp.BoundaryCondition(g, bound_faces, labels))
                param.set_bc_val("flow", bc_val)
            else:
                param.set_bc("flow", pp.BoundaryCondition(g, np.empty(0), np.empty(0)))

            d["param"] = param

        # Assign coupling permeability
        self.gb.add_edge_props("kn")
        for e, d in self.gb.edges():
            g_l = self.gb.nodes_of_edge(e)[0]
            mg = d["mortar_grid"]
            check_P = mg.low_to_mortar_avg()

            gamma = np.power(
                check_P * self.gb.node_props(g_l, "param").get_aperture(),
                1. / (3 - g_l.dim),
            )

            d["kn"] = self.data["kf_n"] / gamma

    # ------------------------------------------------------------------------------#

    def update(self, solver_flow):

        for g, d in self.gb:
            if g.dim == 2:
                # define the non-linear relation with u
                u = np.linalg.norm(d["P0u"], axis=0)

                # to trick the code we need to do the following
                coeff = 1./self.eff_kf_t() + self.data["beta"]*u
                kf = 1./coeff/self.data["aperture"]

                perm = pp.SecondOrderTensor(2, kxx=kf, kyy=kf, kzz=1)
                d["param"].set_tensor("flow", perm)

    # ------------------------------------------------------------------------------#

    def write_network(self, file_name):
        network = "0,0,0,1,1,1\n"
        network += "0.5,0,0,0.5,1,0,0.5,1,1,0.5,0,1\n"
        network += "0,0.5,0,1,0.5,0,1,0.5,1,0,0.5,1\n"
        network += "0,0,0.5,1,0,0.5,1,1,0.5,0,1,0.5\n"
        network += "0.75,0.5,0.5,0.75,1.0,0.5,0.75,1.0,1.0,0.75,0.5,1.0\n"
        network += "0.5,0.5,0.75,1.0,0.5,0.75,1.0,1.0,0.75,0.5,1.0,0.75\n"
        network += "0.5,0.75,0.5,1.0,0.75,0.5,1.0,0.75,1.0,0.5,0.75,1.0\n"
        network += "0.50,0.625,0.50,0.75,0.625,0.50,0.75,0.625,0.75,0.50,0.625,0.75\n"
        network += "0.625,0.50,0.50,0.625,0.75,0.50,0.625,0.75,0.75,0.625,0.50,0.75\n"
        network += "0.50,0.50,0.625,0.75,0.50,0.625,0.75,0.75,0.625,0.50,0.75,0.625"

        with open(file_name, "w") as text_file:
            text_file.write(network)

    # ------------------------------------------------------------------------------#

    def bound_faces_dir(self, g):
        bound_faces = g.get_boundary_faces()
        null = np.zeros(bound_faces.size, dtype=np.bool)
        if bound_faces.size == 0:
            return null, null, null
        else:
            bound_face_centers = g.face_centers[:, bound_faces]

            val = 0.5 - self.tol
            b_in = np.logical_and.reduce(
                tuple(bound_face_centers[i, :] < val for i in np.arange(3))
            )

            val = 0.75 + self.tol
            b_out = np.logical_and.reduce(
                tuple(bound_face_centers[i, :] > val for i in np.arange(3))
            )
            return np.logical_or(b_in, b_out), b_in, b_out

    # ------------------------------------------------------------------------------#
