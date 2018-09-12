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

        self.domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}

        file_name = "network.csv"
        self.write_network(file_name)
        self.gb = pp.importer.dfm_2d_from_csv(file_name, mesh_kwargs, self.domain)
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

            # Permeability
            if g.dim == 2:
                kxx = np.ones(g.num_cells)
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=1)
            else:
                kxx = self.data["kf_t"] * np.ones(g.num_cells)
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=1, kzz=1)
            param.set_tensor("flow", perm)

            # Source term
            param.set_source("flow", np.zeros(g.num_cells))

            # Assign apertures
            aperture = np.power(self.data["aperture"], 2 - g.dim)
            param.set_aperture(np.ones(g.num_cells) * aperture)

            # Boundaries
            bound_faces = g.get_boundary_faces()
            if bound_faces.size != 0:
                bound_face_centers = g.face_centers[:, bound_faces]

                left = bound_face_centers[0, :] < self.domain["xmin"] + self.tol
                right = bound_face_centers[0, :] > self.domain["xmax"] - self.tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[right] = "dir"

                bc_val = np.zeros(g.num_faces)
                bc_val[bound_faces[left]] = -aperture * g.face_areas[bound_faces[left]]
                bc_val[bound_faces[right]] = 1

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
                1. / (2 - g_l.dim),
            )

            d["kn"] = self.data["kf_n"] / gamma

    # ------------------------------------------------------------------------------#

    @staticmethod
    def write_network(file_name):
        network = "FID,START_X,START_Y,END_X,END_Y\n"
        network += "0,0,0.5,1,0.5\n"
        network += "1,0.5,0,0.5,1\n"
        network += "2,0.5,0.75,1,0.75\n"
        network += "3,0.75,0.5,0.75,1\n"
        network += "4,0.5,0.625,0.75,0.625\n"
        network += "5,0.625,0.5,0.625,0.75\n"

        with open(file_name, "w") as text_file:
            text_file.write(network)

    # ------------------------------------------------------------------------------#
