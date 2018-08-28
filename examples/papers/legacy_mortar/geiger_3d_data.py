import numpy as np

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition
from porepy.numerics import elliptic

# ------------------------------------------------------------------------------#


class DarcyModelData(elliptic.EllipticDataAssigner):
    def __init__(self, g, data, **kwargs):
        self.domain = kwargs["domain"]
        self.tol = kwargs["tol"]

        self.apert = kwargs.get("aperture", 1e-4)
        self.km = kwargs.get("km", 1)
        self.km_low = kwargs.get("km_low", 1)
        self.kf = kwargs["kf"]
        self.max_dim = kwargs.get("max_dim", 3)

        # define two pieces of the boundary, useful to impose boundary conditions
        self.flux_boundary = np.empty(0)
        self.pressure_boundary = np.empty(0)

        b_faces = g.get_domain_boundary_faces()
        if b_faces.size > 0:
            b_face_centers = g.face_centers[:, b_faces]

            val = 0.5 - self.tol
            self.b_in = np.logical_and.reduce(
                tuple(b_face_centers[i, :] < val for i in range(self.max_dim))
            )

            val = 0.75 + self.tol
            self.b_out = np.logical_and.reduce(
                tuple(b_face_centers[i, :] > val for i in range(self.max_dim))
            )
            self.b_pressure = self.b_in + self.b_out

        elliptic.EllipticDataAssigner.__init__(self, g, data)

    def aperture(self):
        return np.power(self.apert, self.max_dim - self.grid().dim)

    def bc(self):
        b_faces = self.grid().get_domain_boundary_faces()
        labels = np.array(["neu"] * b_faces.size)
        if self.grid().dim == 3:
            if b_faces.size == 0:
                return BoundaryCondition(self.grid(), np.empty(0), np.empty(0))

            labels[self.b_pressure] = "dir"
        return BoundaryCondition(self.grid(), b_faces, labels)

    def bc_val(self):
        bc_val = np.zeros(self.grid().num_faces)
        if self.grid().dim == 3:
            b_faces = self.grid().get_domain_boundary_faces()
            if b_faces.size > 0:
                bc_val[b_faces[self.b_in]] = 1

        return bc_val

    def low_zones(self):
        zone_0 = np.logical_and(
            self.grid().cell_centers[0, :] > 0.5, self.grid().cell_centers[1, :] < 0.5
        )

        zone_1 = np.logical_and.reduce(
            tuple(
                [
                    self.grid().cell_centers[0, :] > 0.75,
                    self.grid().cell_centers[1, :] > 0.5,
                    self.grid().cell_centers[1, :] < 0.75,
                    self.grid().cell_centers[2, :] > 0.5,
                ]
            )
        )

        zone_2 = np.logical_and.reduce(
            tuple(
                [
                    self.grid().cell_centers[0, :] > 0.625,
                    self.grid().cell_centers[0, :] < 0.75,
                    self.grid().cell_centers[1, :] > 0.5,
                    self.grid().cell_centers[1, :] < 0.625,
                    self.grid().cell_centers[2, :] > 0.5,
                    self.grid().cell_centers[2, :] < 0.75,
                ]
            )
        )

        return np.logical_or.reduce(tuple([zone_0, zone_1, zone_2]))

    def permeability(self):
        dim = self.grid().dim
        if dim == 3:
            kxx = self.km * np.ones(self.grid().num_cells)
            kxx[self.low_zones()] = self.km_low
            return tensor.SecondOrder(dim, kxx=kxx, kyy=kxx, kzz=kxx)
        elif dim == 2:
            kxx = self.kf * np.ones(self.grid().num_cells)
            return tensor.SecondOrder(dim, kxx=kxx, kyy=kxx, kzz=1)
        else:  # dim == 1
            kxx = self.kf * np.ones(self.grid().num_cells)
            return tensor.SecondOrder(dim, kxx=kxx, kyy=1, kzz=1)
