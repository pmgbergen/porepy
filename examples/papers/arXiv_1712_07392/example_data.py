import numpy as np

from porepy.params import tensor
from porepy.params.bc import BoundaryCondition

from porepy.numerics import elliptic

# ------------------------------------------------------------------------------#


class DarcyModelData(elliptic.EllipticDataAssigner):
    def __init__(self, g, data, **kwargs):
        self.domain = kwargs["domain"]
        self.gb = kwargs["gb"]
        self.tol = kwargs["tol"]

        self.apert = kwargs["aperture"]
        self.km = kwargs["km"]
        self.kf_low = kwargs["kf_low"]
        self.kf_high = kwargs["kf_high"]
        self.special_fracture = kwargs["special_fracture"]

        elliptic.EllipticDataAssigner.__init__(self, g, data)

    def aperture(self):
        return np.power(self.apert, self.gb.dim_max() - self.grid().dim)

    def bc(self):

        bound_faces = self.grid().tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size == 0:
            return BoundaryCondition(self.grid(), np.empty(0), np.empty(0))

        bound_face_centers = self.grid().face_centers[:, bound_faces]

        top = bound_face_centers[2, :] > self.domain["zmax"] - self.tol
        bottom = bound_face_centers[2, :] < self.domain["zmin"] + self.tol

        boundary = np.logical_or(top, bottom)

        labels = np.array(["neu"] * bound_faces.size)
        labels[boundary] = ["dir"]

        return BoundaryCondition(self.grid(), bound_faces, labels)

    def bc_val(self):

        bc_val = np.zeros(self.grid().num_faces)
        bound_faces = self.grid().tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size == 0:
            return bc_val

        bound_face_centers = self.grid().face_centers[:, bound_faces]
        bottom = bound_face_centers[2, :] < self.domain["zmin"] + self.tol

        bc_val[bound_faces[bottom]] = 1
        return bc_val


# ------------------------------------------------------------------------------#


class VEMModelData(DarcyModelData):
    def __init__(self, g, data, **kwargs):
        DarcyModelData.__init__(self, g, data, **kwargs)

    def permeability(self):
        if self.grid().dim == 3:
            kxx = self.km * np.ones(self.grid().num_cells)
            return tensor.SecondOrderTensor(self.grid().dim, kxx=kxx, kyy=kxx, kzz=kxx)

        elif self.grid().dim == 2:
            if self.grid().frac_num == self.special_fracture:
                kxx = self.kf_high * np.ones(self.grid().num_cells)
            else:
                kxx = self.kf_low * np.ones(self.grid().num_cells)
            return tensor.SecondOrderTensor(self.grid().dim, kxx=kxx, kyy=kxx, kzz=1)

        else:  # g.dim == 1
            neigh = self.gb.node_neighbors(self.grid(), only_higher=True)
            frac_num = np.array([gh.frac_num for gh in neigh])
            if np.any(frac_num == self.special_fracture):
                if np.any(frac_num == 1):
                    kxx = self.kf_high * np.ones(self.grid().num_cells)
                else:
                    kxx = self.kf_low * np.ones(self.grid().num_cells)
            else:
                kxx = self.kf_low * np.ones(self.grid().num_cells)
            return tensor.SecondOrderTensor(self.grid().dim, kxx=kxx, kyy=1, kzz=1)


# ------------------------------------------------------------------------------#


class TPFAModelData(DarcyModelData):
    def __init__(self, g, data, **kwargs):
        DarcyModelData.__init__(self, g, data, **kwargs)

    def permeability(self):
        if self.grid().dim == 3:
            kxx = self.km * np.ones(self.grid().num_cells)
            return tensor.SecondOrderTensor(3, kxx)

        elif self.grid().dim == 2:
            if self.grid().frac_num == self.special_fracture:
                kxx = self.kf_high * np.ones(self.grid().num_cells)
            else:
                kxx = self.kf_low * np.ones(self.grid().num_cells)
            return tensor.SecondOrderTensor(3, kxx)

        else:  # g.dim == 1
            neigh = self.gb.node_neighbors(self.grid(), only_higher=True)
            frac_num = np.array([gh.frac_num for gh in neigh])
            if np.any(frac_num == self.special_fracture):
                if np.any(frac_num == 1):
                    kxx = self.kf_high * np.ones(self.grid().num_cells)
                else:
                    kxx = self.kf_low * np.ones(self.grid().num_cells)
            else:
                kxx = self.kf_low * np.ones(self.grid().num_cells)
            return tensor.SecondOrderTensor(3, kxx)


# ------------------------------------------------------------------------------#
