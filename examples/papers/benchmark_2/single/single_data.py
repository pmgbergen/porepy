import numpy as np

from porepy.params import tensor, units
from porepy.fracs import importer
from porepy.fracs.fractures import Fracture, FractureNetwork
from porepy.params.bc import BoundaryCondition
from porepy.viz.exporter import Exporter
from porepy.numerics import elliptic, parabolic

#------------------------------------------------------------------------------#

def import_grid(file_geo, tol):

    frac = Fracture(np.array([[0, 10, 10,  0],
                              [0,  0, 10, 10],
                              [8,  2,  2,  8]])*10)
    network = FractureNetwork([frac], tol=tol)

    domain = {'xmin': 0, 'xmax': 100,
              'ymin': 0, 'ymax': 100,
              'zmin': 0, 'zmax': 100}
    network.impose_external_boundary(domain)
    network.find_intersections()
    network.split_intersections()
    network.to_gmsh('dummy.geo')

    gb = importer.dfm_from_gmsh(file_geo, 3, network)
    gb.compute_geometry()

    return gb, domain

#------------------------------------------------------------------------------#

class DarcyModelData(elliptic.EllipticDataAssigner):
    def __init__(self, g, data, **kwargs):
        self.domain = kwargs['domain']
        self.tol = kwargs['tol']

        self.apert = kwargs['aperture']
        self.km_high = kwargs['km_high']
        self.km_low = kwargs['km_low']
        self.kf = kwargs['kf']
        self.max_dim = kwargs.get('max_dim', 3)

        # define two pieces of the boundary, useful to impose boundary conditions
        self.bc_top = np.empty(0)
        self.bc_bottom = np.empty(0)

        b_faces = g.tags['domain_boundary_faces'].nonzero()[0]
        if b_faces.size > 0:
            b_face_centers = g.face_centers[:, b_faces]

            self.bc_top = np.logical_and(b_face_centers[0, :] < 0 + self.tol,
                                         b_face_centers[2, :] > 90 - self.tol)

            self.bc_bottom = np.logical_and(b_face_centers[1, :] < 0 + self.tol,
                                            b_face_centers[2, :] < 10 + self.tol)

        elliptic.EllipticDataAssigner.__init__(self, g, data)

    def aperture(self):
        return np.power(self.apert, self.max_dim-self.grid().dim)

    def bc(self):

        b_faces = self.grid().tags['domain_boundary_faces'].nonzero()[0]
        if b_faces.size == 0:
            return BoundaryCondition(self.grid(), np.empty(0), np.empty(0))

        labels = np.array(['neu'] * b_faces.size)
        labels[self.bc_top] = 'dir'
        labels[self.bc_bottom] = 'dir'
        return BoundaryCondition(self.grid(), b_faces, labels)

    def bc_val(self):
        bc_val = np.zeros(self.grid().num_faces)
        b_faces = self.grid().tags['domain_boundary_faces'].nonzero()[0]
        if b_faces.size > 0:
            bc_val[b_faces[self.bc_top]] = 4 * units.BAR
            bc_val[b_faces[self.bc_bottom]] = 1 * units.BAR

        return bc_val

    def low_zones(self):
        return self.grid().cell_centers[2, :] < 10

    def permeability(self):
        dim = self.grid().dim
        if dim == 3:
            kxx = self.km_high * np.ones(self.grid().num_cells)
            kxx[self.low_zones()] = self.km_low
            kxx /= 1e-3 # viscosity
            return tensor.SecondOrder(dim, kxx=kxx, kyy=kxx, kzz=kxx)
        else:
            kxx = self.kf * np.ones(self.grid().num_cells)
            kxx /= 1e-3 # viscosity
            return tensor.SecondOrder(dim, kxx=kxx, kyy=kxx, kzz=1)

#------------------------------------------------------------------------------#

class AdvectiveProblem(parabolic.ParabolicModel):

    def space_disc(self):
        return self.source_disc(), self.advective_disc()

#------------------------------------------------------------------------------#

class AdvectiveDataAssigner(parabolic.ParabolicDataAssigner):

    def __init__(self, g, data, physics='transport', **kwargs):
        self.domain = kwargs['domain']
        self.tol = kwargs['tol']
        self.max_dim = kwargs.get('max_dim', 3)

        self.phi_high = kwargs['phi_high']
        self.phi_low = kwargs['phi_low']
        self.phi_f = kwargs['phi_f']

        # define two pieces of the boundary, useful to impose boundary conditions
        self.inflow = np.empty(0)

        b_faces = g.tags['domain_boundary_faces'].nonzero()[0]
        if b_faces.size > 0:
            b_face_centers = g.face_centers[:, b_faces]

            self.inflow = np.logical_and(b_face_centers[0, :] < 0 + self.tol,
                                         b_face_centers[2, :] > 90 - self.tol)

        parabolic.ParabolicDataAssigner.__init__(self, g, data, physics)

    def low_zones(self):
        return self.grid().cell_centers[2, :] < 10

    def porosity(self):
        if self.grid().dim == 3:
            phi = self.phi_high * np.ones(self.grid().num_cells)
            phi[self.low_zones()] = self.phi_low
        else:
            phi = self.phi_f * np.ones(self.grid().num_cells)
        return phi

    def bc(self):
        b_faces = self.grid().tags['domain_boundary_faces'].nonzero()[0]
        if b_faces.size == 0:
            return BoundaryCondition(self.grid(), np.empty(0), np.empty(0))
        return BoundaryCondition(self.grid(), b_faces, 'dir')

    def bc_val(self, _):
        bc_val = np.zeros(self.grid().num_faces)
        b_faces = self.grid().tags['domain_boundary_faces'].nonzero()[0]
        if b_faces.size > 0:
            bc_val[b_faces[self.inflow]] = 0.01
        return  bc_val

#------------------------------------------------------------------------------#
