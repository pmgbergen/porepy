import numpy as np

from porepy.numerics import parabolic
from porepy.numerics import time_stepper

from porepy.params.bc import BoundaryCondition


class AdvectiveModel(parabolic.ParabolicModel):
    """
    Inherits from ParabolicProblem
    This class solves equations of the type:
    phi *c_p dp/dt  - \nabla K \nabla p = q
    Init:
    - gb (Grid/GridBucket) Grid or grid bucket for the problem
    - physics (string) Physics key word. See Parameters class for valid physics
    functions:
    discharge(): computes the discharges and saves it in the grid bucket as 'pressure'
    Also see functions from ParabolicProblem
    Example:
    # We create a problem with standard data
    gb = meshing.cart_grid([], [10,10], physdims=[1,1])
    for g, d in gb:
        d['problem'] = SlightlyCompressibleData(g, d)
    problem = SlightlyCompressible(gb)
    problem.solve()
   """

    def __init__(self, gb, **kwargs):
        parabolic.ParabolicModel.__init__(self, gb, **kwargs)

    def space_disc(self):
        return self.advective_disc(), self.source_disc()

    def time_step(self):
        return 0.01

    def end_time(self):
        return 3


class AdvectiveModelData(parabolic.ParabolicDataAssigner):
    def __init__(self, g, data, domain, tol):
        self.domain = domain
        self.tol = tol
        parabolic.ParabolicDataAssigner.__init__(self, g, data)

    def bc_val(self, t):

        bc_val = np.zeros(self.grid().num_faces)
        bound_faces = self.grid().tags["domain_boundary_faces"].nonzero()[0]
        if bound_faces.size == 0:
            return bc_val

        bound_face_centers = self.grid().face_centers[:, bound_faces]
        bottom = bound_face_centers[2, :] < self.domain["zmin"] + self.tol
        bc_val[bound_faces[bottom]] = 1

        return bc_val

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
