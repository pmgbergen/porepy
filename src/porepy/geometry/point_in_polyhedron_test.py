
import numpy as np

class PointInPolyhedronTest(object):

    def __init__(self, vertices, connectivity, tol):
        self.vertices = vertices
        self.connectivity = connectivity
        self.dim = vertices.shape[1]
        self.tol = tol

    def solid_angle(self, R):
        # https://dl.acm.org/doi/pdf/10.1145/2461912.2461916
        r = np.linalg.norm(R, axis=1)

        check_overlapping_vertices = np.any(np.absolute(r) < self.tol)
        if check_overlapping_vertices:
            raise ValueError("Test point coincides with a vertex")

        r_area = 0.5 * np.array(
            [np.linalg.norm(np.cross(R[0], R[1])), np.linalg.norm(np.cross(R[1], R[2])),
             np.linalg.norm(np.cross(R[2], R[0]))])
        check_collinearity = np.any(np.absolute(r_area) < self.tol)
        if check_collinearity:
            raise ValueError("Test point is collinear with the vertices")

        r_volume = np.linalg.norm(np.dot(R[1], np.cross(R[0] - R[1], R[2] - R[1])))
        check_coplanarity = np.any(np.absolute(r_volume) < self.tol)
        if check_coplanarity:
            raise ValueError("Test point is coplanar with the vertices")

        nv = np.dot(R[0], np.cross(R[1], R[2]))
        dv = np.prod(r) + np.dot(R[0], R[1]) * r[2] + np.dot(R[0], R[2]) * r[
            1] + np.dot(
            R[1], R[2]) * r[0]
        angle = 2.0 * np.arctan2(nv, dv)
        return angle

    def winding_number(self, point):
        R = self.vertices - point
        angles = np.array([self.solid_angle(R[indexes]) for indexes in self.connectivity])
        wn = np.sum(angles) / (4.0 * np.pi)
        return wn
