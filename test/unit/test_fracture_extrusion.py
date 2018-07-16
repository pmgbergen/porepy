import unittest
import numpy as np

from porepy.fracs.fractures import Fracture
from porepy.fracs import extrusion


class TestFractureExtrusion(unittest.TestCase):
    def test_single_t_intersection(self):
        # Three fractures meeting in a T, same family tag for two of them
        edges = np.array([[0, 1, 0], [1, 2, 0], [1, 3, 1]]).T
        abut, prim, sec, other = extrusion.t_intersections(edges)

        known_abut = [1]
        known_prim = [0]
        known_sec = [1]
        known_other = [3]

        assert np.all(known_abut == abut)
        assert np.all(known_prim == prim)
        assert np.all(known_sec == sec)
        assert np.all(known_other == other)

    def test_single_t_intersection_different_family(self):
        # Three fractures meeting in a T. Different family on all, so no
        # intersections found
        edges = np.array([[0, 1, 0], [1, 2, 1], [1, 3, 2]]).T
        abut, prim, sec, other = extrusion.t_intersections(edges)

        assert abut.size == 0
        assert prim.size == 0
        assert sec.size == 0
        assert other.size == 0

    def test_single_t_intersection_disregard_family_tag(self):
        # Three fractures meeting in a T. Different family on all, but we
        # disregard family tags, so a T-intersection is found
        edges = np.array([[0, 1, 0], [1, 2, 1], [1, 3, 2]]).T
        abut, prim, sec, other = extrusion.t_intersections(
            edges, remove_three_families=True
        )
        known_abut = [1]
        known_prim = [0]
        known_sec = [1]
        known_other = [3]

        assert np.all(known_abut == abut)
        assert np.all(known_prim == prim)
        assert np.all(known_sec == sec)
        assert np.all(known_other == other)

    def test_disc_radius_center(self):

        p0 = np.array([[0], [0]])
        p1 = np.array([[0], [1]])
        length = np.array([1])
        theta = np.array([np.pi / 2])

        known_cc = np.array([[0], [0.5], [0]])

        rad, cc, _ = extrusion.disc_radius_center(length, p0, p1, theta)
        assert rad[0] == 0.5
        assert np.allclose(cc, known_cc)

    def compare_arrays(self, a, b):
        assert a.shape == b.shape

        for i in range(a.shape[1]):
            ai = a[:, i].reshape((3, 1))
            assert np.min(np.sum((b - ai) ** 2, axis=0)) < 1e-5
            bi = b[:, i].reshape((3, 1))
            assert np.min(np.sum((a - bi) ** 2, axis=0)) < 1e-5

    def test_fracture_rotation_90_deg(self):

        p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f = Fracture(p, check_convexity=False)

        p_rot_1_known = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]).T
        exposure = np.array([[0], [0], [0]])
        extrusion.rotate_fracture(f, [1, 0, 0], np.pi / 2, exposure)
        self.compare_arrays(f.p, p_rot_1_known)

        p_rot_2_known = np.array(
            [[0, 0.5, -0.5], [1, 0.5, -0.5], [1, 0.5, 0.5], [0, 0.5, 0.5]]
        ).T
        exposure = np.array([[0], [0.5], [0]])
        f = Fracture(p, check_convexity=False)
        extrusion.rotate_fracture(f, [1, 0, 0], np.pi / 2, exposure)
        self.compare_arrays(f.p, p_rot_2_known)

    def test_cut_fracture_no_intersection(self):
        p1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T
        f1 = Fracture(p1, check_convexity=False)

        p2 = np.array(
            [[0.5, -0.5, 0.2], [0.5, 0.5, 0.2], [0.5, 0.5, 0.8], [0.5, -0.5, 0.8]]
        ).T
        p_known = p2.copy()
        f2 = Fracture(p2, check_convexity=False)

        ref_pt = np.array([[0], [1], [0]])
        extrusion.cut_fracture_by_plane(f2, f1, ref_pt)

        self.compare_arrays(p_known, f2.p)

    def test_cut_fracture_simple_intersection(self):
        p1 = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]).T
        f1 = Fracture(p1, check_convexity=False)

        p2 = np.array(
            [[0.5, -0.5, 0.2], [0.5, 0.5, 0.2], [0.5, 0.5, 0.8], [0.5, -0.5, 0.8]]
        ).T
        p_known = np.array(
            [[0.5, 0, 0.2], [0.5, 0.5, 0.2], [0.5, 0.5, 0.8], [0.5, 0, 0.8]]
        ).T
        f2 = Fracture(p2, check_convexity=False)

        ref_pt = np.array([[0], [1], [0]])
        extrusion.cut_fracture_by_plane(f2, f1, ref_pt)

        self.compare_arrays(p_known, f2.p)

    def test_cut_fracture_one_inclined(self):
        p1 = np.array([[0, 1, -0.5], [1, 1, -0.5], [1, -1, 1.5], [0, -1, 1.5]]).T
        f1 = Fracture(p1, check_convexity=False)

        p2 = np.array(
            [[0.5, -0.5, 0.2], [0.5, 0.5, 0.2], [0.5, 0.5, 0.8], [0.5, -1.5, 0.8]]
        ).T
        p_known = np.array(
            [[0.5, 0.3, 0.2], [0.5, 0.5, 0.2], [0.5, 0.5, 0.8], [0.5, -0.3, 0.8]]
        ).T
        f2 = Fracture(p2, check_convexity=False)

        ref_pt = np.array([[0], [1], [0]])
        extrusion.cut_fracture_by_plane(f2, f1, ref_pt)

        self.compare_arrays(p_known, f2.p)

    def test_rotation(self):
        # Test a case with a fracture terminated in both ends (H-configuration)
        # This turned out to be problematic for certain cases.
        # The test is simply that it runs - should have something more
        # insightful here
        pt = np.array(
            [
                [0.4587156, 0.76605504, 0.60586278, 0.74934585, 0.46570401, 0.55721055],
                [0.28593274, 0.35321103, 0.31814406, 0.028759, 0.44178218, 0.30749384],
            ]
        )
        edges = np.array([[0, 1], [2, 3], [4, 5]]).T

        np.random.seed(7)
        family = np.zeros(3, dtype=np.int)
        extrusion.fractures_from_outcrop(
            pt, edges, family=family, family_std_incline=[0.3]
        )

    if __name__ == "__main__":
        unittest.main()
