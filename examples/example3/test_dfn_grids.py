import numpy as np

from porepy.fracs.fractures import Fracture, FractureNetwork
from porepy.fracs import meshing


def test_conforming_two_fractures():
    f_1 = Fracture(np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]).T)
    f_2 = Fracture(np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]]).T)
    network = FractureNetwork([f_1, f_2])
    gb = meshing.dfn(network, conforming=True)


def test_non_conforming_two_fractures():
    f_1 = Fracture(np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]).T)
    f_2 = Fracture(np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]]).T)
    network = FractureNetwork([f_1, f_2])
    gb = meshing.dfn(network, conforming=False)


def test_conforming_three_fractures():
    f_1 = Fracture(np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]).T)
    f_2 = Fracture(np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]]).T)
    f_3 = Fracture(np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]]).T)
    network = FractureNetwork([f_1, f_2, f_3])
    gb = meshing.dfn(network, conforming=True)


def test_non_conforming_three_fractures():
    f_1 = Fracture(np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]).T)
    f_2 = Fracture(np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]]).T)
    f_3 = Fracture(np.array([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1]]).T)
    network = FractureNetwork([f_1, f_2, f_3])
    gb = meshing.dfn(network, conforming=False)
