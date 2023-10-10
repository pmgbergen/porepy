"""Expected values for the tests are collected in this file.

For each test file, there should be a dictionary with the name of the test file. Inside
this dictionary, there should be a dictionary with the name of the test function/class,
containing the expected values for the test named according to what they are compared
to.

"""
import numpy as np


# test_mpfa.py
_grad_bound_known = np.array(
    [
        [0.10416667, 0.0, 0.0, 0.0, 0.0, -0.02083333, 0.0, 0.0],
        [0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.10416667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02083333],
        [0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.10416667, 0.0, 0.0, 0.02083333, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.10416667, 0.0, 0.0, 0.0, 0.0, -0.02083333],
        [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
        [-0.02083333, 0.0, 0.0, 0.0, 0.0, 0.10416667, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.02083333, 0.0, 0.0, 0.10416667, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
        [0.02083333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10416667],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0],
        [0.0, 0.0, -0.02083333, 0.0, 0.0, 0.0, 0.0, 0.10416667],
    ]
)
_grad_cell = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]
)
test_mpsa = {
    "MpsaReconstructBoundaryDisplacement": {
        "test_cart_2d": {
            "grad_bound_known": _grad_bound_known,
            "grad_cell_known": _grad_cell,
        },
    }
}
