"""Test functionality related to ellipse_fracture module."""

import numpy as np
import pytest

from porepy.fracs import ellipse_fracture


@pytest.mark.parametrize(
    "ellipse_fracture_params",
    [
        (np.array([3.0, 4.0, 5.0]), 2.0, 1.0, np.pi / 6.0, np.pi / 4.0, np.pi / 8.0),
        (np.array([8.0, 7.0, 6.0]), 2.5, 0.5, np.pi / 6.0, np.pi / 4.0, np.pi / 8.0),
    ],
)
def test_ellipse_fracture_center(ellipse_fracture_params):
    # parse the parameters
    center, major_axis, minor_axis, major_axis_angle, strike_angle, dip_angle = (
        ellipse_fracture_params
    )

    center_known = ellipse_fracture_params[0]
    fracture = ellipse_fracture.EllipticFracture(
        center, major_axis, minor_axis, major_axis_angle, strike_angle, dip_angle
    )
    assert np.allclose(center_known, fracture.center)
