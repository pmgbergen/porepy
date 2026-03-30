import numpy as np


def distance_from_points_to_fracture_plane(
    points_xyz: np.ndarray, center: np.ndarray, strike_angle: float, dip_angle: float
) -> np.ndarray:
    """Calculate the distance from a set of points to a plane defined by its center and
    normal vector.

    The normal vector is calculated from the strike and dip angles, following the
    right-hand rule.

    Parameters:
        points_xyz: An array of shape (N, 3) containing the coordinates of the points.
        center: A 1D array of length 3 containing the coordinates of the center of the
            plane.
        strike_angle: The strike angle of the plane in radians.
        dip_angle: The dip angle of the plane in radians.

    """
    P = np.asarray(points_xyz)
    c = np.asarray(center).ravel()
    phi = float(strike_angle)
    theta = float(dip_angle)

    n = np.array(
        [np.sin(theta) * np.sin(phi), -np.sin(theta) * np.cos(phi), np.cos(theta)],
    )
    n /= np.linalg.norm(n)

    dis_error = (P - c) @ n
    return dis_error
