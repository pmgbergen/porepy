"""
Create a grid based on the Soultz data set.
"""
import numpy as np

from porepy.fracs.fractures import Fracture, EllipticFracture, FractureNetwork
from porepy.fracs import meshing, simplex, split_grid


def create_grid(**mesh_kwargs):
    """
    Create a grid bucket containing grids from gmsh.

    NOTE: The line setting 'path_to_gmsh' *must* be modified for this to work.

    Parameters concerning mesh size, domain size etc. may also be changed, see
    below.

    Returns:
        grid_bucket: A grid_bucket containing the full hierarchy of grids.

    """
    num_fracs = mesh_kwargs.get("num_fracs", 39)

    # If the
    # Don't change the path, or move the file
    data = _soultz_data()
    data = data[:num_fracs, :]

    # Data format of the data file is (frac_num, fracture center_xyz, major
    # axis, minor axis, dip direction, dip angle)
    centers = data[:, 1:4]
    major_axis = data[:, 4]
    minor_axis = data[:, 5]

    dip_direction = data[:, 6] / 180 * np.pi
    dip_angle = data[:, 7] / 180 * np.pi

    # We will define the fractures as elliptic fractures. For this we need
    # strike angle, rather than dip direction.
    strike_angle = dip_direction + np.pi / 2

    # Modifications of the fracture definition:
    # These are carried out to ease the gridding; without these, we will end up
    # with gridding polygons that have very close points. The result may be

    # Minor axis angle. This is specified as zero (the fractures are
    # interpreted as circles), but we rotate them in an attempt to avoid close
    # points in the fracture specification.
    # Also note that since the fractures are defined as circles, any
    # orientation of the approximating polygon is equally correct
    major_axis_angle = np.zeros(num_fracs)
    #   major_axis_angle[14] = 5 * np.pi / 180
    ##    major_axis_angle[24] = 5 * np.pi / 180
    #    major_axis_angle[26] = 5 * np.pi / 180
    ##    major_axis_angle[32-1] = 5 * np.pi / 180

    # Also modify some centers. This may potentially have some impact on the
    # properties of the fracture network, but they been selected as to not
    # modify the fracture network properties.
    #   centers[3, 2] += 30
    if num_fracs > 10:
        centers[11, 2] += 15

    #    centers[8, 2] -= 10
    #    centers[19, 2] -= 20
    #    centers[22, 2] -= 10
    #    centers[23, 1:3] -= 15
    #    centers[24, 2] += 30
    #    centers[25, 2] += 10
    #    centers[29, 2] -= 30
    #    centers[30, 2] += 30
    #    centers[31, 2] += 30
    #    centers[34, 2] -= 10
    #    centers[38, 2] -= 10

    # Create a set of fractures
    frac_list = []
    num_points = mesh_kwargs.get("num_points", 16)
    for fi in range(data.shape[0]):
        frac_new = EllipticFracture(
            centers[fi],
            major_axis[fi],
            minor_axis[fi],
            major_axis_angle[fi],
            strike_angle[fi],
            dip_angle[fi],
            num_points=num_points,
        )
        frac_list.append(frac_new)

    # Create the network, dump to vtu
    network = FractureNetwork(frac_list, verbose=1, tol=1e-4)
    network.to_vtk("soultz_fractures_full.vtu")

    # Impose domain boundaries. These are set large enough to not be in
    # conflict with the network.
    # This may be changed if desirable.
    domain = {
        "xmin": -4000,
        "xmax": 4000,
        "ymin": -3000,
        "ymax": 3000,
        "zmin": 0,
        "zmax": 8000,
    }
    domain = mesh_kwargs.get("domain", domain)
    network.impose_external_boundary(domain)

    # Find intersections, and split these
    network.find_intersections()
    network.split_intersections()

    # This may be changed, if desirable.
    if mesh_kwargs is None:
        mesh_kwargs = {
            "mesh_size_frac": 150,
            "mesh_size_bound": 500,
            "mesh_size_min": 10,
            "file_name": "soultz_fracs",
        }
    # Since we have a ready network (and may want to take this file into
    # jupyter and study the network before gridding), we abuse the workflow
    # slightly by calling simplex_tetrahedral directly, rather than to go the
    # way through meshing.simplex_grid (the latter is, for now, restricted to
    # specifying the grid by individual fractures, rather than networks).
    grids = simplex.tetrahedral_grid(network=network, **mesh_kwargs)

    # Convert the grids into a bucket
    meshing.tag_faces(grids)
    gb = meshing.assemble_in_bucket(grids)
    gb.compute_geometry()

    split_grid.split_fractures(gb)
    return gb


def _soultz_data():
    """
    Hard coded data that  describes the fracture network.
    """
    data = np.array(
        [
            [
                1.00000000e+00,
                -4.26646154e+02,
                1.52215385e+02,  # 0
                1.01200000e+03,
                3.00000000e+02,
                3.00000000e+02,
                1.30000000e+02,
                7.90000000e+01,
            ],
            [
                2.00000000e+00,
                -4.22353846e+02,
                1.50784615e+02,  # 1
                1.19800000e+03,
                6.00000000e+02,
                6.00000000e+02,
                2.47000000e+02,
                7.40000000e+01,
            ],
            [
                3.00000000e+00,
                -3.81022727e+02,
                1.77284091e+02,  # 2
                1.64300000e+03,
                4.00000000e+02,
                4.00000000e+02,
                7.60000000e+01,
                5.80000000e+01,
            ],
            [
                4.00000000e+00,
                -3.20113636e+02,
                2.19920455e+02,  # 3
                2.17900000e+03,
                6.00000000e+02,
                6.00000000e+02,
                2.78000000e+02,
                5.30000000e+01,
            ],
            [
                5.00000000e+00,
                -4.35000000e+01,
                1.74000000e+01,  # 4
                1.01500000e+03,
                3.00000000e+02,
                3.00000000e+02,
                2.70000000e+02,
                4.50000000e+01,
            ],
            [
                6.00000000e+00,
                -5.22857143e+01,
                2.09142857e+01,  # 5
                1.22000000e+03,
                6.00000000e+02,
                6.00000000e+02,
                2.47000000e+02,
                7.40000000e+01,
            ],
            [
                7.00000000e+00,
                -7.80000000e+01,
                3.12000000e+01,  # 6
                1.82000000e+03,
                6.00000000e+02,
                6.00000000e+02,
                2.70000000e+01,
                4.70000000e+01,
            ],
            [
                8.00000000e+00,
                -1.20562500e+02,
                4.81312500e+01,  # 7
                2.81500000e+03,
                4.00000000e+02,
                4.00000000e+02,
                2.30000000e+02,
                7.00000000e+01,
            ],
            [
                9.00000000e+00,
                -1.35862500e+02,
                5.17012500e+01,  # 8
                3.22300000e+03,
                3.00000000e+02,
                3.00000000e+02,
                6.00000000e+01,
                7.50000000e+01,
            ],
            [
                1.00000000e+01,
                -1.45950000e+02,
                5.40550000e+01,  # 9
                3.49200000e+03,
                3.00000000e+02,
                3.00000000e+02,
                2.57000000e+02,
                6.30000000e+01,
            ],
            [
                1.10000000e+01,
                -1.22550000e+02,
                4.85950000e+01,  # 10
                2.86800000e+03,
                3.00000000e+02,
                3.00000000e+02,
                2.90000000e+02,
                7.00000000e+01,
            ],
            [
                1.20000000e+01,
                -4.96440625e+02,
                1.03075000e+02,  # 11
                2.12300000e+03,
                6.00000000e+02,
                6.00000000e+02,
                6.50000000e+01,
                7.00000000e+01,
            ],
            [
                1.30000000e+01,
                -5.20600000e+02,
                1.31800000e+02,  # 12
                3.24200000e+03,
                3.00000000e+02,
                3.00000000e+02,
                8.20000000e+01,
                6.90000000e+01,
            ],
            [
                1.40000000e+01,
                -5.22100000e+02,
                1.36300000e+02,  # 13
                3.34700000e+03,
                3.00000000e+02,
                3.00000000e+02,
                2.31000000e+02,
                8.40000000e+01,
            ],
            [
                1.50000000e+01,
                -5.24485714e+02,
                1.43457143e+02,  # 14
                3.51400000e+03,
                3.00000000e+02,
                3.00000000e+02,
                3.13000000e+02,
                5.70000000e+01,
            ],
            [
                1.60000000e+01,
                -5.30000000e+02,
                1.60000000e+02,  # 15
                3.90000000e+03,
                4.00000000e+02,
                4.00000000e+02,
                2.34000000e+02,
                6.40000000e+01,
            ],
            [
                1.70000000e+01,
                -3.76521739e+02,
                -4.26086957e+01,  # 16
                4.76000000e+03,
                4.00000000e+02,
                4.00000000e+02,
                2.50000000e+02,
                6.50000000e+01,
            ],
            [
                1.80000000e+01,
                -3.52028985e+02,
                -7.18115942e+01,  # 17
                4.89000000e+03,
                3.00000000e+02,
                3.00000000e+02,
                2.50000000e+02,
                6.50000000e+01,
            ],
            [
                1.90000000e+01,
                -3.20000000e+02,
                -1.10000000e+02,  # 18
                5.06000000e+03,
                4.00000000e+02,
                4.00000000e+02,
                2.50000000e+02,
                6.50000000e+01,
            ],
            [
                2.00000000e+01,
                -5.26853741e+02,
                4.29659864e+01,  # 19
                1.57900000e+03,
                3.00000000e+02,
                3.00000000e+02,
                6.90000000e+01,
                7.80000000e+01,
            ],
            [
                2.10000000e+01,
                -5.27840136e+02,
                4.45442177e+01,  # 20
                1.63700000e+03,
                3.00000000e+02,
                3.00000000e+02,
                4.60000000e+01,
                6.80000000e+01,
            ],
            [
                2.20000000e+01,
                -5.30952381e+02,
                4.95238095e+01,  # 21
                1.82000000e+03,
                3.00000000e+02,
                3.00000000e+02,
                4.60000000e+01,
                6.40000000e+01,
            ],
            [
                2.30000000e+01,
                -5.34727891e+02,
                5.55646258e+01,  # 22
                2.04200000e+03,
                3.00000000e+02,
                3.00000000e+02,
                7.20000000e+01,
                6.50000000e+01,
            ],
            [
                2.40000000e+01,
                -5.34795918e+02,
                5.56734694e+01,  # 23
                2.04600000e+03,
                3.00000000e+02,
                3.00000000e+02,
                2.43000000e+02,
                6.90000000e+01,
            ],
            [
                2.50000000e+01,
                -5.35578231e+02,
                5.69251701e+01,  # 24
                2.09200000e+03,
                3.00000000e+02,
                3.00000000e+02,
                9.10000000e+01,
                7.60000000e+01,
            ],
            [
                2.60000000e+01,
                -5.58720930e+02,
                8.08023256e+01,  # 25
                2.97000000e+03,
                4.00000000e+02,
                4.00000000e+02,
                7.70000000e+01,
                8.20000000e+01,
            ],
            [
                2.70000000e+01,
                -6.46220930e+02,
                8.88523256e+01,  # 26
                3.27100000e+03,
                4.00000000e+02,
                4.00000000e+02,
                3.45000000e+02,
                8.50000000e+01,
            ],
            [
                2.80000000e+01,
                -8.99343750e+02,
                1.12031250e+02,  # 27
                4.08900000e+03,
                3.00000000e+02,
                3.00000000e+02,
                2.53000000e+02,
                6.20000000e+01,
            ],
            [
                2.90000000e+01,
                -9.41176471e+02,
                1.14323529e+02,  # 28
                4.77500000e+03,
                3.00000000e+03,
                3.00000000e+03,
                2.34000000e+02,
                7.10000000e+01,
            ],
            [
                3.00000000e+01,
                -5.00000000e+02,
                6.08117647e+01,  # 29
                1.72300000e+03,
                3.00000000e+02,
                3.00000000e+02,
                2.16000000e+02,
                6.90000000e+01,
            ],
            [
                3.10000000e+01,
                -5.00000000e+02,
                6.35647059e+01,  # 30
                1.80100000e+03,
                3.00000000e+02,
                3.00000000e+02,
                2.60000000e+01,
                8.00000000e+01,
            ],
            [
                3.20000000e+01,
                -6.44324324e+02,
                1.11648649e+02,  # 31
                2.81700000e+03,
                3.00000000e+02,
                3.00000000e+02,
                2.42000000e+02,
                8.60000000e+01,
            ],
            [
                3.30000000e+01,
                -1.25135135e+03,
                2.02702703e+02,  # 32
                3.94000000e+03,
                3.00000000e+02,
                3.00000000e+02,
                2.50000000e+02,
                6.80000000e+01,
            ],
            [
                3.40000000e+01,
                -1.47891892e+03,
                2.36837838e+02,  # 33
                4.36100000e+03,
                4.00000000e+02,
                4.00000000e+02,
                2.80000000e+02,
                7.70000000e+01,
            ],
            [
                3.50000000e+01,
                -1.54400000e+03,
                2.58857143e+02,  # 34
                4.62000000e+03,
                6.00000000e+02,
                1.20000000e+03,
                2.85000000e+02,
                7.80000000e+01,
            ],
            [
                3.60000000e+01,
                -1.56240000e+03,
                2.66742857e+02,  # 35
                4.71200000e+03,
                4.00000000e+02,
                4.00000000e+02,
                2.12000000e+02,
                5.00000000e+01,
            ],
            [
                3.70000000e+01,
                -1.61460000e+03,
                2.89114286e+02,  # 36
                4.97300000e+03,
                3.00000000e+02,
                3.00000000e+02,
                2.76000000e+02,
                8.10000000e+01,
            ],
            [
                3.80000000e+01,
                -1.62240000e+03,
                2.92457143e+02,  # 37
                5.01200000e+03,
                3.00000000e+02,
                3.00000000e+02,
                2.57000000e+02,
                8.50000000e+01,
            ],
            [
                3.90000000e+01,
                -1.64000000e+03,
                3.00000000e+02,  # 38
                5.10000000e+03,
                3.00000000e+02,
                3.00000000e+02,
                2.55000000e+02,
                6.90000000e+01,
            ],
        ]
    )
    return data
