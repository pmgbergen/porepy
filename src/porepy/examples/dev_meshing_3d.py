import numpy as np
import porepy as pp
from itertools import combinations

import gmsh

if False:
    gmsh.initialize()

    # An a unit square to gmsh
    # f_1 = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    # Add an elliptical fracture with center at (0, 0, 1), semi-axes 0.5 and 0.2 in the xy-plane
    f_2 = gmsh.model.occ.add_disk(0, 0, 1, 0.5, 0.2)

    # # Add a pentagon fracture at height z=2
    # p_1 = gmsh.model.occ.addPoint(-0.5, 0, 2)
    # p_2 = gmsh.model.occ.addPoint(0, 0.5, 2)
    # p_3 = gmsh.model.occ.addPoint(0.5, 0, 2)
    # p_4 = gmsh.model.occ.addPoint(0.3, -0.5, 2)
    # p_5 = gmsh.model.occ.addPoint(-0.3, -0.5, 2)
    # l_1 = gmsh.model.occ.addLine(p_1, p_2)
    # l_2 = gmsh.model.occ.addLine(p_2, p_3)
    # l_3 = gmsh.model.occ.addLine(p_3, p_4)
    # l_4 = gmsh.model.occ.addLine(p_4, p_5)
    # l_5 = gmsh.model.occ.addLine(p_5, p_1)
    # curve_loop = gmsh.model.occ.addCurveLoop([l_1, l_2, l_3, l_4, l_5])
    # f_3 = gmsh.model.occ.addPlaneSurface([curve_loop])

    p_6 = gmsh.model.occ.addPoint(0, -0.3, 0.2)
    p_7 = gmsh.model.occ.addPoint(0, 0.2, 0.2)
    p_8 = gmsh.model.occ.addPoint(0, 0.2, 1.7)
    p_9 = gmsh.model.occ.addPoint(0, -0.3, 1.7)
    l_6 = gmsh.model.occ.addLine(p_6, p_7)
    l_7 = gmsh.model.occ.addLine(p_7, p_8)
    l_8 = gmsh.model.occ.addLine(p_8, p_9)
    l_9 = gmsh.model.occ.addLine(p_9, p_6)
    curve_loop_2 = gmsh.model.occ.addCurveLoop([l_6, l_7, l_8, l_9])
    f_4 = gmsh.model.occ.addPlaneSurface([curve_loop_2])

    gmsh.model.occ.synchronize()

    # fractures = [f_1, f_2, f_3, f_4]
    fractures = [f_2, f_4]

    isect_info = gmsh.model.occ.fragment([(2, f) for f in fractures], [])

    gmsh.model.occ.synchronize()
    # gmsh.model.mesh.generate(2)

    # bnds = []
    # normals = []
    # for f in fractures:
    #     b = gmsh.model.get_parametrization_bounds(2, f)
    #     bnds.append(b)
    #     midpoint = gmsh.model.get_value(
    #         2, f, [(b[0][0] + b[0][1]) / 2, (b[1][0] + b[1][1]) / 2]
    #     )
    #     n = gmsh.model.get_normal(f, [(b[0][0] + b[0][1]) / 2, (b[1][0] + b[1][1]) / 2])
    #     normals.append(n)

    # # The angle of the normal vectors can now be computed by the cosine formula.
    # angle = np.arccos(
    #     np.dot(normals[0], normals[1])
    #     / (np.linalg.norm(normals[0]) * np.linalg.norm(normals[1]))
    # )
    # if angle > np.pi / 2:
    #     angle = np.pi - angle

    THRESHOLD = 0.4

    fracture_tags = gmsh.model.get_entities(2)

    points_to_add = {}

    fracture_diameters = {}
    for f in fracture_tags:
        mass = gmsh.model.occ.get_mass(*f)
        fracture_diameters[f] = np.sqrt(mass)

    for f1, f2 in combinations(fracture_tags, 2):
        dist = gmsh.model.occ.get_distance(2, f1[1], 2, f2[1])[0]
        if dist > THRESHOLD:
            continue

        elif dist > 1e-8:
            # Fractures are close, but not intersecting.
            for main_frac in [f1, f2]:
                if main_frac not in points_to_add:
                    points_to_add[main_frac] = []

                num_pts = np.ceil(fracture_diameters[main_frac] / dist).astype(int)
                bnds = gmsh.model.get_parametrization_bounds(*main_frac)
                u = np.linspace(bnds[0][0], bnds[1][0], num_pts)
                v = np.linspace(bnds[0][1], bnds[1][1], num_pts)
                grid = np.array(np.meshgrid(u, v)).T.reshape(-1, 2)

                other_frac = f2 if main_frac[1] == f1[1] else f1

                for pt in grid:
                    pt_phys = gmsh.model.get_value(2, main_frac[1], pt)
                    pi = gmsh.model.occ.add_point(pt_phys[0], pt_phys[1], pt_phys[2])
                    d_to_other = gmsh.model.occ.get_distance(0, pi, 2, other_frac[1])[0]
                    if d_to_other < THRESHOLD:
                        points_to_add[main_frac].append((pi, d_to_other))
                    else:
                        gmsh.model.occ.remove([(0, pi)])

        else:
            debug = []
            pass

    f1 = f_2
    f2 = f_4

    intersections = gmsh.model.occ.intersect([(2, f1)], [(2, f2)])
else:
    f_0 = pp.PlaneFracture(np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]).T)
    f_1 = pp.PlaneFracture(
        np.array([[0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5]]).T
    )
    f_2 = pp.PlaneFracture(
        np.array([[0.2, -0.3, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 1], [0.2, -0.3, 1]]).T
    )
    fractures = [f_0]

    domain = pp.Domain(
        {
            "xmin": -2.5,
            "xmax": 3.5,
            "ymin": -2.5,
            "ymax": 3.5,
            "zmin": -2.5,
            "zmax": 3.5,
        }
    )
    network = pp.create_fracture_network(fractures, domain)

    mdg = network.mesh(mesh_args={"mesh_size_frac": 0.05, "mesh_size_bound": 0.1})

    fn = "dev_meshing_3d"

    exp = pp.Exporter(mdg, fn)
    exp.write_vtu()

    import zipfile

    # Identify all files called fn.*.
    import glob

    names = glob.glob(f"{fn}*")

    with zipfile.ZipFile("export.zip", "w") as zf:
        for name in names:
            zf.write(name)

    # Delete the individual files.
    import os

    for name in names:
        os.remove(name)

    print(mdg)

    # Package all files called fn.* into a zip file.

    debug = []
