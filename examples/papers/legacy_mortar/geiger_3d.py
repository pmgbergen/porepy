import numpy as np

from porepy.fracs import importer, mortars, meshing
from porepy.grids.simplex import StructuredTriangleGrid
from porepy.grids.grid import FaceTag
from porepy.numerics import elliptic


test_case = 1
assert test_case == 1 or test_case == 2


# geometrical tolerance
tol = 1e-8
VEM = True

# first line in the file is the domain boundary, the others the fractures
file_dfm = "geiger_3d.csv"

# import the dfm and generate the grids
gb, domain = importer.dfm_3d_from_csv(file_dfm, tol, h_ideal=0.2, h_min=0.2)
gb.compute_geometry()

for _, d in gb.edges_props():
    mg = d["mortar_grid"]
    # print(mg.high_to_mortar.shape[0])

dom_min = 0
dom_max = 1

if True:
    frac_list, network, domain = importer.network_3d_from_csv("geiger_3d.csv")
    # Conforming=False would have been cool here, but that does not split the 1d grids
    gb_new = meshing.dfn(frac_list, conforming=True, h_ideal=0.07, h_min=0.05)

    gmap = {}
    for go in gb.grids_of_dimension(2):
        for gn in gb_new.grids_of_dimension(2):
            if gn.frac_num == go.frac_num:
                gmap[go] = gn
                xf = gn.face_centers
                hit = np.logical_or(
                    np.any(np.abs(xf - dom_min) < tol, axis=0),
                    np.any(np.abs(xf - dom_max) < tol, axis=0),
                )
                domain_tag = FaceTag.DOMAIN_BOUNDARY
                gn.remove_face_tag_if_tag(domain_tag, domain_tag)
                gn.add_face_tag(np.where(hit)[0], domain_tag)

    mortars.replace_grids_in_bucket(gb, gmap)

if VEM:
    internal_flag = FaceTag.FRACTURE
    [g.remove_face_tag_if_tag(FaceTag.BOUNDARY, internal_flag) for g, _ in gb]

if True:
    # select the permeability depending on the selected test case
    if test_case == 1:
        kf = 1e4
    else:
        kf = 1e-4
    data_problem = {
        "domain": domain,
        "tol": tol,
        "aperture": 1e-4,
        "km_low": 1,
        "km": 1,
        "kf": kf,
    }

    from geiger_3d_data import DarcyModelData

    gb.add_node_props(["is_tangential", "problem", "frac_num", "low_zones"])
    for g, d in gb:
        d["problem"] = DarcyModelData(g, d, **data_problem)
        d["is_tangential"] = True
        d["low_zones"] = d["problem"].low_zones()

        if g.dim == 2:
            d["frac_num"] = g.frac_num * np.ones(g.num_cells)
        else:
            d["frac_num"] = -1 * np.ones(g.num_cells)

    # Assign coupling permeability, the aperture is read from the lower dimensional grid
    gb.add_edge_prop("kn")
    for e, d in gb.edges_props():
        mg = d["mortar_grid"]
        g_l = gb.sorted_nodes_of_edge(e)[0]
        aperture = gb.node_prop(g_l, "param").get_aperture()
        d["kn"] = data_problem["kf"] / (mg.low_to_mortar_int * aperture)

    # Create the problem and solve it. Export the pressure and projected velocity for visualization.

    if VEM:
        problem = elliptic.DualEllipticModel(gb)
        up = problem.solve()
        problem.split(gb)
        problem.pressure()
        problem.save("pressure")
    else:
        problem = elliptic.EllipticModel(gb)
        problem.solve()
        problem.split()

        problem.pressure("pressure")
        problem.save(["pressure", "frac_num", "low_zones"])
#    problem.discharge('discharge')
#    problem.project_discharge('P0u')
#    problem.save(['pressure', 'P0u', 'frac_num', 'low_zones'])


# for g in gb.grids_of_dimension(2):
#
#    mx = g.nodes.max(axis=1)
#    mi = g.nodes.min(axis=1)
#    dx = mx - mi
#
#    if dx.max() > 0.6:
#        N = [40, 40]
#    elif dx.max() > 0.3:
#        N = [20, 20]
#    elif dx.max() > 0.2:
#        N = [10, 10]
#    else:
#        N = [5, 5]
#
#
#    active = np.where(dx > 1e-4)[0]
#    passive = np.setdiff1d(np.arange(3), active)
#    gn = StructuredTriangleGrid(N, physdims=dx[active])
#    tmp = gn.nodes.copy()
#    gn.nodes[active] = tmp[:2] + mi[active].reshape((2, 1))
#    gn.nodes[passive] = tmp[2] + mi[passive]
#    gn.compute_geometry()
#    gn.frac_num = g.frac_num
#    gmap[g] = gn
