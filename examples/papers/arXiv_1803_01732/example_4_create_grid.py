import numpy as np

from porepy.fracs import importer
from porepy.fracs import extrusion
from porepy.fracs.fractures import FractureNetwork
from porepy.fracs import meshing
import porepy.utils.comp_geom as cg

# ------------------------------------------------------------------------------#


def create(conforming, tol=1e-4):

    csv_folder = "./"
    csv_file = "example_4_outcrop.csv"
    pts, edges = importer.lines_from_csv(csv_folder + csv_file)

    # A tolerance of 1 seems to be sufficient to recover the T-intersections, but
    # I might have missed some, though, so take a look at the network and modify
    # if necessary.
    snap_pts = cg.snap_points_to_segments(pts, edges, tol=1)

    extrusion_kwargs = {}
    extrusion_kwargs["tol"] = tol
    extrusion_kwargs["exposure_angle"] = np.pi / 4. * np.ones(edges.shape[1])
    # Added an option not to include the points on the exposed surface. This
    # reduces cell refinement somewhat, but setting it True should also be okay
    extrusion_kwargs["outcrop_consistent"] = True

    fractures = extrusion.fractures_from_outcrop(snap_pts, edges, **extrusion_kwargs)
    network = FractureNetwork(fractures, tol=tol)
    # network.to_vtk(folder_export+"network.vtu")
    bounding_box = {
        "xmin": -800,
        "xmax": 600,
        "ymin": 100,
        "ymax": 1500,
        "zmin": -100,
        "zmax": 1000,
    }
    network.impose_external_boundary(
        bounding_box, truncate_fractures=True, keep_box=False
    )

    mesh_kwargs = {}
    h = 30
    mesh_kwargs["mesh_size"] = {
        "mode": "weighted",  # 'distance'
        "value": h,
        "bound_value": h,
        "tol": tol,
    }

    if conforming:
        # Change h_ideal and h_min at will here, but I ran into trouble with h_min < 1.
        gb = meshing.dfn(network, conforming=True, h_ideal=100, h_min=5)
    else:
        # Switch conforming=True to get conforming meshes
        gb = meshing.dfn(network, conforming=False, **mesh_kwargs, keep_geo=True)

    gb.remove_nodes(lambda g: g.dim == 0)
    gb.compute_geometry()
    gb.assign_node_ordering()

    return gb


# ------------------------------------------------------------------------------#
