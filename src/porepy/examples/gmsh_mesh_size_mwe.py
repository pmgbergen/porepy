"""EK's thinking of the different possible cases of local mesh size fields for
fractures.

A fracture can either intersect other fractures, or the boundary, or it is isolated.

Any fracture is divided into segments by its endpoints, and possibly also by
intersection points with other fractures (which will give a finer subdivision of the
fracture).

If a fracture is close to, but not intersecting, another fracture or the boundary, the
closest point on the fracture could be used as a center for a distance field to control
the mesh size locally around that point. For a long fracture, this would be preferrable
to using the whole fracture as a source for the distance field, since this would give a
small mesh size along the whole fracture, which is not desired. If this is done, the
closest point on the fracture will give a further subdivision of the fracture into
segments.

If two fractures are close, but not intersecting, the closest point on at least one of
the fractures is an endpoint of the fracture.

If a fracture is isolated and not close to any other fracture or the boundary, we can do
a gradual transition from the fracture mesh size to the domain mesh size away from the
fracture.

This suggests the following approach:
    1. Loop over all combinations of fractures and boundaries, excluding only the
       boundary-boundary combinations. For each pair, compute the distance. If this is
       larger than a certain threshold (domain size??), skip. Otherwise, for the closest
       points on each fracture, if this is not already an endpoint of the fracture (how
       can this be checked?), add it to the geometry as a point.

       Take note of the triplet (line, closest_point, distance) for each line in the
       pair (is this a dictionary line -> list of (point, distance)?). Since we keep
       track of the distance, it should not be necessary to also keep track of the
       line-line pair.

       Special points:
        a) For pairs of fractures that are close but not intersecting (think two almost
           parallel lines with endpoints that can be shifted, depending on which type of
           problem we want), it is necessary to take the distance from the endpoints of
           both lines to the other line, and possibly add these as well. If both
           endpoints are close, it means we need to have a fine mesh along the whole
           fracture (and the relevant segment of the other fracture). This requires
           special handling, TODO.

        b) For intersection points, we cannot yet set a distance to other features,
           since this should be the minimum of the distances from all other points
           associated with this fracture (i.e., endpoints and closest points to other
           fractures/boundary - Q: Do we need closest points or only intersection
           points). This means that we need to first collect all such points, and then
           in a second loop over all fractures, compute the distances from all such
           points to all other points associated with the fracture, and take the
           minimum. This will be used in step 3.

    2. For all fractures, or boundary lines, that have not been assigned any extra
       points (they were never found in the previous step), the local mesh size is the
       minimum of the fracture mesh size and the length of the fracture. For such
       fractures, we may want to use not a distance field with a threshold, but rather
       an Extend field, see Gmsh documentation (second thoughts: we probably need the
       distance field, since this is the way we can override any local mesh sizes set on
       close points in the next step - these will interpolate towards the domain mesh
       size). NOTE: We may need to do this for all fractures, not isolated ones.
    3. For fractures that were found in step 1, we want to set up distance+threshold
       fields that transition from a small mesh size at the closest point to the
       *domain* mesh size at some distance away. The simplest option would be if we can
       let this distance field go from a fraction of the distance (calculated in step 1)
       to the domain mesh size, with the latter set with the same dist_max as in step 2.
    An assumption here is that we can add points and distance fields around these with
    little cost to the meshing time (relatively speaking), and that the distance field
    will be applied for meshing in all dimensions (e.g., if a point is added on a
    fracture without it being identified with the fracture in any sense, we still see
    its effect in the mesh).

Bonus point:
    4.  3d should be similar, on the first glance, there is no need to extend the
        algorithm in this case.

"""

import itertools
import math
from time import time
import gmsh
import numpy as np

gmsh.initialize()
gmsh.logger.start()
gmsh.model.add("2D Fractures")

factory = gmsh.model.occ


# region Geometry

tailored_geometry = True

# Square domain.
dx = 5
factory.add_rectangle(0, 0, 0, dx, dx, tag=1)
factory.synchronize()

fractures = []
if tailored_geometry:
    # Intersecting fractures (placed in bottom-left quadrant)
    for i in range(3):
        x = 0.5 + i * 0.5
        y = 0.5 + i * 0.5
        start = factory.add_point(x, y, 0)
        end = factory.add_point(x + 0.05, y, 0)
        factory.add_line(start, end, tag=10 + i)
        fractures.append((1, 10 + i))

        start = factory.add_point(x + 0.05, y - 0.1, 0)
        end = factory.add_point(x + 0.05, y + 0.2, 0)
        factory.add_line(start, end, tag=15 + i)
        fractures.append((1, 15 + i))

    # Parallel fractures (placed in top-left quadrant)
    for i in range(3):
        y1 = 3 + i * 0.5
        y2 = 3.05 + i * 0.5
        start1 = factory.add_point(0.2, y1, 0)
        end1 = factory.add_point(0.9, y1, 0)
        factory.add_line(start1, end1, tag=20 + i)
        fractures.append((1, 20 + i))

        start2 = factory.add_point(0.2, y2, 0)
        end2 = factory.add_point(0.9, y2, 0)
        factory.add_line(start2, end2, tag=25 + i)
        fractures.append((1, 25 + i))

    # T-shaped fractures (placed in center)
    for i in range(3):
        x = 2.2 + i * 0.1
        y = 2.2 + i * 0.1
        start3 = factory.add_point(x, y, 0)
        end3 = factory.add_point(x, y + 0.2, 0)
        factory.add_line(start3, end3, tag=30 + i)
        fractures.append((1, 30 + i))

        start4 = factory.add_point(x + 0.02, y + 0.08, 0)
        end4 = factory.add_point(x + 0.2, y + 0.08, 0)
        factory.add_line(start4, end4, tag=35 + i)
        fractures.append((1, 35 + i))

    # Small-angle crossing fractures (placed in top-right quadrant)
    for i in range(3):
        angle = math.radians(5)
        length = 0.3
        center_x, center_y = 3.2 + i * 0.3, 3.4 + i * 0.3
        dx = length * math.cos(angle) / 2
        dy = length * math.sin(angle) / 2

        start5 = factory.add_point(center_x - dx, center_y - dy, 0)
        end5 = factory.add_point(center_x + dx, center_y + dy, 0)
        factory.add_line(start5, end5, tag=40 + i)
        fractures.append((1, 40 + i))

        angle2 = math.radians(-5)
        dx2 = length * math.cos(angle2) / 2
        dy2 = length * math.sin(angle2) / 2
        start6 = factory.add_point(center_x - dx2, center_y - dy2, 0)
        end6 = factory.add_point(center_x + dx2, center_y + dy2, 0)
        factory.add_line(start6, end6, tag=45 + i)
        fractures.append((1, 45 + i))

    # Lone fractures (placed in lower right quadrant)
    for i in range(3):
        start7 = factory.add_point(3.25 + i * 0.5, 0.5 + i * 0.5, 0)
        end7 = factory.add_point(3.5 + i * 0.5, 0.75 + i * 0.5, 0)
        factory.add_line(start7, end7, tag=50 + i)
        fractures.append((1, 50 + i))

    # Boundary-near fractures (placed near edges but spaced)
    boundary_margin = 0.05
    for i in range(3):
        start = factory.add_point(boundary_margin, 1.0 + i * 0.5, 0)
        end = factory.add_point(boundary_margin + 0.5, 1.0 + i * 0.5, 0)
        factory.add_line(start, end, tag=60 + i)
        fractures.append((1, 60 + i))
else:  # Random fractures
    num_fractures = 10
    np.random.seed(42)
    for i in range(num_fractures):
        x = np.random.uniform(0, dx)
        y = np.random.uniform(0, dx)
        start = factory.add_point(x, y, 0)

        end = factory.add_point(np.random.uniform(0, dx), np.random.uniform(0, dx), 0)
        tg = factory.add_line(start, end)
        fractures.append((1, tg))

# Fragment and synchronize
ov, ovv = factory.fragment([(2, 1)], fractures)
factory.synchronize()

# Filter out boundary lines
boundaries = gmsh.model.get_boundary([(2, 1)])
fractures = [f for f in gmsh.model.getEntities(1) if f not in boundaries]

# endregion

# region Mesh size fields and mesh generation

# Loop over all unique pairs of fractures.
for i, (domain_msh_size, fracture_msh_size, boundary_mesh_size) in enumerate(
    [(1.0, 0.2, 0.2), (0.2, 0.2, 1.0), (0.2, 0.05, 0.05), (0.05, 0.05, 0.05)]
):
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    local_msh_size_fields = []

    try:
        gmsh.model.mesh.field.remove(global_msh_size_field)
    except NameError:
        pass
    for local_msh_size_field in local_msh_size_fields:
        try:
            gmsh.model.mesh.field.remove(local_msh_size_field)
        except NameError:
            pass
    gmsh.model.mesh.clear()

    distances = {}
    tic = time()
    line_tags = [tag for _, tag in gmsh.model.getEntities(1)]
    fracture_tags = [tag for _, tag in fractures]
    boundary_tags = [tag for _, tag in boundaries]

    # Create a local mesh size field for each pair of close fractures and/or boundaries.
    for line_1, line_2 in itertools.combinations(line_tags, 2):
        # Skip boundary-boundary pairs.
        if (line_1 in boundary_tags) and (line_2 in boundary_tags):
            continue

        distances = factory.getDistance(1, line_1, 1, line_2)
        dist = distances[0]
        if dist < fracture_msh_size * 2:
            if dist == 0.0:
                dist = fracture_msh_size

            # distances[(line_1, line_2)] = dist

            pts = [
                factory.add_point(*distances[1:4]),
                factory.add_point(*distances[4:]),
            ]

            ds = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(ds, "PointsList", pts)
            tf = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(tf, "InField", ds)
            gmsh.model.mesh.field.setNumber(tf, "DistMin", dist)
            gmsh.model.mesh.field.setNumber(tf, "DistMax", 2 * dist)
            gmsh.model.mesh.field.setNumber(tf, "SizeMin", dist / 3)
            gmsh.model.mesh.field.setNumber(tf, "SizeMax", domain_msh_size)
            local_msh_size_fields.append(tf)

            # Set small mesh size if both fractures are close, else domain mesh size.
            threshold_fields = []

            for fracture in (line_1, line_2):
                distance_field = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(
                    distance_field, "EdgesList", [fracture]
                )
                # This is the way to control the number of sample points used to compute
                # the distance along the fracture. More points means better accuracy,
                # but also more computational cost.
                gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 2)

                threshold_field = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(
                    threshold_field, "InField", distance_field
                )
                gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 2 * dist)
                gmsh.model.mesh.field.setNumber(
                    threshold_field, "DistMax", 3 * dist
                )  # Smooth transition
                gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", 1.0)
                gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", 0.0)
                threshold_fields.append(threshold_field)

            local_msh_size_field = gmsh.model.mesh.field.add("MathEval")
            sum_expr = f"F{threshold_fields[0]} + F{threshold_fields[1]}"
            # Approximate a step function that is 0 if sum_expr <= 1, else 1.
            full_expr = (
                f"{domain_msh_size} - ({domain_msh_size} - {dist / 3}) * "
                f"(0.5 * tanh(10*({sum_expr} - 1.5)) + 0.5)"
            )
            gmsh.model.mesh.field.setString(local_msh_size_field, "F", full_expr)
            # local_msh_size_fields.append(local_msh_size_field)

    # Create a local mesh size field for each fracture and boundary.
    for line in line_tags:
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", [line])
        local_msh_size = (
            fracture_msh_size if line in fracture_tags else boundary_mesh_size
        )
        local_msh_size_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(local_msh_size_field, "InField", distance_field)
        gmsh.model.mesh.field.setNumber(
            local_msh_size_field, "DistMin", 2 * local_msh_size
        )
        gmsh.model.mesh.field.setNumber(
            local_msh_size_field, "DistMax", 3 * local_msh_size
        )
        gmsh.model.mesh.field.setNumber(local_msh_size_field, "SizeMin", local_msh_size)
        gmsh.model.mesh.field.setNumber(
            local_msh_size_field, "SizeMax", domain_msh_size
        )
        local_msh_size_fields.append(local_msh_size_field)

    global_msh_size_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(
        global_msh_size_field, "FieldsList", local_msh_size_fields
    )
    gmsh.model.mesh.field.setAsBackgroundMesh(global_msh_size_field)

    # Disable curvature/point-based sizing for more control.
    # gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    # gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    factory.synchronize()
    print(f"Time for mesh size processing: {time() - tic:.4f} s")

    gmsh.model.mesh.generate(2)

    # gmsh.fltk.run()
    gmsh.write(f"mesh_size_final_algo_option_{i}.msh")


# endregion
