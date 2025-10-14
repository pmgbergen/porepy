import itertools
import math

import gmsh
import numpy as np

gmsh.initialize()
gmsh.logger.start()
gmsh.model.add("2D Fractures")

factory = gmsh.model.occ


# region Geometry

# Square domain.
factory.add_rectangle(0, 0, 0, 5, 5, tag=1)
factory.synchronize()

fractures = []

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
    gmsh.option.setNumber("Mesh.Algorithm", 6)

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

    line_tags = [tag for _, tag in gmsh.model.getEntities(1)]
    fracture_tags = [tag for _, tag in fractures]
    boundary_tags = [tag for _, tag in boundaries]

    # Create a local mesh size field for each pair of close fractures and/or boundaries.
    for line_1, line_2 in itertools.combinations(line_tags, 2):
        # Skip boundary-boundary pairs.
        if (line_1 in boundary_tags) and (line_2 in boundary_tags):
            continue

        dist = factory.getDistance(1, line_1, 1, line_2)[0]
        if dist < fracture_msh_size * 2:
            if dist == 0.0:
                dist = fracture_msh_size

            distances[(line_1, line_2)] = dist

            # Set small mesh size if both fractures are close, else domain mesh size.
            threshold_fields = []

            for fracture in (line_1, line_2):
                distance_field = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(
                    distance_field, "EdgesList", [fracture]
                )
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
            local_msh_size_fields.append(local_msh_size_field)

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
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    factory.synchronize()
    gmsh.model.mesh.generate(2)

    gmsh.fltk.run()
    gmsh.write(f"mesh_size_final_algo_option_{i}.msh")


# endregion
