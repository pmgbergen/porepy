import gmsh

import meshio

import porepy as pp
import numpy as np

gmsh.initialize()

fac = gmsh.model.occ

# Define the domain. This will normally be the task of the FractureNetwork class.
domain = fac.add_rectangle(-1, -1, 0, 4, 4)


# Add a few fractures. This will normally be the task of the individual Fracture objects.

# Create a line [x1, y1, z1], [x2, y2, z2]
p_0 = fac.addPoint(0, 0, 0)
p_1 = fac.addPoint(1, 1, 0)
line1 = fac.add_line(p_0, p_1)

# Create a new line crossing the first one
p_2 = fac.addPoint(0, 1, 0)
p_3 = fac.addPoint(1, 0.5, 0)
line2 = fac.add_line(p_2, p_3)

# Add a third line which crosses the first line, but not the second.
p_4 = fac.addPoint(0.5, -0.8, 0)
p_5 = fac.addPoint(0.5, -0, 0)
line3 = fac.add_line(p_4, p_5)

# List of fractures. This is roughly the expected output after a FractureNetwork object
# has looped over its fractures and had them represented in the gmsh.occ kernel.
lines = [(1, line1), (1, line2), (1, line3)]

# Synchronize the GMSH model before fragmenting.
fac.synchronize()

## This part identifies fractures that are fully outside the domain and kicks them out.

new_lines = {}
lines_removed = []

for ind, line in enumerate(lines):
    truncated_line, b = fac.intersect(
        [line], [(2, domain)], removeTool=False, removeObject=False
    )
    if len(truncated_line) > 0:
        # The line was partly outside the domain. It must be replaced.
        new_lines[ind] = truncated_line[0]
    elif len(b[0]) == 0:
        # The line was fully outside the domain. It will be deleted.
        lines_removed.append(ind)

# Remove the lines from the gmsh representation. Recursive is critical here, or else the
# boundary of 'line' will continue to be present.
for line in lines_removed:
    fac.remove([lines[line]], recursive=True)

for old_line, new_line in new_lines.items():
    fac.remove([lines[old_line]], recursive=True)
    lines[old_line] = new_line

lines = [lines[i] for i in range(len(lines)) if i not in lines_removed]

fac.synchronize()

# Make gmsh calculate the intersections between the lines, using the domain as a
# secondary object (the latter will by magic ensure that the lines are embedded in the
# domain, hence the mesh will conform to the fractures). The removal statements here
# will replace the old (possibly intersecting) lines with new, split lines. Similarly,
# the removal of the domain (removeTool, no idea why) avoids the domain being present
# twice.
_, isect_mapping = fac.fragment(
    lines, [(2, domain)], removeObject=True, removeTool=True
)

fac.synchronize()

# During intersection removal, gmsh will add intersection points and replace the lines
# with non-intersecting polylines (example: Two lines intersecting as a cross become
# four lines with a common point). We need to identify these intersecting points.
# Annoyingly, gmsh also reassigns tags during fragmentation. Hence we need to find
# intersection points on our own, as is done below.
intersection_points = []
# Loop over the mappings from old lines to new segments. The idea is to find the
# boundary points of all segments, identify those that occur more than once - these will
# be intersections - and store the tag that gmsh has assigned them.
for old_line in isect_mapping:
    all_boundary_points_of_segments = []
    for segment in old_line:
        all_boundary_points_of_segments += gmsh.model.get_boundary([segment])

    # Find the unique boundary points and obtain a mapping from the full set of boundary
    # points to the unique ones.
    unique_boundary_points, u2a_ind = np.unique(
        all_boundary_points_of_segments, axis=0, return_index=True
    )
    # Count the number of occurrences of each unique boundary point. Those that occur
    # more than once will be intersections.
    multiple_occs = np.where(np.bincount(u2a_ind) > 1)[0]
    # Store the gmsh representation of the intersection points.
    intersection_points += [unique_boundary_points[i] for i in multiple_occs]
# Finally, we need to uniquify the intersection points, since the same point will have
# been identified in at least two old lines.
unique_intersection_points = np.unique(np.vstack(intersection_points), axis=0)

# Represent the intersection points, fractures and domain as physical points.
fac.synchronize()

# Intersection points can be dealt with right away.
for i, pt in enumerate(unique_intersection_points):
    gmsh.model.addPhysicalGroup(0, [pt[1]], -1, f"FRACTURE_INTERSECTION_POINT_{i}")

fac.synchronize()

# Since fractures may have been split at intersection points, we need to collect all the
# segments (found in isect_mapping) into a single physical group.
for i, line_group in enumerate(isect_mapping):
    all_lines = []
    for line in line_group:
        if line[0] == 1:
            all_lines.append(line[1])
    if all_lines:
        gmsh.model.addPhysicalGroup(1, all_lines, -1, f"FRACTURE_{i}")


gmsh.model.addPhysicalGroup(2, [domain], -1, "DOMAIN")

fac.synchronize()

gmsh.write("tmp.geo_unrolled")

fn = "tmp.msh"

# Create a gmsh mesh
gmsh.model.mesh.generate(3)
gmsh.write(fn)

msh = meshio.read(fn)


mdg = pp.fracture_importer.dfm_from_gmsh(fn, dim=2)
print(mdg)

debug = []
