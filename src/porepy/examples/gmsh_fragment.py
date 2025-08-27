import gmsh

import meshio

import porepy as pp
import numpy as np

gmsh.initialize()

fac = gmsh.model.occ


# Create a line [x1, y1, z1], [x2, y2, z2]
p_0 = fac.addPoint(0, 0, 0)
p_1 = fac.addPoint(1, 1, 0)
line1 = fac.add_line(p_0, p_1)

# Create a new line crossing the first one
p_2 = fac.addPoint(0, 1, 0)
p_3 = fac.addPoint(1, 0, 0)
line2 = fac.add_line(p_2, p_3)

# Add a third line which crosses the first line, but not the second.
p_4 = fac.addPoint(0.5, -1, 0)
p_5 = fac.addPoint(0.5, 0, 0)
line3 = fac.add_line(p_4, p_5)

fac.synchronize()

# Make all the lines added into separate physical entities in the gmsh sense. They will
# have names 'FRACTURE_LINE_{INDEX} where INDEX is 0-offset and increase with the line
# number.
for i, line in enumerate([line1, line2, line3]):
    gmsh.model.addPhysicalGroup(1, [line], -1, f"FRACTURE_LINE_{i}")

# Define a domain [-1, 2] x [-1, 2]
domain = fac.add_rectangle(-1, -1, 0, 4, 4)

# Using the second argument to specify the domain will impose the boundary on the
# domain. Should be done with caution, since we need to keep track of which fractures
# are kicked out.
isect_pts, isect_mapping = fac.intersect(
    [(1, line1), (1, line2), (1, line3)],
    [],
)


fac.synchronize()

gmsh.write("tmp.geo_unrolled")

fn = "tmp.msh"

# Create a gmsh mesh
gmsh.model.mesh.generate(2)
gmsh.write(fn)

msh = meshio.read(fn)

debug = []
