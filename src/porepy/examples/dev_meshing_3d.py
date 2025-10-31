import numpy as np
import porepy as pp

import gmsh


gmsh.initialize()

# An a unit square to gmsh
f_1 = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
# Add an elliptical fracture with center at (0, 0, 1), semi-axes 0.5 and 0.2 in the xy-plane
f_2 = gmsh.model.occ.add_disk(0, 0, 1, 0.5, 0.2)

# Add a pentagon fracture at height z=2
p_1 = gmsh.model.occ.addPoint(-0.5, 0, 2)
p_2 = gmsh.model.occ.addPoint(0, 0.5, 2)
p_3 = gmsh.model.occ.addPoint(0.5, 0, 2)
p_4 = gmsh.model.occ.addPoint(0.3, -0.5, 2)
p_5 = gmsh.model.occ.addPoint(-0.3, -0.5, 2)
l_1 = gmsh.model.occ.addLine(p_1, p_2)
l_2 = gmsh.model.occ.addLine(p_2, p_3)
l_3 = gmsh.model.occ.addLine(p_3, p_4)
l_4 = gmsh.model.occ.addLine(p_4, p_5)
l_5 = gmsh.model.occ.addLine(p_5, p_1)
curve_loop = gmsh.model.occ.addCurveLoop([l_1, l_2, l_3, l_4, l_5])
f_3 = gmsh.model.occ.addPlaneSurface([curve_loop])

gmsh.model.occ.synchronize()
# gmsh.model.mesh.generate(2)

bnds = []
normals = []
for f in [f_1, f_2, f_3]:
    b = gmsh.model.get_parametrization_bounds(2, f)
    bnds.append(b)
    midpoint = gmsh.model.get_value(
        2, f, [(b[0][0] + b[0][1]) / 2, (b[1][0] + b[1][1]) / 2]
    )
    n = gmsh.model.get_normal(f, [(b[0][0] + b[0][1]) / 2, (b[1][0] + b[1][1]) / 2])
    normals.append(n)


# The angle of the normal vectors can now be computed by the cosine formula.
angle = np.arccos(
    np.dot(normals[0], normals[1])
    / (np.linalg.norm(normals[0]) * np.linalg.norm(normals[1]))
)
if angle > np.pi / 2:
    angle = np.pi - angle

# We can do an angle-like interpretation of the refinement/coarsening, like we did in 2d
# If it turns out that the angle is small, we need to do something similar to what we
# did in 2d: From the closest point on (one of the fractures?), move along each of the
# coordinate axes in parametrization space, step length equal to the distance (properly
# converted between physical and parametric coordinates), and add points as long as it
# is close enough. It should also be possible to combine get_distance from this new
# candidate point, and something with the parametrization of the other fracture to see
# if we are outside of the extrusion of the other fracture - this was used as a
# criterion in 2d (e.g., to avoid refinement of two fractures that lie in the same
# plane, but do not intersect). It should be sufficient to add points on one fracture in
# this case.
#
# If the fractures are not close to parallel, we may still need to add points to a
# T-like configuration. The simplest option seems to be to identify which of the
# fractures has its closest point on the boundary (it must be at least one of them).
# Then move along the coordinate axes in parametric space of that fracture, staying on
# the boundary, and add points as needed. Make sure we don't go outside the fracture.

bnds_3 = gmsh.model.get_parametrization_bounds(2, f_3)
# param = gmsh.model.get_parametrization(2, f_1, bnds)

cp = gmsh.model.occ.get_distance(2, f_1, 2, f_2)

p = gmsh.model.get_parametrization(2, f_1, cp[1:4])

debug = []
