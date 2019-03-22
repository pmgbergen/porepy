
"""Robust point-in-polyhedron test.
Given an closed, oriented surface in R^3 described by a triangular mesh, the
code below gives a robust algorithm for determining whether a given point is
inside, on the boundary of, or outside, the surface.  The algorithm should give
correct results even in degenerate cases, and applies to disconnected
polyhedra, non simply-connected surfaces, and so on.  There are no requirements
for the surface to be convex, simple, connected or simply-connected.
More precisely, we give a method for computing the *winding number* of a closed
oriented surface S around a point O that doesn't lie on S.  Roughly speaking,
the winding number of the closed oriented surface S around a point O not on S
is the number of times that the surface encloses that point; for a simple
outward-oriented surface (like that of a convex polyhedron, for example), the
winding number will be 1 for points inside the surface and 0 for points
outside.
For a precise definition of winding number, we can turn to algebraic topology:
our oriented surface is presented as a collection of combinatorial data
defining abstract vertices, edges and triangles, together with a mapping of
those vertices to R^3.  The combinatorial data describe a simplicial complex C,
and assuming that O doesn't lie on the surface, the mapping of the vertices to
R^3 gives a continuous map from the geometric realization of C to R^3 - {O}.
This in turn induces a map on second homology groups:
   H^2(C, Z) -> H^2(R^3 - {O}, Z)
and by taking the usual right-handed orientation in R^3 we identify H^2(R^3 -
{O}, Z) with Z.  The image of [S] under this map gives the winding number.  In
particular, the well-definedness of the winding number does not depend on
topological properties of the embedding: it doesn't matter if the surface is
self-intersecting, or has degenerate triangles.  The only condition is that O
does not lie on the surface S.
Algorithm
---------
The algorithm is based around the usual method of ray-casting: we take a
vertical line L through O and count the intersections of this line with the
triangles of the surface, keeping track of orientations as we go.  Let's ignore
corner cases for a moment and assume that:
(1) O does not lie on the surface, and
(2) for each triangle T (thought of as a closed subset of R^3) touched by
    our vertical line L, L meets the interior of T in exactly one point Q
Then there are four possibilities for each such triangle T:
1. T lies *above* O and is oriented *upwards* (*away* from O).
2. T lies *above* O and is oriented *downwards* (*towards* O).
3. T lies *below* O and is oriented *downwards* (*away* from O).
4. T lies *below* O and is oriented *upwards* (*towards* O).
Let's write N1, N2, N3 and N4 for the counts of triangles satisfying conditions
1, 2, 3 and 4 respectively.  Since we have a closed surface, these numbers
are not independent; they satisfy the relation:
    N1 + N4 == N2 + N3
That is, the number of upward-facing triangles must match the number of
downward-facing triangles.  The winding number w is then given by:
    w = N1 - N2 == N3 - N4
In the code below, we simply compute 2*w = (N1 + N3) - (N2 + N4), so each
triangle oriented away from O contributes 1 to 2w, while each triangle oriented
towards O contributes -1.
Making the algorithm robust
---------------------------
Now we describe how to augment the basic algorithm described above to include:
- correct treatment of corner cases (vertical triangles, cases where L meets an
  edge or vertex directly, etc.)
- detection of the case where the point lies directly on the surface.
It turns out that to make the algorithm robust, all we need to do is be careful
and consistent about classifying vertices, edges and triangles.  We do this as
follows:
- Each vertex of the surface that's not equal to O is considered *positive* if
  its coordinates are lexicographically greater than O, and *negative*
  otherwise.
- For an edge PQ of the surface that's not collinear with O, we first describe
  the classification in the case that P is negative and Q is positive, and
  then extend to arbitrary PQ.
  For P negative and Q positive, there are two cases:
  1. P and Q have distinct x coordinates.  In that case we classify the edge
     PQ by its intersection with the plane passing through O and parallel
     to the yz-plane: the edge is *positive* if the intersection point is
     positive, and *negative* otherwise.
  2. P and Q have the same x coordinate, in which case they must have
     distinct y coordinates.  (If the x and the y coordinates both match
     then PQ passes through O.)  We classify by the intersection of PQ
     with the line parallel to the y-axis through O.
  For P positive and Q negative, we classify as above but reverse the sign.
  For like-signed P and Q, the classification isn't used.
  Computationally, in case 1 above, the y-coordinate of the intersection
  point is:
      Py + (Qy - Py) * (Ox - Px) / (Qx - Px)
  and this is greater than Oy iff
      (Py - Oy) * (Qx - Ox) - (Px - Ox) * (Qy - Oy)
  is positive, so the sign of the edge is the sign of the above expression.
  Similarly, if this quantity is zero then we need to look at the z-coordinate
  of the intersection, and the sign of the edge is given by
      (Pz - Oz) * (Qx - Ox) - (Px - Ox) * (Qz - Oz)
  In case 2, both of the above quantities are zero, and the sign of the edge is
  the sign of
      (Pz - Oz) * (Qy - Oy) - (Py - Oy) * (Qz - Oz)
  Another way to look at this: if P, Q and O are not collinear then the
  matrix
   ( Px Qx Ox )
   ( Py Qy Ox )
   ( Pz Qz Ox )
   (  1  1  1 )
  has rank 3.  It follows that at least one of the three 3x3 minors
   | Px Qx Ox |  | Px Qx Ox |  | Py Qy Oy |
   | Py Qy Oy |  | Pz Qz Oz |  | Pz Qz Oz |
   |  1  1  1 |  |  1  1  1 |  |  1  1  1 |
  is nonzero.  We define the sign of PQ to be the *negative* of the sign of the
  first nonzero minor in that list.
- Each triangle PQR of the surface that's not coplanar with O is considered
  *positive* if its normal points away from O, and *negative* if its normal
  points towards O.
  Computationally, the sign of the triangle PQR is the sign of the 4x4
  determinant
    | Px Qx Rx Ox |
    | Py Qy Ry Oy |
    | Pz Qz Rz Oz |
    |  1  1  1  1 |
  or equivalently of the 3x3 determinant
    | Px-Ox Qx-Ox Rx-Ox |
    | Py-Oy Qy-Oy Ry-Oy |
    | Pz-Oz Qz-Oz Rz-Oz |
Now to compute the contribution of any given triangle to the total winding
number:
1. Classify the vertices of the triangle.  At the same time, we can check that
   none of the vertices is equal to O.  If all vertices have the same sign,
   then the winding number contribution is zero.
2. Assuming that the vertices do not all have the same sign, two of the three
   edges connect two differently-signed vertices.  Classify both those edges
   (and simultaneously check that they don't pass through O).  If the edges
   have opposite classification, then the winding number contribution is zero.
3. Now two of the edges have the same sign: classify the triangle itself.  If
   the triangle is positive it contributes 1/2 to the winding number total; if
   negative it contributes -1/2.  In practice we count contributions of 1 and
   -1, and halve the total at the end.
Note that an edge between two like-signed vertices can never pass through O, so
there's no need to check the third edge in step 2.  Similarly, a triangle whose
edge-cycle is trivial can't contain O in its interior.
To understand what's going on above, it's helpful to step into the world of
homology again. The homology of R^3 - {O} can be identified with that of the
two-sphere S^2 by deformation retract, and we can decompose the two-sphere as a
CW complex consisting of six cells, as follows:
* 0-cells B and F, where B = (-1, 0, 0) and F = (1, 0, 0)
* 1-cells L and R, where
     L = {(cos t, sin t, 0) | -pi <= t <= 0 }
     R = {(cos t, sin t, 0) | 0 <= t <= pi }
* 2-cells U and D, where U is the top half of the sphere (z >= 0)
  and D is the bottom half (z <= 0), both oriented outwards.
And the homology of the CW complex is now representable in terms of cellular
homology:
               d               d
  Z[U] + Z[D] --> Z[L] + Z[R] --> Z[B] + Z[F]
with boundary maps given by:
  d[U] = [L] + [R]; d[D] = -[L] - [R]
  d[R] = [B] - [F]; d[L] = [F] - [B]
Now the original map C -> R^3 - {O} from the geometric realization of the
simplicial complex is homotopic to a map C -> S^2 that sends:
* each positive vertex to F and each negative vertex to B
* each edge with boundary [F] - [B] to L if the edge is negative, and -R if the
  edge is positive
* each edge with boundary [B] - [F] to R if the edge is positive, and -L if the
  edge is negative
* all other edges to 0
* each triangle whose boundary is [L] + [R] to either U or -D,
  depending on whether the triangle is positive or negative
* each triangle whose boundary is -[L] - [R] to either D or -U,
  depending on whether the triangle is positive or negative
* all other triangles to 0
Mapping all of the triangles in the surface this way, and summing the results
in second homology, we end up with (winding number)*([U] + [D]).
"""


def sign(x):
    """
    Return 1 if x is positive, -1 if it's negative, and 0 if it's zero.
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def vertex_sign(P, O):
    """
    Sign of the vertex P with respect to O, as defined above.
    """
    result = sign(P[0] - O[0]) or sign(P[1] - O[1]) or sign(P[2] - O[2])
    if not result:
        raise ValueError("vertex coincides with origin")
    return result


def edge_sign(P, Q, O):
    """
    Sign of the edge PQ with respect to O, as defined above.
    """
    result = (
        sign((P[1] - O[1]) * (Q[0] - O[0]) - (P[0] - O[0]) * (Q[1] - O[1])) or
        sign((P[2] - O[2]) * (Q[0] - O[0]) - (P[0] - O[0]) * (Q[2] - O[2])) or
        sign((P[2] - O[2]) * (Q[1] - O[1]) - (P[1] - O[1]) * (Q[2] - O[2]))
    )
    if not result:
        raise ValueError("vertices collinear with origin")
    return result


def triangle_sign(P, Q, R, O):
    """
    Sign of the triangle PQR with respect to O, as defined above.
    """
    m1_0 = P[0] - O[0]
    m1_1 = P[1] - O[1]
    m2_0 = Q[0] - O[0]
    m2_1 = Q[1] - O[1]
    m3_0 = R[0] - O[0]
    m3_1 = R[1] - O[1]
    result = sign(
        (m1_0 * m2_1 - m1_1 * m2_0) * (R[2] - O[2]) +
        (m2_0 * m3_1 - m2_1 * m3_0) * (P[2] - O[2]) +
        (m3_0 * m1_1 - m3_1 * m1_0) * (Q[2] - O[2]))
    if not result:
        raise ValueError("vertices coplanar with origin")
    return result


def triangle_chain(v1, v2, v3, origin):
    """
    Return the contribution of this triangle to the winding number.
    Raise ValueError if the face contains the origin.
    """
    v1sign = vertex_sign(v1, origin)
    v2sign = vertex_sign(v2, origin)
    v3sign = vertex_sign(v3, origin)

    face_boundary = 0
    if v1sign != v2sign:
        face_boundary += edge_sign(v1, v2, origin)
    if v2sign != v3sign:
        face_boundary += edge_sign(v2, v3, origin)
    if v3sign != v1sign:
        face_boundary += edge_sign(v3, v1, origin)
    if not face_boundary:
        return 0

    return triangle_sign(v1, v2, v3, origin)


class Polyhedron(object):
    def __init__(self, triangles, vertex_positions):
        """
        Initialize from list of triangles and vertex positions.
        """
        # Validate: check the combinatorial data.
        edges = set()
        vertices = set()
        for triangle in triangles:
            vertices.update(triangle)
            P, Q, R = triangle
            for edge in ((P, Q), (Q, R), (R, P)):
                if edge[0] == edge[1]:
                    raise ValueError("Self edge: {!r}".format(edge))
                if edge in edges:
                    raise ValueError("Duplicate edge: {!r}".format(edge))
                edges.add(edge)

        # For each edge that appears, the reverse edge should also appear.
        for P, Q in edges:
            if not (Q, P) in edges:
                raise ValueError("Unmatched edge: {!r}".format((P, Q)))

        # Vertex set should match indices in vertex_positions.
        if vertices != set(range(len(vertex_positions))):
            raise ValueError("Vertex set doesn't match position indices.")

        # Vertex positions in R^3.
        self.vertex_positions = vertex_positions
        # Indices making up each triangle, counterclockwise
        # around the outside of the face.
        self.triangles = triangles

    def triangle_positions(self):
        """
        Triples of vertex positions.
        """
        for triangle in self.triangles:
            yield tuple(self.vertex_positions[vx] for vx in triangle)

    def volume(self):
        """
        Return the volume of this polyhedron.
        """
        acc = 0
        for p1, p2, p3 in self.triangle_positions():
            # Twice the area of the projection onto the x-y plane.
            det = ((p2[1] - p3[1]) * (p1[0] - p3[0]) -
                   (p2[0] - p3[0]) * (p1[1] - p3[1]))
            # Three times the average height.
            height = p1[2] + p2[2] + p3[2]
            acc += det * height
        return acc / 6.0

    def winding_number(self, point):
        """Determine the winding number of *self* around the given point.
        """
        return sum(
            triangle_chain(v1, v2, v3, point)
for v1, v2, v3 in self.triangle_positions()) // 2