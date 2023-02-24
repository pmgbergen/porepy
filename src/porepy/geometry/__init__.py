"""
This package contains functionality for geometry manipulation.

Most of these functions and classes have been developed for the purpose of working with
fracture geometries, notably for meshing purposes. Since the functions have been written
on demand, the coverage is spotwise compared to what would be expected from a full
computational geometry package. Nevertheless, the package contains quite a few useful
functions, and familiarity with the content is highly recommended for users who need
to deal with geometry computations.

Note:
    Many of the functions will have a parameter ``tol``, which is used to determine the
    tolerance of the geometric operations. One example would be how close two points
    must be to be considered equal. In its simplest form, the tolerance is tied to the
    machine precision, however, it is sometimes useful to take a more relaxed approach.
    For example, if points are very close to each other, but not considered equal, this
    may lead to extremely fine meshes in the vicinity of the points, which is usualy not
    desirable. Also, the accuracy in a chain of geometric operations may be considerably
    lower than that of indivdiual operations. The general recommendation is to leave the
    tolerance parameter at its default value(s), which has proven sufficient to obtain
    stable behavior during standard usage of PorePy.

The content of this package is organized as follows:

    :mod:`~porepy.geometry.constrain_geometry` contains functions that can be used to
    impose constraints on geometric objects, including truncation of line segments to a
    polygon and of polygons to a polyhedron.

    :mod:`~porepy.geometry.distances` contains functions for computing distances between
    points, line segments and polygons.

    :mod:`~porepy.geometry.domain` contains the class
    :class:`~porepy.geometry.domain.Domain` for defining domains from bounding boxes
    and general polytopes in 2d and 3d. It also contains several functions that are
    used for manipulation and conversion between bounding boxes and polytopes.

    :mod:`~porepy.geometry.geometry_property_checks` contains functions for inquiries on
    geometric properties, e.g., whether a point is inside a polygon.

    :mod:`~porepy.geometry.half_space` contains a set of functions relating to
    geometries defined by a combination of linear constraints of the type ``ax + by <=
    c`` or the equivalent 3d expression.

    :mod:`~porepy.geometry.intersections` contains functions for computing intersections
    between various combinations of segments and polygons. The module also provides
    functions for intersecting simplex tessalations in 1d and 2d.

    :mod:`~porepy.geometry.map_geometry` is a collection of functions for mapping
    :class:`~porepy.grids.grid.Grid` objcets as well as lines and polygons into
    lower-dimensional spaces. The module also provides functions for computing normal
    and tangential vectors for point clouds.

    :mod:`~porepy.geometry.point_in_polyhedron_test` contains a helper class for testing
    if a point is contained in a (non-convex) polygon.

"""
