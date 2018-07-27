"""
Main module for grid generation in fractured domains in 2d and 3d.

The module serves as the only neccessary entry point to create the grid. It
will therefore wrap interface to different mesh generators, pass options to the
generators etc.

"""
import numpy as np
import scipy.sparse as sps
import time
import logging

from porepy.fracs import structured, simplex, split_grid, non_conforming, tools
from porepy.fracs.fractures import Intersection
from porepy import FractureNetwork
from porepy.fracs.fractures import FractureNetwork as FractureNetwork_full
from porepy.grids.grid_bucket import GridBucket
from porepy.grids import mortar_grid
from porepy.grids.structured import TensorGrid
from porepy.utils import mcolon
from porepy.utils import comp_geom as cg


logger = logging.getLogger(__name__)


def simplex_grid(fracs=None, domain=None, network=None, subdomains=[], **kwargs):
    """
    Main function for grid generation. Creates a fractured simiplex grid in 2
    or 3 dimensions.

    NOTE: For some fracture networks, what appears to be a bug in Gmsh leads to
    surface grids with cells that does not have a corresponding face in the 3d
    grid. The problem may have been resolved (at least partly) by newer
    versions of Gmsh, but can still be an issue for our purposes. If this
    behavior is detected, an assertion error is raised. To avoid the issue,
    and go on with a surface mesh that likely is problematic, kwargs should
    contain a keyword ensure_matching_face_cell=False.

    Parameters
    ----------
    fracs (list of np.ndarray): One list item for each fracture. Each item
        consist of a (nd x n) array describing fracture vertices. The
        fractures may be intersecting.
    domain (dict): Domain specification, determined by xmin, xmax, ...
    subdomains (list of np.ndarray or list of Fractures): One list item
        for each fracture, same format as fracs. Specifies internal boundaries
        for the gridding. Only available in 3D.
    **kwargs: May contain fracture tags, options for gridding, etc.

    Gridding options:
    The mesh parameters are:
        mesh_size_frac (double): Ideal mesh size. Will be added to all points
            that are sufficiently far away from other points.
        mesh_size_min (double): Minimal mesh size; we will make no attempts to
            enforce even smaller mesh sizes upon Gmsh.
        mesh_size_bound (double): Optional boundary mesh size, defaults to the
            value of mesh_size_frac. Will be added to the points
            defining the boundary. In 2d, this parameter dictates the size at
            the boundary corners. In 3d, it is assigned unless there are any
            fractures in the immediate vicinity influencing the size. In other
            words, mesh_size_bound is the boundary point equivalent of
            mesh_size_frac.

    TODO: Update 2d implementation to adhere to 3d in
        porepy.fracs.tools.determine_mesh_size.

    Returns
    -------
    GridBucket: A complete bucket where all fractures are represented as
        lower dim grids. The higher dim fracture faces are split in two,
        and on the edges of the GridBucket graph the mapping from lower dim
        cells to higher dim faces are stored as 'face_cells'. Each face is
        given boolean tags depending on the type:
           domain_boundary_faces: All faces that lie on the domain boundary
               (i.e. should be given a boundary condition).
           fracture_faces: All faces that are split (i.e. has a connection to a
               lower dim grid).
           tip_faces: A boundary face that is not on the domain boundary, nor
               coupled to a lower domentional domain.
        The union of the above three is the tag boundary_faces.

    Examples
    --------
    frac1 = np.array([[1,4],[1,4]])
    frac2 = np.array([[1,4],[4,1]])
    fracs = [frac1, frac2]
    domain = {'xmin': 0, 'ymin': 0, 'xmax':5, 'ymax':5}
    gb = simplex_grid(fracs, domain)

    """
    if domain is None:
        if fracs is not None:
            ndim = fracs[0].shape[0]
        else:
            ndim = network[0].p.shape[0]

    elif "zmax" in domain:
        ndim = 3
    elif "ymax" in domain:
        ndim = 2
    else:
        raise ValueError("simplex_grid only supported for 2 or 3 dimensions")

    logger.info("Construct mesh")
    tm_tot = time.time()
    if ndim == 2:
        assert fracs is not None, "2d requires definition of fractures"
        assert domain is not None, "2d requires definition of domain"
        # Convert the fracture to a fracture dictionary.
        if len(fracs) == 0:
            f_lines = np.zeros((2, 0))
            f_pts = np.zeros((2, 0))
        else:
            f_lines = np.reshape(np.arange(2 * len(fracs)), (2, -1), order="F")
            f_pts = np.hstack(fracs)
        frac_dic = {"points": f_pts, "edges": f_lines}
        grids = simplex.triangle_grid(frac_dic, domain, **kwargs)
    elif ndim == 3:
        grids = simplex.tetrahedral_grid(fracs, domain, network, subdomains, **kwargs)
    else:
        raise ValueError("Only support for 2 and 3 dimensions")

    return grid_list_to_grid_bucket(grids, time_tot=tm_tot, **kwargs)


def dfn(fracs, conforming, intersections=None, keep_geo=False, tol=1e-4, **kwargs):
    """ Create a mesh of a DFN model, that is, only of fractures.

    The mesh can eihter be conforming along fracture intersections, or each
    fracture is meshed independently. The latter case will typically require
    some sort of sewing together external to this funciton.

    TODO: What happens if we give in a non-connected network?

    Parameters:
        fracs (either Fractures, or a FractureNetwork).
        conforming (boolean): If True, the mesh will be conforming along 1d
            intersections.
        intersections (list of lists, optional): Each item corresponds to an
            intersection between two fractures. In each sublist, the first two
            indices gives fracture ids (refering to order in fracs). The third
            item is a numpy array representing intersection coordinates. If no
            intersections provided, intersections will be detected using
            function in FractureNetwork.
        **kwargs: Parameters passed to gmsh.

    Returns:
        GridBucket (if conforming is True): Mixed-dimensional mesh that
            represents all fractures, and intersection poitns and line.

    """

    if isinstance(fracs, FractureNetwork) or isinstance(fracs, FractureNetwork_full):
        network = fracs
    else:
        network = FractureNetwork(fracs)

    # Populate intersections in FractureNetowrk, or find intersections if not
    # provided.

    if intersections is not None:
        logger.warn("FractureNetwork use pre-computed intersections")
        network.intersections = [Intersection(*i) for i in intersections]
    else:
        logger.warn("FractureNetwork find intersections in DFN")
        tic = time.time()
        network.find_intersections()
        logger.warn("Done. Elapsed time " + str(time.time() - tic))

    if conforming:
        logger.warn("Create conforming mesh for DFN network")
        grids = simplex.triangle_grid_embedded(network, find_isect=False, **kwargs)
    else:
        logger.warn("Create non-conforming mesh for DFN network")
        tic = time.time()
        grid_list = []
        neigh_list = []

        for fi in range(len(network._fractures)):
            logger.info("Meshing of fracture " + str(fi))
            # Rotate fracture vertexes and intersection points
            fp, ip, other_frac, rot, cp = network.fracture_to_plane(fi)
            frac_i = network[fi]

            f_lines = np.reshape(np.arange(ip.shape[1]), (2, -1), order="F")
            frac_dict = {"points": ip, "edges": f_lines}
            if keep_geo:
                file_name = "frac_mesh_" + str(fi)
                kwargs["file_name"] = file_name
            # Create mesh on this fracture surface.
            grids = simplex.triangle_grid(frac_dict, fp, verbose=False, **kwargs)

            irot = rot.T
            # Loop over grids, rotate back again to 3d coordinates
            for gl in grids:
                for g in gl:
                    g.nodes = irot.dot(g.nodes) + cp

            # Nodes of main (fracture) grid, in 3d coordinates1
            main_nodes = grids[0][0].nodes
            main_global_point_ind = grids[0][0].global_point_ind
            # Loop over intersections, check if the intersection is on the
            # boundary of this fracture.
            for ind, isect in enumerate(network.intersections_of_fracture(fi)):
                of = isect.get_other_fracture(frac_i)
                if isect.on_boundary_of_fracture(frac_i):
                    dist, _, _ = cg.dist_points_polygon(main_nodes, of.p)
                    hit = np.argwhere(dist < tol).reshape((1, -1))[0]
                    nodes_1d = main_nodes[:, hit]
                    global_point_ind = main_global_point_ind[hit]

                    assert cg.is_collinear(nodes_1d, tol=tol)
                    sort_ind = cg.argsort_point_on_line(nodes_1d, tol=tol)
                    g_aux = TensorGrid(np.arange(nodes_1d.shape[1]))
                    g_aux.nodes = nodes_1d[:, sort_ind]
                    g_aux.global_point_ind = global_point_ind[sort_ind]
                    grids[1].insert(ind, g_aux)

            assert len(grids[0]) == 1, "Fracture should be covered by single" "mesh"

            grid_list.append(grids)
            neigh_list.append(other_frac)

        logger.warn("Finished creating grids. Elapsed time " + str(time.time() - tic))
        logger.warn("Merge grids")
        tic = time.time()
        grids = non_conforming.merge_grids(grid_list, neigh_list)
        logger.warn("Done. Elapsed time " + str(time.time() - tic))

        for g_set in grids:
            if len(g_set) > 0:
                s = (
                    "Created "
                    + str(len(g_set))
                    + " "
                    + str(g_set[0].dim)
                    + "-d grids with "
                )
                num = 0
                for g in g_set:
                    num += g.num_cells
                s += str(num) + " cells"
                logger.info(s)

    return grid_list_to_grid_bucket(grids, check_highest_dim=False)


# ------------------------------------------------------------------------------#


def from_gmsh(file_name, dim, **kwargs):
    """
    Import an already generated grid from gmsh.
    NOTE: Only 2d grid is implemented so far.

    TODO: A similar function is being implemnted in the importer, use this instead.

    Parameters
    ----------
    file_name (string): Gmsh file name.
    dim (int): Spatial dimension of the grid.
    **kwargs: May contain fracture tags, options for gridding, etc.

    Returns
    -------
    Grid or GridBucket: If no fractures are present in the gmsh file a simple
        grid is returned. Otherwise, a complete bucket where all fractures are
        represented as lower dim grids. See the documentation of simplex_grid
        for further details.

    Examples
    --------
    gb = from_gmsh('grid.geo', 2)

    """
    # Call relevant method, depending on grid dimensions.
    if dim == 2:
        if file_name.endswith(".geo"):
            simplex.triangle_grid_run_gmsh(file_name, **kwargs)
            grids = simplex.triangle_grid_from_gmsh(file_name, **kwargs)
        elif file_name.endswith(".msh"):
            grids = simplex.triangle_grid_from_gmsh(file_name, **kwargs)

    #    elif dim == 3:
    #        grids = simplex.tetrahedral_grid_from_gmsh(file_name, **kwargs)
    #   NOTE: function simplex.tetrahedral_grid needs to be split as did for
    #   simplex.triangle_grid
    else:
        raise ValueError("Only support for 2 dimensions")

    # No fractures are specified, return a simple grid
    if len(grids[1]) == 0:
        grids[0][0].compute_geometry()
        return grids[0][0]

    return grid_list_to_grid_bucket(grids)


# ------------------------------------------------------------------------------#


def grid_list_to_grid_bucket(grids, time_tot=None, **kwargs):
    """ Convert a list of grids to a full GridBucket.

    The list can come from several mesh constructors, both simplex and
    structured approaches uses this in 2D and 3D.

    The function can not be used on an arbitrary set of grids; they should
    contain information to glue grids together. This will be included for grids
    created by the standard mixed-dimensional grid constructors. In other
    words: Do *not* use this function directly unless you know what you are
    doing.

    Parameters:
        grids (list of lists of grids): Grids to enter into the bucket.
            Sorted per dimension.
        time_tot (double, optional): Start time for full mesh construction.
            Used for logging. Defaults to None, in which case no information
            on total time consumption is logged.
        **kwargs: Passed on to subfunctions.

    Returns:
        GridBucket: Final mixed-dimensional grid.

    """
    # Tag tip faces
    check_highest_dim = kwargs.get("check_highest_dim", False)
    _tag_faces(grids, check_highest_dim)

    logger.info("Assemble in bucket")
    tm_bucket = time.time()
    gb = _assemble_in_bucket(grids, **kwargs)
    logger.info("Done. Elapsed time " + str(time.time() - tm_bucket))

    logger.info("Compute geometry")
    tm_geom = time.time()
    gb.compute_geometry()
    # Split the grids.
    logger.info("Done. Elapsed time " + str(time.time() - tm_geom))
    logger.info("Split fractures")
    tm_split = time.time()
    split_grid.split_fractures(gb, **kwargs)
    logger.info("Done. Elapsed time " + str(time.time() - tm_split))

    create_mortar_grids(gb, **kwargs)

    gb.assign_node_ordering()

    if time_tot is not None:
        logger.info(
            "Mesh construction completed. Total time " + str(time.time() - time_tot)
        )

    return gb


def cart_grid(fracs, nx, **kwargs):
    """
    Creates a cartesian fractured GridBucket in 2- or 3-dimensions.

    Parameters
    ----------
    fracs (list of np.ndarray): One list item for each fracture. Each item
        consist of a (nd x 3) array describing fracture vertices. The
        fractures has to be rectangles(3D) or straight lines(2D) that
        alignes with the axis. The fractures may be intersecting.
        The fractures will snap to closest grid faces.
    nx (np.ndarray): Number of cells in each direction. Should be 2D or 3D
    **kwargs:
        physdims (np.ndarray): Physical dimensions in each direction.
            Defaults to same as nx, that is, cells of unit size.
        May also contain fracture tags, options for gridding, etc.

    Returns:
    -------
    GridBucket: A complete bucket where all fractures are represented as
        lower dim grids. The higher dim fracture faces are split in two,
        and on the edges of the GridBucket graph the mapping from lower dim
        cells to higher dim faces are stored as 'face_cells'. Each face is
        given boolean tags depending on the type:
           domain_boundary_faces: All faces that lie on the domain boundary
               (i.e. should be given a boundary condition).
           fracture_faces: All faces that are split (i.e. has a connection to a
               lower dim grid).
           tip_faces: A boundary face that is not on the domain boundary, nor
               coupled to a lower domentional domain.
        The union of the above three is the tag boundary_faces.

    Examples
    --------
    frac1 = np.array([[1,4],[2,2]])
    frac2 = np.array([[2,2],[4,1]])
    fracs = [frac1, frac2]
    gb = cart_grid(fracs, [5,5])
    """
    ndim = np.asarray(nx).size
    physdims = kwargs.get("physdims", None)

    if physdims is None:
        physdims = nx
    elif np.asarray(physdims).size != ndim:
        raise ValueError("Physical dimension must equal grid dimension")

    # Call relevant method, depending on grid dimensions
    if ndim == 2:
        grids = structured.cart_grid_2d(fracs, nx, physdims=physdims)
    elif ndim == 3:
        grids = structured.cart_grid_3d(fracs, nx, physdims=physdims)
    else:
        raise ValueError("Only support for 2 and 3 dimensions")

    return grid_list_to_grid_bucket(grids, **kwargs)


def _tag_faces(grids, check_highest_dim=True):
    """
    Tag faces of grids. Three different tags are given to different types of
    faces:
        NONE: None of the below (i.e. an internal face)
        DOMAIN_BOUNDARY: All faces that lie on the domain boundary
            (i.e. should be given a boundary condition).
        FRACTURE: All faces that are split (i.e. has a connection to a
            lower dim grid).
        TIP: A boundary face that is not on the domain boundary, nor
            coupled to a lower domentional domain.

    Parameters:
        grids (list): List of grids to be tagged. Sorted per dimension.
        check_highest_dim (boolean, default=True): If true, we require there is
            a single mesh in the highest dimension. The test is useful, but
            should be waived for dfn meshes.

    """

    # Assume only one grid of highest dimension
    if check_highest_dim:
        assert len(grids[0]) == 1, "Must be exactly" "1 grid of dim: " + str(len(grids))

    for g_h in np.atleast_1d(grids[0]):
        bnd_faces = g_h.get_all_boundary_faces()
        domain_boundary_tags = np.zeros(g_h.num_faces, dtype=bool)
        domain_boundary_tags[bnd_faces] = True
        g_h.tags["domain_boundary_faces"] = domain_boundary_tags
        bnd_nodes, _, _ = sps.find(g_h.face_nodes[:, bnd_faces])
        bnd_nodes = np.unique(bnd_nodes)
        for g_dim in grids[1:-1]:
            for g in g_dim:
                # We find the global nodes of all boundary faces
                bnd_faces_l = g.get_all_boundary_faces()
                indptr = g.face_nodes.indptr
                fn_loc = mcolon.mcolon(indptr[bnd_faces_l], indptr[bnd_faces_l + 1])
                nodes_loc = g.face_nodes.indices[fn_loc]
                # Convert to global numbering
                nodes_glb = g.global_point_ind[nodes_loc]
                # We then tag each node as a tip node if it is not a global
                # boundary node
                is_tip = np.in1d(nodes_glb, bnd_nodes, invert=True)
                # We reshape the nodes such that each column equals the nodes of
                # one face. If a face only contains global boundary nodes, the
                # local face is also a boundary face. Otherwise, we add a TIP tag.
                n_per_face = _nodes_per_face(g)
                is_tip = np.any(
                    is_tip.reshape((n_per_face, bnd_faces_l.size), order="F"), axis=0
                )

                g.tags["tip_faces"][bnd_faces_l[is_tip]] = True
                domain_boundary_tags = np.zeros(g.num_faces, dtype=bool)
                domain_boundary_tags[bnd_faces_l[is_tip == False]] = True
                g.tags["domain_boundary_faces"] = domain_boundary_tags


def _nodes_per_face(g):
    """
    Returns the number of nodes per face for a given grid
    """
    if ("TensorGrid" in g.name or "CartGrid" in g.name) and g.dim == 3:
        n_per_face = 4
    elif "TetrahedralGrid" in g.name:
        n_per_face = 3
    elif ("TensorGrid" in g.name or "CartGrid" in g.name) and g.dim == 2:
        n_per_face = 2
    elif "TriangleGrid" in g.name:
        n_per_face = 2
    elif ("TensorGrid" in g.name or "CartGrid" in g.name) and g.dim == 1:
        n_per_face = 1
    else:
        raise ValueError(
            "Can not find number of nodes per face for grid: " + str(g.name)
        )
    return n_per_face


def _assemble_in_bucket(grids, **kwargs):
    """
    Create a GridBucket from a list of grids.
    Parameters
    ----------
    grids: A list of lists of grids. Each element in the list is a list
        of all grids of a the same dimension. It is assumed that the
        grids are sorted from high dimensional grids to low dimensional grids.
        All grids must also have the mapping g.global_point_ind which maps
        the local nodes of the grid to the nodes of the highest dimensional
        grid.

    Returns
    -------
    GridBucket: A GridBucket class where the mapping face_cells are given to
        each edge. face_cells maps from lower-dim cells to higher-dim faces.
    """

    # Create bucket
    bucket = GridBucket()
    [bucket.add_nodes(g_d) for g_d in grids]

    # We now find the face_cell mapings.
    for dim in range(len(grids) - 1):
        for hg in grids[dim]:
            # We have to specify the number of nodes per face to generate a
            # matrix of the nodes of each face.
            n_per_face = _nodes_per_face(hg)
            fn_loc = hg.face_nodes.indices.reshape(
                (n_per_face, hg.num_faces), order="F"
            )
            # Convert to global numbering
            fn = hg.global_point_ind[fn_loc]
            fn = np.sort(fn, axis=0)

            for lg in grids[dim + 1]:
                cell_2_face, cell = tools.obtain_interdim_mappings(
                    lg, fn, n_per_face, **kwargs
                )
                if cell_2_face.size > 0:
                    face_cells = sps.csc_matrix(
                        (np.ones(cell.size, dtype=bool), (cell, cell_2_face)),
                        (lg.num_cells, hg.num_faces),
                    )

                    bucket.add_edge([hg, lg], face_cells)

    return bucket


# ------------------------------------------------------------------------------#


def create_mortar_grids(gb, ensure_matching_face_cell=True, **kwargs):

    gb.add_edge_props("mortar_grid")
    # loop on all the nodes and create the mortar grids
    for e, d in gb.edges():
        lg = gb.nodes_of_edge(e)[0]
        # d['face_cells'].indices gives mappings into the lower dimensional
        # cells. Count the number of occurences for each cell.
        num_sides = np.bincount(d["face_cells"].indices)
        # Each cell should be found either twice (think a regular fracture
        # that splits a higher dimensional mesh), or once (the lower end of
        # a T-intersection, or both ends of an L-intersection).
        if ensure_matching_face_cell:
            assert np.all(num_sides == 1) or np.all(num_sides == 2)
        else:
            assert np.max(num_sides) < 3

        # If all cells are found twice, create two mortar grids
        if np.all(num_sides > 1):
            # we are in a two sides situation
            side_g = {
                mortar_grid.LEFT_SIDE: lg.copy(),
                mortar_grid.RIGHT_SIDE: lg.copy(),
            }
        else:
            # the tag name is just a place-holder we assume left side
            side_g = {mortar_grid.LEFT_SIDE: lg.copy()}
        d["mortar_grid"] = mortar_grid.MortarGrid(lg.dim, side_g, d["face_cells"])


# ------------------------------------------------------------------------------#
