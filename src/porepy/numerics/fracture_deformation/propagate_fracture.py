"""
Propagation of fractures. Much in common with (and reuse of) split_grid.
For now assumes:
    single fracture
When this assumption is relieved, some (re)structuring will be needed.
The structure for multi-fracture propagation may possibly strongly resemble
that of split_grid.

WARNING: This should be considered experimental code, which cannot be assumed
to be bug free.

"""

import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.grids import mortar_grid
from porepy.numerics.linalg.matrix_operations import sparse_array_to_row_col_data


def propagate_fractures(
    mdg: pp.MixedDimensionalGrid, faces: Dict[pp.Grid, np.ndarray]
) -> None:
    """
    mdg - Mixed-dimensional grid with matrix and fracture grids.
    faces_h - list of list of faces to be split in the highest-dimensional
        grid. The length of the outer list equals the number of fractures.
        Each entry in the list is a list containing the higher-dimensional
        indices of the faces to be split for the extension of the corresponding
        fracture.
    Changes to grids done in-place.
    The call changes:
        Geometry and connectivity fields of the two grids involved.
        The face_cells mapping between them
        Their respective face tags.
    Also adds the following to subdomain data dictionaries:
        new_cells and new_faces tags, for use in e.g. local discretization
        updates.
        partial_update, a boolean flag indicating that the grids have been
        updated.

    """

    dim_primary: int = mdg.dim_max()
    sd_primary: pp.Grid = mdg.subdomains(dim=dim_primary)[0]

    n_old_faces_h: int = sd_primary.num_faces

    # First initialise certain tags to get rid of any existing tags from
    # previous calls
    data_primary: Dict = mdg.subdomain_data(sd_primary)
    data_primary["new_cells"] = np.empty(0, dtype=int)
    data_primary["new_faces"] = np.empty(0, dtype=int)
    data_primary["split_faces"] = np.empty(0, dtype=int)

    # Data structure for keeping track of faces in sd_primary to be split
    split_faces = np.empty(0, dtype=int)

    # By default, we will not update the higher-dimensional grid. This will be
    # changed in the below for loop if the grid gets faces split.
    # This variable can be used e.g. to check if a rediscretization is necessary on
    # the higher-dimensional grid
    data_primary["partial_update"] = False

    # Initialize mapping between old and new faces for sd_primary. We will store the updates
    # from splitting related to each lower-dimensional grid, and then merge towards the
    # end; the split data may be handy for debugging
    face_map_h: List[sps.spmatrix] = [
        sps.dia_matrix(
            (np.ones(sd_primary.num_faces), 0),
            (sd_primary.num_faces, sd_primary.num_faces),
        )
    ]

    # The propagation is divided into two main steps:
    # First, update the geometry of the fracture grids, and, simultaneously, the higher
    # dimensional grid (the former will be updated once, the latter may undergo several
    # update steps, depending on how many fractures propagate).
    # Second, update the mortar grids. This is done after all fractures have been
    # propagated.

    for sd_secondary in mdg.subdomains(dim=dim_primary - 1):
        # The propagation of a fracture consists of the following major steps:
        #   1. Find which faces in sd_primary should be split for this sd_secondary.
        #   2. Add nodes to sd_secondary where the fracture will propagate.
        #   3. Update face-node and cell-face relation in sd_secondary.
        #   4. Update face geometry of sd_secondary.
        #   5. Update cell geometry of sd_secondary.
        #   6. Split the faces in sd_primary to make room for the new fracture.
        #   7. Update geometry in sd_secondary and sd_primary.
        #
        # IMPLEMENTATION NOTE: While point 7 replaces information from 4 and 5, the
        # provisional fields may still be needed in point 6.

        # Initialize data on new faces and cells
        data_secondary = mdg.subdomain_data(sd_secondary)
        data_secondary["new_cells"] = np.empty(0, dtype=int)
        data_secondary["new_faces"] = np.empty(0, dtype=int)

        # Step 1:
        # Uniquify the faces to be split. Among others, this avoids trouble when
        # a face is requested split twice, from two neighboring faces
        faces_h = np.unique(np.atleast_1d(np.array(faces[sd_secondary])))
        split_faces = np.append(split_faces, faces_h)

        if faces_h.size == 0:
            # If there is no propagation for this fracture, we continue
            # No need to update discretization of this grid
            data_secondary["partial_update"] = False

            # Variable mappings are unit mappings
            data_secondary["face_index_map"] = sps.identity(sd_secondary.num_faces)
            data_secondary["cell_index_map"] = sps.identity(sd_secondary.num_cells)

            # Identity mapping of faces in this step
            face_map_h.append(sps.identity(sd_primary.num_faces))

            # Move on to the next fracture
            continue

        # Keep track of original information:
        n_old_faces_l = sd_secondary.num_faces
        n_old_cells_l = sd_secondary.num_cells
        n_old_nodes_l = sd_secondary.num_nodes
        n_old_nodes_h = sd_primary.num_nodes

        # It is convenient to tag the nodes lying on the domain boundary. This
        # helps updating the face tags later:
        pp.utils.tags.add_node_tags_from_face_tags(mdg, "domain_boundary")

        # Step 2:
        # Get the "involved nodes", i.e., the union between the new nodes in
        # the lower dimension and the boundary nodes where the fracture
        # propagates. The former are added to the nodes in sd_secondary - specifically,
        # both node coordinates and global_point_ind of sd_secondary are amended.
        (
            unique_node_indata_secondary,
            unique_node_indata_primary,
        ) = _update_nodes_fracture_grid(sd_primary, sd_secondary, faces_h)

        # Step 3:
        # Update the connectivity matrices (cell_faces and face_nodes) and tag
        # the lower-dimensional faces, including re-classification of (former)
        # tips to internal faces, where appropriate.
        n_new_faces, new_face_centers = _update_connectivity_fracture_grid(
            sd_secondary,
            sd_primary,
            unique_node_indata_secondary,
            unique_node_indata_primary,
            n_old_nodes_l,
            n_old_faces_l,
            n_old_cells_l,
            faces_h,
        )

        # Step 4: Update fracture grid face geometry
        # Note: This simply expands arrays with face geometry, but it does not
        # compute reasonable values for the geometry
        _append_face_geometry_fracture_grid(sd_secondary, n_new_faces, new_face_centers)

        # Step 5: Update fracture grid cell geometry
        # Same for cells. Here the geometry quantities are copied from the
        # face values of sd_primary, thus values should be reasonable.
        new_cells: np.ndarray = _update_cells_fracture_grid(
            sd_primary, sd_secondary, faces_h
        )

        # Step 6: Split sd_primary along faces_h
        _split_fracture_extension(
            mdg,
            sd_primary,
            sd_secondary,
            faces_h,
            unique_node_indata_primary,
            new_cells,
            non_planar=True,
        )

        # Store information on which faces and cells have just been added.
        # Note that we only keep track of the faces and cells from the last
        # propagation call!
        new_faces_l = np.arange(
            sd_secondary.num_faces - n_new_faces, sd_secondary.num_faces
        )
        new_faces_h = sd_primary.frac_pairs[
            1, np.isin(sd_primary.frac_pairs[0], faces_h)
        ]

        # Sanity check on the grid; most likely something will have gone wrong
        # long before if there is a problem.
        assert np.all(new_faces_h >= n_old_faces_h)
        if not np.min(new_cells) >= n_old_cells_l:
            raise ValueError("New cells are assumed to be appended to cell array")
        if not np.min(new_faces_l) >= n_old_faces_l:
            raise ValueError("New faces are assumed to be appended to face array")

        # Update the geometry
        _update_geometry(
            sd_primary, sd_secondary, new_cells, n_old_cells_l, n_old_faces_l
        )

        # Finally, some bookkeeping that can become useful in a larger-scale simulation.

        # Mark both grids for a partial update
        data_primary["partial_update"] = True
        data_secondary["partial_update"] = True

        # Append arrays of new faces (sd_secondary, sd_primary) and cells (sd_secondary)
        data_primary["new_faces"] = np.append(data_primary["new_faces"], new_faces_h)
        data_secondary["new_cells"] = np.append(data_secondary["new_cells"], new_cells)
        data_secondary["new_faces"] = np.append(
            data_secondary["new_faces"], new_faces_l
        )

        # Create mappings between the old and new faces and cells in sd_secondary
        arr = np.arange(n_old_faces_l)
        face_map_l = sps.coo_matrix(
            (np.ones(n_old_faces_l, dtype=int), (arr, arr)),
            shape=(sd_secondary.num_faces, n_old_faces_l),
        ).tocsr()
        arr = np.arange(n_old_cells_l)
        cell_map_l = sps.coo_matrix(
            (np.ones(n_old_cells_l, dtype=int), (arr, arr)),
            shape=(sd_secondary.num_cells, n_old_cells_l),
        ).tocsr()

        # These can be stored directly - there should be no more changes for sd_secondary
        data_secondary["face_index_map"] = face_map_l
        data_secondary["cell_index_map"] = cell_map_l

        # For sd_primary we construct the map of faces for the splitting of this sd_secondary
        # and append it to the list of face_maps

        # The size of the next map should be compatible with the number of faces in
        # the previous map.
        nfh = face_map_h[-1].shape[0]
        arr = np.arange(nfh)
        face_map_h.append(
            sps.coo_matrix(
                (np.ones(nfh, dtype=int), (arr, arr)),
                shape=(sd_primary.num_faces, nfh),
            ).tocsr()
        )

        # Append default tags for the new nodes. Both high and low-dimensional grid
        _append_node_tags(sd_secondary, sd_secondary.num_nodes - n_old_nodes_l)
        _append_node_tags(sd_primary, sd_primary.num_nodes - n_old_nodes_h)

    # The standard node tags are updated from the face tags, which are updated on the
    # fly in the above loop.
    node_tags = ["domain_boundary", "tip", "fracture"]
    for tag in node_tags:
        # The node tag is set to true if at least one neighboring face is tagged
        pp.utils.tags.add_node_tags_from_face_tags(mdg, tag)
    # Done with all splitting.

    # Compose the mapping of faces for sd_secondary
    fm = face_map_h[0]
    for m in face_map_h[1:]:
        fm = m * fm
    data_primary["face_index_map"] = fm
    # Also make a cell-map, this is a 1-1 mapping in this case
    data_primary["cell_index_map"] = sps.identity(sd_primary.num_cells)

    data_primary["split_faces"] = np.array(split_faces, dtype=int)

    ##
    # Second main step of propagation: Update mortar grid.

    # When all faces have been split, we can update the mortar grids
    for intf_old in mdg.subdomain_to_interfaces(sd_primary):
        data_edge = mdg.interface_data(intf_old)
        _, sd_secondary = mdg.interface_to_subdomain_pair(intf_old)
        data_secondary = mdg.subdomain_data(sd_secondary)
        intf_old = _update_mortar_grid(
            sd_primary,
            sd_secondary,
            intf_old,
            data_edge,
            data_secondary["new_cells"],
            data_primary["new_faces"],
        )

        # Get hold of the new interface data dictionary, in case something happened with
        # the mapping when replacing the interface.
        data_edge = mdg.interface_data(intf_old)

        # Mapping of cell indices on the mortar grid is composed by the corresponding
        # map for sd_secondary.
        cell_map = sps.kron(sps.identity(2), data_secondary["cell_index_map"]).tocsr()
        data_edge["cell_index_map"] = cell_map

        # Also update projection operators
        pp.set_local_coordinate_projections(mdg, [intf_old])


def _update_mortar_grid(
    sd_primary: pp.Grid,
    sd_secondary: pp.Grid,
    intf: pp.MortarGrid,
    d_e: Dict[str, Any],
    new_cells,
    new_faces_h,
):
    # Face-cell map. This has been updated during splitting, thus it has
    # the shapes of the new grids
    face_cells = d_e["face_cells"]

    cells, faces, _ = sparse_array_to_row_col_data(face_cells)

    # If this is ever broken, we have a problem
    other_side_old = intf._ind_face_on_other_side

    other_side_new = np.copy(other_side_old)

    # Make sure that the + and - side of the new mortar cells is
    # coherent with those already in place. This may not be strictly
    # necessary, as the normal vectors of the grid will be adjusted
    # locally to the +- convention, however, it will ease the interpretation
    # of results, including debugging.

    #
    for ci in new_cells:
        # Find the occurrences of this new cell in the face-cell map.
        # There should be exactly two of these.
        hit = np.where(ci == cells)[0]
        assert hit.size == 2
        # Find the faces in the higher-dimensional grid that correspond
        # to this new cell
        loc_faces = faces[hit]

        # The new faces will be on each side of the fracture, and
        # there will be at least one node not shared by the faces.
        # We need to pick one of the faces, and find its neighboring
        # faces along the fracture, on the same side of the fracture.
        # The sign of the new face (in the mortar grid) will be the
        # same as the old one

        # We need to focus on split nodes, or else we risk finding neighboring
        # faces on both sides of the fracture.
        # Nodes of both local faces
        local_nodes_0 = sd_primary.face_nodes[:, loc_faces[0]].indices
        local_nodes_1 = sd_primary.face_nodes[:, loc_faces[1]].indices

        # Nodes that belong only to the first local face
        local_nodes_0_only = np.setdiff1d(local_nodes_0, local_nodes_1)

        # Get the other faces of these nodes. These will include both faces
        # on the fracture, and faces internal to sd_primary
        _, other_faces, _ = sparse_array_to_row_col_data(
            sd_primary.face_nodes[local_nodes_0_only]
        )

        # Pick those of the other faces that were not added during splitting
        old_other_faces = np.setdiff1d(other_faces, new_faces_h)

        if np.any(np.isin(old_other_faces, other_side_old)):
            other_side_new = np.append(other_side_new, loc_faces[0])
        else:
            other_side_new = np.append(other_side_new, loc_faces[1])

    # The new mortar grid is constructed to be matching with sd_secondary.
    # If splitting is undertaken for a non-matching grid, all bets are off.
    side_grids = {
        mortar_grid.MortarSides.LEFT_SIDE: sd_secondary,
        mortar_grid.MortarSides.RIGHT_SIDE: sd_secondary,
    }
    mg_new = pp.MortarGrid(
        sd_secondary.dim,
        side_grids,
        d_e["face_cells"],
        face_duplicate_ind=other_side_new,
    )
    # Update old grid with values from the new one. This is similar to redoing initialization.
    fields = ["side_grids", "sides", "num_cells", "cell_volumes", "cell_centers"]
    for field in fields:
        setattr(intf, field, getattr(mg_new, field))
    intf._init_projections(d_e["face_cells"], other_side_new)
    intf._set_projections()
    return intf


def _update_geometry(
    sd_primary: pp.Grid,
    sd_secondary: pp.Grid,
    new_cells: np.ndarray,
    n_old_cells_l: int,
    n_old_faces_l: int,
) -> None:
    # Update geometry on each iteration to ensure correct tags.

    # The geometry of the higher-dimensional grid can be computed straightforwardly.
    sd_primary.compute_geometry()

    if sd_primary.dim == 2:
        # 1d geometry computation is valid also for manifolds
        sd_secondary.compute_geometry()
    else:
        # The implementation of 2d compute_geometry() assumes that the
        # grid is planar. The simplest option is to treat one cell at
        # a time, and then merge the arrays at the end.

        # Initialize arrays for geometric quantities
        fa = np.empty(0)  # Face areas
        fc = np.empty((3, 0))  # Face centers
        fn = np.empty((3, 0))  # Face normals
        cv = np.empty(0)  # Cell volumes
        cc = np.empty((3, 0))  # Cell centers
        # Many of the faces will have their quantities computed twice,
        # once from each side. Keep track of which faces we are dealing with
        face_ind = np.array([], dtype=int)

        for ci in new_cells:
            sub_g, face_indata_secondaryoc, _ = pp.partition.extract_subgrid(
                sd_secondary, ci
            )
            sub_g.compute_geometry()

            fa = np.append(fa, sub_g.face_areas)
            fc = np.append(fc, sub_g.face_centers, axis=1)
            fn = np.append(fn, sub_g.face_normals, axis=1)
            cv = np.append(cv, sub_g.cell_volumes)
            cc = np.append(cc, sub_g.cell_centers, axis=1)

            face_ind = np.append(face_ind, face_indata_secondaryoc)

        # The new cell geometry is composed of values from the previous grid, and
        # the values computed one by one for the new cells
        sd_secondary.cell_volumes = np.hstack(
            (sd_secondary.cell_volumes[:n_old_cells_l], cv)
        )
        sd_secondary.cell_centers = np.hstack(
            (sd_secondary.cell_centers[:, :n_old_cells_l], cc)
        )

        # For the faces, more work is needed
        face_areas = np.zeros(sd_secondary.num_faces)
        face_centers = np.zeros((3, sd_secondary.num_faces))
        face_normals = np.zeros((3, sd_secondary.num_faces))

        # For the old faces, transfer already computed values
        face_areas[:n_old_faces_l] = sd_secondary.face_areas[:n_old_faces_l]
        face_centers[:, :n_old_faces_l] = sd_secondary.face_centers[:, :n_old_faces_l]
        face_normals[:, :n_old_faces_l] = sd_secondary.face_normals[:, :n_old_faces_l]

        for fi in range(n_old_faces_l, sd_secondary.num_faces):
            # Geometric quantities for this face
            hit = np.where(face_ind == fi)[0]
            # There should be 1 or 2 hits
            assert hit.size > 0 and hit.size < 3

            # For areas and centers, the computations based on the two neighboring
            # cells should give the same result. Check, and then use the value.
            mean_area = np.mean(fa[hit])
            mean_center = np.mean(fc[:, hit], axis=1)
            assert np.allclose(fa[hit], mean_area)
            assert np.allclose(fc[:, hit], mean_center.reshape((3, 1)))
            face_areas[fi] = mean_area
            face_centers[:, fi] = mean_center

            # The normal is more difficult, since this is not unique.
            # The direction of the normal vectors computed from subgrids should be
            # consistent with the +- convention in the main grid.

            # Normal vectors found for this global face
            normals = fn[:, hit]
            if normals.size == 3:
                normals = normals.reshape((3, 1))

            # For the moment, use the mean of the two values.
            mean_normal = np.mean(normals, axis=1)

            face_normals[:, fi] = mean_normal / np.linalg.norm(mean_normal) * mean_area

        # Sanity check
        # assert np.allclose(np.linalg.norm(face_normals, axis=0), face_areas)

        # Store computed values
        sd_secondary.face_areas = face_areas
        sd_secondary.face_centers = face_centers
        sd_secondary.face_normals = face_normals


def _update_connectivity_fracture_grid(
    sd_secondary: pp.Grid,  # higher dimensional grid
    sd_primary: pp.Grid,  # lower dimensional grid
    nodes_l: np.ndarray,  # nodes in sd_primary involved in the propagation
    nodes_h: np.ndarray,  # nodes in sd_secondary involved in the propagation
    n_old_nodes_l: int,  # number of nodes in sd_secondary before splitting
    n_old_faces_l: int,  # number of faces in sd_secondary before splitting
    n_old_cells_l: int,  # number of cells in sd_secondary before splitting
    faces_h,  # faces in sd_primary to be split
) -> Tuple[int, np.ndarray]:
    """
    Update of cell_faces of the lower grid after insertion of new cells at the
    higher-dimensional faces_h. Also tags the faces as domain_boundary or tip
    Should be called after initialization of tags
    and geometry of sd_secondary by append_face_geometry and append_face_tags.

    """
    # Extract immediate information

    # Each split face gives a new cell in sd_secondary
    n_new_cells_l = faces_h.size
    # index of the new cells in sd_secondary. These are appended to the existing cells
    new_cells_l = np.arange(n_old_cells_l, n_old_cells_l + n_new_cells_l)

    # Initialize fields for new faces in sd_secondary
    new_faces_l = np.empty((sd_secondary.dim, 0), dtype=int)
    new_face_centers_l = np.empty((3, 0))

    # Counter of total number of faces in sd_secondary
    face_counter_l = n_old_faces_l

    # Copy what is to be updated: Cell-face and face-node relation in sd_secondary
    old_cell_faces = sd_secondary.cell_faces.copy()
    old_face_nodes = sd_secondary.face_nodes.copy()

    # Get the face_node indices to form lower-dimensional faces on the form
    # [[nodes of face 1], [nodes of face 2], ...], i.e., array where each face
    # is represented by the nodes it consists of.
    # ASSUMPTION: This breaks if not all faces have the same number of cells
    # Rewrite is possible, but more technical
    all_faces_l = np.reshape(
        sd_secondary.face_nodes.indices, (sd_secondary.dim, n_old_faces_l), order="F"
    )

    # Initialize indices and values for the cell_faces update
    ind_f, ind_c, cf_val = (
        np.empty(0, dtype=int),
        np.empty(0, dtype=int),
        np.empty(0, dtype=int),
    )
    # and for the face_nodes update
    fn_ind_f, fn_ind_n = np.empty(0, dtype=int), np.empty(0, dtype=int)

    # Loop over all new cells to be created
    for i, c in enumerate(new_cells_l):
        # Find the nodes of the corresponding higher-dimensional face
        face_h = faces_h[i]
        local_nodes_h = sd_primary.face_nodes[:, face_h].nonzero()[0]

        # Find the nodes' place among the active higher-dimensional nodes, that is,
        # nodes that will be split
        in_unique_nodes = pp.utils.setmembership.ismember_rows(
            local_nodes_h, nodes_h, sort=False
        )[1]

        # Find the corresponding lower-dimensional nodes
        local_nodes_l = np.array(nodes_l[in_unique_nodes], dtype=int)

        # Get geometry information
        local_pts = sd_secondary.nodes[:, local_nodes_l]
        # The new cell center is taken as the mean of the node coordinates.
        # This should be okay for simplexes, not sure what we get for general cells.
        local_cell_center = np.mean(local_pts, axis=1)

        # Store face center for the update of sd_secondary.face_centers

        # Faces are defined by one node in 1d and two in 2d. This requires
        # dimension-dependent treatment:
        if sd_secondary.dim == 2:
            # Sort nodes clockwise (!)
            # ASSUMPTION: This assumes that the new cell is star-shaped with respect to the
            # local cell center. This should be okay.
            map_to_sorted = pp.utils.sort_points.sort_point_plane(
                local_pts, local_cell_center
            )
            sorted_nodes_l = local_nodes_l[map_to_sorted]
            sorted_nodes_h = local_nodes_h[map_to_sorted]

            # Define the faces of the new cell c (size: 2 x faces_per_cell_l). "Duplicate"
            # of the higher dimension used for tag identification.
            faces_l = np.vstack(
                (sorted_nodes_l, np.append(sorted_nodes_l[1:], sorted_nodes_l[0]))
            )
            local_faces_h = np.vstack(
                (sorted_nodes_h, np.append(sorted_nodes_h[1:], sorted_nodes_h[0]))
            )

        else:
            # Faces and nodes are 1:1, but ismember_rows (below) requires 2d array
            faces_l = np.atleast_2d(local_nodes_l)
            local_faces_h = np.atleast_2d(local_nodes_h)

        # Now the faces of c are defined by sorted_nodes_l
        # and their arrangement in faces_l.
        n_local_faces_l = faces_l.shape[-1]

        # Check which faces exist in sd_secondary already, either from before propgation
        # or from previous runs through current loop:
        (exist, existing_faces_l) = pp.utils.setmembership.ismember_rows(
            faces_l, all_faces_l
        )
        # The existing faces are no longer tips (but internal).
        sd_secondary.tags["tip_faces"][existing_faces_l] = False

        # Number of genuinely new local faces in sd_secondary created for this cell
        n_new_local_faces_l = int(
            np.sum(~exist)
        )  # mypy complains if int() is not added

        # Index of the new faces, they will be appended to the face array
        new_face_indices_l = np.arange(
            face_counter_l, face_counter_l + n_new_local_faces_l
        )
        # Update face counter to be ready for the next cell
        face_counter_l += n_new_local_faces_l

        ## Assign tags to the new faces
        # First expand tag arrays to make space for new faces
        _append_face_tags(sd_secondary, n_new_local_faces_l)

        # The existing faces are tagged according to the information from the
        # node tags of sd_primary.
        fi = local_faces_h[:, ~exist]

        # The new faces are either on the domain boundary, or tip faces
        domain_boundary_faces = np.all(
            sd_primary.tags["domain_boundary_nodes"][fi], axis=0
        )
        sd_secondary.tags["tip_faces"][new_face_indices_l] = ~domain_boundary_faces
        sd_secondary.tags["domain_boundary_faces"][
            new_face_indices_l
        ] = domain_boundary_faces

        # Expand array of face-nodes in sd_secondary
        all_faces_l = np.append(all_faces_l, faces_l[:, ~exist], axis=1)

        # Find node indices faces to be updated.
        ind_n_local = faces_l[:, ~exist]
        # TODO: What happens here if ~exist is more than one face?
        local_pts = sd_secondary.nodes[:, ind_n_local]
        local_face_centers = np.mean(local_pts, axis=1)

        # New face center set to the mean of the face's vertexes.
        # This is reasonable at least for simplex (and Cartesian) faces
        new_face_centers_l = np.append(
            new_face_centers_l, np.atleast_2d(local_face_centers), axis=1
        )
        new_faces_l = np.append(new_faces_l, ind_n_local, axis=1)

        # Expand face-node and cell-face relations
        # Build index of all local faces (both new, and already existing)
        all_local_faces = np.empty(faces_l.shape[-1])
        all_local_faces[exist] = existing_faces_l
        all_local_faces[~exist] = new_face_indices_l

        # Add both existing and new faces to face-nodes.
        # Why include exist here, they should have been added already?
        # Answer: We could have dropped it, but this will simply add the same
        # information twice to the face-node relation. Since this has boolean
        # data, adding a 2 instead of a 1 will make no difference.
        ind_f_local = np.tile(all_local_faces, sd_secondary.dim)
        fn_ind_f = np.append(fn_ind_f, ind_f_local)
        fn_ind_n = np.append(fn_ind_n, faces_l)

        # Cell-face relation
        # Here all faces should be added, existing or not
        ind_f = np.append(ind_f, all_local_faces)
        ind_c = np.append(ind_c, c * np.ones(n_local_faces_l))

        # To get the sign correct, some work is needed.
        # We distinguish between three cases
        # 1) This is a new face. We will assign positive sign, thus outer normal
        # 2) This is a face which existed before we entered the loop over
        #    cells. The sign will be oposite of that used in the previous occurence
        #    of the face
        # 3) This is a face that has been added before for a previous new cell.
        #    The sign will be oposite of when first added, that is, -1.
        cf_val_loc = np.zeros(n_local_faces_l)
        # The faces that did not exist, are assigned sign 1
        # (should get outer normal)
        cf_val_loc[~exist] = 1

        # Find faces that were in the original grid (before entering the outer loop
        # over cells)
        are_in_original = existing_faces_l < n_old_faces_l

        # Faces that existed before the cell loop
        ind_in_original = existing_faces_l[are_in_original]
        # Index of these faces in cf_val_loc
        indata_secondaryocal = np.isin(all_local_faces, ind_in_original)

        if sd_secondary.cell_faces[ind_in_original].data.size != ind_in_original.size:
            # This situation can happen in 3d (perhaps also 2d).
            # It will likely correspond to a strangly shaped fracture.
            # Implementation of such geometries seems complex, if at all desirable.
            # The suggested solution is to patch the face splitting algorithm so that
            # this does not happen.
            raise ValueError("Cannot split the same lower-dimensional face twice")

        # The sign of this cell should be the oposite of that used in the
        # original grid.
        cf_val_loc[indata_secondaryocal] = -sd_secondary.cell_faces.tocsr()[
            ind_in_original, :
        ].data

        # Faces that were not in the original grid, but were added before this iteration
        # of the cell loop
        ind_not_in_original = existing_faces_l[~are_in_original]
        # Index of these faces in cf_val_loc
        ind_not_local = np.isin(all_local_faces, ind_not_in_original)
        # These are assigned the value -1; since it was given +1 when first added
        # to cf_val (see call cf_val_loc[~exist] above)
        cf_val_loc[ind_not_local] = -1

        # Store signs of cf_val. This effectively is the sign of the normal vectors
        cf_val = np.append(cf_val, cf_val_loc)

    # Done with the expansion of all faces and cells. What is left is to update
    # sd_secondary.cell_faces and face_nodes.

    # Resize and update face_nodes ...
    sd_secondary.face_nodes = sps.csc_matrix(
        (sd_secondary.num_nodes, face_counter_l), dtype=bool
    )
    sd_secondary.face_nodes[:n_old_nodes_l, :n_old_faces_l] = old_face_nodes
    sd_secondary.face_nodes[fn_ind_n, fn_ind_f] = True
    sd_secondary.face_nodes.eliminate_zeros()

    # ... and cell_faces
    sd_secondary.cell_faces = sps.csc_matrix(
        (face_counter_l, n_old_cells_l + n_new_cells_l)
    )
    sd_secondary.cell_faces[0:n_old_faces_l, 0:n_old_cells_l] = old_cell_faces
    sd_secondary.cell_faces[ind_f, ind_c] = cf_val
    sd_secondary.cell_faces.eliminate_zeros()
    n_new_faces = face_counter_l - n_old_faces_l
    return n_new_faces, new_face_centers_l


def _update_cells_fracture_grid(
    sd_primary: pp.Grid, sd_secondary: pp.Grid, faces_h: np.ndarray
) -> np.ndarray:
    """
    Cell information for sd_secondary is inherited directly from the higher-dimensional
    faces we are splitting. The function updates num_cells, cell_centers and
    cell_volumes.
    """
    n_new_cells = sd_secondary.num_cells + faces_h.size
    new_cells = np.arange(sd_secondary.num_cells, n_new_cells)
    sd_secondary.num_cells = n_new_cells
    sd_secondary.cell_centers = np.append(
        sd_secondary.cell_centers, sd_primary.face_centers[:, faces_h], axis=1
    )
    sd_secondary.cell_volumes = np.append(
        sd_secondary.cell_volumes, sd_primary.face_areas[faces_h]
    )
    return new_cells


def _update_nodes_fracture_grid(
    sd_primary: pp.Grid, sd_secondary: pp.Grid, faces_h: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the nodes in the lower-dimensional grid corresponding to the higher-
    dimensional faces to be split. Updates node information in sd_secondary:
        global_point_ind
        nodes
        num_nodes

    Returns:
        unique_nodes_l - numpy array (number of involved nodes x 1) Indices of
            the nodes (as arranged in sd_secondary.nodes).
        unique_nodes_h - same, but corresponding to sd_primary.nodes.

    """
    # Nodes of sd_primary to be split
    nodes_h = sd_primary.face_nodes[:, faces_h].nonzero()[0]
    unique_nodes_h = np.unique(nodes_h)

    # Global index of nodes to split
    unique_global_nodes = sd_primary.global_point_ind[unique_nodes_h]

    # Some of the nodes of the face to be split will be in sd_secondary already (as tip nodes)
    # Find which are present, and which should be added
    # NOTE: This comparison must be done in terms of global_point_ind
    are_old_global_nodes_in_l = np.isin(
        unique_global_nodes, sd_secondary.global_point_ind
    )
    are_new_global_nodes_in_l = np.logical_not(are_old_global_nodes_in_l)

    # Index in sd_primary, of nodes to be added to sd_secondary
    new_node_indices_h = unique_nodes_h[are_new_global_nodes_in_l]
    # Global indices of nodes to be added to sd_secondary
    new_global_node_indices_l = unique_global_nodes[are_new_global_nodes_in_l]

    # Append nodes to sd_secondary and update bookkeeping
    new_nodes_l = sd_primary.nodes[:, new_node_indices_h].copy()
    sd_secondary.nodes = np.append(sd_secondary.nodes, new_nodes_l, axis=1)
    n_new_nodes = new_nodes_l.shape[1]
    sd_secondary.num_nodes += n_new_nodes

    # Append global point indices to sd_secondary
    sd_secondary.global_point_ind = np.append(
        sd_secondary.global_point_ind, new_global_node_indices_l
    )

    # Find index of the updated nodes in sd_secondary that belong to the split faces
    # Order preserving find:
    unique_nodes_l = pp.utils.setmembership.ismember_rows(
        unique_global_nodes, sd_secondary.global_point_ind, sort=False
    )[1]

    return unique_nodes_l, unique_nodes_h


def _append_face_geometry_fracture_grid(
    g: pp.Grid, n_new_faces: int, new_centers: np.ndarray
) -> None:
    """
    Appends and updates faces geometry information for new faces. Also updates
    num_faces.
    """
    g.face_normals = np.append(g.face_normals, np.zeros((3, n_new_faces)), axis=1)
    g.face_areas = np.append(g.face_areas, np.ones(n_new_faces))
    g.face_centers = np.append(g.face_centers, new_centers, axis=1)
    g.num_faces += n_new_faces


def _append_face_tags(g, n_new_faces):
    """
    Initiates default face tags (False) for new faces.
    """
    keys = pp.utils.tags.standard_face_tags()
    new_tags = [np.zeros(n_new_faces, dtype=bool) for _ in range(len(keys))]
    pp.utils.tags.append_tags(g.tags, keys, new_tags)


def _append_node_tags(g, n_new_nodes):
    """
    Initiates default face tags (False) for new faces.
    """
    keys = pp.utils.tags.standard_node_tags()
    new_tags = [np.zeros(n_new_nodes, dtype=bool) for _ in range(len(keys))]
    pp.utils.tags.append_tags(g.tags, keys, new_tags)


def _split_fracture_extension(
    mdg: pp.MixedDimensionalGrid,
    sd_primary: pp.Grid,
    sd_secondary: pp.Grid,
    faces_h: np.ndarray,
    nodes_h: np.ndarray,
    cells_l: np.ndarray,
    non_planar: bool = False,
):
    """
    Split the higher-dimensional grid along specified faces. Updates made to
    face_cells of the grid pair and the nodes and faces of the higher-
    dimensional grid.
    Parameters
    ----------
    mdg                 - A grid mixed-dimensional grid
    sd_primary          - Higher-dimensional grid to be split along specified faces.
    sd_secondary          - Immersed lower-dimensional grid.
    faces_h     - The higher-dimensional faces to be split.
    cells_l     - The corresponding lower-dimensional cells.
    nodes_h     - The corresponding (hisd_primaryer-dimensional) nodes.

    """
    # IMPLEMENTATION NOTE: Part of the following code is likely more general than
    # necessary considering assumptions made before we reach this point - e.g.
    # assumptions in propagate_fractures() and other subfunctions. Specifically,
    # it is unlikely the code will be called with sd_primary.dim != mdg.dim_max().

    # We are splitting faces in sd_primary. This affects all the immersed fractures,
    # as face_cells has to be extended for the new faces_h.
    neigh = np.array(mdg.neighboring_subdomains(sd_primary))

    # Find the neighbours that are lower dimensional
    is_low_dim_grid = np.where([w.dim < sd_primary.dim for w in neigh])
    low_dim_neigh = neigh[is_low_dim_grid]
    sd_pairs = [(sd_primary, w) for w in low_dim_neigh]
    sd_secondary_arr = np.nonzero(low_dim_neigh == sd_secondary)[0]

    # Some work is needed to make mypy happy here
    assert sd_secondary_arr.size == 1
    # This converts form numpy int to a standard int.
    sd_secondary_ind = sd_secondary_arr[0].item()

    if len(sd_pairs) == 0:
        # No lower dim grid. Nothing to do.
        warnings.warn("Unexpected neighbourless sd_primary in fracture propagation")
        return

    face_cell_list: List[sps.spmatrix] = []
    intf_list: List[pp.MortarGrid] = []

    for pair in sd_pairs:
        intf = mdg.subdomain_pair_to_interface(pair)
        data = mdg.interface_data(intf)
        face_cell_list.append(data["face_cells"])
        intf_list.append(intf)

    # We split all the faces that are connected to faces_h
    # The new faces will share the same nodes and properties (normals, etc.)
    face_cell_list = pp.fracs.split_grid.split_specific_faces(
        sd_primary, face_cell_list, faces_h, cells_l, sd_secondary_ind, non_planar
    )

    # Replace the face-cell relation on the MixedDimensionalGrid edge
    for intf, f in zip(intf_list, face_cell_list):
        mdg.interface_data(intf)["face_cells"] = f

    # We now find which lower-dim nodes correspond to which higher-
    # dim nodes. We split these nodes according to the topology of
    # the connected higher-dim cells. At a X-intersection we split
    # the node into four, while at the fracture boundary it is not split.
    pp.fracs.split_grid.split_nodes(sd_primary, [sd_secondary], [nodes_h])

    # Remove zeros from cell_faces
    for g in mdg.subdomains():
        g.cell_faces.eliminate_zeros()


def _tag_affected_cells_and_faces(mdg):
    """
    Tag the lower-dimensional cells and higher-dimensional faces which have
    been affected by the update. Should be the new cells, and both the original
    (defining the split, see e.g. faces_h in propagate_fracture) and the newly
    created faces.
    Assumes only two dimensions.
    """
    dim_primary = mdg.dim_max()
    dim_secondary = mdg.dim_min()
    sd_primary = mdg.subdomains(dim=dim_primary)[0]
    sd_secondary = mdg.subdomains(dim=dim_secondary)[0]
    data_primary = mdg.subdomain_data(sd_primary)
    data_secondary = mdg.subdomain_data(sd_secondary)
    cells_l = data_secondary["new_cells"]
    faces_h = data_primary["new_faces"]
    old_faces_h = sd_primary.frac_pairs[0, np.isin(sd_primary.frac_pairs[1], faces_h)]
    all_faces_h = np.concatenate((old_faces_h, faces_h))
    t = np.zeros(sd_primary.num_faces, dtype=bool)
    t[all_faces_h] = True
    sd_primary.tags["discretize_faces"] = t

    # TODO: Fix tpfa, so that local 1d update is possible (MPFA calls TPFA for
    # 1d). Once fixed, change to t = np.zeros(...) in this line:
    t = np.ones(sd_secondary.num_cells, dtype=bool)
    t[cells_l] = True
    sd_secondary.tags["discretize_cells"] = t


def propgation_angle(K):
    """
    Compute the angle of propagation from already computed SIFs. The
    computation is done in the local coordinate system of the fracture tips,
    and positive angles correspond to.
    Intended for a single fracture grid.

    Parameters:
        K: array of stress intensity factors, with mode along first axis and
            face along second.

    Returns:
        phi: array (number of tip faces, ) of propagation angles in radians.
    """
    A = 140 / 180 * np.pi
    B = -70 / 180 * np.pi
    if K.shape[0] == 2:
        K = np.vstack([K, np.zeros((1, K.shape[1]))])
    aK = np.absolute(K)
    phi = A * aK[1] / (K[0] + aK[1] + aK[2]) + B * np.square(
        aK[2] / (K[0] + aK[1] + aK[2])
    )
    neg_ind = K[1] > 0  # ?
    phi[neg_ind] = -phi[neg_ind]
    return phi
