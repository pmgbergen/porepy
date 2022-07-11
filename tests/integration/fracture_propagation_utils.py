#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:04:14 2018

@author: ivar
Utility functions for propagation of fractures and comparison of the changed
grids, discretizations, parameters and bcs.
Note: Several of the functions assume single-fracture domain with two
dimensions only.
"""


import numpy as np
import porepy as pp
from porepy.utils import tags
from porepy.utils import setmembership as sm
from porepy.numerics.fv import fvutils


def propagate_and_update(mdg, faces, discr, update_bc, update_apertures):
    """
    Perform the grid update defined by splitting of faces in the highest
    dimension. Then re-discretize locally and return the discretization
    matrices.
    Updates on BCs and apertures depend on the problem setup, and are kept
    out of the propagate_fracture module.
    """
    faces = np.array(faces)
    if faces.ndim < 2:
        pp.propagate_fracture.propagate_fractures(mdg, [faces])
    else:
        pp.propagate_fracture.propagate_fractures(mdg, faces)
    pp.propagate_fracture.tag_affected_cells_and_faces(mdg)
    gl = mdg.subdomains(dim=mdg.dim_min())[0]
    update_apertures(mdg, gl, faces)
    update_bc(mdg)
    if discr.physics == "flow":
        lhs_out, rhs_out = discr.matrix_rhs(mdg)
    if discr.physics == "mechanics":
        g = mdg.subdomains(dim=mdg.dim_max())[0]
        d = mdg.node_props(g)
        lhs_out, rhs_out = discr.matrix_rhs(g, d)
    return lhs_out, rhs_out


def compare_updates(
    md_grids, lh, rh, phys="mechanics", parameters=None, fractured_mpsa=False
):
    """
    Assert equivalence the BCs, parameters and discretizations on different
    md_grids. All values are compared to those of the first md-grid in md-grids.
    md-grids     - list of grid md-grids. The field names in parameters should
                be present in the parameter objects stored in the data
                dictionary on the gb nodes.
    lh          - list of corresponding lhs matrices
    rh          - list of corresponding rhs vectors
    parameters  - For the assumed parameter fields, see below. To add other
                fields, expand the function compare_parameters and pass field
                names in a list.
    """

    if parameters is None:
        if phys == "mechanics":
            parameters = ["stiffness", "aperture"]
        elif phys == "flow":
            parameters = ["permeability", "aperture"]

    cmh, cml, fmh, fml = check_equivalent_md_grids(md_grids)
    mdg_0 = md_grids[0]
    n = len(md_grids)
    dh = mdg_0.dim_max()
    dl = mdg_0.dim_min()

    for i in range(1, n):
        mdg_i = md_grids[i]
        for dim in range(dl, dh + 1):
            sd_0 = mdg_0.subdomains(dim=dim)[0]
            sd_i = mdg_i.subdomains(dim=dim)[0]
            data_0 = mdg_0.node_props(sd_0)
            data_i = mdg_i.node_props(sd_i)
            if dim == dl:
                face_map, cell_map = fml[i - 1], cml[i - 1]
            if dim == dh:
                face_map, cell_map = fmh[i - 1], cmh[i - 1]

            compare_parameters(
                data_0, data_i, face_map, cell_map, phys=phys, fields=parameters
            )

            compare_bc(sd_i, data_0, data_i, face_map, phys=phys)

        compare_discretizations(
            sd_i,
            lh[0],
            lh[i],
            rh[0],
            rh[i],
            (cmh[i - 1], cml[i - 1]),
            (fmh[i - 1], fml[i - 1]),
            phys=phys,
            fractured_mpsa=fractured_mpsa,
        )


def compare_parameters(
    data_0, data_1, face_map=None, cell_map=None, phys="flow", fields=[]
):
    """
    Compare following parameters:
        Permeability
        Aperture
    May be extended with more checks.
    """
    p_0 = data_0["param"]
    p_1 = data_1["param"]
    if "permeability" in fields:
        assert np.all(
            np.isclose(
                p_0.get_tensor(phys).perm, p_1.get_tensor(phys).perm[:, :, cell_map]
            )
        )
    if "stiffness" in fields:
        if p_0.dim > 1:
            assert np.all(
                np.isclose(
                    p_0.get_tensor(phys).c, p_1.get_tensor(phys).c[:, :, cell_map]
                )
            )
    if "aperture" in fields:
        assert np.all(np.isclose(p_0.get_aperture(), p_1.get_aperture()[cell_map]))


def compare_bc(sd_1, data_0, data_1, face_map, phys="flow"):
    """
    Compare type and value of BCs.
    """
    BC_0 = data_0["param"].get_bc(phys)
    BC_1 = data_1["param"].get_bc(phys)
    vals_0 = data_0["param"].get_bc_val(phys)
    vals_1 = data_1["param"].get_bc_val(phys)

    assert np.all(BC_0.is_dir == BC_1.is_dir[face_map])
    assert np.all(BC_0.is_neu == BC_1.is_neu[face_map])
    if phys == "mechanics":
        boundary_face_map = fvutils.expand_indices_nd(face_map, sd_1.dim)
    else:
        boundary_face_map = face_map
    assert np.all(vals_0 == vals_1[boundary_face_map])


def compare_discretizations(
    sd_1,
    lhs_0,
    lhs_1,
    rhs_0,
    rhs_1,
    cell_maps,
    face_maps,
    phys="flow",
    fractured_mpsa=False,
):
    """
    Assumes dofs sorted as cell_maps. Not necessarily true for multiple
    fractures.
    """
    if fractured_mpsa:
        # The dofs are at the cell centers
        dof_map_cells = fvutils.expand_indices_nd(cell_maps[0], sd_1.dim)

        # and faces on either side of the fracture. Find the order of the g_1
        # frac faces among the g_0 frac faces.
        frac_faces_loc = sm.ismember_rows(
            face_maps[0], sd_1.frac_pairs.ravel("C"), sort=False
        )[1]
        # And expand to the dofs, one for each dimension for each face. For two
        # faces f0 and f1 in 3d, the order is
        # u(f0). v(f0), w(f0), u(f1). v(f1), w(f1)
        frac_indices = fvutils.expand_indices_nd(frac_faces_loc, sd_1.dim)
        # Account for the cells
        frac_indices += sd_1.num_cells * sd_1.dim
        global_dof_map = np.concatenate((dof_map_cells, frac_indices))
    elif phys == "mechanics":
        global_dof_map = fvutils.expand_indices_nd(cell_maps[0], sd_1.dim)
        global_dof_map = np.array(global_dof_map, dtype=int)
    else:
        global_dof_map = np.concatenate(
            (cell_maps[0], cell_maps[1] + cell_maps[0].size)
        )
    mapped_lhs = lhs_1[global_dof_map][:, global_dof_map]
    assert np.isclose(np.sum(np.absolute(lhs_0 - mapped_lhs)), 0)
    assert np.all(np.isclose(rhs_0, rhs_1[global_dof_map]))


def check_equivalent_md_grids(md_grids, decimals=12):
    """
    Checks agreement between number of cells, faces and nodes, their
    coordinates and the connectivity matrices cell_faces and face_nodes. Also
    checks the face tags.

    """
    dim_h = md_grids[0].dim_max()
    dim_l = dim_h - 1
    num_md_grids = len(md_grids)
    cell_maps_h, face_maps_h = [], []
    cell_maps_l, face_maps_l = num_md_grids * [{}], num_md_grids * [{}]

    # Check that all md-grids have the same number of grids in the lower dimension
    num_grids_l: int = len(md_grids[0].subdomains(dim=dim_h - 1))
    for mdg in md_grids:
        assert len(mdg.subdomains(dim=dim_h - 1)) == num_grids_l

    for dim in range(dim_l, dim_h + 1):
        for target_grid in range(len(md_grids[0].subdomains(dim=dim))):

            n_cells, n_faces, n_nodes = np.empty(0), np.empty(0), np.empty(0)
            nodes, face_centers, cell_centers = [], [], []
            cell_faces, face_nodes = [], []
            for mdg in md_grids:
                sd = mdg.subdomains(dim=dim)[target_grid]
                n_cells = np.append(n_cells, sd.num_cells)
                n_faces = np.append(n_faces, sd.num_faces)
                n_nodes = np.append(n_nodes, sd.num_nodes)
                cell_faces.append(sd.cell_faces)
                face_nodes.append(sd.face_nodes)
                cell_centers.append(sd.cell_centers)
                face_centers.append(sd.face_centers)
                nodes.append(sd.nodes)

            # Check that all md-grids have the same number of cells, faces and nodes
            assert np.unique(n_cells).size == 1
            assert np.unique(n_faces).size == 1
            assert np.unique(n_nodes).size == 1

            # Check that the coordinates agree
            cell_centers = np.round(cell_centers, decimals)
            nodes = np.round(nodes, decimals)
            face_centers = np.round(face_centers, decimals)
            for i in range(1, num_md_grids):
                assert np.all(sm.ismember_rows(cell_centers[0], cell_centers[i])[0])
                assert np.all(sm.ismember_rows(face_centers[0], face_centers[i])[0])
                assert np.all(sm.ismember_rows(nodes[0], nodes[i])[0])

            # Now we know all nodes, faces and cells are in all grids, we map them
            # to prepare cell_faces and face_nodes comparison
            sd_0 = md_grids[0].subdomains(dim=dim)[target_grid]
            for i in range(1, num_md_grids):
                mdg = md_grids[i]
                sd = mdg.subdomains(dim=dim)[target_grid]
                cell_map, face_map, node_map = make_maps(sd_0, sd, mdg.dim_max())
                mapped_cf = sd.cell_faces[face_map][:, cell_map]
                mapped_fn = sd.face_nodes[node_map][:, face_map]

                assert np.sum(np.abs(sd_0.cell_faces) != np.abs(mapped_cf)) == 0
                assert np.sum(np.abs(sd_0.face_nodes) != np.abs(mapped_fn)) == 0
                if sd.dim == dim_h:
                    face_maps_h.append(face_map)
                    cell_maps_h.append(cell_map)
                else:
                    cell_maps_l[i][sd] = cell_map
                    face_maps_l[i][sd] = face_map

                # Also loop on the standard face tags to check that they are
                # identical between the md-grids.
                tag_keys = tags.standard_face_tags()
                for key in tag_keys:
                    assert np.all(np.isclose(sd_0.tags[key], sd.tags[key][face_map]))

    # Mortar grids
    sd_primary_0 = md_grids[0].subdomains(dim=dim_h)[0]
    for target_grid in range(len(md_grids[0].subdomains(dim=dim_l))):
        sd_secondary_0 = md_grids[0].subdomains(dim=dim_l)[target_grid]
        intf_0 = md_grids[0].subdomain_pair_to_interface((sd_primary_0, sd_secondary_0))
        proj_0 = intf_0.primary_to_mortar_int()
        for i in range(1, num_md_grids):
            sd_secondary_i = md_grids[i].subdomains(dim=dim_l)[target_grid]
            sd_primary_i = md_grids[i].subdomains(dim=dim_h)[0]
            intf_i = md_grids[i].subdomain_pair_to_interface(
                (sd_primary_i, sd_secondary_i)
            )
            proj_i = intf_i.primary_to_mortar_int()
            cm = cell_maps_l[i][sd_secondary_i]
            cm_extended = np.append(cm, cm + cm.size)
            fm = face_maps_h[i - 1]
            mapped_fc = proj_i[cm_extended, :][:, fm]
            assert np.sum(np.absolute(proj_0) - np.absolute(mapped_fc)) == 0
    return cell_maps_h, cell_maps_l, face_maps_h, face_maps_l


def make_maps(g0, g1, n_digits=8, offset=0.11):
    """
    Given two grid with the same nodes, faces and cells, the mappings between
    these entities are constructed. Handles non-unique nodes and faces on next
    to fractures by exploiting neighbour information.
    Builds maps from g1 to g0, so g1.x[x_map]=g0.x, e.g.
    g1.tags[some_key][face_map] = g0.tags[some_key].
    g0 Reference grid
    g1 Other grid
    n_digits Tolerance in rounding before coordinate comparison
    offset: Weight determining how far the fracture neighbour nodes and faces
    are shifted (normally away from fracture) to ensure unique coordinates.
    """
    cell_map = sm.ismember_rows(
        np.around(g0.cell_centers, n_digits),
        np.around(g1.cell_centers, n_digits),
        sort=False,
    )[1]
    # Make face_centers unique by dragging them slightly away from the fracture

    fc0 = g0.face_centers.copy()
    fc1 = g1.face_centers.copy()
    n0 = g0.nodes.copy()
    n1 = g1.nodes.copy()
    fi0 = g0.tags["fracture_faces"]
    if np.any(fi0):
        fi1 = g1.tags["fracture_faces"]
        d0 = np.reshape(np.tile(g0.cell_faces[fi0, :].data, 3), (3, sum(fi0)))
        fn0 = g0.face_normals[:, fi0] * d0
        d1 = np.reshape(np.tile(g1.cell_faces[fi1, :].data, 3), (3, sum(fi1)))
        fn1 = g1.face_normals[:, fi1] * d1
        fc0[:, fi0] += fn0 * offset
        fc1[:, fi1] += fn1 * offset
        (ni0, fid0) = g0.face_nodes[:, fi0].nonzero()
        (ni1, fid1) = g1.face_nodes[:, fi1].nonzero()
        un, inv = np.unique(ni0, return_inverse=True)
        for i, node in enumerate(un):
            n0[:, node] += offset * np.mean(fn0[:, fid0[inv == i]], axis=1)
        un, inv = np.unique(ni1, return_inverse=True)
        for i, node in enumerate(un):
            n1[:, node] += offset * np.mean(fn1[:, fid1[inv == i]], axis=1)

    face_map = sm.ismember_rows(
        np.around(fc0, n_digits), np.around(fc1, n_digits), sort=False
    )[1]

    node_map = sm.ismember_rows(
        np.around(n0, n_digits), np.around(n1, n_digits), sort=False
    )[1]
    return cell_map, face_map, node_map
