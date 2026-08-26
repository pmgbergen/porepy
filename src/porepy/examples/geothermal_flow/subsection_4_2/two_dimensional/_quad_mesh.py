"""Unstructured quadrilateral (quad-dominant) meshing for the --md build.

PorePy's gmsh->grid importer (``porepy.fracs.msh_2_grid.create_2d_grids``) reads only the
``"triangle"`` cell block and builds a ``pp.TriangleGrid``, so a gmsh-recombined mesh (all-
or quad-dominant) fails there.  This module adds quad support WITHOUT touching library files,
by monkeypatching -- only for the duration of one ``create_mdg`` call -- two things:

  1. ``gmsh.model.mesh.generate`` : turn on recombination (RecombineAll + blossom) for the 2D
     pass, so gmsh emits quads (and the triangles it cannot pair -- a mixed mesh is expected
     and fine, especially near fracture intersections and acute angles).
  2. ``msh_2_grid.create_2d_grids`` : build the ambient 2D grid as a general ``pp.Grid`` from
     the mixed triangle+quad cells.  Only the grid CONSTRUCTION differs from the library
     version; the gmsh-line -> pp.Grid-face tagging (which drives the mixed-dimensional
     fracture/boundary coupling) is edge-based (every 2D face is a 2-node edge, quad or tri)
     and is reproduced verbatim, so the fracture coupling is unchanged.

The ambient grid is no longer K-orthogonal, so pair --recombine with --consistent (MPFA).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.fracs import msh_2_grid as _m2g
from porepy.fracs import meshing as _meshing

_ORIG_CREATE_2D = _m2g.create_2d_grids
_ORIG_NODES_PER_FACE = _meshing._nodes_per_face
_QUAD_GRID_NAME = "SimplexQuadGrid"


def _polygon_grid(nodes: np.ndarray, polys: list[np.ndarray]) -> pp.Grid:
    """General 2D grid from polygon cells (mixed triangles/quadrilaterals).

    ``nodes`` is ``(3, num_nodes)`` in GLOBAL gmsh numbering; ``polys`` is a list of node-index
    arrays (one per cell, CCW), indexing into ``nodes``.  Faces are the unique cell edges."""
    edges = []
    cell_ptr = [0]
    for nd in polys:
        k = len(nd)
        for i in range(k):
            edges.append((nd[i], nd[(i + 1) % k]))
        cell_ptr.append(cell_ptr[-1] + k)
    edges = np.asarray(edges, dtype=int)
    sign = np.where(edges[:, 1] > edges[:, 0], 1, -1)
    unique_edges, inv = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    n_faces = unique_edges.shape[0]
    n_nodes = nodes.shape[1]

    fn_indptr = np.arange(0, 2 * n_faces + 1, 2)
    face_nodes = sps.csc_matrix(
        (np.ones(2 * n_faces, dtype=bool), unique_edges.ravel(), fn_indptr),
        shape=(n_nodes, n_faces),
    )
    # Repair any inconsistently-oriented shared edge (both cells gave the same sign).
    w = np.bincount(inv, weights=sign, minlength=n_faces)
    for f in np.where(np.abs(w) > 1)[0]:
        sign[np.where(inv == f)[0][-1]] *= -1
    cell_faces = sps.csc_matrix(
        (sign.astype(int), inv, np.asarray(cell_ptr, dtype=int)),
        shape=(n_faces, len(polys)),
    )
    return pp.Grid(2, nodes, face_nodes, cell_faces, _QUAD_GRID_NAME)


def _nodes_per_face(g):
    """Any 2D grid has 2 nodes per face (an edge); teach _assemble_mdg about the quad grid."""
    if g.dim == 2 and _QUAD_GRID_NAME in g.name:
        return 2
    return _ORIG_NODES_PER_FACE(g)


def _create_2d_grids_mixed(pts, cells, phys_names, cell_info, is_embedded=False,
                           surface_tag=None, constraints=None):
    """Drop-in for ``create_2d_grids`` that also consumes ``cells['quad']``. Pure-triangle
    input and the embedded (3D) branch fall back to the library implementation."""
    if is_embedded or "quad" not in cells:
        return _ORIG_CREATE_2D(pts, cells, phys_names, cell_info,
                               is_embedded, surface_tag, constraints)

    tri = np.asarray(cells.get("triangle", np.zeros((0, 3), dtype=int)), dtype=int)
    quad = np.asarray(cells["quad"], dtype=int)
    polys = [row for row in tri] + [row for row in quad]        # triangles first, then quads
    g_2d = _polygon_grid(pts.transpose(), polys)

    # tag_grid maps the ambient dimension to the "triangle" block, so fold the quad cell tags
    # in behind the triangle ones (same order as `polys`) and drop the quad key.
    ci = dict(cell_info)
    ci["triangle"] = np.concatenate([
        np.asarray(ci.get("triangle", np.zeros(0, dtype=int)), dtype=int),
        np.asarray(ci.get("quad", np.zeros(0, dtype=int)), dtype=int)])
    ci.pop("quad", None)
    g_2d = _m2g.tag_grid(g_2d, phys_names, ci)
    g_2d.global_point_ind = np.arange(pts.shape[0])

    # --- gmsh-line -> face tagging: verbatim from msh_2_grid.create_2d_grids (edge-based) ---
    line = np.sort(cells.get("line", np.array([[]])).T, axis=0)
    if line.size == 0:
        return [g_2d]
    faces = np.sort(np.reshape(g_2d.face_nodes.indices, (2, -1), order="F"), axis=0)
    idxf = np.lexsort(faces)
    idxl = np.lexsort(line)
    IC = np.empty(line.shape[1], dtype=int)
    IC[idxl] = np.arange(line.shape[1])
    facestr = np.char.add(np.char.add(faces[0, idxf].astype(str), ","),
                          faces[1, idxf].astype(str))
    linestr = np.char.add(np.char.add(line[0, idxl].astype(str), ","),
                          line[1, idxl].astype(str))
    is_line = np.isin(facestr, linestr, assume_unique=True)
    line2face = idxf[is_line][IC]
    if not np.allclose(faces[:, line2face], line):
        raise RuntimeError("Could not map gmsh lines to pp.Grid faces on the quad mesh.")
    for tag in np.unique(cell_info["line"]):
        tag_name = phys_names[tag].lower() + "_faces"
        g_2d.tags[tag_name] = np.zeros(g_2d.num_faces, dtype=bool)
        g_2d.tags[tag_name][line2face[cell_info["line"] == tag]] = True
    return [g_2d]


def build_recombined_mdg(mesh_args, network, file_name=None):
    """``pp.create_mdg('simplex', ...)`` with gmsh recombination -> quad-dominant ambient grid.
    ``file_name`` is the (per-process, unique) gmsh output path. Patches are installed only
    for the duration of the meshing call, then removed."""
    import gmsh

    orig_generate = gmsh.model.mesh.generate

    def _generate(dim=3):
        if dim >= 2:                                     # recombine the 2D pass only
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)   # 1 = blossom
        return orig_generate(dim)

    gmsh.model.mesh.generate = _generate
    _m2g.create_2d_grids = _create_2d_grids_mixed
    _meshing._nodes_per_face = _nodes_per_face
    try:
        return pp.create_mdg("simplex", mesh_args, network, file_name=file_name)
    finally:
        gmsh.model.mesh.generate = orig_generate
        _m2g.create_2d_grids = _ORIG_CREATE_2D
        _meshing._nodes_per_face = _ORIG_NODES_PER_FACE
