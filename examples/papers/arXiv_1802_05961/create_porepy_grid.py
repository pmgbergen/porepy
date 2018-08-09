#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 10:21:44 2017

@author: eke001
"""

from porepy.fracs import simplex, meshing, split_grid

import grid_to_gmsh


def create(mesh_size=0.01):
    fn = "test_grid"
    grid_to_gmsh.write_geo(fn + ".geo", mesh_size=mesh_size)

    simplex.triangle_grid_run_gmsh(fn)

    grids = simplex.triangle_grid_from_gmsh(fn)

    meshing.tag_faces(grids)

    gb = meshing.assemble_in_bucket(grids)
    gb.compute_geometry()

    split_grid.split_fractures(gb)
    gb.assign_node_ordering()

    return gb


if __name__ == "__main__":
    create()
