"""Test functions in :module:`porepy.fracs.msh_2_grid`.

For now, the tests focus on ensuring that geometries with inclusions are properly
handled, with the inclusion tag correctly applied to grids of the appropriate
dimension.

The functions in the module msh_2_grid are also tested indirectly through the tests of
fracture mesh generation in :mod:`tests.fracs.test_fracture_network_2d` and
:mod:`tests.fracs.test_fracture_network_3d`.

"""

import copy
import pathlib

import numpy as np
import pytest

import gmsh

import porepy as pp
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.fracs.msh_2_grid import (
    create_0d_grids,
    create_1d_grids,
    create_2d_grids,
    create_3d_grids,
)
from porepy.fracs.simplex import _read_gmsh_file
from porepy.grids.mdg_generation import _preprocess_simplex_args


INCLUSION_NAME: str = "inclusion"


@pytest.fixture(scope="module")
def dims(request: pytest.FixtureRequest) -> tuple:
    return request.param


@pytest.fixture(scope="module")
def create_gmsh_file(dims: tuple) -> str:
    """Create a gmsh file of the specified fracture network, with an inclusion
    added.

    The inclusion is a square in 2d and a cube in 3d, placed somewhere in the middle of
    the domain. The inclusion is assigned a physical name, ``INCLUSION_NAME``.

    Parameters:
        dims: The dimensions of the domain. Should be a tuple of length 2 or 3.
    Returns:
        The path to the created gmsh file.

    """
    # The idea behind this function is to create a gmsh geometry for the given
    # fracture network. Then we will add an inclusion, in the form of a square in 2d and
    # a cube in 3d, to the geometry, and finally construct the mesh. The resulting gmsh
    # mesh will have a physical name for the inclusion, which should be picked up
    # during PorePy mesh generation from the gmsh file.

    if len(dims) == 2:
        bounding_box = {
            "xmin": 0,
            "xmax": dims[0],
            "ymin": 0,
            "ymax": dims[1],
        }
        # Two intersecting line fractures.
        fractures = pp.fracture_sets.orthogonal_fractures_2d(size=dims[0])

    if len(dims) == 3:
        bounding_box = {
            "xmin": 0,
            "xmax": dims[0],
            "ymin": 0,
            "ymax": dims[1],
            "zmin": 0,
            "zmax": dims[2],
        }
        # Three intersecting plane fractures creating both a line fracture intersection
        # and a point fracture intersection.
        fractures = pp.fracture_sets.orthogonal_fractures_3d(size=dims[0])
    domain = pp.Domain(bounding_box)
    fracture_network = pp.create_fracture_network(fractures, domain)

    msh_file = pathlib.Path("test.msh")

    # Step 1: Create a gmsh geometry for the fracture network.
    # Use functionality from pp.create_mdg and fracture_network.mesh to create a gmsh
    # file from the fracture_network.

    # The following are going to get shifted to extra_args by _preprocess_simplex_args.
    kwargs: dict = {"file_name": str(msh_file), "write_geo": False}

    # We need to differ between 2d and 3d fracture networks.
    if isinstance(fracture_network, FractureNetwork2d):
        dim = 2
        lower_level_args, *_ = _preprocess_simplex_args(
            {"cell_size": 0.5}, kwargs, FractureNetwork2d.mesh
        )
    elif isinstance(fracture_network, FractureNetwork3d):
        dim = 3
        lower_level_args, *_ = _preprocess_simplex_args(
            {"cell_size": 0.5}, kwargs, FractureNetwork3d.mesh
        )
    # Generate the mesh. We are not really interested in the returned grids, but we
    # need to call the function to get the gmsh geometry created.
    fracture_network.mesh(
        mesh_args=lower_level_args,
        file_name=msh_file,
        finalize_gmsh=False,
        clear_gmsh=False,
    )

    # Step 2: Add an inclusion to the gmsh geometry. Need to open the .geo file created
    # (which we generate by deault when calling fracture_network.mesh above). If we
    # instead had opened the .msh file, we would not be able to add new geometry in a
    # meaningful way.
    geo_file = msh_file.with_suffix(".geo_unrolled")
    gmsh.open(str(geo_file))

    domain_tag = gmsh.model.get_entities(dim)[0][1]

    # Define the geometry for the inclusion.
    domain = fracture_network.domain
    x_min = domain.bounding_box["xmin"]
    x_max = domain.bounding_box["xmax"]
    y_min = domain.bounding_box["ymin"]
    y_max = domain.bounding_box["ymax"]
    dx = x_max - x_min
    dy = y_max - y_min

    box_size = 0.1
    box_offset = 0.1

    if dim == 3:
        z_min = domain.bounding_box["zmin"]
        z_max = domain.bounding_box["zmax"]
        dz = z_max - z_min
        box_min = np.array(
            [x_min + box_offset * dx, y_min + box_offset * dy, z_min + box_offset * dz]
        )
        box_max = box_min + box_size * np.array([dx, dy, dz])
    else:
        z_min = 0.0
        z_max = 0.0
        dz = 0.0
        box_min = np.array([x_min + box_offset * dx, y_min + box_offset * dy])
        box_max = box_min + box_size * np.array([dx, dy])

    # Hard-coded geometry for the inclusion. It is placed away from the fractures.
    if dim == 2:
        p_0 = gmsh.model.geo.addPoint(box_min[0], box_min[1], z_min)
        p_1 = gmsh.model.geo.addPoint(box_max[1], box_min[1], z_min)
        p_2 = gmsh.model.geo.addPoint(box_max[1], box_max[1], z_min)
        p_3 = gmsh.model.geo.addPoint(box_min[0], box_max[1], z_min)
        line_0 = gmsh.model.geo.addLine(p_0, p_1)
        line_1 = gmsh.model.geo.addLine(p_1, p_2)
        line_2 = gmsh.model.geo.addLine(p_2, p_3)
        line_3 = gmsh.model.geo.addLine(p_3, p_0)
        # Embed the lines in the domain so that the grid conforms to the inclusion.
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(1, [line_0, line_1, line_2, line_3], 2, domain_tag)
        loop = gmsh.model.geo.addCurveLoop([line_0, line_1, line_2, line_3])
        inclusion = gmsh.model.geo.addPlaneSurface([loop])

    else:  # dim == 3
        # Add points spanning a cube.
        p_0 = gmsh.model.geo.addPoint(box_min[0], box_min[1], box_min[2])
        p_1 = gmsh.model.geo.addPoint(box_max[1], box_min[1], box_min[2])
        p_2 = gmsh.model.geo.addPoint(box_max[1], box_max[1], box_min[2])
        p_3 = gmsh.model.geo.addPoint(box_min[0], box_max[1], box_min[2])
        p_4 = gmsh.model.geo.addPoint(box_min[0], box_min[1], box_max[2])
        p_5 = gmsh.model.geo.addPoint(box_max[1], box_min[1], box_max[2])
        p_6 = gmsh.model.geo.addPoint(box_max[1], box_max[1], box_max[2])
        p_7 = gmsh.model.geo.addPoint(box_min[0], box_max[1], box_max[2])
        # Add lines forming the wire basket of the cube.
        line_0 = gmsh.model.geo.addLine(p_0, p_1)
        line_1 = gmsh.model.geo.addLine(p_1, p_2)
        line_2 = gmsh.model.geo.addLine(p_2, p_3)
        line_3 = gmsh.model.geo.addLine(p_3, p_0)
        line_4 = gmsh.model.geo.addLine(p_4, p_5)
        line_5 = gmsh.model.geo.addLine(p_5, p_6)
        line_6 = gmsh.model.geo.addLine(p_6, p_7)
        line_7 = gmsh.model.geo.addLine(p_7, p_4)
        line_8 = gmsh.model.geo.addLine(p_0, p_4)
        line_9 = gmsh.model.geo.addLine(p_1, p_5)
        line_10 = gmsh.model.geo.addLine(p_2, p_6)
        line_11 = gmsh.model.geo.addLine(p_3, p_7)
        # For each of the bounding surfaces, create a curve loop and define the surface.
        loop_0 = gmsh.model.geo.addCurveLoop([line_0, line_1, line_2, line_3])
        loop_1 = gmsh.model.geo.addCurveLoop([line_4, line_5, line_6, line_7])
        loop_2 = gmsh.model.geo.addCurveLoop([line_0, line_9, -line_4, -line_8])
        loop_3 = gmsh.model.geo.addCurveLoop([line_2, line_11, -line_6, -line_10])
        loop_4 = gmsh.model.geo.addCurveLoop([line_1, line_10, -line_5, -line_9])
        loop_5 = gmsh.model.geo.addCurveLoop([line_3, line_8, -line_7, -line_11])
        surface_0 = gmsh.model.geo.addPlaneSurface([loop_0])
        surface_1 = gmsh.model.geo.addPlaneSurface([loop_1])
        surface_2 = gmsh.model.geo.addPlaneSurface([loop_2])
        surface_3 = gmsh.model.geo.addPlaneSurface([loop_3])
        surface_4 = gmsh.model.geo.addPlaneSurface([loop_4])
        surface_5 = gmsh.model.geo.addPlaneSurface([loop_5])
        # Embed the surfaces in the domain so that the grid conforms to the inclusion.
        gmsh.model.geo.synchronize()

        # EK: To ensure that the inclusion is properly represented in the mesh, and
        # specifically to pass the test that the cell centers of cells with the
        # inclusion tag are indeed inside the inclusion, we should embed the surfaces
        # bounding the inclusion in the domain (as is done for dim=2 above). However,
        # doing so in 3d, with the lines commented out below, gave a host of errors from
        # gmsh (the actual error depends on which technical solution is used for the
        # embedding, and geometry definition in general). In EK's understanding, the
        # error is connected with the inclusion being imposed on top of the domain,
        # instead of being added as a carved-out piece of the domain. In the future,
        # when the full geometry is treated using Gmsh's open cascade kernel, it could
        # be that we can do this properly using boolean operations (only available
        # through said kernel). For now, this turned out to be technically infeasible,
        # and, since the meshes seem to be fine without the embedding, we leave it out
        # for now.
        #
        # gmsh.model.mesh.embed(
        #     2,
        #     [surface_0, surface_1, surface_2, surface_3, surface_4, surface_5],
        #     3,
        #     domain_tag,
        # )

        surface_loop = gmsh.model.geo.addSurfaceLoop(
            [surface_0, surface_1, surface_2, surface_3, surface_4, surface_5]
        )
        inclusion = gmsh.model.geo.addVolume([surface_loop])

    # There were some issues with gmsh assigning the same tag (numerical value) to
    # different objects. To circumvent this, make sure to assign a new tag that is
    # higher than all existing tags (assuming gmsh uses consecutive numbering, which
    # it does).
    gmsh.model.geo.synchronize()
    num_tags = len(gmsh.model.get_entities())
    gmsh.model.add_physical_group(
        dim, [inclusion], tag=num_tags + 1, name=INCLUSION_NAME
    )

    # Step 3: Generate the mesh, and write the gmsh file.
    gmsh.model.geo.synchronize()
    gmsh.write("final.geo_unrolled")
    gmsh.model.mesh.generate(dim)
    gmsh.write(str(msh_file))
    gmsh.clear()

    return str(msh_file), box_min, box_max


@pytest.mark.parametrize(
    "create_function, expected_grid_type",
    [
        (create_0d_grids, pp.PointGrid),
        (create_1d_grids, pp.Grid),
        (create_2d_grids, pp.Grid),
        (create_3d_grids, pp.Grid),
    ],
)
@pytest.mark.parametrize("dims", [(2, 2), (2, 2, 2)], indirect=True)
def test_create_grids_with_high_dim_inclusion(
    create_function,
    expected_grid_type,
    create_gmsh_file: str,
    dims,
) -> None:
    """Test that create_nd_grids functions produce grids with correct tags.

    The test is designed to verify that the inclusion tag is correctly applied
    to the generated grids based on their dimensions. See the function create_gmsh_file
    for details on the inclusion and its geometry.

    The test can be extended to include other types of tagging (though it is a bit
    unclear to EK what this means at the moment), but  this should be done on an
    on-demand basis.

    Parameters:
        create_function: The function to create the grids.
        expected_grid_type: The expected type of the created grids.
        create_gmsh_file: Fixture that creates a gmsh file with an inclusion.
        dims: The dimensions of the domain.

    """
    # Read the gmsh file to get points, cells, cell_info and physical names.
    msh_file, box_min, box_max = create_gmsh_file

    pts, cells, cell_info, phys_names = _read_gmsh_file(msh_file)

    nd = len(dims)

    # By default we need no extra kwargs.
    kwargs: dict = {}

    if nd == 2 and create_function.__name__ == "create_3d_grids":
        # If the target geometry is 2d, we should not try to make a 3d grid (this will
        # fail with a key error since the gmsh file contains no tetrahedra in this
        # case).
        return
    if nd == 3 and create_function.__name__ == "create_2d_grids":
        # If the target dimension is 3d, 2d grids should be embedded.
        kwargs = {"is_embedded": True}

    # Create grids of the desired type.
    grids = create_function(pts, cells, phys_names, cell_info, **kwargs)

    if isinstance(grids, tuple):
        grids, _ = grids

    # Loop over the generated grids. If they are nd, they should have the inclusion tag.
    # If they are of lower dimension, they should not have the inclusion tag.
    for g in grids:
        assert isinstance(g, expected_grid_type)
        if g.dim == nd:
            # Check that we preserved the inclusion tag.
            assert INCLUSION_NAME in g.tags

            # Also check that the cells with the inclusion tag are indeed inside the
            # inclusion box. This is complicated, since we add the inclusion in a way
            # that seems not to be compatible with the ways of gmsh: The inclusion was
            # added on top of the existing domain, instead of by carving out a piece of
            # the domain and then adding the inclusion. This has two consequences:
            # 1) The inclusion is somehow "floating" on top of the existing domain,
            #    which means that the inclusion is covered by two sets of cells (don't
            #    ask!). Hence, we cannot do a 1-1 comparison between those cells that
            #    are inside the box and those that have the inclusion tag. This should
            #    not be a problem for a properly constructed geometry, but it does give
            #    some hints on how to construct the geometry if we want to avoid this.
            # 2) Somehow, the overlapping cells are oriented in a way that does not fit
            #    with our geometry computation, which results in some cells having
            #    negative volume (EK did a quick check, and it seems that the
            #    overlapping cells break with some tacit assumptions in our geometry
            #    computations; however, since we have not encountered any practical
            #    problems related to this, it seems highly plausible that this is caused
            #    by Gmsh's reaction to our non-standard way of adding the inclusion).
            #    Therefore, to get the cell centers, we cannot use g.compute_geometry(),
            #    but rather need to compute the cell centers manually based on the
            #    average coordinates of the cells' nodes.
            cn = np.reshape(g.cell_nodes().tocsc().indices, (-1, g.dim + 1))
            cx = np.mean(g.nodes[0, cn], axis=1)
            cy = np.mean(g.nodes[1, cn], axis=1)
            cz = np.mean(g.nodes[2, cn], axis=1)

            if nd == 2:
                inside_box = np.logical_and(
                    np.logical_and(cx >= box_min[0], cx <= box_max[0]),
                    np.logical_and(cy >= box_min[1], cy <= box_max[1]),
                )
            else:
                # If in the future this test fail, see comment in helper function
                # create_gmsh_file regarding embedding of the surfaces bounding the
                # inclusion.
                inside_box = np.logical_and(
                    np.logical_and(cx >= box_min[0], cx <= box_max[0]),
                    np.logical_and(cy >= box_min[1], cy <= box_max[1]),
                    np.logical_and(cz >= box_min[2], cz <= box_max[2]),
                )

            assert np.all(
                np.isin(np.where(g.tags[INCLUSION_NAME]), np.where(inside_box))
            )

        else:
            assert INCLUSION_NAME not in g.tags
