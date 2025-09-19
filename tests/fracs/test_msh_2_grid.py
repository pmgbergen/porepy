"""Test functions in :module:`porepy.fracs.msh_2_grid`."""

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
    gmsh_element_types,
    tag_grid,
)
from porepy.fracs.simplex import _read_gmsh_file
from porepy.grids.mdg_generation import _preprocess_simplex_args

rng: np.random.Generator = np.random.default_rng(seed=42)
dirname: pathlib.Path = pathlib.Path(__file__).parent


@pytest.fixture
def dims(request: pytest.FixtureRequest) -> tuple:
    return request.param


@pytest.fixture(scope="function")
def fracture_network(dims: tuple) -> pp.fracture_network:
    if len(dims) == 2:
        bounding_box = {
            "xmin": 0,
            "xmax": dims[0],
            "ymin": 0,
            "ymax": dims[1],
        }
        # Two intersecting line fractures
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
    return pp.create_fracture_network(fractures, domain)


@pytest.fixture(scope="function")
def create_gmsh_file(
    fracture_network: pp.fracture_network, tmp_path: pathlib.Path
) -> str:
    """Create a gmsh file of the specified fracture network, with an inclusion
    added.

    The inclusion is a square in 2d and a cube in 3d, placed somewhere in the middle of
    the domain. The inclusion is assigned a physical name, ``INCLUSION_NAME``.

    Parameters:
        fracture_network: The fracture network to create a gmsh file for.
        tmp_path: Temporary path to store the gmsh file.

    Returns:
        The path to the created gmsh file.

    """
    # The idea behind the this function is to create a gmsh geometry for the given
    # fracture network. Then we will add an inclusion, in the form of a square in 2d and
    # a cube in 3d, to the geometry, and finally construct the mesh. The resulting gmsh
    # mesh will have a physical name for the inclusion, which should be picked up
    # during PorePy mesh generation from the gmsh file.

    msh_file: pathlib.Path = tmp_path / "test.msh"

    # Step 1: Create a gmsh geometry for the fracture network.
    # Use functionality from pp.create_mdg and fracture_network.mesh to create a gmsh
    # file from the fracture_network.

    # The following are going to get shifted to extra_args by _preprocess_simplex_args.
    kwargs: dict = {"file_name": str(msh_file), "write_geo": False}

    dim: int

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

    # Step 2: Add an inclusion to the gmsh geometry.
    gmsh.open(str(msh_file))

    # Define the geometry for the inclusion.
    domain = fracture_network.domain
    x_min = domain.bounding_box["xmin"]
    x_max = domain.bounding_box["xmax"]
    y_min = domain.bounding_box["ymin"]
    y_max = domain.bounding_box["ymax"]
    dx = x_max - x_min
    dy = y_max - y_min
    if dim == 3:
        z_min = domain.bounding_box["zmin"]
        z_max = domain.bounding_box["zmax"]
        dz = z_max - z_min
    else:
        z_min = 0.0
        z_max = 0.0
        dz = 0.0

    # Hard-coded geometry for the inclusion. It is placed away from the fractures.
    if dim == 2:
        p_0 = gmsh.model.geo.addPoint(x_min + 0.1 * dx, y_min + 0.1 * dy, 0)
        p_1 = gmsh.model.geo.addPoint(x_min + 0.2 * dx, y_min + 0.1 * dy, 0)
        p_2 = gmsh.model.geo.addPoint(x_min + 0.2 * dx, y_min + 0.2 * dy, 0)
        p_3 = gmsh.model.geo.addPoint(x_min + 0.1 * dx, y_min + 0.2 * dy, 0)
        line_0 = gmsh.model.geo.addLine(p_0, p_1)
        line_1 = gmsh.model.geo.addLine(p_1, p_2)
        line_2 = gmsh.model.geo.addLine(p_2, p_3)
        line_3 = gmsh.model.geo.addLine(p_3, p_0)
        loop = gmsh.model.geo.addCurveLoop([line_0, line_1, line_2, line_3])
        inclusion = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
        gmsh.model.add_physical_group(2, [inclusion], name=INCLUSION_NAME)

    else:  # dim == 3
        # Add points spanning a cube.
        p_0 = gmsh.model.geo.addPoint(
            x_min + 0.1 * dx, y_min + 0.1 * dy, z_min + 0.1 * dz
        )
        p_1 = gmsh.model.geo.addPoint(
            x_min + 0.2 * dx, y_min + 0.1 * dy, z_min + 0.1 * dz
        )
        p_2 = gmsh.model.geo.addPoint(
            x_min + 0.2 * dx, y_min + 0.2 * dy, z_min + 0.1 * dz
        )
        p_3 = gmsh.model.geo.addPoint(
            x_min + 0.1 * dx, y_min + 0.2 * dy, z_min + 0.1 * dz
        )
        p_4 = gmsh.model.geo.addPoint(
            x_min + 0.1 * dx, y_min + 0.1 * dy, z_min + 0.2 * dz
        )
        p_5 = gmsh.model.geo.addPoint(
            x_min + 0.2 * dx, y_min + 0.1 * dy, z_min + 0.2 * dz
        )
        p_6 = gmsh.model.geo.addPoint(
            x_min + 0.2 * dx, y_min + 0.2 * dy, z_min + 0.2 * dz
        )
        p_7 = gmsh.model.geo.addPoint(
            x_min + 0.1 * dx, y_min + 0.2 * dy, z_min + 0.2 * dz
        )
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
        surface_loop = gmsh.model.geo.addSurfaceLoop(
            [surface_0, surface_1, surface_2, surface_3, surface_4, surface_5]
        )
        inclusion = gmsh.model.geo.addVolume([surface_loop])
        gmsh.model.geo.synchronize()

        # There were some issues with gmsh assigning the same tag (numerical value) to
        # different objects. To circumvent this, make sure to assign a new tag that is
        # higher than all existing tags (assuming gmsh uses consecutive numbering, which
        # it does).
        num_tags = len(gmsh.model.get_entities())
        gmsh.model.add_physical_group(
            3, [inclusion], tag=num_tags + 1, name=INCLUSION_NAME
        )

    # Step 3: Generate the mesh, and write the gmsh file.
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(dim)
    gmsh.write(str(msh_file))
    gmsh.clear()

    return str(msh_file)


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
@pytest.mark.parametrize("num_phys_names", [0, 1, 5], indirect=True)
def test_create_grids(
    create_function,
    expected_grid_type,
    create_gmsh_file: str,
    dims,
    num_phys_names,
) -> None:
    """Test that create_nd_grids functions produce grids with correct tags."""
    pts, cells, cell_info, phys_names = _read_gmsh_file(create_gmsh_file)

    if len(dims) == 2 and create_function.__name__ == "create_3d_grids":
        # If the target geometry is 2d, we should not try to make a 3d grid (this will
        # fail with a key error since the gmsh file contains no tetrahedra in this
        # case).
        return

    grids = create_function(pts, cells, phys_names, cell_info)

    if isinstance(grids, tuple):
        grids, _ = grids

    for g in grids:
        assert isinstance(g, expected_grid_type)

        if g.dim == nd:
            assert INCLUSION_NAME in g.tags
        else:
            assert INCLUSION_NAME not in g.tags
