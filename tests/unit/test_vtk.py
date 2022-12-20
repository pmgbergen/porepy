"""Tests of the export functionalities of Exporter, FractureNetwork2d, and
FractureNetwork3d.

The tests focus on the write-to-file capabilities of the Exporter. It is tested
for various sorts of relevant meshes in 1d, 2d, and 3d, including single domain
as well as mixed-dimensional domains. In addition, the export capability of  2d
and 3d fracture networks is tested. All tests have a similar character and are
based on a simple comparison with reference vtu files. It should be noted that
failure of any test indicates that something in the export filter, or in the vtk
python bindings has changed. If the change is external to PorePy, this does not
necessarily mean that something is wrong.
"""
import os
import shutil
from collections import namedtuple
from pathlib import Path

import meshio
import numpy as np
import pytest
from deepdiff import DeepDiff

import porepy as pp

# Globally store location of reference files
folder_reference = (
    os.path.dirname(os.path.realpath(__file__)) + "/" + "test_vtk_reference"
)


class ExporterTestSetup:
    """Class to define where to store vtu files, and test the export functionality
    of the Exporter, FractureNetwork2d, and FractureNetwork3d.

    """

    def __init__(self):
        # Define ingredients of where to store vtu files for exporting during testing.
        self.folder = "./test_vtk"
        self.file_name = "grid"
        self.folder_reference = folder_reference


@pytest.fixture
def setup():
    """Method to deliver a setup to all tests, and remove any temporary directory."""

    # Setup
    setup = ExporterTestSetup()
    yield setup

    # Teardown: remove temporary directory for vtu files.
    full_path = Path.cwd() / Path.resolve(Path(setup.folder)).name
    shutil.rmtree(full_path)


def _compare_vtu_files(
    test_file: str, reference_file: str, overwrite: bool = False
) -> bool:
    """Determine whether the contents of two vtu files are identical.

    Helper method to determine whether two vtu files, accessed by their
    paths, are identical. Returns True if both files are identified as the
    same, False otherwise. This is the main auxiliary routine used to compare
    down below whether the Exporter produces identical outputs as stored
    reference files.

    .. note:
        It is implicitly assumed that Gmsh returns the same grid as
        for the reference grid; thus, if this test fails, it should be
        rerun with an older version of Gmsh to test for failure due to
        external reasons.

    Parameters:
        test_file: Name of the test file.
        reference_file: Name of the reference file
        overwrite: Whether to overwrite the reference file with the test file. This
            should only ever be done if you are changing the "truth" of the test.

    Returns:
        Boolean. True iff files are identical.

    """
    if overwrite:
        shutil.copy(test_file, reference_file)
        return True

    # Trust meshio to read the vtu files
    test_data = meshio.read(test_file)
    reference_data = meshio.read(reference_file)

    # Determine the difference between the two meshio objects.
    # Ignore differences in the data type if values are close. To judge whether values
    # are close, only consider certain number of significant digits and base the
    # comparison in exponential form.
    # Also ignore differences in the subdomain_id and interface_id, as these are
    # very sensitive to the order of grid creation, which may depend on pytest assembly
    # and number of tests run.
    excludePaths = [
        "root['cell_data']['subdomain_id']",
        "root['cell_data']['interface_id']",
    ]
    diff = DeepDiff(
        reference_data.__dict__,
        test_data.__dict__,
        significant_digits=8,
        number_format_notation="e",
        ignore_numeric_type_changes=True,
        exclude_paths=excludePaths,
    )

    # If the difference is empty, the meshio objects are identified as identical.
    return diff == {}


@pytest.fixture(scope="function")
def subdomain(request):
    """Helper for parametrization of test_single_subdomains.

    Define collection of single subdomains incl. 1d, 2d, 3d grids, of simplicial,
    Cartesian and polytopal element type.

    """

    # Construct 2d polytopal grid
    sd_polytop_2d = pp.StructuredTriangleGrid([2] * 2, [1] * 2)
    sd_polytop_2d.compute_geometry()
    pp.coarsening.generate_coarse_grid(sd_polytop_2d, [0, 1, 3, 3, 1, 1, 2, 2])

    # Construct 3d polytopal grid
    sd_polytop_3d = pp.CartGrid([3, 2, 3], [1] * 3)
    sd_polytop_3d.compute_geometry()
    pp.coarsening.generate_coarse_grid(
        sd_polytop_3d, [0, 0, 1, 0, 1, 1, 0, 2, 2, 3, 2, 2, 4, 4, 4, 4, 4, 4]
    )

    # Define data type for a single subdomain
    SingleSubdomain = namedtuple("SingleSubdomain", ["grid", "ref_vtu_file"])

    # Define the collection of subdomains
    subdomains = [
        # 1d grid
        SingleSubdomain(
            pp.CartGrid(3, 1),
            f"{folder_reference}/single_subdomain_1d.vtu",
        ),
        # 2d simplex grid
        SingleSubdomain(
            pp.StructuredTriangleGrid([3] * 2, [1] * 2),
            f"{folder_reference}/single_subdomain_2d_simplex_grid.vtu",
        ),
        # 2d Cartesian grid
        SingleSubdomain(
            pp.CartGrid([4] * 2, [1] * 2),
            f"{folder_reference}/single_subdomain_2d_cart_grid.vtu",
        ),
        # 2d polytopal grid
        SingleSubdomain(
            sd_polytop_2d,
            f"{folder_reference}/single_subdomain_2d_polytop_grid.vtu",
        ),
        # 3d simplex grid
        SingleSubdomain(
            pp.StructuredTetrahedralGrid([3] * 3, [1] * 3),
            f"{folder_reference}/single_subdomain_3d_simplex_grid.vtu",
        ),
        # 3d Cartesian grid
        SingleSubdomain(
            pp.CartGrid([4] * 3, [1] * 3),
            f"{folder_reference}/single_subdomain_3d_cart_grid.vtu",
        ),
        # 3d polytopal grid
        SingleSubdomain(
            sd_polytop_3d,
            f"{folder_reference}/single_subdomain_3d_polytop_grid.vtu",
        ),
    ]
    return subdomains[request.param]


@pytest.mark.parametrize("subdomain", np.arange(7), indirect=True)
def test_single_subdomains(setup, subdomain):
    """Test of the Exporter for single subdomains of different dimensionality
    and different grid type. Exporting of scalar and vectorial data is tested.

    """

    # Define grid
    sd = subdomain.grid
    sd.compute_geometry()

    # Define data
    dummy_scalar = np.ones(sd.num_cells) * sd.dim
    dummy_vector = np.ones((3, sd.num_cells)) * sd.dim

    # Export data
    save = pp.Exporter(
        sd,
        setup.file_name,
        setup.folder,
        export_constants_separately=False,
    )
    save.write_vtu([("dummy_scalar", dummy_scalar), ("dummy_vector", dummy_vector)])

    # Check that exported vtu file and reference file are the same
    assert _compare_vtu_files(
        f"{setup.folder}/{setup.file_name}_{sd.dim}.vtu",
        f"{subdomain.ref_vtu_file}",
    )


@pytest.mark.parametrize("subdomain", np.arange(7), indirect=True)
def test_single_subdomains_import(setup, subdomain):
    # Test of the import routine of the Exporter for single subdomains.
    # Consistent with test_single_subdomains.

    # Define grid
    sd = subdomain.grid
    sd.compute_geometry()

    # Define exporter
    save = pp.Exporter(
        sd,
        setup.file_name,
        setup.folder,
        export_constants_separately=False,
    )

    # Define keys (here corresponding to all data stored in the vtu file to pass the test).
    keys = ["dummy_scalar", "dummy_vector"]

    # Import data
    save.import_from_vtu(
        keys=keys,
        file_names=f"{subdomain.ref_vtu_file}",
        automatic=False,
        dims=sd.dim,
    )

    # Perform comparison on vtu level (seems the easiest as it only involves a
    # comparison of dictionaries). This requires test_single_subdomains to pass
    # all tests.
    save.write_vtu(keys)

    # Check that exported vtu file and reference file are the same
    assert _compare_vtu_files(
        f"{setup.folder}/{setup.file_name}_{sd.dim}.vtu",
        f"{subdomain.ref_vtu_file}",
    )


def test_mdg(setup):
    """Test Exporter for 2d mixed-dimensional grids for a two-fracture domain.

    Exporting of scalar and vectorial data, separately defined on both subdomains and
    interfaces.

    """

    # Define grid
    mdg, _ = pp.md_grids_2d.two_intersecting(
        [4, 4], y_endpoints=[0.25, 0.75], simplex=False
    )

    # Define data
    for sd, sd_data in mdg.subdomains(return_data=True):
        pp.set_state(
            sd_data,
            {
                "dummy_scalar": np.ones(sd.num_cells) * sd.dim,
                "dummy_vector": np.ones((3, sd.num_cells)) * sd.dim,
            },
        )

    for intf, intf_data in mdg.interfaces(return_data=True):
        pp.set_state(
            intf_data,
            {
                "dummy_scalar": np.zeros(intf.num_cells),
                "unique_dummy_scalar": np.zeros(intf.num_cells),
            },
        )

    # Export data
    save = pp.Exporter(
        mdg,
        setup.file_name,
        setup.folder,
        export_constants_separately=False,
    )
    save.write_vtu(["dummy_scalar", "dummy_vector", "unique_dummy_scalar"])

    # Check that exported vtu files and reference files are the same.
    for appendix in ["1", "2", "mortar_1"]:
        assert _compare_vtu_files(
            f"{setup.folder}/{setup.file_name}_{appendix}.vtu",
            f"{setup.folder_reference}/mdg_grid_{appendix}.vtu",
        )


@pytest.mark.parametrize("addendum", ["", "nontrivial_data_"])
def test_mdg_import(setup, addendum):
    # Test of the import routine of the Exporter for 2d mixed-dimensional grids.
    # Consistent with test_mdg.

    # Define grid
    mdg, _ = pp.md_grids_2d.two_intersecting(
        [4, 4], y_endpoints=[0.25, 0.75], simplex=False
    )

    # Define exporter
    save = pp.Exporter(
        mdg,
        setup.file_name,
        setup.folder,
        export_constants_separately=False,
    )

    # Define keys (here corresponding to all data stored in the vtu file to pass the test).
    keys = ["dummy_scalar", "dummy_vector", "unique_dummy_scalar"]

    # Import data
    save.import_from_vtu(
        keys=keys,
        file_names=[
            f"{setup.folder_reference}/mdg_{addendum}grid_2.vtu",
            f"{setup.folder_reference}/mdg_{addendum}grid_1.vtu",
            f"{setup.folder_reference}/mdg_{addendum}grid_mortar_1.vtu",
        ],
    )

    # Perform comparison on vtu level (seems the easiest as it only involves a
    # comparison of dictionaries). This requires test_mdg to pass all tests.
    save.write_vtu(keys)

    # Check that exported vtu files and reference files are the same.
    for appendix in ["1", "2", "mortar_1"]:
        assert _compare_vtu_files(
            f"{setup.folder}/{setup.file_name}_{appendix}.vtu",
            f"{setup.folder_reference}/mdg_{addendum}grid_{appendix}.vtu",
        )


def test_mdg_data_selection(setup):
    """Test Exporter for 2d mixed-dimensional grids for a two-fracture domain.

    Exporting of scalar and vectorial data, separately defined on both subdomains and
    interfaces. Furthermore, the different possibilities of how to export data are
    tested: addressing selected data associated to all subdomains and interfaces, single
    ones, or defining external data (here simply cell centers).

    """

    # Define grid
    mdg, _ = pp.md_grids_2d.two_intersecting(
        [4, 4], y_endpoints=[0.25, 0.75], simplex=False
    )

    # Define data
    for sd, sd_data in mdg.subdomains(return_data=True):
        pp.set_state(
            sd_data,
            {
                "dummy_scalar": np.ones(sd.num_cells) * sd.dim,
                "dummy_vector": np.ones((3, sd.num_cells)) * sd.dim,
            },
        )

    for intf, intf_data in mdg.interfaces(return_data=True):
        pp.set_state(
            intf_data,
            {
                "dummy_scalar": np.zeros(intf.num_cells),
                "unique_dummy_scalar": np.zeros(intf.num_cells),
            },
        )

    # Fetch separate subdomains
    subdomains_1d = mdg.subdomains(dim=1)
    subdomains_2d = mdg.subdomains(dim=2)
    sd_2d = subdomains_2d[0]

    # Export data
    save = pp.Exporter(
        mdg,
        setup.file_name,
        setup.folder,
        export_constants_separately=False,
    )
    save.write_vtu(
        [
            (subdomains_1d, "dummy_scalar"),
            "dummy_vector",
            "unique_dummy_scalar",
            (sd_2d, "cc", sd_2d.cell_centers),
        ]
    )

    # Check that exported vtu files and reference files are the same.
    for appendix in ["1", "2", "mortar_1"]:
        assert _compare_vtu_files(
            f"{setup.folder}/{setup.file_name}_{appendix}.vtu",
            f"{setup.folder_reference}/mdg_data_selection_grid_{appendix}.vtu",
        )


def test_constant_data(setup):
    """Test Exporter functionality to distinguish between constant and non-constant
    data during exporting.

    """

    # Define grid
    g = pp.StructuredTriangleGrid([3] * 2, [1] * 2)
    g.compute_geometry()

    # Define data
    dummy_scalar = np.ones(g.num_cells) * g.dim
    dummy_vector = np.ones((3, g.num_cells)) * g.dim

    # Export data
    save = pp.Exporter(
        g,
        setup.file_name,
        setup.folder,
    )
    # Add additional constant data (cell centers)
    save.add_constant_data([(g, "cc", g.cell_centers)])
    save.write_vtu([("dummy_scalar", dummy_scalar), ("dummy_vector", dummy_vector)])

    # Check that exported vtu files and reference files are the same
    for appendix in ["2", "constant_2"]:
        assert _compare_vtu_files(
            f"{setup.folder}/{setup.file_name}_{appendix}.vtu",
            f"{setup.folder_reference}/constant_data_test_grid_{appendix}.vtu",
        )


def test_fracture_network_2d(setup):
    """Test of the export functionality of FractureNetwork2d."""

    # Define network
    p = np.array([[0, 2, 1, 2, 1], [0, 0, 0, 1, 2]])
    e = np.array([[0, 2, 3], [1, 3, 4]])
    domain = {"xmin": -2, "xmax": 3, "ymin": -2, "ymax": 3}
    network_2d = pp.FractureNetwork2d(p, e, domain)

    # Define data
    dummy_scalar = np.ones(network_2d.num_frac())
    dummy_vector = np.ones((3, network_2d.num_frac()))
    data = {"dummy_scalar": dummy_scalar, "dummy_vector": dummy_vector}

    # Make directory if not existent
    if not os.path.exists(setup.folder):
        os.makedirs(setup.folder)

    # Export data
    network_2d.to_file(
        setup.folder + "/" + setup.file_name + ".vtu",
        data=data,
    )

    # Check that exported vtu file and reference file are the same.
    assert _compare_vtu_files(
        f"{setup.folder}/{setup.file_name}.vtu",
        f"{setup.folder_reference}/fractures_2d.vtu",
    )


def test_fracture_network_3d(setup):
    """Test of the export functionality of FractureNetwork3d."""

    # Define network
    f_1 = pp.PlaneFracture(np.array([[0, 1, 2, 0], [0, 0, 1, 1], [0, 0, 1, 1]]))
    f_2 = pp.PlaneFracture(
        np.array([[0.5, 0.5, 0.5, 0.5], [-1, 2, 2, -1], [-1, -1, 2, 2]])
    )
    domain = {"xmin": -2, "xmax": 3, "ymin": -2, "ymax": 3, "zmin": -3, "zmax": 3}
    network_3d = pp.FractureNetwork3d([f_1, f_2], domain=domain)

    # Define data
    num_frac = len(network_3d._fractures)
    dummy_scalar = [[1] for _ in range(num_frac)]
    dummy_vector = [[np.ones(3)] for _ in range(num_frac)]
    data = {"dummy_scalar": dummy_scalar, "dummy_vector": dummy_vector}

    # Make directory if not existent.
    if not os.path.exists(setup.folder):
        os.makedirs(setup.folder)

    # Export data
    network_3d.to_file(
        setup.folder + "/" + setup.file_name + ".vtu",
        data=data,
    )

    # Check that exported vtu file and reference file are the same.
    assert _compare_vtu_files(
        f"{setup.folder}/{setup.file_name}.vtu",
        f"{setup.folder_reference}/fractures_3d.vtu",
    )
