"""Tests of the export functionalities of Exporter, FractureNetwork2d, and
FractureNetwork3d.

The tests focus on the write-to-file capabilities of the Exporter. It is tested for
various sorts of relevant meshes in 1d, 2d, and 3d, including single domain as well as
mixed-dimensional domains. In addition, the export capability of  2d and 3d fracture
networks is tested. All tests have a similar character and are based on a simple
comparison with reference vtu files. It should be noted that failure of any test
indicates that something in the export filter, or in the vtk python bindings has
changed. If the change is external to PorePy, this does not necessarily mean that
something is wrong.
"""
from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET
from collections import namedtuple
from pathlib import Path
from typing import Union

import meshio
import numpy as np
import pytest
from deepdiff import DeepDiff

import porepy as pp
from porepy.fracs.utils import pts_edges_to_linefractures


# Globally store location of reference files
folder_reference = (
    os.path.dirname(os.path.realpath(__file__)) + "/" + "test_vtk_reference"
)

# Data structure for defining paths
PathLike = Union[str, Path]


class ExporterTestSetup:
    """Class to define where to store vtu files, and test the export functionality of
    the Exporter, FractureNetwork2d, and FractureNetwork3d.

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
    test_file: PathLike, reference_file: PathLike, overwrite: bool = False
) -> bool:
    """Determine whether the contents of two vtu files are identical.

    Helper method to determine whether two vtu files, accessed by their paths, are
    identical. Returns True if both files are identified as the same, False otherwise.
    This is the main auxiliary routine used to compare down below whether the Exporter
    produces identical outputs as stored reference files.

    .. note:
        It is implicitly assumed that Gmsh returns the same grid as for the reference
        grid; thus, if this test fails, it should be rerun with an older version of
        Gmsh to test for failure due to external reasons.

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

    # Determine the difference between the two meshio objects. Ignore differences in
    # the data type if values are close. To judge whether values are close, only
    # consider certain number of significant digits and base the comparison in
    # exponential form. Also ignore differences in the subdomain_id and interface_id,
    # as these are very sensitive to the order of grid creation, which may depend on
    # pytest assembly and number of tests run.
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


def _compare_pvd_files(
    test_file: PathLike, reference_file: PathLike, overwrite: bool = False
) -> bool:
    """ """

    if overwrite:
        shutil.copy(test_file, reference_file)
        return True

    # Read pvd files which are xml files and compare.
    tree_test = ET.parse(test_file)
    tree_ref = ET.parse(reference_file)

    # NOTE: Here, we strictly assume that the pvd files subject to the comparison are
    # created using the Exporter. Thus, they have a non-hierarchical xml-structure.
    # Finally, there is just two different types of xml structures, either created by
    # write_pvd() or _export_mdg_pvd(). The first contains the keyword "timestep",
    # whereas the second does not. This characteristic is used to determine the type
    # of pvd files. Assume consistency, and that the first entry is sufficient to check.
    for dataset in tree_test.iter("DataSet"):
        data = dataset.attrib
        test_originates_from_write_pvd = "time" in data
    for dataset in tree_ref.iter("DataSet"):
        data = dataset.attrib
        ref_originates_from_write_pvd = "time" in data
    pvd_files_compatible = (
        test_originates_from_write_pvd == ref_originates_from_write_pvd
    )

    if not pvd_files_compatible:
        return False

    # Here, we make a simple brute-force comparison, and search for each item in the
    # test file a matching item in the reference file.
    def _check_xml_subtrees(
        tree1: ET.ElementTree, tree2: ET.ElementTree, keys: list[str]
    ) -> bool:
        """Check whether tree1 is a subtree of tree2."""
        # Check each item of tree1
        for dataset1 in tree1.iter("DataSet"):
            data1 = dataset1.attrib

            # Initialize item success
            found_data1 = False

            # Try to find corresponding entry in tree2
            for dataset2 in tree2.iter("DataSet"):
                data2 = dataset2.attrib
                found_data1 = all([data1[key] == data2[key] for key in keys])

                if found_data1:
                    break

            # Failure, if item not part of tree2.
            if not found_data1:
                return False

        # Success, as each item of tree1 has been identified in tree2.
        return True

    def _check_xml_tree_equality(
        tree1: ET.ElementTree, tree2: ET.ElementTree, keys: list[str]
    ) -> bool:
        """Check whether tree1 and tree2 are subtress of each other."""
        return _check_xml_subtrees(tree1, tree2, keys) and _check_xml_subtrees(
            tree2, tree1, keys
        )

    # Check both directions, to check equality. The keys are chosen depending on the
    # origin.
    if test_originates_from_write_pvd:
        keys = ["part", "timestep", "file"]
    else:
        keys = ["part", "file"]
    return _check_xml_tree_equality(tree_test, tree_ref, keys)


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
    """Test of the Exporter for single subdomains of different dimensionality and
    different grid type. Exporting of scalar and vectorial data is tested.

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
def test_import_state_from_vtu_single_subdomains(setup, subdomain):
    # Test of the import routine of the Exporter for single subdomains. Consistent
    # with test_single_subdomains.

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

    # Define keys (here corresponding to all data stored in the vtu file to pass the
    # test).
    keys = ["dummy_scalar", "dummy_vector"]

    # Import data
    save.import_state_from_vtu(
        vtu_files=f"{subdomain.ref_vtu_file}",
        keys=keys,
        automatic=False,
        dims=sd.dim,
    )

    # Perform comparison on vtu level (seems the easiest as it only involves a
    # comparison of dictionaries). This requires test_single_subdomains to pass all
    # tests.
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
        pp.set_solution_values(
            name="dummy_scalar",
            values=np.ones(sd.num_cells) * sd.dim,
            data=sd_data,
            time_step_index=0,
        )

        pp.set_solution_values(
            name="dummy_vector",
            values=np.ones((3, sd.num_cells)) * sd.dim,
            data=sd_data,
            time_step_index=0,
        )

    for intf, intf_data in mdg.interfaces(return_data=True):
        pp.set_solution_values(
            name="dummy_scalar",
            values=np.zeros(intf.num_cells),
            data=intf_data,
            time_step_index=0,
        )

        pp.set_solution_values(
            name="unique_dummy_scalar",
            values=np.zeros(intf.num_cells),
            data=intf_data,
            time_step_index=0,
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


@pytest.mark.parametrize("case", np.arange(2))
def test_import_from_pvd_mdg(setup, case):
    """Test import-from-pvd functionality of the Exporter for 2d mixed-dimensional grids
    for a two-fracture domain.

    Here, purely reading functionality is tested, given a fixed set of input pvd files.

    Two cases are considered, testing two functionalities: importing from a mdg pvd
    file and a (conventional) pvd file, originating from pp.Exporter._export_mdg_pvd()
    and pp.Exporter.write_pvd(). These correspond to case equal 1 and 0, respectively.

    Exporting of scalar and vectorial data, separately defined on both subdomains and
    interfaces.

    """

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

    # Assume the following has been run for a previous simulation
    # save.write_vtu(
    #     ["dummy_scalar", "dummy_vector", "unique_dummy_scalar"],
    #     timestep=1
    # )
    # Yet, then the simulation crashed, now it is restarted from pvd file, picking up
    # the latest available timestep.
    pvd_file = Path(f"{setup.folder_reference}/restart/previous_grid.pvd")
    if case == 0:
        # Test restart from conventional pvd file.
        time_index = save.import_from_pvd(
            pvd_file=pvd_file,
            keys=["dummy_scalar", "dummy_vector", "unique_dummy_scalar"],
        )

    elif case == 1:
        # Test restart from
        time_index = save.import_from_pvd(
            pvd_file=Path(f"{setup.folder_reference}/restart/grid_000001.pvd"),
            is_mdg_pvd=True,
            keys=["dummy_scalar", "dummy_vector", "unique_dummy_scalar"],
        )

    # The above setup has been created such that the imported data corresponds to some
    # first time step. In case 0, this is encoded in the content of previous_grid.pvd.
    # In case 1, this is encoded in the appendix of grid_000001.pvd. Test whether the
    # time index has been identified correctly.
    assert time_index == 1

    # To trick the test, copy the current pvd file to the temporary folder before
    # continuing writing it through appending the next time step.
    Path(f"{setup.folder}").mkdir(parents=True, exist_ok=True)
    shutil.copy(pvd_file, Path(f"{setup.folder}/{setup.file_name}.pvd"))

    # Imitate the initialization of a simulation, i.e., export the initial condition.
    save.write_vtu(["dummy_scalar", "dummy_vector", "unique_dummy_scalar"], time_step=1)
    save.write_pvd(append=True)

    # Now, export both the vtu and the pvd (continuing using the previous one).
    # NOTE: Typically, the data would be modified by running the simulation
    # for one more timestep. This is irrelevant for testing the restarting capabilities.
    save.write_vtu(["dummy_scalar", "dummy_vector", "unique_dummy_scalar"], time_step=2)
    save.write_pvd(append=True)

    # Check that newly exported vtu files and reference files are the same.
    for appendix in ["1", "2", "mortar_1"]:
        assert _compare_vtu_files(
            f"{setup.folder}/{setup.file_name}_{appendix}_000002.vtu",
            f"{setup.folder_reference}/restart/grid_{appendix}_000002.vtu",
        )

    # Check that the newly exported pvd files and reference file are the same.
    for appendix in ["_000002", ""]:
        assert _compare_pvd_files(
            f"{setup.folder}/{setup.file_name}{appendix}.pvd",
            f"{setup.folder_reference}/restart/grid{appendix}.pvd",
        )


@pytest.mark.parametrize("addendum", ["", "nontrivial_data_"])
def test_import_state_from_vtu_mdg(setup, addendum):
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

    # Define keys (here corresponding to all data stored in the vtu file to pass the
    # test).
    keys = ["dummy_scalar", "dummy_vector", "unique_dummy_scalar"]

    # Import data
    save.import_state_from_vtu(
        vtu_files=[
            Path(f"{setup.folder_reference}/mdg_{addendum}grid_2.vtu"),
            Path(f"{setup.folder_reference}/mdg_{addendum}grid_1.vtu"),
            Path(f"{setup.folder_reference}/mdg_{addendum}grid_mortar_1.vtu"),
        ],
        keys=keys,
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
        pp.set_solution_values(
            name="dummy_scalar",
            values=np.ones(sd.num_cells) * sd.dim,
            data=sd_data,
            time_step_index=0,
        )

        pp.set_solution_values(
            name="dummy_vector",
            values=np.ones((3, sd.num_cells)) * sd.dim,
            data=sd_data,
            time_step_index=0,
        )

    for intf, intf_data in mdg.interfaces(return_data=True):
        pp.set_solution_values(
            name="dummy_scalar",
            values=np.zeros(intf.num_cells),
            data=intf_data,
            time_step_index=0,
        )

        pp.set_solution_values(
            name="unique_dummy_scalar",
            values=np.zeros(intf.num_cells),
            data=intf_data,
            time_step_index=0,
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
    fractures = pts_edges_to_linefractures(p, e)
    domain = pp.Domain({"xmin": -2, "xmax": 3, "ymin": -2, "ymax": 3})
    network_2d = pp.create_fracture_network(fractures, domain)

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
    bbox = {"xmin": -2, "xmax": 3, "ymin": -2, "ymax": 3, "zmin": -3, "zmax": 3}
    network_3d = pp.create_fracture_network([f_1, f_2], domain=pp.Domain(bbox))

    # Define data
    num_frac = len(network_3d.fractures)
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
