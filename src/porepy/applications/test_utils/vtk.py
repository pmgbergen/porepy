"""Test helpers for the vtk files.
"""
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import meshio

# Data structure for defining paths
PathLike = str | Path


def compare_vtu_files(
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

    from deepdiff import DeepDiff

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


def compare_pvd_files(
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
