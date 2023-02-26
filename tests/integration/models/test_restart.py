from __future__ import annotations

import pytest

import porepy as pp

# TODO use _compare_vtu_files from unit.test_vtk after debugging
import meshio
from deepdiff import DeepDiff
from ...unit.test_vtk import _compare_pvd_files

# from ...unit.test_vtk import _compare_pvd_files, _compare_vtu_files
from .setup_utils import (
    TimeDependentMechanicalBCsDirNorthSouth,
    RectangularDomainThreeFractures,
)
from .test_poromechanics import (
    NonzeroFractureGapPoromechanics,
    BoundaryConditionsMassAndEnergyDirNorthSouth,
)


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
    if diff != {}:
        print(diff)
    return diff == {}


class DynamicConstitutiveLawsPoromechanics(
    pp.constitutive_laws.CubicLawPermeability,
    pp.poromechanics.ConstitutiveLawsPoromechanics,
):
    pass


class DynamicPoromechanics(
    pp.poromechanics.EquationsPoromechanics,
    pp.poromechanics.VariablesPoromechanics,
    DynamicConstitutiveLawsPoromechanics,
    pp.poromechanics.BoundaryConditionsPoromechanics,
    pp.poromechanics.SolutionStrategyPoromechanics,
    pp.ModelGeometry,
    pp.DataSavingMixin,
):
    pass


class RectangularPoromechanics(
    RectangularDomainThreeFractures,
    DynamicPoromechanics,
):
    pass


class TailoredPoromechanics(
    NonzeroFractureGapPoromechanics,
    TimeDependentMechanicalBCsDirNorthSouth,
    BoundaryConditionsMassAndEnergyDirNorthSouth,
    RectangularPoromechanics,
):
    pass


def create_fractured_setup(solid_vals, fluid_vals, uy_north):
    """Create a setup for a fractured domain.

    This is an enhanced copy of .test_poromechanics.create_fractured_setup.
    It enables multiple time steps, and the export of the solution.

    Parameters:
        solid_vals (dict): Parameters for the solid mechanics model.
        fluid_vals (dict): Parameters for the fluid mechanics model.
        uy_north (float): Displacement in y-direction on the north boundary.

    Returns:
        TailoredPoromechanics: A setup for a fractured domain.

    """
    # Instantiate constants and store in params.
    solid_vals["fracture_gap"] = 0.042
    solid_vals["residual_aperture"] = 1e-10
    solid_vals["biot_coefficient"] = 1.0
    fluid_vals["compressibility"] = 1
    solid = pp.SolidConstants(solid_vals)
    fluid = pp.FluidConstants(fluid_vals)

    params = {
        "suppress_export": False,  # Suppress output for tests
        "material_constants": {"solid": solid, "fluid": fluid},
        "uy_north": uy_north,
        "max_iterations": 20,
        # TODO fix DataSavingMixin.write_pvd and use [0,1] with dt_init=0.5 here.
        "time_manager": pp.TimeManager(schedule=[0, 2], dt_init=1, constant_dt=True),
        "restart_options": {
            "restart": True,
            "reuse_dt": True,
            "file": "./restart_reference/previous_data.pvd",
        },
    }
    setup = TailoredPoromechanics(params)
    return setup


@pytest.mark.parametrize(
    "solid_vals,north_displacement",
    [
        ({"porosity": 0.5}, 0.1),
    ],
)
def test_restart_2d_single_fracture(solid_vals, north_displacement):
    """Restart version of .test_poromechanics.test_2d_single_fracture.

    Provided the exported data from a previous time step, restart the
    simulaton, continue running and compare the final state and exported
    vtu/pvd files with reference files.

    Parameters:
        solid_vals (dict): Dictionary with keys as those in :class:`pp.SolidConstants`
            and corresponding values.
        north_displacement (float): Value of displacement on the north boundary.
        expected_x_y (tuple): Expected values of the displacement in the x and y.
            directions. The values are used to infer sign of displacement solution.

    """
    # Setup and run model
    setup = create_fractured_setup(solid_vals, {}, north_displacement)
    pp.run_time_dependent_model(setup, {})

    # Compare final solution with reference solution
    for ending in ["_000002", ""]:
        assert _compare_pvd_files(
            f"./visualization/data{ending}.pvd",
            f"./restart_reference/data{ending}.pvd",
        )

    for ending in ["000001", "000002"]:
        for i in ["1", "2"]:
            # TODO make assert again
            print(
                i,
                ending,
                _compare_vtu_files(
                    f"./visualization/data_{i}_{ending}.vtu",
                    f"./restart_reference/data_{i}_{ending}.vtu",
                ),
            )

        print(
            ending,
            _compare_vtu_files(
                f"./visualization/data_mortar_1_{ending}.vtu",
                f"./restart_reference/data_mortar_1_{ending}.vtu",
            ),
        )

    assert False
    # TODO rm visualization
