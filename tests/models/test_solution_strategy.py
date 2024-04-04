"""Test for solution strategy part of models.

We test:
    - Restart: Here, exemplarily for a mixed-dimensional poromechanics model with
    time-varying boundary conditions.

    -Targeted rediscretization: Targeted rediscretization is a feature that allows
    rediscretization of a subset of discretizations defined in the
    nonlinear_discretization property. The list is accessed by
    :meth:`porepy.models.solution_strategy.SolutionStrategy.rediscretize`  called at the
    beginning of a nonlinear iteration.

    - Equation parsing: The tests here considers equations that are present in the
    global system to be solved, opposed to simpler relations (typically constitutive
    relations) that are used to construct these global equations. Tests discretization
    and assembly of the equations. TODO: Might become obsolete if full coverage is
    achieved for the test approach in test_single_phase_flow.py.

"""

from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils import models
from porepy.applications.test_utils.vtk import compare_pvd_files, compare_vtu_files

from .test_poromechanics import TailoredPoromechanics, create_fractured_setup

# Store current directory, directory containing reference files, and temporary
# visualization folder.
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
reference_dir = current_dir / Path("restart_reference")
visualization_dir = Path("visualization")


def create_restart_model(
    solid_vals: dict, fluid_vals: dict, uy_north: float, restart: bool
):
    # Create fractured setup
    fractured_setup = create_fractured_setup(solid_vals, fluid_vals, uy_north)

    # Fetch parameters for enhancing them
    params = fractured_setup.params

    # Enable exporting
    params["times_to_export"] = None

    # Add time stepping to the setup
    params["time_manager"] = pp.TimeManager(
        schedule=[0, 1], dt_init=0.5, constant_dt=True
    )

    # Add restart possibility
    params["restart_options"] = {
        "restart": restart,
        "pvd_file": reference_dir / Path("previous_data.pvd"),
        "times_file": reference_dir / Path("previous_times.json"),
    }

    # Redefine setup
    setup = TailoredPoromechanics(params)
    return setup


@pytest.mark.parametrize(
    "solid_vals,north_displacement",
    [
        ({"porosity": 0.5}, 0.1),
    ],
)
def test_restart(solid_vals, north_displacement):
    """Restart version of .test_poromechanics.test_2d_single_fracture.

    Provided the exported data from a previous time step, restart the simulaton,
    continue running and compare the final state and exported vtu/pvd files with
    reference files.

    This test also serves as minimal documentation of how to restart a model in a
    practical situation.

    Parameters:
        solid_vals (dict): Dictionary with keys as those in :class:`pp.SolidConstants`
            and corresponding values.
        north_displacement (float): Value of displacement on the north boundary.
        expected_x_y (tuple): Expected values of the displacement in the x and y.
            directions. The values are used to infer sign of displacement solution.

    """
    # Setup and run model for full time interval. With this generate reference files
    # for comparison with a restarted simulation. At the same time, this generates the
    # restart files.
    setup = create_restart_model(solid_vals, {}, north_displacement, restart=False)
    pp.run_time_dependent_model(setup, {})

    # The run generates data for initial and the first two time steps. In order to use
    # the data as restart and reference data, move it to a reference folder.
    pvd_files = list(visualization_dir.glob("*.pvd"))
    vtu_files = list(visualization_dir.glob("*.vtu"))
    json_files = list(visualization_dir.glob("*.json"))
    for f in pvd_files + vtu_files + json_files:
        dst = reference_dir / Path(f.stem + f.suffix)
        shutil.move(f, dst)

    # Now use the reference data to restart the simulation. Note, the restart
    # capabilities of the models automatically use the last available time step for
    # restart, here the restart files contain information on the initial and first
    # time step. Thus, the simulation is restarted from the first time step.
    # Recompute the second time step which will serve as foundation for the comparison
    # to the above computed reference files.
    setup = create_restart_model(solid_vals, {}, north_displacement, restart=True)
    pp.run_time_dependent_model(setup, {})

    # To verify the restart capabilities, perform five tests.

    # 1. Check whether the states have been correctly initialized at restart time.
    # Visit all dimensions and the mortar grids for this.
    for i in ["1", "2"]:
        assert compare_vtu_files(
            visualization_dir / Path(f"data_{i}_000001.vtu"),
            reference_dir / Path(f"data_{i}_000001.vtu"),
        )
    assert compare_vtu_files(
        visualization_dir / Path(f"data_mortar_1_000001.vtu"),
        reference_dir / Path(f"data_mortar_1_000001.vtu"),
    )

    # 2. Check whether the successive time step has been computed correctely.
    # Visit all dimensions and the mortar grids for this.
    for i in ["1", "2"]:
        assert compare_vtu_files(
            visualization_dir / Path(f"data_{i}_000002.vtu"),
            reference_dir / Path(f"data_{i}_000002.vtu"),
        )
    assert compare_vtu_files(
        visualization_dir / Path(f"data_mortar_1_000002.vtu"),
        reference_dir / Path(f"data_mortar_1_000002.vtu"),
    )

    # 3. Check whether the mdg pvd file is defined correctly.
    assert compare_pvd_files(
        visualization_dir / Path(f"data_000002.pvd"),
        reference_dir / Path(f"data_000002.pvd"),
    )

    # 4. Check whether the pvd file is compiled correctly, combining old and new data.
    assert compare_pvd_files(
        visualization_dir / Path(f"data.pvd"),
        reference_dir / Path(f"data.pvd"),
    )

    # 5. the logging of times and step sizes is correct.
    restarted_times_json = open(visualization_dir / Path("times.json"))
    reference_times_json = open(reference_dir / Path("times.json"))
    restarted_times = json.load(restarted_times_json)
    reference_times = json.load(reference_times_json)
    for key in ["time", "dt"]:
        assert np.all(
            np.isclose(np.array(restarted_times[key]), np.array(reference_times[key]))
        )
    restarted_times_json.close()
    reference_times_json.close()

    # Remove temporary visualization folder
    shutil.rmtree(visualization_dir)

    # Clean up the reference data
    for f in pvd_files + vtu_files + json_files:
        src = reference_dir / Path(f.stem + f.suffix)
        src.unlink()


class RediscretizationTest:
    """Class to short-circuit the solution strategy to a single iteration.

    The class is used as a mixin which partially replaces the SolutionStrategy class.

    Relevant parts of simulation flow:
    1. Discretize the problem
    2. Assemble the linear system
    3. Solve the linear system
    4. Update the state variables
    5. Check convergence first time (expected not to pass)
    6. Rediscretize at the beginning of the next iteration
    7. Assemble the linear system
    8. Check convergence second time (passes by hard-coding)

    Then, quit the simulation and compare stored linear systems.

    """

    def check_convergence(self, *args, **kwargs):
        if self.nonlinear_solver_statistics.num_iteration > 1:
            return 0.0, 0.0, True, False
        else:
            # Call to super is okay here, since the full model used in the tests is
            # known to have a method of this name.
            return super().check_convergence(*args, **kwargs)

    def rediscretize(self):
        if self.params["full_rediscretization"]:
            return super().discretize()
        else:
            return super().rediscretize()

    def assemble_linear_system(self) -> None:
        """Store all assembled linear systems for later comparison."""
        super().assemble_linear_system()
        if not hasattr(self, "stored_linear_system"):
            self.stored_linear_system = []
        self.stored_linear_system.append(copy.deepcopy(self.linear_system))


# Non-trivial solution achieved through BCs.
class RandomPressureBCs(
    pp.model_boundary_conditions.BoundaryConditionsMassDirNorthSouth,
    pp.model_boundary_conditions.BoundaryConditionsEnergyDirNorthSouth,
):
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries. We
        set the nonzero random pressure on the north boundary, and 0 on the south
        boundary.

        Parameters:
            boundary_grid: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.

        """
        domain_sides = self.domain_boundary_sides(boundary_grid)
        vals_loc = np.zeros(boundary_grid.num_cells)
        # Fix the random seed to make it possible to debug in the future.
        np.random.seed(0)
        vals_loc[domain_sides.north] = np.random.rand(domain_sides.north.sum())
        return vals_loc


# No need to test momentum balance, as it contains no discretizations that need
# rediscretization.
model_classes = [
    models._add_mixin(RandomPressureBCs, models.MassBalance),
    models._add_mixin(RandomPressureBCs, models.MassAndEnergyBalance),
    models._add_mixin(RandomPressureBCs, models.Poromechanics),
    models._add_mixin(RandomPressureBCs, models.Thermoporomechanics),
]


@pytest.mark.parametrize("model_class", model_classes)
def test_targeted_rediscretization(model_class):
    """Test that targeted rediscretization yields same results as full discretization."""
    params_full = {
        "full_rediscretization": True,
        "fracture_indices": [0, 1],
        "cartesian": True,
        # Make flow problem non-linear:
        "material_constants": {"fluid": pp.FluidConstants({"compressibility": 1})},
    }
    # Finalize the model class by adding the rediscretization mixin.
    rediscretization_model_class = models._add_mixin(RediscretizationTest, model_class)
    # A model object with full rediscretization.
    model_full = rediscretization_model_class(params_full)
    pp.run_time_dependent_model(model_full, params_full)

    # A model object with targeted rediscretization.
    params_targeted = params_full.copy()
    params_targeted["full_rediscretization"] = False
    # Set up the model.
    model_targeted = rediscretization_model_class(params_targeted)
    pp.run_time_dependent_model(model_targeted, params_targeted)

    # Check that the linear systems are the same.
    assert len(model_full.stored_linear_system) == 2
    assert len(model_targeted.stored_linear_system) == 2
    for i in range(len(model_full.stored_linear_system)):
        A_full, b_full = model_full.stored_linear_system[i]
        A_targeted, b_targeted = model_targeted.stored_linear_system[i]

        # Convert to dense array to ensure the matrices are identical.
        assert np.allclose(A_full.A, A_targeted.A)
        assert np.allclose(b_full, b_targeted)

    # Check that the discretization matrix changes between iterations. Without this
    # check, missing rediscretization may go unnoticed.
    tol = 1e-2
    diff = model_full.stored_linear_system[0][0] - model_full.stored_linear_system[1][0]
    assert np.linalg.norm(diff.todense()) > tol


@pytest.mark.parametrize(
    "model_type,equation_name,only_codimension",
    [
        ("mass_balance", "mass_balance_equation", None),
        ("mass_balance", "interface_darcy_flux_equation", None),
        ("momentum_balance", "momentum_balance_equation", 0),
        ("momentum_balance", "interface_force_balance_equation", 1),
        ("momentum_balance", "normal_fracture_deformation_equation", 1),
        ("momentum_balance", "tangential_fracture_deformation_equation", 1),
        ("energy_balance", "energy_balance_equation", None),
        ("energy_balance", "interface_enthalpy_flux_equation", None),
        ("energy_balance", "interface_fourier_flux_equation", None),
        # Energy balance inherits mass balance equations. Test one of these as well.
        ("energy_balance", "mass_balance_equation", None),
        ("poromechanics", "mass_balance_equation", None),
        ("poromechanics", "momentum_balance_equation", 0),
        ("poromechanics", "interface_force_balance_equation", 1),
        ("thermoporomechanics", "interface_fourier_flux_equation", None),
    ],
)
# Run the test for models with and without fractures. We skip the case of more than one
# fracture, since it seems unlikely this will uncover any errors that will not be found
# with the simpler setups. Activate more fractures if needed in debugging.
@pytest.mark.parametrize(
    "num_fracs",
    [  # Number of fractures
        0,
        1,
        pytest.param(2, marks=pytest.mark.skipped),
        pytest.param(3, marks=pytest.mark.skipped),
    ],
)
@pytest.mark.parametrize("domain_dim", [2, 3])
def test_parse_equations(
    model_type, equation_name, only_codimension, num_fracs, domain_dim
):
    """Test that equation parsing works as expected.

    Currently tested are the relevant equations in the mass balance and momentum balance
    models. To add a new model, add a new class in setup_utils.py and expand the
    test parameterization accordingly.

    Parameters:
        model_type: Type of model to test. Currently supported are "mass_balance" and
            "momentum_balance". To add a new model, add a new class in setup_utils.py.
        equation_name: Name of the method to test.
        domain_inds: Indices of the domains for which the method should be called.
            Some methods are only defined for a subset of domains.

    """
    if only_codimension is not None:
        if only_codimension == 0 and num_fracs > 0:
            # If the test is to be run on the top domain only, we need not consider
            # models with fractures (note that since we iterate over num_fracs, the
            # test will be run for num_fracs = 0, which is the top domain only)
            return
        elif only_codimension == 1 and num_fracs == 0:
            # If the point of the test is to check the method on a fracture, we need
            # not consider models without fractures.
            return
        else:
            # We will run the test, but only on the specified codimension.
            dimensions_to_assemble = domain_dim - only_codimension
    else:
        # Test on subdomains or interfaces of all dimensions
        dimensions_to_assemble = None

    if domain_dim == 2 and num_fracs > 2:
        # The 2d models are not defined for more than two fractures.
        return

    # Set up an object of the prescribed model
    setup = models.model(model_type, domain_dim, num_fracs=num_fracs)
    # Fetch the relevant method of this model and extract the domains for which it is
    # defined.
    method = getattr(setup, equation_name)
    domains = models.subdomains_or_interfaces_from_method_name(
        setup.mdg, method, dimensions_to_assemble
    )

    # Call the discretization method.
    setup.equation_system.discretize()

    # Assemble the matrix and right hand side for the given equation. An error here will
    # indicate that something is wrong with the way the conservation law combines
    # terms and factors (e.g., grids, parameters, variables, other methods etc.) to form
    # an Ad operator object.
    setup.equation_system.assemble({equation_name: domains})
