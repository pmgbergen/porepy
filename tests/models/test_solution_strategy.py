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
from typing import Any, Optional

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.applications.test_utils import models
from porepy.applications.test_utils.vtk import compare_pvd_files, compare_vtu_files

from .test_poromechanics import TailoredPoromechanics, create_model_with_fracture
from porepy.applications.md_grids.mdg_library import square_with_orthogonal_fractures

# Store current directory, directory containing reference files, and temporary
# visualization folder.
current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
reference_dir = current_dir / Path("restart_reference")
visualization_dir = Path("visualization")


def create_restart_model(
    solid_vals: dict, fluid_vals: dict, uy_north: float, restart: bool
) -> TailoredPoromechanics:
    # Create model with a fractured geometry
    model = create_model_with_fracture(solid_vals, fluid_vals, {}, uy_north)

    # Fetch parameters for enhancing them
    params = model.params

    # Enable exporting
    params["times_to_export"] = None

    # Add time stepping to the model
    params["time_manager"] = pp.TimeManager(
        schedule=[0, 1], dt_init=0.5, constant_dt=True
    )

    # Add restart possibility
    params["restart_options"] = {
        "restart": restart,
        "pvd_file": reference_dir / Path("previous_data.pvd"),
        "times_file": reference_dir / Path("previous_times.json"),
    }

    # Redefine model
    model = TailoredPoromechanics(params)
    return model


@pytest.mark.parametrize(
    "solid_vals,north_displacement",
    [
        ({"porosity": 0.5}, 0.1),
    ],
)
def test_restart(solid_vals: dict, north_displacement: float):
    """Restart version of .test_poromechanics.test_2d_single_fracture.

    Provided the exported data from a previous time step, restart the simulaton,
    continue running and compare the final state and exported vtu/pvd files with
    reference files.

    This test also serves as minimal documentation of how to restart a model in a
    practical situation.

    Parameters:
        solid_vals: Dictionary with keys as those in :class:`pp.SolidConstants` and
            corresponding values.
        north_displacement: Value of displacement on the north boundary.

    """
    # Set up and run model for full time interval. With this generate reference files
    # for comparison with a restarted simulation. At the same time, this generates the
    # restart files.
    model = create_restart_model(solid_vals, {}, north_displacement, restart=False)
    pp.run_time_dependent_model(model)

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
    # restart, here the restart files contain information on the initial and first time
    # step. Thus, the simulation is restarted from the first time step. Recompute the
    # second time step which will serve as foundation for the comparison to the above
    # computed reference files.
    model = create_restart_model(solid_vals, {}, north_displacement, restart=True)
    pp.run_time_dependent_model(model)

    # To verify the restart capabilities, perform five tests.

    # 1. Check whether the states have been correctly initialized at restart time.
    # Visit all dimensions and the mortar grids for this.
    for i in ["1", "2"]:
        assert compare_vtu_files(
            visualization_dir / Path(f"data_{i}_000001.vtu"),
            reference_dir / Path(f"data_{i}_000001.vtu"),
        )
    assert compare_vtu_files(
        visualization_dir / Path("data_mortar_1_000001.vtu"),
        reference_dir / Path("data_mortar_1_000001.vtu"),
    )

    # 2. Check whether the successive time step has been computed correctly.
    # Visit all dimensions and the mortar grids for this.
    for i in ["1", "2"]:
        assert compare_vtu_files(
            visualization_dir / Path(f"data_{i}_000002.vtu"),
            reference_dir / Path(f"data_{i}_000002.vtu"),
        )
    assert compare_vtu_files(
        visualization_dir / Path("data_mortar_1_000002.vtu"),
        reference_dir / Path("data_mortar_1_000002.vtu"),
    )

    # 3. Check whether the mdg pvd file is defined correctly.
    assert compare_pvd_files(
        visualization_dir / Path("data_000002.pvd"),
        reference_dir / Path("data_000002.pvd"),
    )

    # 4. Check whether the pvd file is compiled correctly, combining old and new data.
    assert compare_pvd_files(
        visualization_dir / Path("data.pvd"),
        reference_dir / Path("data.pvd"),
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

    # Remove temporary visualization folder.
    shutil.rmtree(visualization_dir)

    # Clean up the reference data.
    for f in pvd_files + vtu_files + json_files:
        src = reference_dir / Path(f.stem + f.suffix)
        src.unlink()


class RediscretizationTest(pp.PorePyModel):
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
            return True, False
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
    def bc_values_pressure(self, bg: pp.BoundaryGrid) -> np.ndarray:
        """Boundary condition values for Darcy flux.

        Dirichlet boundary conditions are defined on the north and south boundaries. We
        set the nonzero random pressure on the north boundary, and 0 on the south
        boundary.

        Parameters:
            bg: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.

        """
        domain_sides = self.domain_boundary_sides(bg)
        vals_loc = np.zeros(bg.num_cells)
        # Fix the random seed to make it possible to debug in the future.
        np.random.seed(0)
        vals_loc[domain_sides.north] = np.random.rand(domain_sides.north.sum())
        return vals_loc


# No need to test momentum balance, as it contains no discretizations that need
# rediscretization.
model_classes: list[type[pp.PorePyModel]] = [
    models.add_mixin(RandomPressureBCs, models.MassBalance),
    models.add_mixin(RandomPressureBCs, models.MassAndEnergyBalance),
    models.add_mixin(RandomPressureBCs, models.Poromechanics),
    models.add_mixin(RandomPressureBCs, models.Thermoporomechanics),
]


@pytest.mark.parametrize("model_class", model_classes)
def test_targeted_rediscretization(model_class):
    """Test that targeted rediscretization yields same results as full
    discretization."""
    model_params = {
        "fracture_indices": [0, 1],
        "full_rediscretization": True,
        "cartesian": True,
        # Make flow problem non-linear:
        "material_constants": {"fluid": pp.FluidComponent(compressibility=1.0)},
        "times_to_export": [],
    }
    # Finalize the model class by adding the rediscretization mixin.
    rediscretization_model_class = models.add_mixin(RediscretizationTest, model_class)
    # A model object with full rediscretization.
    full_model: pp.PorePyModel = rediscretization_model_class(model_params)
    pp.run_time_dependent_model(full_model)

    # A model object with targeted rediscretization.
    targeted_model_params = model_params.copy()
    targeted_model_params["full_rediscretization"] = False
    # Set up the model.
    targeted_model = rediscretization_model_class(targeted_model_params)
    pp.run_time_dependent_model(targeted_model)

    # Check that the linear systems are the same.
    assert len(full_model.stored_linear_system) == 2
    assert len(targeted_model.stored_linear_system) == 2
    for i in range(len(full_model.stored_linear_system)):
        A_full, b_full = full_model.stored_linear_system[i]
        A_targeted, b_targeted = targeted_model.stored_linear_system[i]

        # Convert to dense array to ensure the matrices are identical.
        assert np.allclose(A_full.toarray(), A_targeted.toarray())
        assert np.allclose(b_full, b_targeted)

    # Check that the discretization matrix changes between iterations. Without this
    # check, missing rediscretization may go unnoticed.
    tol = 1e-2
    diff = full_model.stored_linear_system[0][0] - full_model.stored_linear_system[1][0]
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
        ("poromechanics", "normal_fracture_deformation_equation", 1),
        ("thermoporomechanics", "interface_fourier_flux_equation", None),
    ],
)
# Run the test for models with and without fractures. We skip the case of more than one
# fracture, since it seems unlikely this will uncover any errors that will not be found
# with the simpler models. Activate more fractures if needed in debugging.
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
    model_type: str,
    equation_name: str,
    only_codimension: Optional[int],
    num_fracs: int,
    domain_dim: int,
):
    """Test that equation parsing works as expected.

    Currently tested are the relevant equations in the mass balance and momentum balance
    models. To add a new model, add a new class in setup_utils.py and expand the test
    parameterization accordingly.

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
    model = models.model(model_type, domain_dim, num_fracs=num_fracs)
    # Fetch the relevant method of this model and extract the domains for which it is
    # defined.
    method = getattr(model, equation_name)
    domains = models.subdomains_or_interfaces_from_method_name(
        model.mdg, method, dimensions_to_assemble
    )

    # Call the discretization method.
    model.equation_system.discretize()

    # Assemble the matrix and right hand side for the given equation. An error here will
    # indicate that something is wrong with the way the conservation law combines
    # terms and factors (e.g., grids, parameters, variables, other methods etc.) to form
    # an Ad operator object.
    model.equation_system.assemble({equation_name: domains})


class CheckConvergenceTest(pp.SolutionStrategy):
    """Class to ."""

    def __init__(self, params: dict):
        super().__init__(params)
        self.nonlinear_solver_statistics = pp.SolverStatistics()


@pytest.fixture
def check_convergence_test_model() -> CheckConvergenceTest:
    return CheckConvergenceTest({})


@pytest.mark.parametrize(
    "nonlinear_increment,residual,expected",
    [
        # Case 1: Both increment and residual are below tolerance.
        (np.array([1e-6, 1e-6]), np.array([1e-6]), (True, False)),
        # Case 2: Increment is above tolerance.
        (np.array([1e-6, 1]), np.array([1e-6]), (False, False)),
        # Case 3: Residual is above tolerance.
        (np.array([1e-6, 1e-6]), np.array([1]), (False, False)),
        # Case 4: Increment is nan.
        (np.array([np.nan, 0.1]), np.array([1e-6]), (False, True)),
    ],
)
def test_check_convergence(
    nonlinear_increment: np.ndarray,
    residual: np.ndarray,
    expected: tuple[bool, bool],
    check_convergence_test_model: CheckConvergenceTest,
):
    """Test that ``SolutionStrategy.check_convergence`` returns the right
    diverged/converged values.

    """
    # ``reference_residual`` is not used by ``check_convergence``. We pass a zero
    # arrray.
    nl_params: dict[str, Any] = {
        "nl_convergence_tol": 1e-5,
        "nl_convergence_tol_res": 1e-5,
    }
    converged, diverged = check_convergence_test_model.check_convergence(
        nonlinear_increment, residual, np.zeros(1), nl_params
    )
    assert (converged, diverged) == expected


@pytest.mark.parametrize(
    "params",
    [
        # Mass balance must be nonlinear, no matter with or without fractures.
        dict(model_name="mass_balance", num_fracs=0, is_nonlinear=True),
        dict(model_name="mass_balance", num_fracs=1, is_nonlinear=True),
        # Momentum balance without fractures is linear.
        dict(model_name="momentum_balance", num_fracs=0, is_nonlinear=False),
        dict(model_name="momentum_balance", num_fracs=1, is_nonlinear=True),
        dict(model_name="mass_and_energy_balance", num_fracs=0, is_nonlinear=True),
        dict(model_name="poromechanics", num_fracs=0, is_nonlinear=True),
        # There was a bug here once #1350.
        dict(model_name="thermoporomechanics", num_fracs=0, is_nonlinear=True),
        dict(model_name="thermoporomechanics", num_fracs=1, is_nonlinear=True),
        dict(model_name="contact_mechanics", num_fracs=1, is_nonlinear=True),
    ],
)
def test_linear_or_nonlinear_model(params: dict):
    """Tests that the base models are properly tagged as linear or nonlinear."""
    model_name: str = params["model_name"]
    num_fracs: int = params["num_fracs"]
    is_nonlinear: bool = params["is_nonlinear"]

    model = models.model(model_type=model_name, dim=2, num_fracs=num_fracs)
    assert model._is_nonlinear_problem() == is_nonlinear


# ---------------------------------------------------------------------
# Tests for block‐permutation and inversion in SolutionStrategy
# ---------------------------------------------------------------------


@pytest.fixture(
    params=[
        dict(
            A=sps.csr_matrix(
                np.array(
                    [
                        [1, 0, 2, 0, 0, 0],  # cell 0, var types 0,1,2
                        [0, 1, 0, 3, 0, 0],  # cell 1, var types 0,1,2
                        [1, 0, 4, 0, 0, 0],
                        [0, 1, 0, 5, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                    ]
                ),
                dtype=np.float64,
            ),
            block_sizes=[2, 2, 1, 1],
            row_perm=[0, 2, 1, 3, 4, 5],
            col_perm=[0, 2, 1, 3, 4, 5],
        ),
        dict(
            A=sps.csr_matrix(
                np.array(
                    [
                        [1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 2, 0],
                        [0, 0, 0, 1, 0, 3],
                        [0, 0, 1, 0, 4, 0],
                        [0, 0, 0, 1, 0, 5],
                    ]
                ),
                dtype=np.float64,
            ),
            block_sizes=[1, 1, 2, 2],
            row_perm=[0, 1, 2, 4, 3, 5],
            col_perm=[0, 1, 2, 4, 3, 5],
        ),
        dict(
            A=sps.csr_matrix(
                np.array(
                    [
                        [0, 0, 1, 1],
                        [0, 0, 0, 1],
                        [1, 1, 0, 0],
                        [0, 1, 0, 0],
                    ]
                ),
                dtype=np.float64,
            ),
            block_sizes=[2, 2],
            row_perm=[0, 1, 2, 3],
            col_perm=[2, 3, 0, 1],
        ),
    ]
)
def non_block_diag_matrix(request) -> dict[str, Any]:
    """Fixture to provide a non-diagonal block matrix for testing."""
    return request.param


def test_generate_block_permutations(non_block_diag_matrix: dict[str, Any]):
    """
    Test that generate_block_permutations correctly identifies the 4 diagonal
    blocks in a 2-cell, 3-variable/equation-per-cell system and returns the expected
    row/column permutations and block sizes.
    """

    # Build a 6×6 non-diagonal block matrix for 2 cells (each with 3 eqns/vars)

    strategy = pp.SolutionStrategy({})
    row_perm, col_perm, block_sizes = strategy.generate_block_permutations(
        non_block_diag_matrix["A"]
    )

    # Expect four blocks: two blocks of size 2 (for vars 0&1 on each cell),
    # and two singleton blocks (for var type 2 on each cell)
    assert list(block_sizes) == non_block_diag_matrix["block_sizes"]

    # The permutations  groups eqns/vars by cell:
    assert list(row_perm) == non_block_diag_matrix["row_perm"]
    assert list(col_perm) == non_block_diag_matrix["col_perm"]


def test_invert_non_diagonal_matrix(non_block_diag_matrix: dict[str, Any]):
    """Test that invert_non_diagonal_matrix correctly inverts a block-structured
    sparse matrix by permuting to block-diagonal, inverting each block, and
    undoing the permutations.
    """

    strategy = pp.SolutionStrategy({})
    A = non_block_diag_matrix["A"]
    A_inv = strategy.invert_non_diagonal_matrix(A)

    # Verify that A * A_inv is the identity matrix
    approx_identity = A.dot(A_inv).toarray()
    assert np.allclose(approx_identity, np.eye(A.shape[0]))


def test_invert_non_diagonal_matrix_on_mdg_problem():
    """
    Construct a mixed-dimensional system (two orthogonal fractures in a Cartesian grid),
    register variables and equations on subdomains and interfaces, then build and invert its
    Schur-complement secondary block via block-diagonal permutations.
    """
    # Create a 2‐fracture “square” MD grid (Cartesian, cell size 0.5)
    mdg, _ = square_with_orthogonal_fractures("cartesian", {"cell_size": 0.1}, [0, 1])

    # Instantiate an EquationSystem on the MD grid
    es = pp.ad.EquationSystem(mdg)

    # Create four “subdomain” variables (p1, p2, s1, s2), one DOF per cell
    p1 = es.create_variables(name="p1", subdomains=mdg.subdomains())
    p2 = es.create_variables(name="p2", subdomains=mdg.subdomains())
    s1 = es.create_variables(name="s1", subdomains=mdg.subdomains())
    s2 = es.create_variables(name="s2", subdomains=mdg.subdomains())

    # Create three “interface” variables (pf, sf1, sf2), one DOF per interface cell
    pf = es.create_variables(name="pf",  interfaces=mdg.interfaces())
    sf1 = es.create_variables(name="sf1", interfaces=mdg.interfaces())
    sf2 = es.create_variables(name="sf2", interfaces=mdg.interfaces())

    # Define equation‐to‐grid-entity mapping: one equation per cell, none on faces or nodes
    eq_per_gridEntity = {"cells": 1, "faces": 0, "nodes": 0}

    # On each subdomain cell, register 4 equations
    expr_p1 = p1 + s1 * 2.0 - 1.0
    expr_p1.set_name("eq_p1")
    es.set_equation(expr_p1, mdg.subdomains(), eq_per_gridEntity)

    expr_p2 = p2 * 1.0 + s2 * 2.0 - 2.0
    expr_p2.set_name("eq_p2")
    es.set_equation(expr_p2, mdg.subdomains(), eq_per_gridEntity)

    expr_s1 = p1 * 3.0 + s1 * 1.0 - 3.0
    expr_s1.set_name("eq_s1")
    es.set_equation(expr_s1, mdg.subdomains(), eq_per_gridEntity)

    expr_s2 = p2 * 3.0 + s2 * 1.0 - 4.0
    expr_s2.set_name("eq_s2")
    es.set_equation(expr_s2, mdg.subdomains(), eq_per_gridEntity)

    # On each interface, register 3 equations 
    eq_pf = pf**2.0 - 2.0
    eq_pf.set_name("eq_p_f")
    es.set_equation(eq_pf, mdg.interfaces(), eq_per_gridEntity)

    eq_sf1 = pf + sf2 + sf1 * 2.5 - 1.0
    eq_sf1.set_name("eq_s_f_1")
    es.set_equation(eq_sf1, mdg.interfaces(), eq_per_gridEntity)

    eq_sf2 = pf + sf2 * sf1 + sf1 - 10.0
    eq_sf2.set_name("eq_s_f_2")
    es.set_equation(eq_sf2, mdg.interfaces(), eq_per_gridEntity)

    # Define “primary” lists of equations & variables
    primaryEqList = ["eq_p1", "eq_p2", "eq_p_f", "eq_s1", "eq_s2"]
    primaryVarList = ["p1",   "p2",   "pf",   "s1",   "s2"]

    # Use a dummy initial state (zeros) for assembling the Schur complement
    dummy_state = np.zeros(es.num_dofs())

    strategy = pp.SolutionStrategy({})

    # Apply Schur complement to eliminate sf1 and sf2
    _, _ = es.assemble_schur_complement_system(
        primaryEqList,
        primaryVarList,
        state=dummy_state,
        inverter=strategy.invert_non_diagonal_matrix
    )

    # Extract secondary block matrix and its inverse from the Schur complement. 
    inv_A_ss, _, _, _, _ = es._Schur_complement
    A_ss = es.A_secondary_block

    # Verify that A_ss * inv_A_ss is the identity matrix
    approx_identity = A_ss.dot(inv_A_ss).toarray()
    assert np.allclose(approx_identity, np.eye(A_ss.shape[0]))
