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
import shutil
from pathlib import Path
from typing import Any, Callable, Optional, cast

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.applications.md_grids.mdg_library import (
    cube_with_orthogonal_fractures,
    square_with_orthogonal_fractures,
)
from porepy.applications.test_utils import models
from porepy.applications.test_utils.models import add_mixin
from porepy.applications.test_utils.vtk import compare_pvd_files, compare_vtu_files

from ..functional.setups.linear_tracer import TracerFlowModel_3p
from .test_poromechanics import TailoredPoromechanics, create_model_with_fracture

# Store current directory, directory containing reference files, and temporary
# visualization folder.
current_dir = Path(__file__).parent
reference_dir = current_dir / "restart_reference"
visualization_dir = Path("visualization")


def create_restart_model(
    solid_vals: dict, fluid_vals: dict, uy_north: float, restart: bool
) -> TailoredPoromechanics:
    # Create model with a fractured geometry.
    model = create_model_with_fracture(
        solid_vals, fluid_vals, {}, uy_north, TailoredPoromechanics
    )

    # Fetch parameters for enhancing them.
    params = model.params

    # Enable exporting
    params["times_to_export"] = None

    # Add time stepping to the model
    params["time_manager"] = pp.TimeManager(
        schedule=[0, 1], dt_init=0.5, constant_dt=True
    )

    # Add restart possibility.
    params["restart_options"] = {
        "restart": restart,
        "pvd_file": reference_dir / "previous_data.pvd",
        "times_file": reference_dir / "previous_times.json",
    }

    # Redefine model.
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
        # Case 5: Residual is above divergence tolerance.
        (np.array([1e-6, 1e-6]), np.array([2e4]), (False, True)),
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
        "nl_divergence_tol": 1e4,
    }
    converged, diverged = check_convergence_test_model.check_convergence(
        nonlinear_increment, residual, nl_params
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


def _get_primary_equ_and_vars_cf(model: pp.PorePyModel) -> tuple[list[str], list[str]]:
    """Returns the primary equations and variables of a CF model."""

    equ_names: list[str] = []
    if isinstance(model, pp.fluid_mass_balance.FluidMassBalanceEquations):
        equ_names += [
            pp.fluid_mass_balance.FluidMassBalanceEquations.primary_equation_name()
        ]
    if isinstance(model, pp.energy_balance.TotalEnergyBalanceEquations):
        equ_names += [
            pp.energy_balance.TotalEnergyBalanceEquations.primary_equation_name()
        ]
    if isinstance(model, pp.compositional_flow.ComponentMassBalanceEquations):
        equ_names += model.component_mass_balance_equation_names()

    for n in model.equation_system.equations.keys():
        if "_flux" in n:
            equ_names.append(n)

    var_names: list[str] = []
    if isinstance(model, pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow):
        var_names += [model.pressure_variable]

    if isinstance(
        model, pp.compositional_flow.SolutionStrategyExtendedFluidMassAndEnergy
    ):
        var_names += [model.enthalpy_variable]
    elif isinstance(model, pp.energy_balance.SolutionStrategyEnergyBalance):
        var_names += [model.temperature_variable]

    if isinstance(model, pp.compositional.CompositionalVariables):
        var_names += model.overall_fraction_variables
        var_names += model.tracer_fraction_variables

    for var in model.equation_system.variables:
        if "_flux" in var.name:
            var_names.append(var.name)

    return list(set(equ_names)), list(set(var_names))


@pytest.mark.parametrize(
    "test_model_class,get_primary_equs_vars",
    [
        (TracerFlowModel_3p, _get_primary_equ_and_vars_cf),
    ],
)
@pytest.mark.parametrize(
    "mdg",
    [
        square_with_orthogonal_fractures("cartesian", {"cell_size": 0.25}, [0, 1])[0],
        cube_with_orthogonal_fractures("cartesian", {"cell_size": 0.25}, [0, 1, 2])[0],
    ],
)
def test_schur_complement_inverter_on_model(
    mdg: pp.MixedDimensionalGrid,
    test_model_class: type[pp.PorePyModel],
    get_primary_equs_vars: Callable[[pp.PorePyModel], tuple[list[str], list[str]]],
):
    """Tests the block-diagonal inverter for the secondary block of the linear system
    of a porepy model."""

    # NOTE: Depending on what which model class the tests are performed, the local
    # geometry mixin and model parameters need adaption in order to overwrite the
    # geometry and parametrization already contained within the tested model class.
    # The adaption must be consistent with the mdg's the tests are performed on.
    class LocalGeometry(pp.PorePyModel):
        def create_mdg(self) -> None:
            self.mdg = mdg

        def set_domain(self) -> None:
            self._domain = nd_cube_domain(
                mdg.dim_max(), self.units.convert_units(1.0, "m")
            )

    model_params = {
        "apply_schur_complement_reduction": True,
        "equilibrium_condition": "dummy",
        "meshing_arguments": {
            "cell_size": 0.1,
        },
    }

    model_class = add_mixin(LocalGeometry, test_model_class)

    model = model_class(model_params)
    model = cast(pp.SolutionStrategy, model)
    model.prepare_simulation()
    prim_equs, prim_vars = get_primary_equs_vars(model)

    # Set primary equations and variables to get the secondary ones for assembly of
    # secondary block.
    model.schur_complement_primary_equations = prim_equs
    model.schur_complement_primary_variables = prim_vars

    secondary_equs = list(
        set(model.equation_system.equations.keys()).difference(set(prim_equs))
    )
    secondary_vars = list(
        set([var.name for var in model.equation_system.variables]).difference(
            set(prim_vars)
        )
    )

    N = model.equation_system.dofs_of(secondary_vars).size
    model.equation_system.set_variable_values(
        np.ones(model.equation_system.num_dofs()), iterate_index=0, time_step_index=0
    )

    model.before_nonlinear_loop()
    model.before_nonlinear_iteration()
    model.assemble_linear_system()
    inv_A_ss = model.equation_system._Schur_complement[0]

    # NOTE A_ss is not stored explicitly as part of the linear system assembly.
    A_ss, _ = model.equation_system.assemble(
        equations=secondary_equs,
        variables=secondary_vars,
    )

    # NOTE: Do not convert to dense array, 3D test case will run out of memory.
    approx_identity = cast(sps.csr_matrix, A_ss @ inv_A_ss)
    identity = cast(sps.csr_matrix, sps.eye(N, format="csr"))

    assert (
        np.all(approx_identity.indices == identity.indices)
        and np.all(approx_identity.indptr == identity.indptr)
        and np.allclose(
            approx_identity.data,
            np.ones(N),
            rtol=0.0,
            atol=1e-16,
        )
        and np.all(approx_identity.shape == identity.shape)
    )
