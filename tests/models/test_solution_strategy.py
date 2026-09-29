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

import porepy as pp
from porepy.applications.md_grids.domains import nd_cube_domain
from porepy.applications.md_grids.mdg_library import (
    cube_with_orthogonal_fractures,
    square_with_orthogonal_fractures,
)
from porepy.applications.test_utils import models
from porepy.applications.test_utils.models import add_mixin
from porepy.applications.test_utils.vtk import compare_pvd_files, compare_vtu_files
from porepy.models.metric import EuclideanMetric
from porepy.numerics.ad.equation_system import GridEntity
from porepy.numerics.solvers.convergence_check import (
    ConvergenceCriteria,
    ConvergenceStatus,
    DivergenceCriteria,
)

from ..functional.setups.linear_tracer import TracerFlowModel_3p
from .test_fluid_mass_balance import WellModel
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


@pytest.fixture(scope="module")
def restarted_poromechanics(tmp_path_factory) -> tuple[Any, Any]:
    """Run a fractured poromechanics case, then resume it from its own output.

    The unit system is scaled and the times are large and unevenly spaced on purpose:
    both are properties of a realistic simulation that a restart has to survive, and
    both are absent from the reference-file test below. Length is what is scaled,
    because displacement is the only variable this setup drives to a magnitude worth
    comparing - its pressures sit at 1e-24, where any discrepancy hides inside the
    tolerance of the comparison.

    Returns:
        The model that was run, and the model restarted from its output.

    """
    directory = tmp_path_factory.mktemp("restart")

    def build(restart: bool):
        model = create_model_with_fracture(
            {"porosity": 0.5}, {}, {}, 0.1, TailoredPoromechanics
        )
        params = model.params
        params["units"] = pp.Units(m=1e2)
        params["folder_name"] = str(directory)
        params["times_to_export"] = None
        params["time_manager"] = pp.TimeManager(
            schedule=[0, 1.8e8], dt_init=9e7, constant_dt=True
        )
        params["restart_options"] = {
            "restart": restart,
            "pvd_file": directory / "data.pvd",
            "times_file": directory / "times.json",
        }
        return TailoredPoromechanics(params)

    original = build(restart=False)
    pp.run_time_dependent_model(original)
    restarted = build(restart=True)
    restarted.prepare_simulation()
    return original, restarted


def test_restart_resumes_from_the_exported_state(restarted_poromechanics):
    """A restarted model must resume from the state it was exported with.

    Exported values are written in SI units, while the model works in the scaled units
    given by ``units``. A failure here means the conversion back is missing, so the
    restart resumes from values off by the scaling factor -- invisibly, since the
    resumed simulation runs perfectly happily on the wrong numbers.

    Every variable is compared, so this also covers the vector-valued ones
    (displacement, contact traction) and those living on interfaces.

    """
    original, restarted = restarted_poromechanics

    # The two models build the same variables in the same order, on equal grids.
    largest = 0.0
    for exported_variable, resumed_variable in zip(
        original.equation_system.variables,
        restarted.equation_system.variables,
        strict=True,
    ):
        assert exported_variable.name == resumed_variable.name
        exported = original.equation_system.get_variable_values(
            variables=[exported_variable], time_step_index=0
        )
        resumed = restarted.equation_system.get_variable_values(
            variables=[resumed_variable], time_step_index=0
        )
        assert np.allclose(exported, resumed), (
            f"{exported_variable.name} did not survive the restart"
        )
        largest = max(largest, float(np.max(np.abs(exported))))

    # Without this the test passes on a state of all zeros, which is what a comparison
    # of quantities this setup never drives away from zero amounts to.
    assert largest > 1e-3, "no variable is large enough for the comparison to mean much"


def test_restart_resumes_at_the_exported_time(restarted_poromechanics):
    """A restarted model must resume at the time it stopped at.

    The time index is read from the pvd file, whose entries are labelled by simulation
    time. A failure here means the time is being used as an index into the exported
    history, or that the entries are being ordered as strings rather than as numbers.

    """
    original, restarted = restarted_poromechanics
    assert restarted.time_manager.time == original.time_manager.time
    assert restarted.time_manager.dt == original.time_manager.dt


@pytest.fixture(scope="module")
def restarted_well_flow(tmp_path_factory) -> dict[str, Any]:
    """Run a compressible flow case with a well crossing a fracture three time steps,
    once uninterrupted, and once restarted after the second step.

    The well is coupled to the fracture through an interface of codimension two, and
    the compressible fluid makes the problem both transient and nonlinear, so that a
    restart has to reproduce the state of every variable exactly for the continued run
    to take the same Newton path as the uninterrupted one.

    Returns:
        The uninterrupted run, the run that was restarted from, the restarted run, and
        the folder of the restarted run.

    """
    directory = tmp_path_factory.mktemp("restart_well")

    def run(name: str, end_time: float, restart_from: Optional[str] = None):
        params = {
            "fracture_indices": [2],
            "material_constants": {
                "solid": pp.SolidConstants(permeability=1e-6, well_radius=0.01),
                "fluid": pp.FluidComponent(compressibility=1e-7),
            },
            "time_manager": pp.TimeManager(
                schedule=[0, end_time], dt_init=1.0, constant_dt=True
            ),
            "times_to_export": None,
            "folder_name": str(directory / name),
        }
        if restart_from is not None:
            params["restart_options"] = {
                "restart": True,
                "pvd_file": directory / restart_from / "data.pvd",
                "times_file": directory / restart_from / "times.json",
            }
        model = WellModel(params)
        pp.run_time_dependent_model(model)
        return model

    return {
        "uninterrupted": run("uninterrupted", 3.0),
        "first_part": run("first_part", 2.0),
        "restarted": run("restarted", 3.0, restart_from="first_part"),
        "restarted_folder": directory / "restarted",
    }


def test_restarted_well_flow_continues_as_if_uninterrupted(restarted_well_flow):
    """A run restarted from its own output must continue exactly as the uninterrupted
    run does: along the same Newton path, to the same state.

    The Newton path is what reveals a state that was not restored. A converged step
    forgets its initial guess, so a variable restored wrongly, such as a well flux
    between the well and the fracture it crosses restored as zero because it was not
    exported, changes the converged state only within the solver tolerance - but it
    changes the increments and residuals of every iteration on the way.

    """
    uninterrupted = restarted_well_flow["uninterrupted"]
    restarted = restarted_well_flow["restarted"]
    assert restarted.time_manager.time == uninterrupted.time_manager.time
    expected_path = uninterrupted.nonlinear_solver_statistics.convergence_info
    actual_path = restarted.nonlinear_solver_statistics.convergence_info
    for norm in ("inc_abs", "res_abs"):
        assert np.allclose(actual_path[norm], expected_path[norm], rtol=1e-8), (
            f"the Newton path differs after the restart ({norm})"
        )
    well_fluxes = 0.0
    for variable, resumed_variable in zip(
        uninterrupted.equation_system.variables,
        restarted.equation_system.variables,
        strict=True,
    ):
        expected = uninterrupted.equation_system.get_variable_values(
            variables=[variable], time_step_index=0
        )
        actual = restarted.equation_system.get_variable_values(
            variables=[resumed_variable], time_step_index=0
        )
        scale = max(float(np.max(np.abs(expected))), 1e-300)
        assert np.allclose(actual, expected, rtol=0, atol=1e-10 * scale), (
            f"{variable.name} differs after the restart"
        )
        if variable.domain in uninterrupted.mdg.interfaces(codim=2):
            well_fluxes = max(well_fluxes, float(np.max(np.abs(expected))))
    # Without this the test passes when the flux through the codimension-two interface
    # is zero anyway, which would not tell a lost variable from a restored one.
    assert well_fluxes > 0, "no flux through the codimension-two interface"


def test_restart_resumes_the_time_step_count(restarted_well_flow):
    """A restarted run must continue counting time steps where it stopped.

    A failure means the time index restarts from zero, which misplaces everything
    indexed by it, such as the time step dependent boundary data of some models.

    """
    first_part = restarted_well_flow["first_part"]
    restarted = restarted_well_flow["restarted"]
    uninterrupted = restarted_well_flow["uninterrupted"]
    assert first_part.time_manager.time_index == 2
    assert restarted.time_manager.time_index == uninterrupted.time_manager.time_index


def test_restarted_pvd_file_labels_every_step_with_its_time(restarted_well_flow):
    """The pvd file continued by a restarted run must label each exported step with
    the time it was exported at, before and after the restart.

    A failure means that the steps exported after the restart are paired with times
    from the start of the history, so that a later restart from this file, or a look at
    it in ParaView, finds the wrong state at a given time.

    """
    folder = restarted_well_flow["restarted_folder"]
    times = json.loads((folder / "times.json").read_text())["time"]
    labelled = {}
    for line in (folder / "data.pvd").read_text().splitlines():
        if "<DataSet" in line:
            time = float(line.split('timestep="')[1].split('"')[0])
            step = int(line.split('file="')[1].split('"')[0].split("_")[-1][:-4])
            labelled.setdefault(step, set()).add(time)
    assert sorted(labelled) == list(range(len(times)))
    for step, labels in labelled.items():
        assert all(label == pytest.approx(times[step]) for label in labels)


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
    pp.ModelRunner(model).run()

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
    pp.ModelRunner(model).run()

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
    """Mixin that selects full or targeted rediscretization."""

    def rediscretize(self):
        if self.params["full_rediscretization"]:
            return super().discretize()
        else:
            return super().rediscretize()


class RecordingLinearSolver(pp.solvers.LinearSolverDirect):
    """Direct linear solver that records each system before solving it.

    The recorded systems are used to compare full and targeted rediscretization across
    nonlinear iterations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear_systems: list[pp.solvers.LinearSystem] = []

    def solve_linear_system(
        self, linear_system: pp.solvers.LinearSystem
    ) -> tuple[np.ndarray, pp.solvers.LinearSolverStatus]:
        self.linear_systems.append(copy.deepcopy(linear_system))
        return super().solve_linear_system(linear_system)


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
    solver_params = {
        "nl_convergence_inc_atol": 0,
        "nl_convergence_res_atol": 0,
        "nl_max_iterations": 2,
    }

    def run_and_collect(model: pp.PorePyModel) -> list[pp.solvers.LinearSystem]:
        """Run a model and return the linear systems passed to its solver."""
        linear_solver = RecordingLinearSolver()
        nonlinear_solver = pp.solvers.NewtonSolver(
            params=solver_params, linear_solver=linear_solver
        )
        try:
            pp.ModelRunner(
                model, solver_params, nonlinear_solver=nonlinear_solver
            ).run()
        except RuntimeError:
            # RuntimeError expected due to unattainable convergence criteria
            # with only two iterations and zero tolerances.
            pass
        return linear_solver.linear_systems

    # Finalize the model class by adding the rediscretization mixin.
    rediscretization_model_class = models.add_mixin(RediscretizationTest, model_class)
    # A model object with full rediscretization.
    full_model: pp.PorePyModel = rediscretization_model_class(model_params)
    full_systems = run_and_collect(full_model)

    # A model object with targeted rediscretization.
    targeted_model_params = model_params.copy()
    targeted_model_params["full_rediscretization"] = False
    # Set up the model.
    targeted_model = rediscretization_model_class(targeted_model_params)
    targeted_systems = run_and_collect(targeted_model)

    # Check that the linear systems are the same.
    assert len(full_systems) == 2
    assert len(targeted_systems) == 2
    for full_system, targeted_system in zip(full_systems, targeted_systems):
        A_full, b_full = full_system.matrix, full_system.rhs
        A_targeted, b_targeted = targeted_system.matrix, targeted_system.rhs

        # Convert to dense array to ensure the matrices are identical.
        assert A_full is not None
        assert A_targeted is not None
        assert np.allclose(A_full.toarray(), A_targeted.toarray())
        assert np.allclose(b_full, b_targeted)

    # Check that the discretization matrix changes between iterations. Without this
    # check, missing rediscretization may go unnoticed.
    tol = 1e-2
    first_matrix = full_systems[0].matrix
    second_matrix = full_systems[1].matrix
    assert first_matrix is not None
    assert second_matrix is not None
    diff = first_matrix - second_matrix
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


@pytest.mark.parametrize(
    "nonlinear_increment,residual,expected",
    [
        # Case 1: Both increment and residual are below tolerance.
        (np.array([1e-6, 1e-6]), np.array([1e-6]), ConvergenceStatus.CONVERGED),
        # Case 2: Increment is above tolerance.
        (np.array([1e-6, 1]), np.array([1e-6]), ConvergenceStatus.CONTINUE_ITERATING),
        # Case 3: Residual is above tolerance.
        (np.array([1e-6, 1e-6]), np.array([1]), ConvergenceStatus.CONTINUE_ITERATING),
        # Case 4: Increment is nan.
        (np.array([np.nan, 0.1]), np.array([1e-6]), ConvergenceStatus.FAILED),
        # Case 5: Residual is above divergence tolerance.
        (np.array([1e-6, 1e-6]), np.array([2e4]), ConvergenceStatus.FAILED),
    ],
)
def test_check_convergence(
    nonlinear_increment: np.ndarray,
    residual: np.ndarray,
    expected: tuple[bool, bool],
):
    """Test that ``SolutionStrategy.check_convergence`` returns the right
    diverged/converged values.

    """
    # Standard setup of a convergence absolute convergence criterion - typically
    # orchestrated by a nonlinear solver.
    metric = EuclideanMetric()
    convergence_criteria = ConvergenceCriteria(
        {
            "inc_abs": pp.solvers.IncrementBasedAbsoluteCriterion(
                tol=1e-5, metric=metric
            ),
            "res_abs": pp.solvers.ResidualBasedAbsoluteCriterion(
                tol=1e-5, metric=metric
            ),
        }
    )
    divergence_criteria = DivergenceCriteria(
        {
            "inc_nan": pp.solvers.IncrementBasedNanCriterion(),
            "res_max": pp.solvers.ResidualBasedAbsoluteDivergenceCriterion(
                tol=1e4, metric=metric
            ),
        }
    )
    # Check convergence.
    convergence_status, _ = convergence_criteria.check(
        increment=nonlinear_increment, residual=residual
    )
    divergence_status = divergence_criteria.check(
        increment=nonlinear_increment, residual=residual
    )

    # Condense the two statuses into one.
    status = convergence_status.union(divergence_status)
    if expected == ConvergenceStatus.CONVERGED:
        assert status.is_converged()
    elif expected == ConvergenceStatus.CONTINUE_ITERATING:
        assert status.is_iterating()
    elif expected == ConvergenceStatus.FAILED:
        assert status.is_failed()


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
