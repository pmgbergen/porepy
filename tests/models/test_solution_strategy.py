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
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils import models
from porepy.applications.test_utils.vtk import compare_pvd_files, compare_vtu_files
from porepy.models.metric import EuclideanMetric
from porepy.numerics.solvers.convergence_check import (
    ConvergenceCriteria,
    ConvergenceStatus,
    DivergenceCriteria,
)
from tests.models.test_poromechanics import PoromechanicsWell


class RestartModel(PoromechanicsWell):
    """Poromechanics model with a well and a fracture, and displacement boundary values
    that vary in time.

    """

    def set_wells(self) -> None:
        # Scale the geometry of the well, to not lose it to the length unit. Place it
        # away from the intersection of the fractures.
        ls = self.units.convert_units(1, "m")
        self._wells = [pp.Well(np.array([[0.3, 0.3], [0.3, 0.3], [0.2, 1.0]]) * ls)]

    def bc_values_displacement(self, bg: pp.BoundaryGrid) -> np.ndarray:
        # Nonzero and varying in time, such that the boundary reference state, i.e.,
        # the values at the start, differs from the values at later times.
        values = super().bc_values_displacement(bg)
        return values + 1e-3 * (1 + self.time_manager.time)


class RestartModelWithReference(RestartModel):
    """:class:`RestartModel` with nonzero reference state of primary variables, stress
    and boundary values.

    """

    def initialize_operator_reference_values_from_initial_state(self) -> None:
        """Shift the reference values of the pressure, the interface displacement, the
        interface Darcy flux and the well flux by small random numbers, and set a
        constant nonzero reference stress.

        The shift makes the reference state nonzero, such that a restart that fails to
        recover it from file gives a different stress.

        """
        super().initialize_operator_reference_values_from_initial_state()
        es = self.equation_system
        # Shifting all variables, e.g., the contact traction, hinders convergence.
        variables = es.get_variables(
            [
                self.pressure_variable,
                self.interface_displacement_variable,
                self.interface_darcy_flux_variable,
                self.well_flux_variable,
            ]
        )
        values = es.get_variable_values(variables, iterate_index=0)
        rng = np.random.default_rng(seed=0)
        es.set_variable_values(
            values + 1e-3 * rng.random(values.size), variables, reference=True
        )
        for sd, data in self.mdg.subdomains(return_data=True, dim=self.nd):
            pp.set_solution_values(
                self.reference_stress_key,
                1e-3 * np.ones(sd.num_faces * self.nd),
                data,
                reference=True,
            )

    def set_equations(self) -> None:
        super().set_equations()
        # The boundary values are set, and the equations (with the boundary condition
        # type filters) are present only now.
        self.set_boundary_reference_values()


def restart_model_params(
    folder: Path, end_time: float, restart_options: Optional[dict] = None
) -> dict[str, Any]:
    """Parameters of :class:`RestartModel`.

    Parameters:
        folder: Folder to write to.
        end_time: Time to run until, with time step size :data:`RESTART_DT`.
        restart_options: Restart options of the model. If ``None``, the run starts from
            the initial state.

    Returns:
        The model parameters.

    """
    params: dict[str, Any] = {
        # Two intersecting fractures, one of which the well crosses.
        "fracture_indices": [0, 2],
        "well_flux": -1e-2,
        "folder_name": str(folder),
        "times_to_export": None,  # Means export all.
        "time_manager": pp.TimeManager(
            schedule=[0, end_time], dt_init=RESTART_DT, constant_dt=True
        ),
        # Exported values are in SI units, while the model works in scaled units, so a
        # restart must convert the imported state back.
        "units": pp.Units(m=10, kg=1e3),
    }
    if restart_options is not None:
        params["restart_options"] = restart_options
    return params


def run_restart_model(
    folder: Path, end_time: float, restart_options: Optional[dict] = None
) -> tuple[RestartModel, list[np.ndarray]]:
    """Run :class:`RestartModelWithReference`, writing to a folder.

    Parameters:
        folder: Folder to write to.
        end_time: Time to run until, with time step size :data:`RESTART_DT`.
        restart_options: Restart options of the model. If ``None``, the run starts from
            the initial state. A restarted run uses the model without the nonzero
            reference state, which must hence be read from file.

    Returns:
        The model that was run, and its initial values of the variables.

    """
    params = restart_model_params(folder, end_time, restart_options)
    model_class = RestartModel if restart_options else RestartModelWithReference
    model = model_class(params)
    runner = pp.ModelRunner(model)
    initial_state = [
        model.equation_system.get_variable_values(variables=[v], time_step_index=0)
        for v in model.equation_system.variables
    ]
    runner.run()
    return model, initial_state


@pytest.fixture(scope="module")
def restart_runs(tmp_path_factory) -> dict[str, Any]:
    """Run a poromechanics case with a well, once uninterrupted, and once restarted
    after an earlier time step.

    Returns:
        The uninterrupted run, its initial state, the run that was restarted from, the
        restarted run, and the folders of the latter two.

    """
    directory = tmp_path_factory.mktemp("restart")
    first_part_folder = directory / "first_part"
    uninterrupted, initial_state = run_restart_model(
        directory / "uninterrupted", RESTART_END_TIME
    )
    first_part, _ = run_restart_model(first_part_folder, RESTART_FROM_TIME)
    restarted, _ = run_restart_model(
        directory / "restarted",
        RESTART_END_TIME,
        {
            "restart": True,
            "pvd_file": first_part_folder / "data.pvd",
            "times_file": first_part_folder / "times.json",
        },
    )
    return {
        "uninterrupted": uninterrupted,
        "initial_state": initial_state,
        "first_part": first_part,
        "restarted": restarted,
        "first_part_folder": first_part_folder,
        "restarted_folder": directory / "restarted",
    }


def test_restart(restart_runs):
    """The output of a restarted run is the same as the stored output of an
    uninterrupted run.

    This includes the exported states of the last time step, the pvd file combining old
    and new steps, the time history, and the reference state written once at the start
    of the simulation.

    Run this module as a script to generate the stored files, see
    :data:`restart_data_dir`.

    """
    folder = restart_runs["restarted_folder"]
    compared = 0
    # RESTART_DATA_DIR contains *.vtu, *.pvd and times.json files. The data files are
    # only stored for time step 3, so the actual simulation's output is a superset of
    # what is stored in RESTART_DATA_DIR. We iterate over stored files and compare them
    # with the actual simulation's
    for stored in sorted(RESTART_DATA_DIR.iterdir()):
        if stored.is_dir():
            # Skipping the ./reference directory that is handled later. If some other
            # directories are added in the future, they are also skipped.
            continue
        file = folder / stored.name
        if file.suffix == ".vtu":
            assert compare_vtu_files(file, stored), file.name
            compared += 1
        elif file.suffix == ".pvd":
            assert compare_pvd_files(file, stored), file.name
        else:
            times = json.loads(file.read_text())
            expected_times = json.loads(stored.read_text())
            assert times.keys() == expected_times.keys()
            for key in times:
                np.testing.assert_allclose(times[key], expected_times[key], err_msg=key)
    # The stored states are those of the last time step, written after the restart.
    assert compared > 0

    # The reference state is not written again by the restarted run, but it is read
    # from the folder it restarts from.
    reference_folder = restart_runs["first_part_folder"] / "reference"
    for file in sorted(reference_folder.iterdir()):
        stored = RESTART_DATA_DIR / "reference" / file.name
        if file.suffix == ".vtu":
            assert compare_vtu_files(file, stored), file.name
        elif file.suffix == ".pvd":
            assert compare_pvd_files(file, stored), file.name
        else:
            with np.load(file) as saved, np.load(stored) as expected:
                assert saved.files == expected.files
                for key in saved.files:
                    np.testing.assert_allclose(saved[key], expected[key], err_msg=key)


def test_restart_continues_as_if_uninterrupted(restart_runs):
    """A run restarted from its own output must continue exactly as the uninterrupted
    run does.

    The Newton residuals are not tested for the following reason. Conversion to SI and
    back causes slight round-off errors which amplify in Newton residuals. Numerical
    instability of solvers (possibly, PyPardiso in particular) in different environments
    (NREC vs GitHub CI) lead to different Newton behavior, hindering a meaningful test
    and causing false positive alerts.

    This behavior is substantial for conversion of the length units: pp.Units(m=10), and
    insignificant for the mass units: pp.Units(kg=1e3). Investigation could not point to
    any related error in exporter/importer, so the conclusion was that it is truly the
    round-off errors that cause it, and changing the length unit has a larger effect on
    the problem's condition number than the mass unit.

    """
    uninterrupted = restart_runs["uninterrupted"]
    restarted = restart_runs["restarted"]
    initial_state = restart_runs["initial_state"]

    # Time bookkeeping.
    assert restart_runs["first_part"].time_manager.time_index == RESTART_FROM_STEP
    expected_time = uninterrupted.time_manager
    restarted_time = restarted.time_manager
    assert restarted_time.time == expected_time.time
    assert restarted_time.time_index == expected_time.time_index
    assert restarted_time.dt == expected_time.dt
    assert restarted_time.exported_times == expected_time.exported_times
    assert restarted_time.exported_dt == expected_time.exported_dt
    assert restarted_time.exported_time_indices == expected_time.exported_time_indices

    # The reference state, of variables, stress and boundary values, is the one of the
    # uninterrupted run, which sets it in the model setup. The reference stress is
    # checked to be nonzero, since the restarted model has a zero one by default.
    for sd, data in uninterrupted.mdg.subdomains(
        return_data=True, dim=uninterrupted.nd
    ):
        reference_stress = pp.get_solution_values(
            uninterrupted.reference_stress_key, data, reference=True
        )
        assert np.any(reference_stress != 0), "the reference stress is zero"
    restarted_reference = restarted._get_grids_and_data_from_md_grid()
    for key, (
        _,
        expected_data,
    ) in uninterrupted._get_grids_and_data_from_md_grid().items():
        expected_values = expected_data[pp.REFERENCE_SOLUTIONS]
        restarted_values = restarted_reference[key][1][pp.REFERENCE_SOLUTIONS]
        assert restarted_values.keys() == expected_values.keys(), key
        for name, values in expected_values.items():
            # The reference state is stored in the units of the model, without any
            # conversion, hence the values are equal.
            assert np.array_equal(restarted_values[name], values), (
                f"the reference {name} of {key} differs after the restart"
            )

    # Final state of all variables is the same in the restarted and the uninterrupted
    # runs.
    well_fluxes = 0.0
    largest_change = 0.0
    for variable, resumed_variable, initial_values in zip(
        uninterrupted.equation_system.variables,
        restarted.equation_system.variables,
        initial_state,
        # Strict raises ValueError if lists are of different lengths.
        strict=True,
    ):
        assert variable.name == resumed_variable.name
        expected = uninterrupted.equation_system.get_variable_values(
            variables=[variable], time_step_index=0
        )
        restarted_val = restarted.equation_system.get_variable_values(
            variables=[resumed_variable], time_step_index=0
        )
        scale = max(float(np.max(np.abs(expected))), 1e-300)
        np.testing.assert_allclose(
            restarted_val,
            expected,
            rtol=0,
            atol=1e-10 * scale,
            err_msg=f"{variable.name} differs after the restart",
        )
        largest_change = max(
            largest_change, float(np.max(np.abs(expected - initial_values)))
        )
        if variable.domain in uninterrupted.mdg.interfaces(codim=2):
            well_fluxes = max(well_fluxes, float(np.max(np.abs(expected))))
    # Without this the test passes if the state never evolves, which would not tell a
    # restart from the initial state from a restart from the exported one.
    assert largest_change > 0, "the simulation did not evolve the state"
    # Without this the test passes when the flux through the codimension-two interface
    # is zero anyway, which would not tell a lost variable from a restored one.
    assert well_fluxes > 0, "no flux through the codimension-two interface"

    # The stress depends on the reference state of variables, stress and boundary
    # values.
    def stress(model):
        sd = model.mdg.subdomains(dim=model.nd)[0]
        return model.equation_system.evaluate(model.stress([sd]))

    np.testing.assert_allclose(
        stress(restarted), stress(uninterrupted), atol=1e-8, rtol=1e-8
    )


def test_restarted_pvd_file_labels_every_step_with_its_time(restart_runs):
    """The pvd file continued by a restarted run must label each exported step with
    the time it was exported at, before and after the restart.

    """
    folder = restart_runs["restarted_folder"]
    times = json.loads((folder / "times.json").read_text())["time"]
    labelled: dict[int, list[float]] = {}
    # Collect, for each exported step (the suffix of the vtu file names), the times it
    # is labelled with in the pvd file. All files of a step should carry one time.
    for dataset in ET.parse(folder / "data.pvd").iter("DataSet"):
        step = int(Path(dataset.attrib["file"]).stem.split("_")[-1])
        # There are multiple files corresponding to the same step (data_0_*, data_1_*,
        # data_mortar_0_*, etc.). All these data should contain the same simulation
        # time.
        labelled.setdefault(step, list()).append(float(dataset.attrib["timestep"]))
    # Every step of the history, before and after the restart, is in the pvd file,
    # labelled with the time the history records for it.
    assert sorted(labelled) == list(range(len(times)))
    for step, labels in labelled.items():
        assert all(label == times[step] for label in labels)


@pytest.mark.parametrize("allow_missing", [True, False])
def test_restart_without_reference_state(restart_runs, tmp_path, allow_missing):
    """A history written without the reference state, e.g., by an older version, can
    only be restarted from if ``allow_missing_exported_files`` is True. Then, a warning
    is issued, and the reference state of the model setup is kept. Otherwise, a
    ValueError is raised.

    """
    folder = tmp_path / "first_part"
    shutil.copytree(
        restart_runs["first_part_folder"],
        folder,
        ignore=shutil.ignore_patterns("reference"),
    )
    params = restart_model_params(
        tmp_path / "restarted",
        RESTART_END_TIME,
        {
            "restart": True,
            "pvd_file": folder / "data.pvd",
            "times_file": folder / "times.json",
            "allow_missing_exported_files": allow_missing,
        },
    )
    if allow_missing:
        with pytest.warns(UserWarning, match="No reference state found"):
            RestartModel(params).prepare_simulation()
    else:
        with pytest.raises(ValueError, match="No reference state found"):
            RestartModel(params).prepare_simulation()


@pytest.mark.parametrize("allow_missing", [True, False])
def test_restart_without_time_indices(restart_runs, tmp_path, allow_missing):
    """ "times.json" may not contain field "time_index" (older versions). If
    ``allow_missing_exported_files`` is False, a ValueError is raised. If it is True, a
    warning is issued, and the time manager's time index should remain unchanged.

    """
    # Copy the restart folder and delete "time_index" from "times.json", to mimic the
    # older version of "times.json".
    folder = tmp_path / "first_part"
    shutil.copytree(restart_runs["first_part_folder"], folder)
    times_file = folder / "times.json"
    history = json.loads(times_file.read_text())
    assert "time_index" in history
    del history["time_index"]
    times_file.write_text(json.dumps(history))

    # Set up the model. The reference state is present, so only the missing time
    # indices are under test.
    params = restart_model_params(
        tmp_path / "restarted",
        RESTART_END_TIME,
        {
            "restart": True,
            "pvd_file": folder / "data.pvd",
            "times_file": times_file,
            "allow_missing_exported_files": allow_missing,
        },
    )
    model = RestartModel(params)

    if not allow_missing:
        with pytest.raises(ValueError, match="does not contain exported_time_indices"):
            model.prepare_simulation()
        return

    with pytest.warns(UserWarning, match="does not contain exported_time_indices"):
        model.prepare_simulation()

    time_manager = model.time_manager
    assert time_manager.time == pytest.approx(RESTART_FROM_TIME)
    # Time index remains as it was.
    assert time_manager.time_index == 0
    # The history before the restart state has unknown indices, while the restart state
    # is exported again with the unchanged time index.
    assert time_manager.exported_time_indices == [None] * RESTART_FROM_STEP + [0]
    assert len(time_manager.exported_times) == RESTART_FROM_STEP + 1


# Directory containing the stored output of an uninterrupted run of RestartModel, to
# compare the output of a restarted run with. It is generated by running
# ``python -m tests.models.test_solution_strategy`` from the root of the repository.
RESTART_DATA_DIR = Path(__file__).parent / "restart_reference"

# Time step size, deliberately not one, such that time and time step index differ.
RESTART_DT = 1.7e5

# Time steps of the uninterrupted run, and of the run it is restarted from, with the
# corresponding times.
RESTART_END_STEP = 3
RESTART_FROM_STEP = 2
RESTART_END_TIME = RESTART_END_STEP * RESTART_DT
RESTART_FROM_TIME = RESTART_FROM_STEP * RESTART_DT


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
        np.testing.assert_allclose(A_full.toarray(), A_targeted.toarray())
        np.testing.assert_allclose(b_full, b_targeted)

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


if __name__ == "__main__":
    # Generate the stored output of an uninterrupted run.
    shutil.rmtree(RESTART_DATA_DIR, ignore_errors=True)
    run_restart_model(RESTART_DATA_DIR, RESTART_END_TIME)
    # Keep the exported states of the last time step only, the pvd file and the time
    # history of the whole run, and the reference state.
    for file in RESTART_DATA_DIR.glob("data_*"):
        if int(file.stem.split("_")[-1]) < RESTART_END_STEP:
            file.unlink()
