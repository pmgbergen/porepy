# Review: model initialization (branch `initialization-4` vs `develop`)

Scope: the diff to `develop` and `runscript.py` (the pipeline, formerly `yura2.py`).

**Status (after commits `3e088f2cb`, `db8ffabd3`, `d18e350b8`, `8c929b471`):** §2.1, §2.2, §3.1 (core
part) and §3.2 are fixed. §3.3 and §3.5 have an agreed-upon proposal but are not implemented.
Everything else is open. The next issues are in §0. Items marked **(verify)** have not been
confirmed by a run.

Goal: `initialize(model)` works as a black box on any PorePy model, ends in a verified steady
state, and leaves the model in its original form (constitutive laws, folders, time manager).

Severity: **P0** = wrong results or violates a stated requirement. **P1** = fragile or not
fail-proof. **P2** = cleanup.

---

## 0. Next issues (priority order)

What a full run of `runscript.py` showed (THM example, coarse mesh): the init run stopped after 12
steps at t = 4.6e7 s (about 1.45 years, i.e. just after `earliest_stop_time = 1 year`), the residual
check passed, and the main simulation started and took steps. The pipeline works for this one model.
It does not yet meet "fail-proof black box".

1. ~~Boundary/source time during init (§3.3)~~ implemented (`d882539a4`); remaining gaps in §3.3.
2. ~~Stale state after the class swap (§3.5)~~ `SolutionStrategy.rebuild_equations()` (`c9c54b77f`),
   with a residual/Jacobian comparison test. Note: in the tested THM model the nonlinear
   discretization list did **not** grow with the old inline rebuild, so the stale-list worry was not
   reproduced; clearing is kept as a safe guard.
3. ~~Weak residual check (§3.1)~~ (`ba79ab896`): max-norm residual per equation relative to
   `|J| @ tol`, plus a trial Newton step relative to variable tolerances. Both unit-independent.
4. ~~Stopping criterion (§3.7)~~ (`f1af5efdc`): stops on the steady residual and trial step, not on
   per-step changes.
5. ~~State copying (§3.4)~~: `initialize_previous_iterate_and_time_step_values()` and
   `update_derived_quantities()` are used. `ContactIndicators` is stateless (computed from
   variables), so it needs no copying.

**New findings from the stricter checks (THM example):**
- With the old criterion the run "converged" at about 1.45 years, but the steady trial step showed
  pressure still 17% of its 100 Pa tolerance away and temperature 1.5e-3 of 1 K. The previous
  "success" was not an equilibrium in the strict sense.
- Later, the trial-step ratios for pressure and temperature **grow** with time (pressure 5e-3 at
  about 10 years, 1.9e-2 at about 60 years). The transient is drifting away from the steady problem
  instead of approaching it. Possible causes to investigate: very large time steps (up to 18
  years), temperature-dependent density with gravity (convection in the high-permeability
  fractures), or boundary data that do not admit a steady state. Not understood yet.
- `TRIAL_STEP_RTOL = 1e-1` and `RESIDUAL_RTOL = 1e-3` are judgement calls. At `1e-3` the example
  never stopped. The residual ratio is much more lenient than the trial step (mass balance 6e-5
  while pressure was still moving), so the trial step is the check that matters.
- Variables without an explicit tolerance use 1% of their own magnitude; flux variables of passive
  wells use the magnitude of the interface fluxes (`SCALE_LIKE`). This is a heuristic and model
  specific (§3.6).
- The trial step costs one linear solve per time step after `earliest_stop_time` (still 1 year, set
  in the call).

Next: (a) investigate the drift above, (b) tests for the restore and failure paths (§4.2) and a
test for the new checks, (c) generality and the library API (§3.6, §3.9), (d) persistence (§4.1),
(e) API breaks and TODOs (§2.6, §2.7).

---

## 1. Summary

The physics side is mostly coherent. The reference-state mechanism has four parts: a
`reference_stress`, `perturbation_from_reference()` in the stress, porosity and shear-dilation
laws, and `set_boundary_reference_values`. The pipeline in `runscript.py` is a prototype:

| Requirement | Status |
|---|---|
| Residual at the end is ~0 and validated | **Partly.** `InitializationError` is raised on failure or non-zero residual, but the tolerance is absolute and RMS-based (§3.1). |
| Shear dilation = 0, constant porosity during init | Works for the one model it was written for. Not generic (§3.6). |
| Boundary grids get reference values | Done. The reference time is not controlled (§3.3). The Tpsa stress is now consistent (§2.2, fixed). |
| Constitutive laws restored afterwards | `__class__` is restored in a `finally`. Stale operators and nonlinear discretizations survive (§3.5). |
| Output goes to `<dir>_initialization`, then reverts | **Fixed** (§3.2). Checked on the THM example: init files go to `<dir>_initialization`, the main run and the t=0 export to `<dir>`. |
| Fail-proof, black-box | Not yet (§0, §3, §4). |

---

## 2. Library changes (`src/porepy`)

### 2.1 ~~P0: `cached_method` is disabled~~ FIXED (`3e088f2cb`)
`numerics/ad/operators.py:2131` has a bare `return func` at the top of `cached_method`. It looks
like a debugging hack. It silently disables all caching, in `geometry.py` and
`fluid_property_library.py`, and makes everything slower. Revert it. If it is deliberate (e.g. to
survive the class swap), say so and handle the cache properly instead. See §3.5: `_operator_cache`
is keyed by `__qualname__`, so it is the natural place to drop stale entries.

### 2.2 ~~P1: `ThreeFieldLinearElasticMechanicalStress` perturbs only part of the stress~~ FIXED (`db8ffabd3`)
- `LinearElasticMechanicalStress.mechanical_stress` perturbs the **whole** expression, including
  `bound_stress @ boundary_operator`.
- The 3-field (Tpsa) version perturbs `displacement`, `interface_displacement`, `rotation_stress`
  and `total_pressure`, but leaves `discr.bound_stress() @ boundary_operator` absolute.
- `runscript.py` sets `reference_stress` with boundary values counted in absolute terms. After
  `set_boundary_reference_values()` the boundary term in the 2-field law vanishes as a
  perturbation, so the reference stress carries it. In the 3-field law the boundary term stays
  absolute, so the total stress would count it twice.
- Make both laws consistent. Perturb the boundary term (`.perturbation_from_reference()` on the
  whole sum) or document that it is intentionally absolute. Add a test that runs the init
  pipeline on a Tpsa poromechanics model. **(verify)**

### 2.3 P1: `reference_stress` plumbing is fragile
- The reference stress is stored under `data[REFERENCE_SOLUTIONS]["reference_stress"]` by
  `InitialConditionsMomentumBalance.initial_condition`. A model that overrides `initial_condition`
  without a super-call, or composes a different IC mixin, hits a `KeyError` at the first
  `reference_stress` evaluation. Fall back to zeros (or raise a clear message) in the operator.
- Models that override `stress()` and do not add `reference_stress` get silently wrong
  equilibrium. Examples: user mixins and `applications/`. Grep for `def stress` and
  `mechanical_stress(` callers and add a check or a test.
- `reference_stress` is not exported to vtu/json. Restart from file (`reset_state_from_file`)
  restores variables but not the reference stress or the boundary references, so a restarted
  initialized model is no longer in equilibrium. See §4.1.
- The pipeline calls `model.stress(domains)` at the end of init, before the boundary references
  are set. That is correct today. It depends on call order, so put it in one method on the model,
  e.g. `set_reference_state_from_current()`, instead of in the script.
- Shape is `(nd * num_faces,)`, flattened F-order or C-order? State the convention in the
  docstring and test it.

### 2.4 P1: `set_boundary_reference_values` copies *every* iterate entry
It loops over all keys in `data[ITERATE_SOLUTIONS]` of every boundary grid. That includes the BC
type filters (Dirichlet/Neumann/Robin masks). Copying the filters is probably intended (see the
docstring), but it is not obvious for non-numeric or conditionally present keys.
- It copies only index 0. `_shift_to_reference_solutions` ignores `max_index`, so this is fine,
  but the name "shift" is misleading.
- It must run after `update_all_boundary_conditions` for the same time as the converged state. In
  the pipeline it is run once, at the final init time. See §3.3 for why that time matters.
- The docstring claims "Must be called after the equations are set". It does not depend on the
  equations. It depends on `update_time_dependent_ad_arrays()` having been called. Fix the doc.

### 2.5 P1: porosity changes
- `PoroMechanicsPorosity` now takes `displacement_divergence(...).perturbation_from_reference()`.
  The Tpsa path does the same with `(total_pressure + alpha p).perturbation_from_reference()`.
  Check that the new porosity equals `reference_porosity` at the reference state **for every
  porosity mixin** (thermal expansion, fractures, wells), not only the one used by the THM
  example. The leftover `diff = porosity - reference_porosity` block in `runscript.py` was probably
  meant to check this. It checks nothing, because there is no assert.
- The new comment `# TODO: Is this correct?` at `constitutive_laws.py:~5118` is on a
  consistency-by-perturbation path. Resolve it before merging.
- Tests in `tests/models/test_poromechanics.py` (`test_without_fracture`) rely on the old
  behaviour. Run them. **(verify)**

### 2.6 P1: breaking public API changes without tests/deprecation
- `pp.EuclideanMetric.__call__` now returns `dict[str, float]` (`{"norm": ...}`) instead of
  `float`. `tests/models/test_metric.py::test_euclidean_metric_basic` and
  `tests/numerics/solvers/test_nonlinear_solvers.py` call it as a float, so they will fail.
  The criteria in `convergence_check.py` accept dicts, but any user code that uses it as a float
  will break.
- Removed from the public API: `pp.LebesgueMetric`, `ConvergenceMetricType`, and
  `_IntervalMap` moved to `time_step_control.py`. Check docs and examples. Either keep aliases
  or list this in the changelog.
- The `Schedule` dataclass became a plain class. `dataclass` provided `__eq__` and `__repr__`.
  Check that nothing compares or serializes schedules (`TimeManager.write_time_information`,
  `Schedule.get_array`, tests). `_IntervalMap` is built at construction, so mutating
  `schedule.intervals` afterwards makes it stale. **(verify)**
- `TimeScheduler` no longer builds its own map but asks `schedule.t_snap`. `TimeScheduler`'s
  `atol=self.t_snap` is validated against the schedule's own `t_snap` (default `1e-8`). Two
  sources of truth for one tolerance.

### 2.7 P2: cleanup in the library
- Unfinished markers: `TODO YZ` in `model_runner.py` (docstrings for `early_stop_criteria`, the
  loop comment), `metric.py` (`"""TODO YZ."""`), and `time_step_control.py` (`Schedule`
  docstring and `t_snap` doc).
- `# TODO: Is copying a good idea since it also copies the operator id?` in
  `_operator_states.py`. The `_cached_key = None` fix in `_get_reference` is good. Add a unit
  test: the reference copy and the original must evaluate differently when states differ.
- `solution_strategy.prepare_simulation(self, )` is a stray formatting artifact.
- The ANSI colour escape in `time_stepper.py` log (`\033[92m`) pollutes log files. Remove it.
- `newton_solver.py` fix (`isinstance(..., str)` -> `(int, float)`) is a real bug fix. Keep it
  in its own commit.
- `units.py` `to_si: bool = False`: harmless.
- `ModelRunner._run_stationary` has `# TODO YZ: model.before_time_step is never called...`. If
  that is a real bug it is out of scope here. File it.
- `EarlyStopCriterion` has no docstring. It is exported in `__init__`, so document the contract:
  return `None` to continue, or a `ModelRunnerStatus` to stop.

---

## 3. The pipeline (`runscript.py`)

### 3.1 P0 (core fixed in `8c929b471`, remainder open): failure and success are not distinguished
**Done in `8c929b471`:** `InitializationError` (with `status` and `residual_norms`) is raised if
the run does not end on `SteadyStateModelRunnerSuccess` (a failed step or reaching `t_end` count as
failure), or if any residual of the original equations exceeds the tolerance. The reference state is
only overwritten on success, the `for/else` is gone, and the restore runs in a `finally`. Verified:
a 3-day schedule raises and leaves the class and exporter restored; the full THM run passes.
**Still open:** the tolerance/norm bullet below, a result object returned on success, and the
trial-step validation. After a failure the model keeps the init equations and variables, so the
caller must discard it (document, or rebuild in the `finally`, see §3.5).
The bullets below describe the original problems.

- `except RuntimeError as e: status = e.args[0]` swallows a failed init run. The code goes on
  to overwrite the reference state with a non-converged state. `status` is never inspected.
- If the schedule reaches `t_end = 100 years` without the early-stop criterion firing, the runner
  returns a plain `ModelRunnerStatusSuccess`. The pipeline treats that as success even though
  equilibrium was never established. Check
  `isinstance(status, SteadyStateModelRunnerSuccess)`.
- The final residual check uses `for ... else`. The `else` branch has no `break`, so
  "Initialization complete" is **always** logged, even right after the "Equilibration failed"
  warnings.
- Warnings are not enough for "fail-proof". Raise a dedicated exception
  (e.g. `InitializationError`) carrying the per-equation residual norms, unless the caller opts
  out. Also return a result object (status, time reached, residual norms), not only the model.
- The tolerance `residual_tol = 1e-9` is absolute and unit-dependent. This example sets
  `kg=1e9`, which rescales the equations. `EquationBasedEuclideanMetric` also divides by
  `sqrt(n)` so a localized residual is hidden. Use a relative tolerance (residual vs. the
  size of the largest term, or vs. the residual at the unequilibrated start) and also check the
  max-norm. Check *all* equations, including interface and contact ones.
- The residual is evaluated **after** rebuilding the equations of the original model. That is
  the right order and is the point of the feature. It also needs `ad_time_step` and the
  time-dependent arrays to be in a defined state (see §3.3).
- Idea to validate more strictly: after the residual check, take one trial step of the original
  model with `dt` and confirm the state change is below tolerance.

### 3.2 ~~P0: output folder is not reverted~~ FIXED (`d18e350b8`)
**Done in `d18e350b8`:** the folder name is built with `Path`; after restoring `folder_name` the
pipeline calls `set_nonlinear_solver_statistics()` and `initialize_data_saving()` (new exporter,
iteration exporter, statistics path); the initial state is exported at t=0 at the end of the
pipeline. Checked on the example: the exporter points at the original folder after init and the
t=0 files are there. Not covered by an automated test (§4.2). The bullets below describe the
original problem.

`restore_original_model()` puts `params["folder_name"]` back, but the objects built in
`prepare_simulation` keep the init path:
- `model.exporter` (the `pp.Exporter` is created once with the init folder; its pvd history and
  `_time_step_counter` continue).
- `model.nonlinear_solver_statistics.path` (when `solver_statistics_file_name` is set).
- `IterationExporting.iteration_exporter` (its folder is `<folder>_iterations`, so it is
  `..._initialization_iterations`).
- `time_manager.write_time_information` uses `params["folder_name"]` at call time, so
  `times.json` lands in the **original** folder while the vtu/pvd land in the init folder. A
  split output.
- The main run does not export the initial state at t=0, because `prepare_simulation` is skipped
  (`params={"prepare_simulation": False}`) and so is `save_data_time_step()`.

Fix: after restoring the params, call `model.initialize_data_saving()` and
`model.set_nonlinear_solver_statistics()`, or reset the new objects explicitly. Then export the
initialized state at t=0 with the restored time manager. Also:
- `folder_name` may be a `Path` or have a trailing slash. Build the name with `Path`
  (`folder.with_name(folder.name + "_initialization")`), the same way `IterationExporting` does.
- Restore is only done `if original_folder_name is not None`. The default is always set by
  `SolutionStrategy.__init__`, so the `else: "initialization"` branch is dead. If it ever ran,
  it would not be restored.
- `restore_original_model` is never called on exceptions. Wrap the init in `try/finally` (or a
  context manager) so a failure does not leave the model with the swapped class, time manager
  and folder.

### 3.3 ~~P0: time of the boundary/source data used for the reference state~~ IMPLEMENTED (core)
**Implemented** in `InitializedModel.update_time_dependent_ad_arrays` as proposed below, with one
change: the frozen time is the *first target time* of the main run, `t0 + dt` (override with
`params["initialization_data_time"]`), not `t0`. The example's lithostatic boundary conditions are
zero at the initial time itself and full afterwards, so freezing at `t0` would initialize an
unstressed state. Checked: during init all boundary updates happen at 86400 s and the clock is
restored; the full THM run still passes the residual check. **Still open:** code that reads
`time_manager.time` outside this hook (see below), and an automated test with a ramped boundary
condition.

Original analysis and proposal. Resetting `time_manager.time` to 0 after each step does not
work: `TimeStepper` sets `time = accepted + dt` *before* the step, so boundary values would still be
evaluated at `dt`; the scheduler picks the `dt` interval from `time`, so `dt` would never grow; and
exported times and the time index would collide.
The clock is shared by the stepper and by model code, so decouple them in `InitializedModel`:

```python
def update_time_dependent_ad_arrays(self):
    t = self.time_manager.time
    self.time_manager.time = t0          # original_time_manager.time
    try:
        super().update_time_dependent_ad_arrays()
    finally:
        self.time_manager.time = t
```

This freezes boundary values, boundary-type filters and sources updated through that hook at `t0`,
so the boundary references, the reference stress and the final residual check all refer to `t0`.
It does not cover code that reads `time_manager.time` elsewhere (`update_derived_quantities`
overrides, time-dependent well protocols); those must be disabled or frozen separately. Cleaner
long term: a `data_time` property on the model (default `time_manager.time`) that boundary/source
code reads. Add a test with a ramped boundary condition.

During init the model clock runs 0 -> up to 100 years. Boundary values and sources are
re-evaluated every step (`update_time_dependent_ad_arrays` in `before_time_step`), so the final
state, the stored boundary references and the reference stress all correspond to BC/source
values **at the last init time**, not at the original `t = 0`.
- For the example (lithostatic stress is 0 at t=0 and jumps for t>0) this happens to give the
  intended state. For a model with a time-dependent protocol (injection starting at day 5, a
  ramped BC, time-dependent source) the initialization would equilibrate to the protocol value
  at year 100, and the references would be wrong for the main run.
- Decide the semantics explicitly. Most likely: during init, freeze the BC/source time at the
  original start time `t0` (or at a model-provided `initialization_time`). Add a hook such as
  `model.initialization_time()` and make `update_time_dependent_ad_arrays` read it, rather than
  the time manager, while initializing.
- Wells with injection/production protocols are another time-dependent source. They must be
  off, or fixed at their `t0` values, during init. The example's well is passive, so this is
  not tested.
- After `set_boundary_reference_values()` and the restore, nothing calls
  `update_time_dependent_ad_arrays()` for the restored time manager. The first main step does
  it in `before_time_step`, which is fine. The residual check, however, runs on arrays from the
  last init step. Make that explicit and tested.

### 3.4 P1: state copying after init is incomplete
```python
steady_state = model.equation_system.get_variable_values(time_step_index=0)
set_variable_values(reference=True, ...); (time_step_index=0, ...); (iterate_index=0, ...)
```
- Only time step 0 and iterate 0 are written. Use
  `model.initialize_previous_iterate_and_time_step_values()` (it covers
  `iterate_indices` and `time_step_indices`, which may have more entries, e.g. BDF2).
- Anything else stored in data dictionaries that depends on history is untouched: contact
  indicators (`ContactIndicators` is in the example), previous-time-step caches, cached
  discretization-dependent states, `update_derived_quantities`. Call
  `update_derived_quantities()` after setting the state.
- The reference is taken from `time_step_index=0` while the iterate is the more current value
  after a converged step. They are equal after `update_time_step_solution`, but
  `get_variable_values(iterate_index=0)` would be less surprising.
- Secondary/derived reference values (`initialize_operator_reference_values_from_initial_state`
  only handles pressure and temperature, with the comment "temporary bridge from PR #1696") are
  not recomputed. The pipeline overwrites references for all variables, which covers this, but
  the temporary method should be removed or reconciled so there is one path.

### 3.5 P1 (open, solution proposed): swapping `__class__` leaves stale state
**Proposed solution (not implemented).** Add `SolutionStrategy.rebuild_equations()`, called by
`prepare_simulation` and by the pipeline, which: removes all equations; clears
`_nonlinear_discretizations`, `_nonlinear_diffusive_flux_discretizations` and `_operator_cache`;
then runs `set_equations`, `update_discretization_parameters`, `discretize`,
`set_nonlinear_discretizations`. Safety test: build a pristine model and an init-then-restored model
at the same state and compare the assembled residual and Jacobian; they must be identical. The
alternative that avoids `__class__` mutation altogether is a flag read by the two overrides plus the
same rebuild. `cached_method` is enabled again (§2.1), so stale cache entries are no longer hidden.
**(verify)** that the stale lists are really non-empty after the swap; this was found by reading.

`model.__class__ = InitializedModel`, then `model.__class__ = original`:
- `_nonlinear_discretizations` (and the diffusive-flux list) keep `MergedOperator`s created
  by the init equations. `set_nonlinear_discretizations()` only appends, with an identity check,
  so after re-setting the equations the list contains both stale and new entries. The stale
  discretizations are re-evaluated on every `discretize` and every nonlinear iteration (wasted
  work, possibly wrong data). Reset these lists before `set_equations()`.
- `_operator_cache` entries from the init class are retained (`cached_method` is currently a
  no-op, which hides this; once §2.1 is fixed it becomes real). Clear the cache before the
  rebuild.
- Removing equations by looping over `equation_system.equations` and re-running
  `set_equations` / `update_discretization_parameters` / `discretize` /
  `set_nonlinear_discretizations` re-implements part of `prepare_simulation`. Move that into a
  model method (`reset_equations()` or `rebuild_equations()`) so it is not duplicated and cannot
  drift from `prepare_simulation`.
- Prefer not to mutate `__class__` of the user's instance. A cleaner alternative: a
  `model.params["initialization"] = True` flag read by hooks in the library
  (`shear_dilation_gap`, `matrix_porosity`), or an `InitializationMixin` that is part of the
  composition of the default model classes. Dynamic class swapping breaks for models using
  `__slots__`, metaclasses, `super()` in unexpected places, or pickling.
  If the swap stays, build the subclass with `type(...)` once, and document the requirement
  that `restore` runs in a `finally`.

### 3.6 P1: the overrides are not generic
```python
def shear_dilation_gap(...) -> Scalar(0)
def matrix_porosity(...) -> reference_porosity * ones
```
- Models without `ShearDilation` or without `matrix_porosity`/`reference_porosity` (pure flow,
  pure mechanics, models with `DarcyFlux`-only setups) do not need or have these. The override
  adds attributes that nothing calls (harmless) or breaks if `reference_porosity` has a
  different signature in a user model. Make the overrides conditional (`hasattr`) or move
  them into the library as documented, optional "initialization hooks" with a default no-op.
- Fracture and well porosity (`porosity` for lower-dimensional subdomains) are not touched.
  Check whether they depend on the unknowns (e.g. aperture-dependent, fracture porosity) and
  whether those also need freezing.
- Permeability with `CubicLawPermeability` depends on aperture, which depends on the
  displacement jump and shear dilation. With dilation zero, the aperture is still
  state-dependent; this is fine only if the aperture at the reference state is what the original
  model would give. State that assumption.
- `matrix_porosity` is also used by the energy balance. If a user model overrides it to depend
  on temperature or pressure, init uses a different porosity than the main run at the same state.
  That is fine **only because** reference = steady state. Verify with a test that residuals of the
  original model vanish at the end (they do by construction only if the rebuilt equations are
  zero at the reference state, so the final residual check is the real test).

### 3.7 P1: steady-state criterion (`SteadyStateEarlyStopCriterion`)
**New observation from the full run:** init stopped at t = 4.6e7 s, 12 steps, right after
`earliest_stop_time = 1 year`. It is unknown whether equilibrium was reached earlier or the
tolerances (e.g. pressure `100 Pa` converted to `kg = 1e9` units) are loose. Log the first time at
which all variables were within tolerance.

- It compares the solution change between two consecutive time steps with an absolute
  tolerance, not a rate. With adaptive `dt` (the schedule grows to 1 year steps) a loose
  threshold can pass while the state still drifts at `tol/dt`, and a tight one is unreachable
  at 1 s steps. Use `|Δx| / dt`, or better the residual of the **steady** equations, or the
  accumulation terms (the time-derivative part) only.
- The norm is RMS per variable (`/ sqrt(n)`). A localized, still-evolving region (a fracture, the
  well) is averaged away. Add a max-norm or a relative one.
- Tolerances are hard-coded for `pressure`, `temperature`, `u`; everything else (contact
  traction, interface displacement, interface fluxes, well fluxes, other names) falls to
  `default_tolerance=1e-3`, an absolute number in model units. For the example's
  `kg = 1e9` unit system that is meaningless, and for other models it may be unreachable (stress
  variables in Pa or in scaled units). Derive tolerances from the reference values and
  characteristic scales (`NumericalConstants`, `reference_variable_values`) and let the user
  override per variable.
- `"unknown_key": -1` in the example tolerances looks like a test of the warning path. It makes
  the warning fire on **every step**. Remove it from the production call.
- The `variable_tags` argument is accepted but unused by the caller. `values.min()` /
  `.max()` raise `ValueError` on a variable with zero dofs. Guard against it.
- `earliest_stop_time = 1 year` is a magic number. Express it relative to the schedule.
- The first call only stores the previous solution and returns (so at least two steps always
  run). Fine, but document it.
- `self.previous_solution = current_solution` is stored by reference to the array returned by
  `get_variable_values` (probably a copy). **(verify)** `np.ndarray` mutation could otherwise make
  the diff always zero.
- Failure mode: if the nonlinear solver never fails but equilibrium is slow (diffusion over a
  1 km domain, as the comments in `thm_fractured_reservoir.py` note), 100 years may not be
  enough, and time is not a measure of equilibrium for models without diffusion. Offer
  alternatives: direct stationary solve of the steady equations (drop the accumulation terms)
  as a first stage, with time stepping only as fallback. This is also what makes the result
  independent of an arbitrary `t_end`.

### 3.8 P1: time schedule is hard-coded
- `make_initialization_time_manager` hard-codes seconds -> 100 years. For a different domain
  scale (a lab sample, a reservoir with 1e-20 m² permeability) it is wrong. Parametrize
  (`t_end`, first `dt`, growth) or derive from the model's characteristic time.
- Interval names are shifted: the interval that starts at `0` is called "2 seconds" but it
  covers `[0, 2 s)`, then the one starting at `2 s` is named "2 hours", and so on. Names and
  starts are off by one; only the last name is used by constants. `SCHEDULE_INTERVAL_EQUILIBRATION`
  is defined and used once, but never read elsewhere. Remove it or use it.
- Also, `t_start=2*pp.SECOND, dt_start=pp.HOUR` jumps from 1 s to 3600 s steps after 2 s, a
  factor of 3600, and `TargetNonlinearIterations` may shrink it back. Check that `dt_min/dt_max`
  defaults permit this. Smooth the progression.
- The init run advances from the model's `t=0` regardless of the original `time_manager`
  start/restart. Note the interplay with `restart_options`: restarting from file plus init is
  undefined. Decide and document, ideally raise.

### 3.9 P1: robustness and API shape
- **Import-level problems**: `from turtle import mode` is a stray IDE import. It fails on headless
  systems without `tkinter`. Remove it. `import pp_solvers` and `ThmFracturedReservoir` at module
  level couple the library function to a private solver package and to the example script.
- The linear solver is hard-wired to `pp_solvers.IterativeLinearSolver()`, and the nonlinear
  solver params to `make_solver_params()` (same dict as in the example). `initialize` must
  accept `nonlinear_solver` / `linear_solver` / `solver_params`, with a library default using
  `pp.solvers.NewtonSolver` and a direct solver.
- Same dict `make_solver_params` is duplicated in `thm_fractured_reservoir.set_solver_params`.
- `prepare_simulation` is called by `ModelRunner` (the `model.prepare_simulation()` call is
  commented out). If the user already prepared the model (common), the grid, variables and
  data are created a second time. Detect it (e.g. `model.mdg` exists / a flag) and either skip
  or raise. The main run then passes `{"prepare_simulation": False}`: good, but it silently
  assumes `initialization_pipeline` has been run. Provide one entry point,
  e.g. `pp.initialize_model(model, ...) -> InitializationResult`, and let `ModelRunner` call it
  when `params["initialize"]` is true.
- `ModelRunner.run()` calls `model.after_simulation()` at the end of the init run
  (`save_statistics()`), then writes statistics for init; the main run also calls it. Check that
  the files do not overwrite each other (separate folders only if §3.2 is fixed).
- The runner object for init is built with `params=None`, so `progressbars` etc. are not
  inherited.
- Non-time-dependent models (`_is_time_dependent() == False`) go through `_run_stationary`, which
  never calls the early-stop criteria. Init of a stationary model is a no-op or an endless one.
  Handle explicitly.
- Leftover debugging after `set_nonlinear_discretizations()`: the whole `for domain in ...`
  block evaluates stress, reference stress, mechanical stress, pressure stress, thermal stress and
  porosity for every matrix subdomain, and discards the results. That costs time and nothing is
  asserted. Turn it into real assertions (e.g. `porosity == reference_porosity`, the mechanical
  perturbation is zero) or delete it.
- Unused variables: `boundary_faces`, `internal_faces`, `diff`, `run_main_simulation.status`,
  `SteadyStateModelRunnerSuccess` is the only way to tell success, but not exported or exposed.
- `logging.basicConfig(...)` at import time and in `__main__` twice.
- Naming and placement. Suggest (no code moved here):
  - `src/porepy/models/initialization.py` for `initialize_model(...)`, `InitializationResult`,
    `InitializationError`, `SteadyStateEarlyStopCriterion`, and the time-manager factory.
  - Hooks as small mixin/protocol methods in the model library (`initialization_overrides`,
    `set_reference_state_from_current`, `rebuild_equations`) rather than in the script.
  - Export through `pp.__init__` (`pp.initialize_model`, `pp.EarlyStopCriterion` is already).
  - Move the THM example to `src/porepy/examples/` or a docs example, and drop `yura.py` /
    `runscript.py`.

---

## 4. Missing pieces

### 4.1 Persisting the result
Init is expensive (up to 100 simulated years of time stepping). There is no way to save and reload
the initialized state. It would need: variable values, `reference_stress`, boundary reference
values, contact/other history. Note that restart (`load_data_from_vtu`/`pvd`) only reloads
variables. Offer `save_initial_state(path)` / `params["initial_state_file"]`, and a hash of the
model setup (grid, params) to refuse a stale file.

### 4.2 Tests (none were added)
Needed at least:
1. Unit: `reference_stress` zero by default, round trip with `set_solution_values`.
2. Unit: boundary reference: operator `bound_stress @ bc` is zero as a perturbation after
   `set_boundary_reference_values`, and not zero before.
3. Unit: porosity equals `reference_porosity` at the reference state for each porosity mixin,
   for fractures and for Tpsa.
4. Integration (small 2D mixed-dimensional model with one fracture, poromechanics and THM):
   after init the residual of the original equations is below a relative tolerance **and** one
   main step with unchanged BC leaves the state unchanged.
5. Integration: constitutive laws, `time_manager`, `params["folder_name"]`, `exporter` folder and
   `nonlinear_solver_statistics.path` are identical to the originals after init (this would catch
   §3.2, §3.5).
6. Integration: init with a time-dependent BC gives the same result as with the BC frozen at
   `t0` (§3.3).
7. Failure path: forced non-convergence raises `InitializationError` and leaves the model
   restored (use `try/finally`).
8. `EuclideanMetric` and `Schedule` behaviour tests updated for the API changes (§2.6).

### 4.3 Documentation
- User docs: what the init does, what must be true of the model, what is changed temporarily and
  what is permanent (reference values are replaced for **all** variables, reference stress,
  boundary references, initial condition).
- Say that after init, `initial_condition()` / `reset_state_from_file` values are replaced.
  A model's own `initial_condition` (hydrostatic, thermal gradient in the example) is only the
  starting guess.
- Resolve every `TODO YZ`.

### 4.4 Scope of "reference == steady state"
- Things that are not primary variables and may carry their own reference or history: plastic
  displacement jump, contact state/indicators, aperture/porosity "previous" values, well
  states. Check each against "residual = 0 at reference".
- Fixed-stress or other splitting schemes, and `SequentialNonlinearSolver` (commented in the
  example) may use their own variable tags. The init solver is a full Newton on everything.
  That is simple and fine, but then the main-run solver choice is independent. Document.
- Models with chemical/multiphase (`eliminate_reference_phase`, flash) have secondary
  variables whose reference values the pipeline does not set explicitly.
- Units: tolerances in §3.7 and the residual tolerance must be unit-aware.

---

## 5. The example, `thm_fractured_reservoir.py`

- `WellZeroNeumannTemperatureBC.bc_type_fluid_flux` and `bc_type_darcy_flux` delegate to
  `super().bc_type_enthalpy_flux(sd)`, a copy-paste bug. The overrides return the right
  Neumann object for well grids but call the wrong parent for the others. **(verify)**
- ~100 lines of commented-out solver configurations and `run_example`. Remove or move.
- `set_model_params` still has a `time_manager` for a 31-day run with an "injection" interval
  name, and `"initialize_operator_reference_from_initial_values": True`; see §3.4.
- The module docstring describes the old "relax from unstressed state" idea; update it to the
  init procedure.
- `"unknown_key"`, `"# YZ: increase diffusion ... ????"` comments: either decide or delete.

---

## 6. Suggested order of work

Done: `cached_method` (§2.1), Tpsa stress (§2.2), output folder/exporter (§3.2), failure handling
(§3.1 core), `try/finally` restore.

1. Init-time semantics for boundary/source data (§3.3) and `rebuild_equations()` (§3.5), each with a
   test.
2. Stronger validation: relative/max-norm residual, trial-step check, rate-based and unit-aware
   stopping criterion (§3.1 remainder, §3.7); parametrize the schedule (§3.8).
3. State copying and derived quantities (§3.4).
4. Move the pipeline into the library with hooks, one entry point and `InitializationResult`; make
   overrides optional and generic (§3.6, §3.9).
5. Resolve the `TODO: Is this correct?` in the porosity (§2.5), API breaks (§2.6), cleanup (§2.7).
6. Persistence of the initialized state (§4.1), tests (§4.2), docs (§4.3), example fixes (§5).
