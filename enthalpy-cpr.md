# The enthalpy reference temperature and the CPR preconditioner

**Setup.** Code references are to PorePy `develop` (`284329fff`, paths relative to
`src/porepy/`) and to `porepy-iterative-solvers` (`pp_solvers`) `develop` (`c1ef10a`).

The test case is `examples/geothermal_reservoir.py`. Its shipped schedule is a 7.5-year
initialisation with the wells closed, followed by about 100 days of injection at 300 K.
It is modified as follows:

- **Injection:** 30 years at 285 K (290–320 K for comparison), with time steps of at most
  one year.
- **Wells:** they end at their fracture crossing instead of continuing below it.
- **Darcy flux:** `DarcysLawAd` on a `TpfaAd` base instead of lagged MPFA.
- **Local Newton modifications, not on `develop`:**
  - convergence is checked per equation, on row-scaled residuals;
  - the Newton update is capped so that temperature stays within 273.16–647 K and
    pressure stays positive;
  - a Krylov solve that stops at its iteration cap is applied as an inexact Newton step.

The linear solver is `pp_solvers.thm_factory` with its defaults: GMRES, rtol 1e-12,
restart 100, at most 300 iterations. The default fracture mesh has 120 fracture cells.

**In short.** PorePy measures the fluid's specific enthalpy from the model's reference
temperature, h = c_p (T − T_ref). Because fluid mass is conserved, the choice of the
enthalpy reference does not change the solution. It does change the Jacobian: the energy
rows carry h times the mass balance's pressure operator. Where the flowing fluid is far
from T_ref, that is a large, elliptic pressure coupling in the energy rows. The CPR in
`thm_factory` leaves it to its stage-2 ILU, which handles it badly.

Measuring the fluid enthalpy from a separate reference that follows the fluid removes the
coupling exactly, with no change to the solver or to the solution. At 285 K injection it
turned a 9.4 h run with 66 failed time-step attempts into the same trajectory as PorePy's
default direct solver (pypardiso), in 0.39 h. A single reference cannot remove *local*
offsets, and those still break the solver at the well opening on a refined fracture mesh.

## 1. Why the standard formulation is hard for this CPR

**The discrete equations.** For each cell, with M = φρ the fluid mass per volume and F the
upwinded Darcy mass fluxes:

- mass: r_m = (M − Mⁿ)/Δt + Σ F − q;
- energy: r_E = (Mh − Mⁿhⁿ)/Δt + Σ h_up F + (pressure-work, Fourier and solid terms)
  − q h_in.

Every term of r_E that contains h is h times the matching term of r_m. This includes the
source, the boundary inflow and the interface and well enthalpy fluxes, which are h_up
times the corresponding mass fluxes. The remaining terms do not involve the enthalpy
reference: the pressure work −φp in the fluid internal energy, the Fourier flux, and the
solid enthalpy, which keeps T_ref.

**Where the critical terms are assembled.**

*The enthalpy and its reference:*
- `models/fluid_property_library.py:1358`, `FluidEnthalpyFromTemperature.specific_enthalpy_of_phase`:
  h = `fluid_specific_heat_capacity` (a `Scalar`, line 1342) ×
  `perturbation_from_thermodynamic_state("temperature")`.
- `models/abstract_equations.py:509`, `perturbation_from_thermodynamic_state`: T − T_ref,
  with T_ref from `reference_variable_values.temperature`. This is the same T_ref that
  the density, the thermal stress and the solid enthalpy use.
- `compositional/compositional_mixins.py:1163`, `FluidMixin`: binds the method to the
  phase as `phase.specific_enthalpy`. Every use of the fluid enthalpy goes through this
  binding, so overriding one method changes all of them.

*The energy rows* (`models/energy_balance.py`, `TotalEnergyBalanceEquations`):

| Term | Method | Discretisation |
|---|---|---|
| Accumulation | `energy_balance_equation` (line 165) → `total_internal_energy` = `fluid_internal_energy` (184, φ(ρh − p)) + `solid_internal_energy` (201) | `volume_integral`; time difference in `balance_equation` (`abstract_equations.py:87`) |
| Advective flux in subdomains | `enthalpy_flux` (292), weight `advection_weight_energy_balance` (268) = ρh/μ | `AdvectiveFlux.advective_flux` (`constitutive_laws.py:2646`): `darcy_flux * (upwind @ weight)`, `UpwindAd` from `enthalpy_discretization` (`constitutive_laws.py:2787`) |
| Interface enthalpy flux | `interface_enthalpy_flux_equation` (353) | `interface_advective_flux` (`constitutive_laws.py:2696`): `interface_darcy_flux * (upwind of weight)`, `UpwindCouplingAd` |
| Well enthalpy flux | `well_enthalpy_flux_equation` (377) | `well_advective_flux` (`constitutive_laws.py:2738`): `well_flux * (upwind of weight)`, `UpwindCouplingAd` |
| Sources from interfaces and wells | `energy_source` (401) | mortar projections of the two flux unknowns above |
| Inflow at the well head | `enthalpy_flux`'s Dirichlet boundary operator, `advection_weight_energy_balance` evaluated on boundary grids from `bc_values_temperature` (`examples/geothermal_reservoir.py:98`) | `bound_transport_dir` of the same `UpwindAd` |

*The mass rows* (`models/fluid_mass_balance.py`, `FluidMassBalanceEquations`):

| Term | Method | Discretisation |
|---|---|---|
| Accumulation | `mass_balance_equation` (147) → `fluid_mass` (167), ρφ | `volume_integral`, `balance_equation` |
| Advective flux in subdomains | `fluid_flux` (210), weight `advection_weight_mass_balance` (191) = ρ/μ | the same `advective_flux`, `UpwindAd` from `mobility_discretization` (`fluid_property_library.py:253`) |
| Interface and well fluxes | `interface_fluid_flux` (307), `well_fluid_flux` (325), sources in `fluid_source` (343) | the same pattern with `UpwindCouplingAd` |

*The Darcy flux, which carries ∂F/∂p:*
- In subdomains: `DarcysLawAd.darcy_flux` (`constitutive_laws.py:1858`) →
  `AdTpfaFlux.diffusive_flux` (line 1204). This is a TPFA flux differentiated with respect
  to the permeability. The `TpfaAd` base is selected through `FluxDiscretization`
  (`applications/discretizations/flux_discretization.py:8`) with
  `params["darcy_flux_discretization"] = "tpfa"`.
- Across interfaces: `DarcysLaw.interface_darcy_flux_equation` (line 1055), with
  `DarcysLawAd.pressure_trace` (line 1881) on the higher-dimensional side.
- Into wells: `PeacemanWellFlux.well_flux_equation` (line 1940).

**Why the identity holds face by face.** The mass flux and the enthalpy flux use two
different upwind discretizations (keywords `mobility` and `enthalpy`). Both are fed the
same Darcy flux values: `update_flux_values` (`fluid_mass_balance.py:962`, called from
`update_derived_quantities`, `solution_strategy.py:782`) stores the flux under every
keyword in `darcy_flux_storage_keywords` (`energy_balance.py:1194`). `rediscretize`
then rebuilds both upwind matrices (`set_nonlinear_discretizations`,
`energy_balance.py:1204` and `fluid_mass_balance.py:1027`).

So the upstream cell is the same for both on every face, and upwind(ρh/μ) =
h_up · upwind(ρ/μ) exactly. The upstream direction is held fixed within the Jacobian; it
is refreshed after each iteration.

**The enthalpy reference is a free constant.** Replace T_ref in h by any constant T₀ and
the energy residual changes by an exact multiple of the mass residual:

  r_E′ = r_E − c_p (T₀ − T_ref) r_m.

The Newton system changes by the same row operation, J′ = L J and r′ = L r, with L the
identity plus −c_p(T₀ − T_ref) times "mass row into energy row". (The interface and well
enthalpy-flux unknowns also change by a multiple of their mass fluxes, which is a change
of variables.) In exact arithmetic the Newton iterates and the converged states are the
same. Only the matrix the preconditioner sees changes.

**What the reference puts into the energy rows.** Split the upwind enthalpy on each face
as h_up = h_i + (h_up − h_i). The pressure derivative of cell i's energy residual is then

  ∂r_E,i/∂p = h_i ∂r_m,i/∂p + Σ_faces (h_up − h_i) ∂F/∂p.

- The second term is the physical coupling. It is independent of T₀, and nonzero only
  where temperature varies between neighbouring cells, such as at a front.
- The first term is a copy of the mass balance's pressure operator, scaled by
  c_p(T_i − T₀). Its size is set entirely by the arbitrary reference.

In code, the first term comes from differentiating `darcy_flux * (upwind @ weight)` in
`advective_flux` with respect to pressure through `darcy_flux`. The factor
`upwind @ (ρh/μ)` then multiplies the TPFA ∂F/∂p from `AdTpfaFlux.diffusive_flux`. The
same h-scaled derivatives arise in `interface_advective_flux` and `well_advective_flux`
through the interface and well flux unknowns, and in the accumulation through ∂(ρφ)/∂p
inside `fluid_internal_energy`.

**Where that operator is elliptic.** In the fractures, where the flow is, it is elliptic
and global: face Péclet ≈ 1e7 and storage ratio ≈ 1e-5 (the accumulation coefficient over
the cell's flux coefficients), so steady advection with flux-dominated mass rows. In the
storage-dominated rock matrix (storage ratio ≈ 55 at one-year steps) it is nearly
diagonal.

**Why the CPR does not cope.** `thm_factory` (`pp_solvers/preconditioners.py:1291`) first
transforms the system (`ContactLinearTransformation`, and `ScaleSpecificVolume`, which
scales the energy rows by the inverse specific volume; `transformations.py:173`, `:276`).
It then runs GMRES around `nested_schur_complements` (line 910), which eliminates the
blocks in this order:

1. contact (`BlockDiagonalPreconditioner`, `BlockDiagonalInverter`);
2. the interface and well flux unknowns, including the interface and well enthalpy
   fluxes (`ILU`, `DiagonalInverter`, line 131: PETSc `selfp`, i.e. S ≈ A₁₁ − A₁₀
   diag(A₀₀)⁻¹ A₀₁);
3. mechanics (`AMG`, `FixedStressInverter`, line 208).

What remains is the temperature–pressure block, preconditioned by a multiplicative
`CompositePreconditioner` (line 498), a two-stage CPR:

- stage 1: a `FieldSplit` (line 562) with `Identity` on the energy rows (`cpr0_energy`,
  line 410) and `AMG` on the mass balance's pressure block (`cpr0_mass`);
- stage 2: `ILU` (`cpr1`) on the whole T–p block, interleaved cell by cell by
  `PythonPermutationWrapper` (line 799).

There is no decoupling step, so stage 1 never removes the pressure operator from the
energy rows. Stage 2 therefore receives an energy block that contains c_p(T − T₀) × (an
elliptic pressure operator over the fracture network). An incomplete factorisation
represents such a global coupling poorly. CPR rests on the assumption that what reaches
stage 2 is local, and that assumption fails.

**The offset also reaches the interface elimination.** The interface enthalpy-flux
equations couple each enthalpy-flux unknown to its Darcy-flux unknown with a factor
−h_up. That is an off-diagonal entry inside the eliminated interface block, and
`DiagonalInverter` drops it. The Schur complement handed to the CPR therefore misses an
h-proportional term. Exact LU on `cpr1` alone removes the difficulty, so this is not the
dominant effect, but it grows with the offset in the same way. Not tested separately.

**Confidence.** The causal link is supported by four independent tests:

- the cost follows the offset;
- moving the reference moves the cost;
- removing the reference exactly removes the cost;
- an exact factorisation of the same block removes the cost.

That ILU's poor approximation of the elliptic part is the precise failure mode is an
inference that fits the evidence but has not been proved. No spectral analysis has been
done.

## 2. Evidence

**Krylov cost follows the fracture's distance from T_ref** (T_ref = 300 K; first solves of
the step at 16 years). The fracture temperature is nearly uniform within each state (the
10th–90th percentile spread is at most 3 K), so one number describes it:

| Injection | Fracture \|T − T_ref\| | Krylov iterations per solve | Median over the 30-year run |
|---|---|---|---|
| 300 K | 0.4 K | 64–65 | ~65 |
| 310 K | 10 K | 99 | ~100 |
| 320 K | 20 K | 190 | ~190 |
| 285 K | 14 K | 294–300 (capped) | ~300 |

**Péclet numbers and storage ratios do not discriminate.** Both are the same in every
state of the table.

The matrix offset (8–19 K) is also the same in every state, including the easy 300 K one.
Offsets in storage-dominated cells are harmless at this level.

**The GMRES history: a plateau, then a fast burst.** The table counts iterations to each
decade of reduction (1e-1, 1e-2, 1e-6, 1e-12) of the GMRES estimate at the first Newton
iteration of the step at 16 years:

| State | 1e-1 | 1e-2 | 1e-6 | 1e-12 |
|---|---|---|---|---|
| 300 K, offset 0.4 K | 46 | 49 | 56 | 64 |
| 285 K, shifted reference (fixed 285 K) | 38 | 46 | 60 | 69 |
| 285 K, shifted reference (dynamic) | 12 | 33 | 58 | 73 |
| 285 K, standard (T_ref = 300 K) | 90 | 98 | 194 | 294 |

- **Under the standard formulation the plateau is longer than about 90 iterations.**
  Restart 100 discards the Krylov space just as the burst begins, so every restart cycle
  repeats the plateau.
- **A longer plateau stagnates restarted GMRES completely.** From about 17 years on,
  almost every solve ends at the 300-iteration cap with relative residual about 1.
- **With the reference shifted, the solve finishes inside one cycle.**

**Moving the reference moves the difficulty.** The 300 K state with T_ref = 320 K takes
199 and then 300 (capped) iterations. Caveat: that test changes T_ref itself, which is
also the density, thermal-stress and solid-enthalpy reference, so it changes the physics
slightly. The fluid-only shift below does not.

**Exact LU on `cpr1` removes it.** At 16 years it takes 31–36 iterations per solve instead
of about 300. At 36 years, in the stagnating phase, it takes 28–31, and the step converges
in 3 Newton iterations instead of failing. The weak component is that block's ILU, not
the AMG stage or the Schur eliminations.

## 3. The fix: a separate, dynamic fluid-enthalpy reference

**How it enters.** `FluidEnthalpyFromTemperature.specific_enthalpy_of_phase`
(`models/fluid_property_library.py:1358`) returns c_p ·
`perturbation_from_thermodynamic_state("temperature")`, i.e. c_p(T − T_ref). The test
replaces that method on the model instance before `prepare_simulation`, and `FluidMixin`
then binds the replacement as `phase.specific_enthalpy`. The new method returns
c_p (T − h_ref), with h_ref a new quantity. Nothing else is overridden: every row listed
in section 1 picks up the new enthalpy through `advection_weight_energy_balance` and
`fluid_internal_energy`.

- **Fluid enthalpy only.** It is used everywhere the fluid enthalpy appears: accumulation,
  upwind advective flux, interface and well enthalpy fluxes, injection and boundary
  enthalpy. T_ref keeps every other role (density, thermal stress, solid enthalpy).
- **Dynamic.** h_ref is a `TimeDependentDenseArray` ("enthalpy_reference") on all
  subdomains and boundary grids:
  - at `initial_condition` it is set to T_ref;
  - at `before_nonlinear_loop`, i.e. once per time step and before any assembly, it is set
    to the arithmetic mean temperature of the fracture cells;
  - the value is written at both the current iterate and the previous time level, so the
    accumulation term (Mh − Mⁿhⁿ) uses one reference. The shift of that term is then
    c_pΔT₀ (M − Mⁿ), an exact multiple of the mass accumulation.
- **Why the reference must be constant in space.** Σ h_ref F = h_ref Σ F only for a
  uniform h_ref: a cell-wise reference would alter the discrete equations, not just their
  algebra. In time it may change between steps.

**What it fixes.** Full 285 K runs, 30 years of injection. "Failed" counts failed
time-step attempts; "capped" counts solves that reached the 300-iteration cap.

| Run | Steps | Failed | Newton | Krylov total (median per solve) | Capped solves | Wall time |
|---|---|---|---|---|---|---|
| Standard formulation | 209 | 66 | 2973 | 838 721 (301) | 2501 | 9.4 h |
| pypardiso (reference trajectory) | 41 | 0 | 230 | — | — | 0.28 h |
| **Dynamic fluid-enthalpy reference** | **41** | **0** | **219** | **16 475 (80)** | **0** | **0.39 h** |
| Dynamic reference, rtol 1e-6 | 41 | 0 | 227 | 15 019 (72) | 0 | 0.34 h |
| Standard formulation, rtol 1e-6 + restart 200 | 41 | 0 | 232 | 23 510 (109) | 0 | 0.56 h |

- **The solver settings are unchanged**, rtol 1e-12 included. Loosening rtol to 1e-6
  on top saves only 9 % of the Krylov iterations: once the plateau is gone, the strict
  tolerance costs a few iterations per solve.
- **The trajectory of h_ref:** 300 K at the start, up to 311 K during the closed-well
  initialisation (the fracture warms to the rock), 299 K at the well opening, and about
  286 K from 12 years on.
- **The converged states are the same.**
  - With a fixed h_ref = 285 K against the standard run at 16 years, the relative
    differences are 2e-12 in pressure, 1e-11 in temperature, 1e-10 in displacement and
    1e-8 in contact traction.
  - With the dynamic reference, the difference is 5e-9.
  - Only the interface and well enthalpy-flux unknowns differ, since they contain the
    reference.

**Side effects to be aware of.**

- **The energy residual itself changes**, by c_p(h_ref − T_ref) r_m. So do the Newton
  convergence check on the energy block and the Krylov tolerance relative to ‖b‖. At
  16 years the dynamic run converged in 6 Newton iterations against 9 for the standard
  formulation and the fixed shift, with the same converged state. That difference is
  plausibly the criterion seeing a different energy residual; it has not been
  investigated. Over the full run, 219 against pypardiso's 230 is within run-to-run
  noise.
- **When h_ref changes at the start of a step, the stored iterate of the enthalpy-flux
  unknowns still carries the old reference.** That is a worse initial guess in variables
  that enter linearly, not an error.
- **The rule "mean fracture temperature" fits this case**, whose fractures carry the
  flow and are nearly isothermal (within about 3 K). It is not a general rule
  (section 5).

## 4. What still fails

**A fixed reference.** h_ref = 285 K throughout fails in the first one-year step: cut ten
times, with every solve stalled. The fractures then sit at about 311 K, an offset of 26 K,
which by the table above is beyond what restart 100 survives. The matrix offset is also
large then (8–38 K), but the evidence in section 2 says matrix offsets of that kind are
harmless, so the fracture offset is the likely cause. The run does not separate the two.

**Local offsets.** These tests use refined fracture meshes at the well opening, with
300 K injection and otherwise the shipped example (MPFA, lagged Darcy flux, shipped
wells), plus the update cap. On the 48 m fracture mesh (292 fracture cells), a global
shift does not help:

- h_ref = 311 K, the fracture mean, still fails after 25 Newton iterations, every solve
  capped;
- the first Newton update clips 16 unknowns to the temperature bounds, leaving a few
  cells hundreds of kelvin from any single reference;
- exact LU on `cpr1` passes the step in 10 Newton iterations, and so does rtol 1e-6 with
  restart 200 (13);
- so the block is the same one, and the offset is local.

The same would be expected wherever a strongly connected flow path spans a large
temperature range, such as a cooling front lying across a fracture. This case does not
contain one at 285 K, so it is untested.

**Algebraic decoupling.** Forming the same row operation from the assembled matrices,
R = I − w × (mass row into energy row) with w = h or the quasi-IMPES ratio of pressure
diagonals, loses accuracy in both forms tried:

- *applied to the operator:* the large terms h ∂r_m/∂p cancel down to the small physical
  remainder, and rounding of the large terms swamps it;
- *applied inside the preconditioner only,* CPR built on R A and applied as CPR(RA)⁻¹ R
  to GMRES/FGMRES on A: GMRES's estimate converges in about 80 iterations, but the true
  residual stalls at 0.1–1. The multipliers of size h plausibly limit the attainable
  accuracy; that has not been analysed.

The same operation done at the discretisation level never forms the large terms, which is
why it works.

**The plateau that remains.** With no offset at all (300 K injection), GMRES still spends
45–60 iterations before the first decade of reduction, then gains a decade every one to
two iterations. That shape suggests a few dozen poorly preconditioned modes, which GMRES
has to resolve before it converges superlinearly. The shift removes the offset's
lengthening of that plateau, not the plateau itself. On the meshes above the plateau also
grows with the number of fracture cells:

| Fracture mesh | Fracture cells | Krylov iterations per solve |
|---|---|---|
| Default | 120 | 66–84 |
| 64 m | 178 | 94–100 |
| 48 m | 292 | capped |

But that series sits at an 11 K offset, so it mixes cell count with the offset. Whether
the shift alone rescues the 64 m mesh has not been tested.

**Unexplained.**

- Why an offset below T_ref (285 K, 14 K) costs more than a larger one above (320 K,
  20 K).
- Why lagged MPFA is easier for this preconditioner than TPFA (about 93 against 197
  iterations per solve, at the same state).

## 5. Open questions

1. **Where a fluid-enthalpy reference belongs in PorePy.** A quantity separate from
   T_ref, used only by the fluid enthalpy, with T_ref as its default so that nothing
   changes unless asked. The dynamic update would be a solution-strategy hook.
2. **How to choose it in general.** A flux-weighted mean temperature over the
   flux-dominated cells (fractures and wells here) is the natural generalisation of
   "fracture mean". With a large temperature spread along the flow paths, no constant
   suffices, and the remedy has to come from the solver side.
3. **Solver-side counterparts, untested:**
   - a stronger `cpr1` (fill levels, flow-aligned ordering, AIR), since exact LU shows the
     ceiling;
   - a CPR whose first stage also treats temperature (AMG on T as well as on p);
   - a decoupling whose weights come from the discretisation rather than from the
     assembled matrix.
4. **Verification still to do:**
   - the dynamic reference at 290–320 K and under other boundary stress states, to
     confirm it never hurts;
   - on the 64 m mesh.
