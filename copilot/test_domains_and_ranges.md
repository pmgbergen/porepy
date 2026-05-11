# Test Coverage Audit: AD Operator Domains and Ranges

**Branch:** `operators_have_domains`  
**Reference plan:** `copilot/ad_sources_and_ranges.md`  
**Test files introduced:**
- `tests/numerics/ad/test_grid_entity.py` (274 lines, ~30 tests)
- `tests/numerics/ad/test_operator_space.py` (1653 lines, ~163 tests)

---

## Summary

Overall test coverage is **good**: every stage in the implementation plan has dedicated
tests, all pass, and mypy reports no errors. However, there are identifiable redundancies,
a few genuine gaps, and some early mock-based tests that were never updated to also
exercise the real implementations they were anticipating.

---

## 1. Coverage Gaps

### 1a. `MixedDimensionalVariable` — not tested

`MixedDimensionalVariable.__init__` explicitly sets `_source = None` and
`_target = None` (see `operators.py` ~line 2272), because an MD variable spans
multiple grids that cannot be summarised as a single `OperatorSpace`. There is no test
asserting this.

**Recommendation:** Add a test that constructs a `MixedDimensionalVariable` from two
`Variable` instances and asserts `source is None` and `target is None`.

### 1b. `SurrogateOperator` / `SurrogateFactory` `source` — not tested

`SurrogateOperator.__init__` calls `super().__init__(..., source=op_space,
range_=op_space)`, correctly propagating a `dof_info`-derived space when one is
provided. The `test_grid_entity.py` tests for `SurrogateFactory` only verify that the
object can be *created* without error; they do not assert that the resulting
`SurrogateOperator` carries the expected `source` and `target`.

**Recommendation:** Add tests that:
1. Create a `SurrogateFactory` with `dof_info={GridEntity.cells: 1}` and sample one of
   the produced `SurrogateOperator` instances, asserting its `source` is set.
2. Assert that `dof_info=None` (the default) leaves `source` as `None`.

### 1c. `Variable` on interfaces and boundary grids — not tested

`TestVariableSpace` only tests `Variable` on a subdomain grid. Variables can also be
defined on mortar grids (interfaces) and boundary grids. The `OperatorSpace.from_domains`
dispatch for `MortarGrid` and `BoundaryGrid` is tested indirectly via the mortar/boundary
projection tests, but not for the `Variable` leaf itself.

**Recommendation:** Add tests constructing `Variable` on a `MortarGrid` and on a
`BoundaryGrid`, asserting the correct `DomainType` and `dof_info`.

### 1d. Vector `Divergence` and `Trace` (`dim > 1`) — not tested

`TestDivergenceSpaces` and `TestTraceSpaces` use only `dim=1`. When `dim > 1` the
`dof_info` should be `{GridEntity.faces: dim}` (domain) and `{GridEntity.cells: dim}`
(range) for `Divergence`, and the transpose for `Trace`. This is especially important
because `Divergence` is heavily used in vector-mechanics models.

**Recommendation:** Add parametrised tests for `dim=2` and `dim=3`.

### 1e. `Tpsa` via `MergedOperator` — integration test missing

The `TestMergedOperatorWithConcreteDiscretization` class covers `Mpfa`, `Tpfa`, `Mpsa`,
and `Upwind` (via `MergedOperator` and the AD wrappers). `Tpsa` has no corresponding
integration test — only direct unit tests on `get_row/col_dof_info`.

**Recommendation:** Add at least one `MergedOperator`-level test for a `Tpsa` matrix
(e.g. `stress_displacement`), verifying that `source` and `target` are
populated when 2D grids are supplied.

### 1f. `DenseArray.__neg__` — not tested explicitly

`SparseArray.__neg__` and `DenseArray.__neg__` are separate overrides, both added in the
Stage 6 regression fixes. The `test_unary_minus_preserves_spaces` test uses `SparseArray`
exclusively. The bug that triggered the fix (unary minus dropping source/target) was
identified for both classes, but only one is exercised by the current tests.

**Recommendation:** Add `test_unary_minus_dense_array_preserves_spaces` mirroring the
existing `SparseArray` test.

### 1g. `sum_operator_list` and `sum_projection_list` — not tested for spaces

The implementation plan (§ Assumptions #6) notes that these utility functions should
benefit automatically from the updated arithmetic operators. There is no test verifying
that the result of `sum_operator_list([op1, op2, ...])` carries correct `source`
and `target`.

**Recommendation:** Add a basic test that sums two `SparseArray` instances with
compatible spaces and asserts that the result propagates those spaces.

### 1h. Model-level integration test — not implemented

The plan mentions (§ Integration/regression tests) an optional test that constructs a
simple model, checks the assembled equation operator for consistent `source`/
`target`, and raises `ValueError` on an intentional mismatch. This was flagged
as optional and has not been added. Given the maturity of the implementation, this would
now be a valuable regression guard.

**Recommendation:** Add at least one integration test that uses a small
`FluidMassBalance`-like model and verifies that a well-typed equation's operator chain
has non-`None` `source`/`target` end-to-end.

---

## 2. Redundant Tests

### 2a. `TestDomainRangePropagation` vs `TestInferDomainRange` — substantial overlap

`TestDomainRangePropagation` was written as part of Stage 2 and `TestInferDomainRange`
was added in Stage 5. Many of their tests cover the same scenarios:

| Scenario | `TestDomainRangePropagation` | `TestInferDomainRange` |
|---|---|---|
| `add` with compatible spaces | `test_add_same_space_propagates` | `test_add_compatible` |
| `sub` with compatible spaces | `test_sub_same_space_propagates` | `test_sub_compatible` |
| `mul` with compatible spaces | `test_mul_same_space_propagates` | `test_mul_compatible` |
| `div` with compatible spaces | `test_div_same_space_propagates` | `test_div_compatible` |
| Scalar + non-scalar | `test_scalar_inherits_other_space` | `test_add_with_scalar_lhs/rhs`, `test_mul_with_scalar` |
| None + known → known | `test_none_space_propagates_other` | `test_none_plus_known_inherits_known` |
| Both None → None | `test_both_none_gives_none` | `test_both_none_stays_none` |
| Incompatible domains raise | `test_incompatible_domains_raises` | `test_add_incompatible_domain`, `test_sub_incompatible_domain`, etc. |
| Matmul compatible | `test_matmul_propagates_outer_spaces` | `test_matmul_compatible` |
| Matmul incompatible raises | `test_matmul_incompatible_range_domain_raises` | `test_matmul_incompatible` |

The differences are minor (different fixture machinery, slightly different assertions).
The duplication does not cause harm, but when the test suite is next reorganised the two
classes could be merged into one authoritative set under the Stage 5/6 heading.

**Recommendation:** No immediate action required. Consider consolidating the two classes
into a single `TestDomainRangeArithmetic` class during any future test-suite
housekeeping.

---

## 3. Tests That Should Also Exercise Real Implementations

### 3a. `TestMergedOperatorSpaces.test_custom_dof_info_gives_space`

Written before Stage 7, this test creates a bespoke `ConcreteDiscr` subclass with a
hardcoded `get_row_dof_info` → `{cells:1}` and `get_col_dof_info` → `{faces:1}`. Its
purpose was to verify that `MergedOperator` reads those methods and sets the operator
space correctly.

Now that `pp.Mpfa`, `pp.Mpsa`, `pp.Upwind`, and `pp.Tpsa` all have real implementations,
the same behaviour is indirectly confirmed by `TestMergedOperatorWithConcreteDiscretization`.
The mock-based test is still a valid *unit* test of the protocol (it isolates `MergedOperator`
from the specific discretisation logic), but it no longer serves as a proxy for real
implementations.

**Status:** Keep the mock test; supplement with Tpsa integration test per §1e above.

### 3b. `TestDiscretizationStubs.test_get_row_dof_info_overridable`

Uses a custom `CustomDiscr` class. Written to anticipate the Stage 7 override protocol.
With concrete implementations now in place, this test is still a valid unit test of the
base-class override mechanism.

**Status:** Keep as-is.

### 3c. `TestSurrogateFactoryBackwardCompatibility`

The two tests (`test_string_dof_info`, `test_enum_dof_info`) in `test_grid_entity.py`
check that `SurrogateFactory` can be *instantiated* with either key type. They were
written for Stage 1 (backwards compat of `GridEntity`) and correctly test that legacy
string keys are still accepted. However, as noted in §1b, they do not verify the
resulting operator's `source`.

**Status:** Extend with an assertion on `source` (see §1b).

---

## 4. Plan Items Covered Well

The following plan requirements have thorough test coverage:

| Plan item | Tests |
|---|---|
| `GridEntity` enum members and backward compat | `test_grid_entity.py` all classes |
| `OperatorSpace` equality, hash, `from_domains`, `scalar()` | `TestOperatorSpaceScalar`, `TestOperatorSpaceFromDomains`, `TestOperatorSpaceEquality` |
| `DomainType` values | `TestDomainType` |
| `Scalar`, `Variable`, `DenseArray`, `SparseArray` leaf spaces | `TestScalarSpace`, `TestVariableSpace`, `TestDenseArraySpace`, `TestSparseArraySpace` |
| `TimeDependentDenseArray` optional `dof_info` | `TestTimeDependentDenseArraySpaces` |
| All 5 grid operator classes | `TestSubdomainProjectionSpaces`, `TestMortarProjectionSpaces`, `TestBoundaryProjectionSpaces`, `TestTraceSpaces`, `TestDivergenceSpaces` |
| `Discretization` base-class stubs | `TestDiscretizationStubs` |
| `MergedOperator` + `InterfaceDiscretization` guard | `TestMergedOperatorSpaces` |
| `infer_source_target` (all ops, scalar, None) | `TestInferDomainRange` |
| Multi-step compound operator chains | `TestCompoundOperatorSpaces` |
| `FVElliptic` (`Mpfa`/`Tpfa`) dof_info | `TestFVEllipticDofInfo` |
| `Mpsa` dof_info | `TestMpsaDofInfo` |
| `Biot` dof_info + inheritance from `Mpsa` | `TestBiotDofInfo` |
| `Upwind` dof_info | `TestUpwindDofInfo` |
| `Tpsa` dof_info (all 14 matrices, nrot formula) | `TestTpsaDofInfo` |
| `MergedOperator` integration (Mpfa, Tpfa, Mpsa, Upwind) | `TestMergedOperatorWithConcreteDiscretization` |
| `create_variables`, `set_equation`, `SurrogateFactory` backward compat | `TestCreateVariablesBackwardCompatibility`, `TestSetEquationBackwardCompatibility`, `TestSurrogateFactoryBackwardCompatibility` |

---

## 5. Priority Ranking

| Priority | Gap / Issue |
|---|---|
| High | §1f — `DenseArray.__neg__` untested (a real bug was fixed there) |
| High | §1b — `SurrogateOperator` `source` unverified |
| High | §1e — `Tpsa` has no `MergedOperator` integration test |
| Medium | §1a — `MixedDimensionalVariable` `source is None` not asserted |
| Medium | §1d — `Divergence`/`Trace` `dim > 1` not tested |
| Low | §1c — `Variable` on mortar/boundary grids |
| Low | §1g — `sum_operator_list` space propagation |
| Low | §1h — end-to-end model integration test |
| Cleanup | §2a — consolidate `TestDomainRangePropagation` with `TestInferDomainRange` |
