# Implementation Plan: AD Operator Domains and Ranges

**Reference:** GitHub Discussion [#1601](https://github.com/pmgbergen/porepy/discussions/1601)

## Problem Statement

The AD operators in `src/porepy/numerics/ad/operators.py` currently have partial domain
support. The goal is to give all atomic operators a well-defined *domain* and *range* in
the mathematical sense, enabling:

1. Early validation of operator compositions (catching mismatches at construction time
   rather than during parsing).
2. A proper foundation for the project described in issue #1497.
3. Future extensions such as unit propagation and richer diagnostics.

---

## Current State

| Concept | Current Implementation |
|---|---|
| `Operator._domains` | A `GridLikeSequence` (list of `pp.Grid`, `pp.MortarGrid`, or `pp.BoundaryGrid`) |
| `Operator._domain_type` | A `Literal["subdomains", "interfaces", "boundary grids", "scalar"]` string attribute inferred from `_domains` |
| `GridEntity` | A `Literal["cells", "faces", "nodes"]` **type alias** (not an enum) in `equation_system.py` line 88 |
| `dof_info` | A `dict[GridEntity, int]` used in `EquationSystem.create_variables` and `SurrogateOperator`, but **not** on the base `Operator` class |
| Domain/range distinction | Only `Projection` implicitly represents a domain-to-range map; the concept is absent from the base `Operator` |
| Validation in `__check_domains` | Only compares `self.domains == other.domains` for add/sub/mul/div; matmul is not validated at all |

Key classes to understand:
- `Operator` (base class, `operators.py`)
- `Variable`, `MixedDimensionalVariable` (`operators.py`)
- `MergedOperator` (discretization wrapper, `ad_utils.py`)
- `SurrogateOperator` (`surrogate_operator.py`)
- `TimeDependentDenseArray` (`operators.py`)
- `SubdomainProjections`, `MortarProjections`, `BoundaryProjection`, `Divergence`, `Trace` (`grid_operators.py`)
- `EquationSystem` (`equation_system.py`)

---

## Implementation Plan

### Step 1 — Convert `GridEntity` to an enum

**File:** `src/porepy/numerics/ad/equation_system.py`

- Replace the `GridEntity = Literal["cells", "faces", "nodes"]` type alias (line 88)
  with a proper `enum.Enum`:

  ```python
  class GridEntity(enum.Enum):
      cells = "cells"
      faces = "faces"
      nodes = "nodes"
  ```

- Add a `void` member to represent the absence of grid entities (used for scalars):

  ```python
  class GridEntity(enum.Enum):
      cells = "cells"
      faces = "faces"
      nodes = "nodes"
      void = "void"    # sentinel for scalar / domain-less operators
  ```

- Export `GridEntity` from the module's `__all__` and from `porepy.numerics.ad.__init__`
  so it is accessible as `pp.ad.GridEntity`.

- Update **all** sites that currently use the string literals `"cells"`, `"faces"`,
  `"nodes"` in the context of `GridEntity` (primarily `EquationSystem.admissible_dof_types`,
  `_variable_dof_type`, `_equation_image_size_info`, `create_variables`, and
  `set_equation_image_info`). Backward-compatible string comparisons should be preserved
  where they appear, or removed where the enum can be used directly. Search for
  `dof_info`, `admissible_dof_types`, and `GridEntity` across the codebase.

**Impact on existing code:**
- `EquationSystem.create_variables` accepts `dof_info: dict[GridEntity, int]` — callers
  pass `{"cells": 1}`. Python enums allow `GridEntity["cells"]` or
  `GridEntity.cells`, so accept both string keys and enum keys during a transition
  period by normalising the input at the boundary of `create_variables`.
- `SurrogateOperator._dof_info` currently stores `dict[GridEntity, int]` with string
  keys — update to enum keys.
- `admissible_dof_types` in `EquationSystem` is a tuple of strings; update to a tuple
  of `GridEntity` members.

---

### Step 2 — Introduce the `OperatorSpace` dataclass

**File:** `src/porepy/numerics/ad/operators.py` (or a new
`src/porepy/numerics/ad/operator_space.py`)

Create a dataclass (or namedtuple) called `OperatorSpace` that bundles:

```python
from __future__ import annotations
import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import porepy as pp
    from porepy.numerics.ad.equation_system import GridEntity

@dataclasses.dataclass(frozen=True)
class OperatorSpace:
    """Represents the domain or range of an AD operator.

    Attributes:
        domain_type: Enum value indicating whether the grids are subdomains,
            interfaces, boundary grids, or void (for scalars).
        grids: Tuple of grid objects. Empty for scalars.
        dof_info: Mapping from grid entity type to the number of DOFs
            per entity on each grid. Empty dict for void/scalar spaces.
    """
    domain_type: GridEntity | None          # None == "void" / scalar
    grids: tuple[pp.Grid | pp.MortarGrid | pp.BoundaryGrid, ...]
    dof_info: dict[GridEntity, int]         # e.g. {GridEntity.cells: 1}
```

Note: Use a *tuple* (not a list) for `grids` so that frozen dataclasses support
equality and hashing correctly without extra work.

Provide:
- A `__eq__` check (already provided by `@dataclasses.dataclass(frozen=True)`).
- A helper constructor `OperatorSpace.scalar()` → `OperatorSpace(None, (), {})`.
- A helper constructor `OperatorSpace.from_domains(domains, dof_info)` that infers
  `domain_type` from the grid types in `domains`.

Introduce a `DomainType` enum (can live in the same file) to capture the four cases
currently represented by strings `"subdomains"`, `"interfaces"`, `"boundary grids"`,
and `"scalar"`:

```python
class DomainType(enum.Enum):
    subdomains = "subdomains"
    interfaces = "interfaces"
    boundary_grids = "boundary grids"
    scalar = "scalar"
```

Use `DomainType` as the type of `OperatorSpace.domain_type`.

The `OperatorSpace.dof_info` dict uses `GridEntity` keys (from Step 1).

---

### Step 3 — Add `domain` and `range_` to `Operator`

**File:** `src/porepy/numerics/ad/operators.py`

Modify the `Operator.__init__` signature to accept **mandatory** `domain` and `range_`
arguments of type `OperatorSpace`. Both are always stored as distinct attributes — they
will often hold equal values (e.g., for a cell-centred variable), but they are never
aliased.

The existing `domains` parameter (a flat list of grids) is kept during the transition but
must no longer be used to infer `domain`/`range_` — callers must supply them explicitly.

```python
def __init__(
    self,
    name: Optional[str] = None,
    domains: Optional[GridLikeSequence] = None,   # legacy; kept for backward compat
    operation: Optional[Operations] = None,
    children: Optional[Sequence[Operator]] = None,
    domain: OperatorSpace = ...,                   # mandatory
    range_: OperatorSpace = ...,                   # mandatory; 'range' is a Python keyword
) -> None:
```

Internal storage — two independent attributes:
- `self._source: OperatorSpace`
- `self._target: OperatorSpace`

Constructor logic:
1. `self._source = domain` (always set; no fallback or default).
2. `self._target = range_` (always set separately; no fallback or default).
3. Both must be provided by every subclass and every compound-operator construction site.

Add two public read-only properties:
```python
@property
def source(self) -> OperatorSpace: ...

@property
def target(self) -> OperatorSpace: ...
```

Since `domain` is now mandatory, the following subclasses must be updated **before**
this change is merged (see Step 4). The compound-operator construction path (arithmetic
dunders) must also pass the inferred source/target (see Step 6).

Keep the existing `domains`, `subdomains`, `interfaces`, and `domain_type` properties
working by reading from `_source` where possible.

---

### Step 4 — Assign `domain` and `range_` to all atomic leaf operators

For each atomic (leaf) operator subclass, add the `domain` and `range_` arguments to
their constructors and pass them through to the parent `Operator.__init__`. Where the
information is statically known, set sensible defaults.

#### 4a — `Scalar` (`operators.py`)

- `domain = OperatorSpace.scalar()`, `range_ = OperatorSpace.scalar()`.
- Constructor change is internal only; the public API is unchanged.

#### 4b — `DenseArray` and `SparseArray` (`operators.py`)

- Currently no domains. Add optional `domain` and `range_` parameters.
- Users wrapping a cell-centred array would write:
  ```python
  pp.ad.DenseArray(values, source=OperatorSpace.from_domains(subdomains, {"cells": 1}))
  ```
- No default is enforced (these arrays are general-purpose); `None` means "unspecified".
- **Note:** `SparseArray` is frequently used to wrap matrices that are effectively
  projections (e.g., from `SubdomainProjections`). At this stage, leave `domain` and
  `range_` optional and unspecified for `SparseArray`, and refine in Step 4e.

#### 4c — `TimeDependentDenseArray` (`operators.py`)

- These are always tied to a set of domains (`self._domains`).
- `domain = OperatorSpace.from_domains(domains, {"cells": 1})` is a reasonable default;
  accept an explicit `dof_info` argument so the user can override.
- `range_` defaults to `domain` (square/diagonal-like operator).

#### 4d — `Variable` (`operators.py`)

- Already has `ndof: dict[Literal["cells","faces","nodes"], int]` (equivalent to
  `dof_info`).
- Construct `domain = OperatorSpace.from_domains([domain_grid], ndof)` and
  `range_ = domain`.

#### 4e — `MergedOperator` (`ad_utils.py`)

- Represents a discretization matrix. Its `domain` (column space) and `range_` (row
  space) come from the discretization itself.
- The `_discr` object (the underlying non-AD discretization) knows the DOF layout.
  Add a method `get_dof_info()` to the non-AD `Discretization` base class in
  `src/porepy/numerics/discretization.py` that returns a `dict[GridEntity, int]` for
  the row and column spaces of the discretization matrices.
- Call `self._discr.get_dof_info()` inside `MergedOperator.__init__` to populate
  `domain` and `range_`.
- In the short term, if `get_dof_info()` is not yet available on all discretizations,
  default to `domain = None` and `range_ = None`.

#### 4f — `SurrogateOperator` (`surrogate_operator.py`)

- Already has `_dof_info`. Use it to construct `domain` and `range_` (both equal, like
  `Variable`).

#### 4g — Grid operators (`grid_operators.py`)

Grid operators such as `SubdomainProjections`, `MortarProjections`, `Divergence`,
`Trace`, and `BoundaryProjection` wrap `SparseArray` objects. These are the most
important cases where `domain != range_`:

- `SubdomainProjections.cell_restriction(subdomains)` maps **from** the cells of all
  registered subdomains **to** the cells of `subdomains` (a subset).
- `Divergence` maps face fluxes to cell sources.
- `MortarProjections.mortar_to_primary_int` maps mortar DOFs to primary subdomain faces.

For each factory method that returns a `SparseArray`, construct the appropriate
`OperatorSpace` for both `domain` and `range_` based on the grid lists and DOF
information available in the method, then pass them to the `SparseArray` constructor.

---

### Step 5 — Validate operands in arithmetic operators

**File:** `src/porepy/numerics/ad/operators.py`

Replace/extend the `Operator.__check_domains` private method with a public
`validate_operands(other: Operator, op: Operations) -> OperatorSpace` method.

Validation rules:
- **`add`, `sub`, `mul`, `div`, `pow`** (elementwise):
  - Both operands must have the same `domain`; and the same `range_`.
  - Because `domain` and `range_` are always separate attributes, both must be
    compared independently even when they happen to be equal.
  - The scalar space (`OperatorSpace.scalar()`) is compatible with any space —
    elementwise operations with a `Scalar` are always valid.
  - Result carries the common `domain` and `range_` as separate attributes.
- **`matmul` (`@`)**:
  - The `range_` of the *right* operand (other) must equal the `domain` of the *left*
    operand (self). In standard matrix notation: `(A @ B)` requires `range(B) == domain(A)`.
  - Result `domain = other.source`, `range_ = self.target` (two
    distinct attributes on the new operator).
- Raise a descriptive `ValueError` on mismatch (including the operator names and
  source/target values for debugging).

Update all `__add__`, `__sub__`, `__mul__`, `__rmul__`, `__truediv__`, `__rtruediv__`,
`__pow__`, `__rpow__`, `__matmul__`, and `__rmatmul__` methods to:
1. Call `validate_operands` (or the new `infer_source_target` — see Step 6).
2. Pass the inferred source/target to the resulting compound `Operator`.

---

### Step 6 — Infer `source` and `target` for compound operators

Add a method `infer_source_target(self, other: Operator, op: Operations) -> tuple[OperatorSpace, OperatorSpace]` on `Operator`. This is called after (or combined with) validation to compute the **separate** source and target of the result:

- `add`, `sub`, `mul`, `div`, `pow`: result's `source` = common source of operands;
  result's `target` = common target of operands. The two are stored as independent
  attributes even if equal.
- `matmul`: result's `source = other.source`; result's `target = self.target`.
  These are guaranteed to be different in all non-trivial cases.
- When one operand is a `Scalar` (void space), the result inherits the other operand's
  source and target (separately).

Pass the pair `(source, target)` to the compound `Operator` constructor as separate
keyword arguments (`source=...`, `target=...`).

---

### Step 7 — Get `dof_info` from discretizations

**File:** `src/porepy/numerics/discretization.py`

Add an abstract (or mixin) method to the `Discretization` base class:

```python
def get_row_dof_info(self) -> dict[GridEntity, int]:
    """Returns DOF info for rows of the discretization matrix."""
    raise NotImplementedError

def get_col_dof_info(self) -> dict[GridEntity, int]:
    """Returns DOF info for columns of the discretization matrix."""
    raise NotImplementedError
```

Implement these for the standard discretizations (e.g., Mpfa flux has face rows and
cell columns). This is noted in the discussion as a "minor change."

In the short term, a default implementation returning `{}` (unknown) is acceptable as a
placeholder.

---

## File Change Summary

| File | Change |
|---|---|
| `src/porepy/numerics/ad/equation_system.py` | Convert `GridEntity` to enum; update all usages |
| `src/porepy/numerics/ad/operators.py` | Add `OperatorSpace`, `DomainType`; update `Operator.__init__`; update all leaf subclasses; update `validate_operands`/`infer_source_target`; update arithmetic dunder methods |
| `src/porepy/numerics/ad/ad_utils.py` | Update `MergedOperator.__init__` to accept and pass `domain`/`range_` |
| `src/porepy/numerics/ad/grid_operators.py` | Assign `domain`/`range_` to `SparseArray` objects returned by `SubdomainProjections`, `MortarProjections`, `Divergence`, `Trace`, `BoundaryProjection` |
| `src/porepy/numerics/ad/surrogate_operator.py` | Construct `OperatorSpace` from existing `_dof_info` |
| `src/porepy/numerics/ad/__init__.py` | Export `GridEntity`, `DomainType`, `OperatorSpace` |
| `src/porepy/numerics/discretization.py` | Add `get_row_dof_info`/`get_col_dof_info` |

---

## Testing Plan

Tests should be placed in `tests/numerics/ad/` alongside existing tests.

### Unit tests for `OperatorSpace` / `DomainType`

- Equality of two `OperatorSpace` instances with the same content.
- Inequality for different grids, different `dof_info`, or different `domain_type`.
- `OperatorSpace.scalar()` constructor.
- `OperatorSpace.from_domains(...)` correctly infers `domain_type`.
- Hashing (required for dict keys and set membership).

### Unit tests for `GridEntity` enum

- `GridEntity.cells`, `.faces`, `.nodes`, `.void` have the expected values.
- String-to-enum conversion works (for backward compatibility path in `create_variables`).

### Unit tests for `validate_operands` / `infer_source_target`

- Compatible add/sub/mul/div: no error, result has correct source/target.
- Incompatible add: mismatching domains raises `ValueError`.
- Compatible matmul: `range_(right) == domain(left)` passes; result source/target correct.
- Incompatible matmul: raises `ValueError`.
- Operations involving a `Scalar`: always valid; result inherits non-scalar space.
- Operations with `None` source/target: skips validation (backward-compatible).

### Integration / regression tests

- All existing tests in `tests/numerics/ad/` must continue to pass.
- Run model-level tests (e.g., `test_example_params.py`, tutorial tests) to catch
  regressions introduced by the domain propagation logic.
- Optionally add a test that constructs a simple flow model, checks that the final
  equation operator's `source` and `target` are consistent, and that
  an intentional domain mismatch raises a `ValueError`.

---

## Assumptions and Open Questions

1. **Gradual roll-out:** Many existing operator constructions do not supply source/target.
   The plan uses `None` as "unspecified" to avoid breaking the codebase, then enforces
   the constraint once all leaf operators are updated.

2. **`dof_info` in `MergedOperator`:** The discretization classes do not currently
   expose their DOF layout. Step 7 adds a lightweight accessor, but until all
   discretizations implement it, `MergedOperator` may have `domain = None`.

3. **`__rsub__` and `__rtruediv__`:** These currently do not call `__check_domains` for
   the domain of the result. They should be updated to use `infer_source_target`.

4. **Vector quantities:** The discussion notes that vector (multi-component) quantities
   should be handled. For now, `dof_info` already supports `{"cells": Nd}`, so
   vector quantities are naturally accommodated.

5. **`MixedDimensionalVariable`:** This class bypasses the `Operator.__init__` entirely.
   Its `_domains`, `_domain_type` etc. are set manually. When adding `_source`
   and `_target`, ensure they are also set in `MixedDimensionalVariable.__init__`.

6. **`sum_operator_list` and `sum_projection_list`:** These utility functions create
   compound operators without explicit source/target propagation. They should benefit
   automatically from the updated arithmetic operators.

7. **Backward compatibility:** The existing `domains` parameter in `Operator.__init__`
   is kept. The internal `_domains` / `_domain_type` attributes can continue to be
   derived from `_source` during the transition, ensuring `subdomains` and
   `interfaces` properties still work.
