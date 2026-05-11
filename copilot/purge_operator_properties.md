# Plan: Purging Redundant Operator Properties

## Background

`Operator` currently exposes four properties derived from `_source`:

| Property | Returns |
|---|---|
| `domains` | All grids (subdomains, interfaces, or boundary grids) |
| `domain_type` | The `DomainType` enum value for the domain |
| `subdomains` | Grids filtered to `DomainType.subdomains` only |
| `interfaces` | Grids filtered to `DomainType.interfaces` only |

The question is whether all four are necessary, or whether some are thin wrappers that add API surface without adding value.

---

## Usage Audit

### `domains`

Used extensively throughout the AD module:

- `operators.py` – `TimeDependentDenseArray._key` (line 1788), `.parse()` (line 1820), `.__repr__()` (line 1851), `MixedDimensionalVariable._key` (line 2144)
- `surrogate_operator.py` – `__repr__` (line 225), `_evaluate` (lines 286, 306)
- `ad_utils.py` – `MergedOperator._key` (line 623), `MergedOperator.parse()` (lines 648, 656)

**Verdict: Keep.** This is the primary accessor for the operator's grid list and is used ubiquitously.

---

### `domain_type`

Used moderately:

- `operators.py` – `TimeDependentDenseArray.parse()` (line 1819) dispatches on domain type
- `surrogate_operator.py` – `__repr__` (line 224)
- `tests/numerics/ad/test_operators.py` – Asserts `sd_array.domain_type`, `intf_array.domain_type`, `bg_array.domain_type`

Without this property, callers would write `self._source.domain_type if self._source else None` — verbose and error-prone (forgetting the `None` guard is a real bug risk). The property is a clean shortcut.

**Verdict: Keep.** Its value is clear and it prevents repetitive None-guard boilerplate.

---

### `subdomains`

Three call sites, all shallow:

1. `src/porepy/viz/diagnostics_mixin.py:384`:
   ```python
   grids = variable_on_grid.subdomains + variable_on_grid.interfaces
   ```
   Since every `Variable` lives on exactly one domain type, this is always equivalent to `variable_on_grid.domains`.

2. `src/porepy/numerics/ad/ad_utils.py:607–609` (`MergedOperator.__repr__`):
   ```python
   if len(self.interfaces) == 0:
       s += f"{len(self.subdomains)} subdomains"
   ```
   Can be replaced with `self.domain_type` and `len(self.domains)`.

**No tests assert `op.subdomains` directly** (only `space.domain_type == DomainType.subdomains` via `OperatorSpace`, which is unrelated).

**Verdict: Remove.** All three call sites have simple substitutions.

---

### `interfaces`

Four call sites, all shallow:

1. `src/porepy/viz/diagnostics_mixin.py:384` — Same combined call as above (`subdomains + interfaces` → `domains`).
2. `src/porepy/numerics/ad/ad_utils.py:607` (`MergedOperator.__repr__`):
   ```python
   if len(self.interfaces) == 0:
   ```
   → `if self.domain_type != DomainType.interfaces:`
3. `src/porepy/numerics/ad/ad_utils.py:612`:
   ```python
   s += f"{len(self.interfaces)} edges"
   ```
   → `len(self.domains)`
4. `tests/viz/test_diagnostics_mixin.py:88`:
   ```python
   if len(var.interfaces) > 0
   ```
   → `if var.domain_type == DomainType.interfaces`

**Verdict: Remove.** All four call sites have straightforward substitutions.

---

## Summary Decision

| Property | Decision | Reason |
|---|---|---|
| `domains` | **Keep** | Widely used; the primary grid accessor |
| `domain_type` | **Keep** | Used in internal dispatch and tests; prevents None-guard boilerplate |
| `subdomains` | **Remove** | Only 3 call sites, all trivially replaceable by `domains` + `domain_type` |
| `interfaces` | **Remove** | Only 4 call sites, all trivially replaceable by `domains` + `domain_type` |

---

## Implementation Plan

### Stage 1 — Update callers of `subdomains` and `interfaces`

Update all code that calls `.subdomains` or `.interfaces` as `Operator` properties (not `mdg.subdomains()`, `discr.subdomains`, etc.):

**`src/porepy/viz/diagnostics_mixin.py` (line 384)**

```python
# Before
grids: list[GridLike] = (
    variable_on_grid.subdomains + variable_on_grid.interfaces
)
# After
grids: list[GridLike] = variable_on_grid.domains
```

**`src/porepy/numerics/ad/ad_utils.py` (lines 607–612, `MergedOperator.__repr__`)**

```python
# Before
if len(self.interfaces) == 0:
    s = f"Operator with key {self._discretization_matrix_key} defined on "
    s += f"{len(self.subdomains)} subdomains"
else:
    s = f"Operator with key {self._discretization_matrix_key} defined on "
    s += f"{len(self.interfaces)} edges"
# After
domain_label = self.domain_type.value if self.domain_type is not None else "unknown"
s = (
    f"Operator with key {self._discretization_matrix_key} defined on "
    f"{len(self.domains)} {domain_label}"
)
```

**`tests/viz/test_diagnostics_mixin.py` (line 88)**

```python
# Before
var.name for var in model.equation_system.variables if len(var.interfaces) > 0
# After
var.name for var in model.equation_system.variables
    if var.domain_type == DomainType.interfaces
```

### Stage 2 — Remove the properties from `Operator`

Remove the `subdomains` property (lines 512–523) and the `interfaces` property (lines 499–510) from `Operator` in `src/porepy/numerics/ad/operators.py`.

### Stage 3 — Verify

Run:
```
python -m mypy src/ --no-error-summary
pytest tests/numerics/ad/ tests/viz/ -q
```

All tests should pass and mypy should report no errors.
