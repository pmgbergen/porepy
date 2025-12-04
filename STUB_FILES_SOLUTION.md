# Stub Files Solution for PorePyModel Inheritance

## Overview

This document explains the stub file approach used to remove runtime inheritance from `pp.PorePyModel` while preserving type checking capabilities.

## Problem

Mixin classes previously inherited from `pp.PorePyModel` for typing purposes, which:
1. Created runtime dependency on PorePyModel
2. Could lead to diamond inheritance issues
3. Mixed runtime and type-checking concerns

## Solution

### Approach

**Removed runtime inheritance** from all 47 mixin classes and **created stub files** (`.pyi`) that declare:
1. Method signatures for each mixin class
2. PorePyModel attributes used by each mixin (declared as `Any` for duck typing)

### Example

**Runtime implementation (`.py` file):**
```python
class InitialConditionMixin:
    """Mixin for setting initial conditions."""
    
    def initial_condition(self) -> None:
        self.equation_system.set_variable_values(
            np.zeros(self.equation_system.num_dofs()), iterate_index=0
        )
        self.set_initial_values_primary_variables()
    
    def set_initial_values_primary_variables(self) -> None:
        pass
```

**Type stub (`.pyi` file):**
```python
from typing import Any

class InitialConditionMixin:
    # Attributes expected from PorePyModel (via duck typing)
    equation_system: Any
    mdg: Any
    
    def initial_condition(self) -> None: ...
    def set_initial_values_primary_variables(self) -> None: ...
```

## How It Works

1. **Runtime**: Mixin classes have no base class (inherit from `object` implicitly)
2. **Type Checking**: Mypy reads the stub files, which declare:
   - Method signatures (so type checkers know what methods exist)
   - Attributes as `Any` type (so type checkers don't complain about missing attributes)

3. **Duck Typing**: When mixins are used in multiple inheritance, the attributes they use (like `equation_system`, `mdg`) come from other mixins or the final model class

## Benefits

1. **No Diamond Inheritance**: Runtime classes don't inherit from Protocol
2. **Type Safety**: Stubs provide method signatures for type checking
3. **Flexibility**: Duck typing approach allows mixins to work with any class that provides the expected attributes
4. **Separation of Concerns**: Type information separate from runtime behavior
5. **No Protocol Issues**: Avoids the `__slots__` issues mentioned in protocol.py warning

## Files Modified

### Python Source Files (10)
- `abstract_equations.py` - 2 classes
- `boundary_condition.py` - 1 class
- `compositional_flow.py` - 1 class  
- `constitutive_laws.py` - 28 classes
- `contact_mechanics.py` - 1 class
- `fluid_property_library.py` - 7 classes
- `fracture_damage.py` - 3 classes
- `geometry.py` - 1 class
- `initial_condition.py` - 1 class
- `solution_strategy.py` - 2 classes

### Stub Files Created (10)
Corresponding `.pyi` files for each of the above

## Type Checking

To verify type checking with stubs:
```bash
# Check specific module
python -m mypy -p porepy.models.initial_condition

# Check all models
python -m mypy src/porepy/models/
```

The stub files ensure that type checkers understand:
- What methods each mixin provides
- What attributes each mixin expects (via duck typing)
- No new type errors introduced (same baseline as before)

## Notes

- Stub files are only used by type checkers (mypy, pyright, etc.)
- They do not affect runtime behavior
- The `Any` type for attributes provides maximum flexibility for duck typing
- This approach follows Python's typing best practices for mixin classes
