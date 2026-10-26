"""Typed results shared by reports and plots for both convergence studies."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import overload

ErrorValue = tuple[float, int]
"""l2 error and number of excluded samples."""

FieldErrors = dict[str, ErrorValue]
CategoryErrors = dict[str, FieldErrors]
SampleErrors = dict[str, CategoryErrors]
SampleCounts = dict[str, dict[str, int]]


@dataclass
class RefinementResult:
    """Errors at a single resolution, keyed by sample set/category/field."""

    level: int
    spacing: tuple[float, float, float]
    node_count: int
    errors: SampleErrors
    n_points: SampleCounts = field(default_factory=dict)
    repairs: dict[str, dict[str, float]] = field(default_factory=dict)
    out_of_bounds: dict[str, float] = field(default_factory=dict)


@dataclass
class ConvergenceResult(Sequence[RefinementResult]):
    """Coarse-to-fine sequence of refinement results, with metric and reference metadata."""

    refinements: list[RefinementResult]
    fields: tuple[str, ...]
    normalization: str
    category_levels: tuple[int, ...]
    reference_level: int | None = None
    outside_counts: dict[str, int] = field(default_factory=dict)
    component: str | None = None

    def __len__(self) -> int:
        return len(self.refinements)

    def __iter__(self) -> Iterator[RefinementResult]:
        return iter(self.refinements)

    @overload
    def __getitem__(self, index: int) -> RefinementResult: ...

    @overload
    def __getitem__(self, index: slice) -> list[RefinementResult]: ...

    def __getitem__(self, index):
        return self.refinements[index]

    def for_gradient_component(
        self, axis: str, fields: Sequence[str]
    ) -> ConvergenceResult:
        """Select one derivative component, keeping the plotting field names.

        Parameters:
            axis: Derivative component, ``"z"``, ``"h"`` or ``"p"``.
            fields: Field names to keep.

        Returns:
            The result for that component.

        Raises:
            ValueError: If ``axis`` is not ``"z"``, ``"h"`` or ``"p"``.

        """
        if axis not in ("z", "h", "p"):
            raise ValueError("Gradient component must be z, h or p.")
        refinements = [
            replace(
                result,
                errors={
                    sample: {
                        category: {name: rows[f"d({name})/d{axis}"] for name in fields}
                        for category, rows in categories.items()
                    }
                    for sample, categories in result.errors.items()
                },
            )
            for result in self.refinements
        ]
        return replace(
            self, refinements=refinements, fields=tuple(fields), component=axis
        )
