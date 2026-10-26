"""Text reports independent of the numerical study producing the results."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from porepy.examples.geothermal_flow.obl_convergence_analysis.convergence_metrics import (
    observed_convergence_order,
)
from porepy.examples.geothermal_flow.obl_convergence_analysis.convergence_results import (
    ConvergenceResult,
)

if TYPE_CHECKING:
    from .interpolation_convergence import SampleSet


def report_convergence(
    results: ConvergenceResult, sample_sets: Sequence["SampleSet"]
) -> None:
    """Print each category's errors per level, followed by the orders between levels.

    Parameters:
        results: Refinement results, coarse to fine.
        sample_sets: Sample sets to report.

    Raises:
        ValueError: If there are no results.

    """
    if not results:
        raise ValueError("No refinement results to report.")
    print(f"\nError measure: {results.normalization}")
    if results.reference_level is not None:
        print(
            f"Reference: L{results.reference_level}; excluded from error/order columns."
        )
        print(
            "Orders are finite-reference estimates, "
            "not orders against exact derivatives."
        )
    category_levels = " and ".join(f"L{level}" for level in results.category_levels)
    for sample_set in sample_sets:
        name = sample_set.name
        print(f"\n{'=' * 76}\nsample set: {name} ({len(sample_set)} points)")
        print(sample_set.description)
        if name in results.outside_counts:
            print(
                f"Outside at least one table (excluded): {results.outside_counts[name]}"
            )
        elif any(r.out_of_bounds.get(name, 0.0) > 0 for r in results):
            fraction = max(r.out_of_bounds.get(name, 0.0) for r in results)
            print(
                f"WARNING: up to {fraction:.1%} outside a table; "
                "errors include its out-of-bounds rule."
            )
        for mask, fraction in results[-1].repairs.get(name, {}).items():
            if fraction > 0:
                print(f"  {mask}: touches {fraction:.1%} at L{results[-1].level}")
        descriptions = {
            "all points": "all selected samples",
            "clean": "no NaN-repaired cell corners"
            + (
                "; cell corners share one phase region"
                if sample_set.single_phase_cells
                else ""
            )
            + f" at {category_levels}",
            "near repair": f"cells touching NaN-repaired nodes at {category_levels}",
        }
        for category in results[-1].errors[name]:
            count = results[-1].n_points[name][category]
            description = descriptions.get(category, category)
            print(f"\n--- {category} ({count} points): {description}")
            if count == 0:
                continue
            print("\nL2 errors (normalisation stated above)")
            print(
                f"{'field / component':24s}"
                + "".join(f"{'L' + str(r.level):>13s}" for r in results)
                + f"{'excluded*':>12s}"
            )
            for name_field in results.fields:
                errors = [r.errors[name][category][name_field] for r in results]
                print(
                    f"{name_field:24s}"
                    + "".join(f"{error:13.4e}" for error, _ in errors)
                    + f"{errors[-1][1]:12d}"
                )
            print(
                "* Excluded at the last displayed level; "
                "sample-category counts precede field exclusions."
            )
            if len(results) < 2:
                continue
            pairs = list(zip(results, results[1:]))
            print("\nObserved convergence orders")
            print(
                f"{'field / component':24s}"
                + "".join(
                    f"{'L' + str(a.level) + '->L' + str(b.level):>13s}"
                    for a, b in pairs
                )
            )
            for name_field in results.fields:
                orders = [
                    observed_convergence_order(
                        a.errors[name][category][name_field][0],
                        b.errors[name][category][name_field][0],
                    )
                    / (b.level - a.level)
                    for a, b in pairs
                ]
                print(
                    f"{name_field:24s}" + "".join(f"{order:13.3f}" for order in orders)
                )
    print("\nTable sizes (comparison levels)")
    for result in results:
        print(
            f"  L{result.level}: spacing={result.spacing}, nodes={result.node_count:,}"
        )
