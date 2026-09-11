# Changelog

All changes to PorePy are documented in this file. It is updated by
every pull request (PR) that changes user-facing behavior, and is used as the
basis for release notes.

Format: each entry is one line, `PR: Summary of the change.`
Keep summaries to 1-2 sentences and written for a PorePy user, not just the
author.

The lists are sorted on PR numbers.

## Unreleased

### Changes
PR 1772: SPEED: Faster construction of grid topologies on 3d simplex grids.
PR 1770: Simplify PR template by deferring to CONTRIBUTING.md for details on code style and conventions.
PR 1767: Bugfix in THM manufactured setup.
PR 1766: Fix bug in generation of Cartesian grids not anchored in the origin.
PR 1732: Added support for evaluating restricted variable subsystems without evaluating
    the full Jacobian and then slicing columns.
PR 1748: Refactor time-step control with interval-based scheduling and extensible
    constraints, including nonlinear-iteration and Courant-number constraints.




### Breaking changes

Changes to the front end (user-facing code: multiphysics models, solvers, grids, and
similar) that require users to update their own code when upgrading.

PR 1748: `time_manager.schedule` is not an array anymore, it's an object. Preserve the
    old behavior by replacing it with `time_manager.schedule.get_array()`.