# Changelog

All changes to PorePy are documented in this file. It is updated by
every pull request (PR) that changes user-facing behavior, and is used as the
basis for release notes.

Format: each entry is one line, `#PR: Summary of the change.`
Keep summaries to 1-2 sentences and written for a PorePy user, not just the
author.

## Unreleased

### Changes
PR 1747: Ad Operators domain, range and dof info is specified by OperatorSpace objects.

### Breaking changes
PR 1747: DenseAdArray, SparseAdArray must be initialized with OperatorSpace objects.
Other Ad Operators are by default treated as mapping from and to the same grid, with one
degree of freedom per cell. To change this default, explicit OperatorSpaces must be
assigned.

Changes to the front end (user-facing code: multiphysics models, solvers, grids, and
similar) that require users to update their own code when upgrading.
