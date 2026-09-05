# Changelog

All changes to PorePy are documented in this file. It is updated by
every pull request (PR) that changes user-facing behavior, and is used as the
basis for release notes.

Format: each entry is one line, `#PR: Summary of the change.`
Keep summaries to 1-2 sentences and written for a PorePy user, not just the
author.

## Unreleased

### Changes
1770: Simplify PR template by deferring to CONTRIBUTING.md for details on code style and conventions.
#1732: Added support for evaluating restricted variable subsystems without evaluating
    the full Jacobian and then slicing columns.

### Breaking changes

Changes to the front end (user-facing code: multiphysics models, solvers, grids, and
similar) that require users to update their own code when upgrading.
