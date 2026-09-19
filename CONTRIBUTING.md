# Contributing to PorePy

Contributions to PorePy are always welcome, in the form of [pull requests](https://github.com/pmgbergen/porepy/pulls),
[issues](https://github.com/pmgbergen/porepy/issues/new), or [questions](https://github.com/pmgbergen/porepy/discussions/new/choose).

By submitting a pull request, you license your code under PorePy's [license](https://github.com/pmgbergen/porepy/blob/develop/LICENSE),
and affirm that you own the copyright of the code or have permission to distribute it under that license.

## Getting started

For installation instructions, see [Install.md](Install.md).

New feature and bugfix branches should be based on, and target, `develop` (`main` tracks releases only).


## Code style

All code must pass the static checks that also run in CI (`.github/workflows/run-static-checks.yml`):

* **[ruff](https://docs.astral.sh/ruff/)** for linting and formatting: `ruff check src tests` and `ruff format --check src tests`.
* **[isort](https://pycqa.github.io/isort/)** for import ordering: `isort --check src tests`.
* **[mypy](https://mypy-lang.org/)** for static type checking: `mypy src`. All new code should carry type hints.

Beyond what tools can check, we also value:

* Expressive function, class, and variable names over comments that explain what the code does.
* Small, focused functions and classes. If you need a comment to separate "sections" of a function, it is
  probably two functions.
* As a rule of thumb, each function should only increase the level of detail by one,
  with further implementation details left to subfunctions (use nested functions if
  relevant). This must be balanced against proliferation of helper functions.  
* No speculative generality: avoid abstractions, configuration options, or error handling for cases the code
  does not actually need to support yet.

## Models and mixins

Models are assembled by multiple inheritance from a large number of mixin classes, and the method resolution
order (MRO) decides which implementation wins. A few conventions keep that tractable:

* Inherit `pp.PorePyModel` in every model mixin, and do not re-declare members the protocol already provides.
  At runtime the protocol is replaced by a placeholder that contributes nothing to the MRO, so inheriting it is
  safe in any base position: it can neither reorder sibling mixins nor cause an inconsistent-MRO `TypeError`.
* Prefer `pp.PorePyModel` over a concrete mixin (`pp.BalanceEquation`, `pp.VariableMixin`, `pp.SolutionStrategy`,
  ...) for a mixin that is only bundled into other mixins. A concrete base imposes a category ordering on every
  model the bundle is mixed into, which may contradict the ordering that model already uses.
* Never import `typing.Protocol` outside an `if TYPE_CHECKING:` block.
* `isinstance(model, pp.PorePyModel)` raises `TypeError` by design; annotate with the protocol instead of testing
  against it.
* Mypy and the interpreter compute different MROs, since the protocol is a base class only for mypy. This is why
  `# type: ignore[safe-super]` is occasionally needed on `super()` calls into sibling mixins. The divergence is
  safe only as long as every implementation of a protocol member stays signature-compatible with its declaration,
  which `mypy src` enforces. See the `Note` in `porepy.models.protocol` for details.

## Documentation

All public functions, classes, and modules should be documented following the
[PorePy docstring guidelines](https://pmgbergen.github.io/porepy/html/docsrc/howto/howto-docstring.html).

Note that a `docs/` directory must never be committed to `develop` or `main`; CI will reject it (the rendered
documentation is built and hosted separately).

Comments should focus on the actual code, not what was changed in a specific update. The exception is warnings against plausible future misunderstandings.

## Testing

* New functionality should be covered by both **unit tests** (isolated, fast, targeting a single function or
  class) and, where relevant, **integration tests** (verifying that components work together correctly).
* Use `pytest` fixtures and parametrization to avoid duplicated test logic rather than copy-pasting near-identical
  test functions.
* Do not contribute tests that were merely useful while developing a feature but are unnecessarily fine-grained
  or brittle for long-term maintenance (e.g. tests that pin incidental implementation details). If you are unsure
  what constitutes appropriate coverage, ask in the PR.
* Write a short docstring or comment for non-trivial tests explaining what a failure most likely indicates. This
  saves time when the test fails in CI six months from now.
* The test directory structure should mirror the source structure: tests for `src/porepy/path/to/file.py` normally
  belong
  in `tests/path/to/test_file.py`. If this rule of thumb makes no sense in your case, ask in the PR.

## Commit messages

Prefer several small, self-contained commits over one large commit; each commit should ideally leave the
repository in a working state. Start the commit message with one of the following prefixes, describing the
main intent of the commit:

| Prefix   | Meaning                                                        |
|----------|-----------------------------------------------------------------|
| `API`    | Changes to the public API, in particular incompatible ones     |
| `BLD`    | Changes related to building PorePy                              |
| `BUG`    | Bug fix                                                          |
| `DEP`    | Deprecate functionality, or remove something already deprecated |
| `DEV`    | Development tools or utilities (not shipped with PorePy)        |
| `DOC`    | Documentation only                                               |
| `ENH`    | Enhancement: new functionality                                   |
| `MAINT`  | Maintenance: refactoring, cleanup, typos, etc.                   |
| `MOVE`   | Move or rename a file, without other changes                    |
| `REL`    | Related to releasing a new version of PorePy                    |
| `REV`    | Revert an earlier commit                                        |
| `SPEED`  | Changes mainly targeting computational performance               |
| `STY`    | Style fixes (whitespace, ruff, mypy, PEP8) with no logic change |
| `TST`    | Addition or modification of tests                                |
| `TUT`    | Addition or modification of tutorials                            |

For example: `TST: Add regression test for MPFA on simplex grids`. If a commit genuinely mixes concerns, pick
the prefix for its main purpose rather than inventing a new one.


## Pull requests

* Aim for reasonably sized PRs: a PR that does one thing is easier and faster to review than one that mixes,
  say, a refactoring with a new feature.
* All pull requests undergo code review and are run against the full test suite and static checks.
* Document your changes in [CHANGELOG.md](https://github.com/pmgbergen/porepy/blob/develop/CHANGELOG.md) and fill in the PR template.
* Response times may vary depending on maintainers' other commitments, but all contributions will be followed up.

## Reporting issues

If reporting an issue, please provide a minimal working example that reproduces the problem, together with any
other information useful for debugging (PorePy version, Python version, OS).
