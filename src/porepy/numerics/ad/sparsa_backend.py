"""Adapter that lets PorePy consume the external ``sparsa`` AD engine.

This is the ONLY place that knows about both PorePy and sparsa. sparsa stays a
host-agnostic library (no PorePy references). Here we provide a drop-in replacement for
``_ad_parser.AdParser`` -- :class:`SparsaParser` -- that *lowers* a PorePy ``Operator``
DAG onto sparsa's local-by-design AD. Inject it by setting ``params["ad_backend"] =
"sparsa"`` on a model.

Node coverage (no fallback to PorePy's global-width assembly):
- variables (current; previous iterate/time become constants),
- constant leaves (Scalar / DenseArray / SparseArray / discretization matrices),
- algebra add/sub/mul/div/pow/matmul (+ reversed),
- ``ProjectionList`` (sum of sliced matmuls),
- operator-functions and surrogates (``Operations.evaluate``): handled by seeding each
  differentiable argument with a LOCAL identity (width = sum of argument sizes, never
  the global dof count), calling the node's own ``func`` to obtain its local partials,
  and composing them in sparsa via :func:`sparsa.compose`.

Only the function's own derivative values are consumed (the surrogate's stored table /
the function's analytic derivative); all composition + assembly is done by sparsa.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sps

import porepy as pp


def _require_sparsa():
    try:
        import sparsa
    except ImportError as err:  # pragma: no cover
        raise ImportError("The sparsa backend requires the 'sparsa' package.") from err
    return sparsa


def _dofs_key(dofs: np.ndarray) -> tuple[int, int, int]:
    return (int(dofs[0]), int(dofs[-1]), int(dofs.size))


def _seed_variables(equation_system, state, sparsa):
    """One sparsa variable per atomic PorePy variable, in global-dof order, so sparsa's
    column layout matches PorePy's global numbering. Returns (tape, var_ad)."""
    tape = sparsa.Tape()
    var_ad: dict[tuple[int, int, int], Any] = {}
    atomic = sorted(
        equation_system.variables, key=lambda v: int(equation_system.dofs_of([v])[0])
    )
    for v in atomic:
        dofs = equation_system.dofs_of([v])
        if dofs.size == 0:
            continue  # e.g. interface variables when there are no interfaces
        ad = tape.variable(state[dofs], name=v.name)
        var_ad[_dofs_key(dofs)] = ad
    if tape.layout.ndof != equation_system.num_dofs():
        raise NotImplementedError(
            "sparsa backend needs contiguous per-variable dof blocks "
            f"({tape.layout.ndof} != {equation_system.num_dofs()})."
        )
    return tape, var_ad


def _is_constant(x, sparsa) -> bool:
    return not isinstance(x, (sparsa.LocalAd, list))


def _lower_function(op, kids, sparsa):
    """Lower an ``Operations.evaluate`` node (operator-function / surrogate).

    Seed each differentiable child with a local identity, call ``op.func`` to obtain the
    node's local Jacobian, split it per child, and compose locally with sparsa.
    """
    diff = [(i, k) for i, k in enumerate(kids) if isinstance(k, sparsa.LocalAd)]
    if not diff:
        vals = [k.val if isinstance(k, sparsa.LocalAd) else k for k in kids]
        return op.func(*vals)  # purely constant -> plain value

    # Surrogate fast path: the value and the per-dependency DIAGONAL derivatives are
    # already stored as plain arrays. Fetch them directly and compose -- never build or
    # slice a sparse Jacobian (which is exactly what op.func/get_jacobian would do, and
    # what PorePy's global parser is forced to do). fetch_data(.,.,True) has shape
    # (num_dependencies, num_dofs): one diagonal partial per dependency/child.
    if hasattr(op, "_fetch_data") and hasattr(op, "domains"):
        value = np.hstack([op._fetch_data(op, g, False) for g in op.domains])
        derivs = np.atleast_2d(np.hstack([op._fetch_data(op, g, True) for g in op.domains]))
        args = [k for _, k in diff]
        jacs = [derivs[i] for i, _ in diff]  # diagonal vector per dependency
        return sparsa.compose(value, args, jacs)

    sizes = [k.n for _, k in diff]
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    total = int(offsets[-1])
    seed_pos = {i: (int(offsets[j]), int(sizes[j])) for j, (i, _) in enumerate(diff)}

    seeded = []
    for i, k in enumerate(kids):
        if isinstance(k, sparsa.LocalAd):
            off, n = seed_pos[i]
            ident = sps.csr_matrix(
                (np.ones(n), (np.arange(n), off + np.arange(n))), shape=(n, total)
            )
            seeded.append(pp.ad.AdArray(k.val, ident))
        else:
            seeded.append(k)  # constant passes straight through to func

    res = op.func(*seeded)
    if not isinstance(res, pp.ad.AdArray):
        return res  # function returned a constant
    jac_full = res.jac.tocsr()
    args = [k for _, k in diff]
    jacs = []
    for (i, _) in diff:
        off, n = seed_pos[i]
        block = jac_full[:, off:off + n]
        diag = _as_diagonal_vector(block, n)
        jacs.append(diag if diag is not None else block)  # vector if diagonal, else sparse
    return sparsa.compose(res.val, args, jacs)


def _as_diagonal_vector(block, n):
    """Return ``block``'s diagonal as a 1-D array if it is square (n x n) and purely
    diagonal (all nonzeros on the diagonal); otherwise None. Diagonal partials are the
    common EOS/closure case and let sparsa keep the cheap diagonal path."""
    if block.shape[0] != n:
        return None
    c = block.tocoo()
    if c.nnz <= n and bool(np.all(c.row == c.col)):
        return block.diagonal()
    return None


def _combine(op, kids, sparsa):
    name = op.operation.name
    if name == "add":
        return kids[0] + kids[1]
    if name == "sub":
        return kids[0] - kids[1]
    if name == "mul":
        return kids[0] * kids[1]
    if name == "rmul":
        return kids[1] * kids[0]
    if name == "div":
        return kids[0] / kids[1]
    if name == "rdiv":
        return kids[1] / kids[0]
    if name == "pow":
        return kids[0] ** kids[1]
    if name == "rpow":
        return kids[1] ** kids[0]
    if name in ("matmul", "rmatmul"):
        left, right = (kids[0], kids[1]) if name == "matmul" else (kids[1], kids[0])
        if isinstance(left, list):  # ProjectionList -> sum of sliced matmuls
            return sum(_apply_slicer(c, right, sparsa) for c in left)
        if isinstance(left, pp.matrix_operations.ArraySlicer):  # single Projection
            return _apply_slicer(left, right, sparsa)
        if isinstance(right, sparsa.LocalAd):
            if not sps.issparse(left) and not (
                isinstance(left, np.ndarray) and left.dtype != object
            ):
                raise NotImplementedError(
                    f"matmul left unsupported: type={type(left).__name__} "
                    f"dtype={getattr(left, 'dtype', None)} op={op!r} "
                    f"children={[type(c).__name__ for c in op.children]}"
                )
            return sparsa.linop(left) @ right
        if isinstance(left, sparsa.LocalAd):
            # LocalAd @ constant matrix (right side) -> apply transpose on the left
            return sparsa.linop(np.asarray(right).T if not sps.issparse(right)
                                else right.T) @ left
        return left @ right  # both constant
    if name == "evaluate":
        return _lower_function(op, kids, sparsa)
    raise NotImplementedError(f"sparsa backend: operation '{name}' not supported.")


def _apply_slicer(slicer, x, sparsa):
    """Apply an ArraySlicer (from a ProjectionList) to a LocalAd or constant."""
    if isinstance(x, sparsa.LocalAd):
        n = x.n
        sel = (slicer @ sps.identity(n, format="csr")).tocsr()  # materialize selection
        return sparsa.linop(sel) @ x
    return slicer @ x


def _lower(op, equation_system, var_ad, mdg, cache, sparsa):
    oid = id(op)
    if oid in cache:
        return cache[oid]
    if isinstance(op, pp.ad.ProjectionList):
        res: Any = [c.parse(mdg) for c in op.children]
    elif op.is_leaf():
        if isinstance(op, pp.ad.Variable):  # also MixedDimensionalVariable
            if op.is_previous_iterate or op.is_previous_time:
                if isinstance(op, pp.ad.MixedDimensionalVariable):
                    subs = op.sub_vars
                    res = (np.concatenate([sv.parse(mdg) for sv in subs])
                           if subs else np.zeros(0))
                else:
                    res = op.parse(mdg)  # atomic previous variable: stored values
            else:
                dofs = equation_system.dofs_of([op])
                if dofs.size == 0:
                    res = sparsa.LocalAd(np.zeros(0), {})  # empty (e.g. no interfaces)
                else:
                    res = var_ad[_dofs_key(dofs)]
        else:
            res = op.parse(mdg)
    else:
        kids = [_lower(c, equation_system, var_ad, mdg, cache, sparsa) for c in op.children]
        res = _combine(op, kids, sparsa)
    cache[oid] = res
    return res


def _finalize(ad, layout, equation_system, sparsa):
    """Turn a lowered result into a PorePy AdArray with a global-width Jacobian."""
    if isinstance(ad, sparsa.LocalAd):
        val, jac = sparsa.to_global(ad, layout)
        return pp.ad.AdArray(val, jac.tocsr())
    # constant promotion, mirroring _ad_parser.evaluate
    if isinstance(ad, (int, float)):
        ad = np.array([float(ad)])
    if isinstance(ad, np.ndarray) and ad.ndim == 1:
        return pp.ad.AdArray(ad, sps.csr_matrix((ad.shape[0], equation_system.num_dofs())))
    return ad


# ---------------------------------------------------------------------------------------
#  Compile-once / replay: walk the (fixed-structure) DAG once into a sparsa Program, then
#  replay it every assemble, refreshing only variable seeds + surrogate value/partials.
#  A tagged result is ("reg", id) for a register-valued (LocalAd) node, ("const", value)
#  for a constant, ("proj", [slicers]) for a ProjectionList, or ("empty",).
# ---------------------------------------------------------------------------------------
class _Bundle:
    __slots__ = ("program", "layout", "var_leaves", "surrogates")

    def __init__(self, program, layout, var_leaves, surrogates):
        self.program = program          # sparsa.Program
        self.layout = layout            # sparsa.Layout (global columns)
        self.var_leaves = var_leaves    # list of (reg, vid, dofs)
        self.surrogates = surrogates    # list of (op, value_reg, partial_regs, dep_rows)


def _combine_compile(op, kids, rec, surr, sparsa):
    name = op.operation.name
    REG = lambda x: x[0] == "reg"
    CONST = lambda x: x[0] == "const"

    if name in ("add", "sub", "mul", "div"):
        a, b = kids
        if REG(a) and REG(b):
            return ("reg", rec.emit(name, [a[1], b[1]]))
        if REG(a) and CONST(b):
            return ("reg", rec.emit(f"{name}_const", [a[1]], const=b[1]))
        if CONST(a) and REG(b):
            if name == "add":
                return ("reg", rec.emit("add_const", [b[1]], const=a[1]))
            if name == "mul":
                return ("reg", rec.emit("mul_const", [b[1]], const=a[1]))
            if name == "sub":
                return ("reg", rec.emit("rsub_const", [b[1]], const=a[1]))
            if name == "div":
                return ("reg", rec.emit("rdiv_const", [b[1]], const=a[1]))
        return ("const", {"add": lambda: a[1] + b[1], "sub": lambda: a[1] - b[1],
                          "mul": lambda: a[1] * b[1], "div": lambda: a[1] / b[1]}[name]())
    if name == "pow":
        a, b = kids
        return ("reg", rec.emit("pow_const", [a[1]], const=b[1])) if REG(a) else \
               ("const", a[1] ** b[1])
    if name in ("matmul", "rmatmul"):
        left, right = (kids[0], kids[1]) if name == "matmul" else (kids[1], kids[0])
        slicers = None
        if left[0] == "proj":
            slicers = left[1]
        elif isinstance(left[1], pp.matrix_operations.ArraySlicer):
            slicers = [left[1]]
        if slicers is not None:
            if REG(right):
                # materialized lazily on first replay using the operand size (cached after)
                return ("reg", rec.emit("matmul_proj", [right[1]], const=slicers))
            res = None  # projection @ constant -> constant
            for sl in slicers:
                m = sl @ right[1]
                res = m if res is None else res + m
            return ("const", res)
        if REG(right):
            return ("reg", rec.emit("matmul_const", [right[1]], const=left[1]))
        return ("const", left[1] @ right[1])
    if name == "evaluate":
        diff = [(i, k) for i, k in enumerate(kids) if REG(k)]
        if not diff:
            return ("const", op.func(*[k[1] for k in kids]))
        if not (hasattr(op, "_fetch_data") and hasattr(op, "domains")):
            raise _CompileUnsupported(f"non-surrogate function node {op!r}")
        value_reg = rec.leaf()
        partial_regs = [rec.leaf() for _ in diff]
        arg_regs = [k[1] for _, k in diff]
        out = rec.emit("compose", [value_reg, *arg_regs, *partial_regs], const=len(diff))
        surr.append((op, value_reg, partial_regs, [i for i, _ in diff]))
        return ("reg", out)
    raise _CompileUnsupported(f"operation {name}")


class _CompileUnsupported(Exception):
    pass


def _compile(op, rec, equation_system, var_by_key, surr, mdg, cache, sparsa):
    oid = id(op)
    if oid in cache:
        return cache[oid]
    if isinstance(op, pp.ad.ProjectionList):
        res = ("proj", [c.parse(mdg) for c in op.children])
    elif op.is_leaf():
        if isinstance(op, pp.ad.Variable):
            if op.is_previous_iterate or op.is_previous_time:
                if isinstance(op, pp.ad.MixedDimensionalVariable):
                    subs = op.sub_vars
                    res = ("const", np.concatenate([sv.parse(mdg) for sv in subs])
                           if subs else np.zeros(0))
                else:
                    res = ("const", op.parse(mdg))
            else:
                dofs = equation_system.dofs_of([op])
                if dofs.size == 0:
                    res = ("const", np.zeros(0))
                else:
                    res = ("reg", var_by_key[_dofs_key(dofs)])
        else:
            res = ("const", op.parse(mdg))
    else:
        kids = [_compile(c, rec, equation_system, var_by_key, surr, mdg, cache, sparsa)
                for c in op.children]
        res = _combine_compile(op, kids, rec, surr, sparsa)
    cache[oid] = res
    return res


def _build_bundle(ops, equation_system, state, mdg, sparsa):
    rec = sparsa.Recorder()
    tape = sparsa.Tape()
    var_by_key, var_leaves = {}, []
    for v in sorted(equation_system.variables, key=lambda v: int(equation_system.dofs_of([v])[0])):
        dofs = equation_system.dofs_of([v])
        if dofs.size == 0:
            continue
        seed = tape.variable(state[dofs], v.name)
        vid = next(iter(seed.blocks))
        reg = rec.leaf()
        var_by_key[_dofs_key(dofs)] = reg
        var_leaves.append((reg, vid, dofs))
    surr, cache, out_regs = [], {}, []
    for o in ops:
        tag = _compile(o, rec, equation_system, var_by_key, surr, mdg, cache, sparsa)
        if tag[0] != "reg":
            raise _CompileUnsupported("equation root is not a register")
        out_regs.append(tag[1])
    return _Bundle(rec.build(out_regs), tape.layout, var_leaves, surr)


def _replay(bundle, equation_system, state, mdg, sparsa):
    leaves = {}
    for reg, vid, dofs in bundle.var_leaves:
        leaves[reg] = sparsa.LocalAd(state[dofs], {vid: sparsa.DiagBlock(np.ones(dofs.size))})
    for op, value_reg, partial_regs, dep_rows in bundle.surrogates:
        value = np.hstack([op._fetch_data(op, g, False) for g in op.domains])
        derivs = np.atleast_2d(np.hstack([op._fetch_data(op, g, True) for g in op.domains]))
        leaves[value_reg] = value
        for preg, row in zip(partial_regs, dep_rows):
            leaves[preg] = derivs[row]
    outs = bundle.program.run(leaves)
    return [_finalize(ad, bundle.layout, equation_system, sparsa) for ad in outs]


class SparsaParser:
    """Drop-in replacement for ``_ad_parser.AdParser`` backed by sparsa.

    For derivative assembles of a (structurally fixed) equation list, the DAG is compiled
    into a sparsa Program once and replayed thereafter; other calls use the eager path.
    """

    def __init__(self, mdg: pp.MixedDimensionalGrid) -> None:
        self.mdg = mdg
        self._sparsa = _require_sparsa()
        self._bundles: dict = {}

    def clear_cache(self) -> None:
        pass

    def evaluate(self, op, equation_system, derivative, state=None):
        sparsa = self._sparsa
        if state is None:
            state = equation_system.get_variable_values(iterate_index=0)

        # Compile-once fast path: derivative assemble of an equation list.
        if derivative and isinstance(op, list):
            key = tuple(id(o) for o in op)
            bundle = self._bundles.get(key)
            if bundle is None and key not in self._bundles:
                try:
                    bundle = _build_bundle(op, equation_system, state, self.mdg, sparsa)
                except _CompileUnsupported:
                    bundle = None  # structurally uncompilable -> eager path
                self._bundles[key] = bundle  # cache success or None
            if bundle is not None:
                return _replay(bundle, equation_system, state, self.mdg, sparsa)

        # Eager path.
        ops = op if isinstance(op, list) else [op]
        tape, var_ad = _seed_variables(equation_system, state, sparsa)
        lowered = [_lower(o, equation_system, var_ad, self.mdg, {}, sparsa) for o in ops]
        if derivative:
            out = [_finalize(ad, tape.layout, equation_system, sparsa) for ad in lowered]
        else:
            out = [ad.val if isinstance(ad, sparsa.LocalAd) else ad for ad in lowered]
        return out if isinstance(op, list) else out[0]


def assemble(equation_system, state: np.ndarray | None = None):
    """Convenience: assemble ``(A, b)`` for an EquationSystem via sparsa (b = -residual)."""
    parser = SparsaParser(equation_system.mdg)
    ad_list = parser.evaluate(list(equation_system.equations.values()), equation_system, True, state)
    A = ad_list[0].jac if len(ad_list) == 1 else sps.vstack([a.jac for a in ad_list], format="csr")
    res = ad_list[0].val if len(ad_list) == 1 else np.concatenate([a.val for a in ad_list])
    return A.tocsr(), -res
