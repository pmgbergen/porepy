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

import logging
import os
from typing import Any

import numpy as np
import scipy.sparse as sps

import porepy as pp

logger = logging.getLogger(__name__)

# lower the replay Program to a compiled (numba) kernel with a baked global scatter.
# On by default; set SPARSA_COMPILED=0 to force the Python register-machine replay (used for
# A/B comparison and as the automatic fallback when a program cannot be lowered).
_USE_COMPILED = os.environ.get("SPARSA_COMPILED", "1") not in ("0", "false", "False")

# A compiled kernel bakes the sparsity of every matrix. Some discretizations -- notably
# UPWIND -- change their nonzero PATTERN with the iterate (which cell is upstream flips with
# the flux sign; see numerics/fv/upwind.py). A compiled kernel is therefore valid only for
# the exact structure it was built for. We key compiled kernels by a signature of the
# re-parsed matrices' structure and keep one per distinct pattern, bounded by this cap; once
# the cap is hit, new patterns fall back to the (always-correct) Python replay. Newton
# iterates typically cycle through a small, stabilizing set of upwind patterns -> cache hits.
_COMPILE_CACHE_CAP = int(os.environ.get("SPARSA_COMPILE_CACHE_CAP", "24"))


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
    __slots__ = ("program", "layout", "var_leaves", "surrogates", "const_leaves", "baked",
                 "compiled_cache", "compile_enabled", "cap_logged")

    def __init__(self, program, layout, var_leaves, surrogates, const_leaves, baked):
        self.program = program          # sparsa.Program (compiled structure + sparsity)
        self.layout = layout            # sparsa.Layout (global columns)
        self.var_leaves = var_leaves    # list of (reg, vid, dofs): refreshed from state
        self.surrogates = surrogates    # list of (op, value_reg, partial_regs, dep_rows)
        self.const_leaves = const_leaves  # list of (op, reg): RE-PARSED each replay
        # ^ discretizations / previous-iterate / BC values change every iteration even
        #   though the op structure is fixed -- they must NOT be baked at compile time.
        self.baked = baked              # list of (reg, matrix): geometric projection
        #   selection matrices, constant across iterations -> filled once, reused.
        # Tier-2: {structure_signature: sparsa.CompiledProgram}. compile_enabled is cleared
        # if the program turns out not to be lowerable (then always the Python replay).
        self.compiled_cache: dict[bytes, Any] = {}
        self.compile_enabled = _USE_COMPILED
        self.cap_logged = False


def _combine_compile(op, kids, rec, surr, sparsa):
    # Tags: "reg" = LocalAd-valued (depends on variables); "creg" = constant register
    # (a refreshed leaf: matrix / array / scalar). A result is "reg" iff any LocalAd
    # operand feeds it. Projection lists carry the tag "proj".
    name = op.operation.name
    is_reg = lambda t: t[0] == "reg"
    rtag = lambda *ts: "reg" if any(is_reg(t) for t in ts) else "creg"

    if name in ("add", "sub", "mul", "div"):
        a, b = kids
        return (rtag(a, b), rec.emit(name, [a[1], b[1]]))
    if name == "pow":
        a, b = kids
        return (rtag(a, b), rec.emit("pow", [a[1], b[1]]))
    if name in ("matmul", "rmatmul"):
        # Projections are baked into constant matrices in _compile, so every matmul
        # operand is a plain register (matrix / LocalAd) -- no special slicer handling.
        left, right = (kids[0], kids[1]) if name == "matmul" else (kids[1], kids[0])
        return (rtag(left, right), rec.emit("matmul", [left[1], right[1]]))
    if name == "evaluate":
        if not (hasattr(op, "_fetch_data") and hasattr(op, "domains")):
            raise _CompileUnsupported(f"non-surrogate function node {op!r}")
        diff = [(i, k) for i, k in enumerate(kids) if is_reg(k)]  # current-var deps
        value_reg = rec.leaf()
        if not diff:
            # Args are all constant/previous-time (e.g. an accumulation term at the old
            # time level): zero Jacobian wrt current variables. Compile to a refreshed
            # value-only leaf (plain array, re-fetched each replay), exactly mirroring the
            # eager path's `op.func(*constants)` -> plain value. Tag "creg" so downstream
            # treats it as a constant (a LocalAd here would break `matrix @ value`).
            surr.append((op, value_reg, [], []))
            return ("creg", value_reg)
        partial_regs = [rec.leaf() for _ in diff]
        arg_regs = [k[1] for _, k in diff]
        out = rec.emit("compose", [value_reg, *arg_regs, *partial_regs], const=len(diff))
        surr.append((op, value_reg, partial_regs, [i for i, _ in diff]))
        return ("reg", out)
    raise _CompileUnsupported(f"operation {name}")


class _CompileUnsupported(Exception):
    pass


def _materialize_projection(slicers):
    """Turn one or more :class:`ArraySlicer` projections into a single constant CSR
    selection matrix (``sum_k slicer_k @ I``). Projections are geometric and fixed across
    iterations, so this is baked once at compile time and reused every replay."""
    M = None
    for sl in slicers:
        m = (sl @ sps.identity(sl.domain_size, format="csr")).tocsr()
        M = m if M is None else (M + m)
    return M


def _compile(op, rec, equation_system, var_by_key, surr, const, baked, mdg, cache, sparsa):
    oid = id(op)
    if oid in cache:
        return cache[oid]
    if isinstance(op, (pp.ad.ProjectionList, pp.ad.Projection)):
        # Geometric projections are fixed -> materialize once into a constant selection
        # matrix and bake it. A baked matrix works uniformly as a matmul operand AND as a
        # binary operand (e.g. ``Scalar(-1.0) * Projection``), unlike a deferred slicer
        # list (which is not indexable as a register).
        parsed = op.parse(mdg)
        slicers = parsed if isinstance(parsed, list) else [parsed]
        reg = rec.leaf()
        baked.append((reg, _materialize_projection(slicers)))
        res = ("creg", reg)
    elif op.is_leaf():
        if isinstance(op, pp.ad.Variable) and not (
            op.is_previous_iterate or op.is_previous_time
        ) and equation_system.dofs_of([op]).size > 0:
            res = ("reg", var_by_key[_dofs_key(equation_system.dofs_of([op]))])
        else:
            # Everything else (constants, discretizations, previous-iterate/time vars,
            # empty interface vars, BC/time-dependent arrays) is a REFRESHED leaf: its
            # value is re-parsed every replay, so nothing stale gets baked in.
            reg = rec.leaf()
            const.append((op, reg))
            res = ("creg", reg)
    else:
        kids = [_compile(c, rec, equation_system, var_by_key, surr, const, baked, mdg, cache, sparsa)
                for c in op.children]
        res = _combine_compile(op, kids, rec, surr, sparsa)
    cache[oid] = res
    return res


def _parse_const_leaf(op, mdg, sparsa):
    """Re-parse a refreshed constant leaf's CURRENT value (called every replay)."""
    if isinstance(op, pp.ad.Variable):
        if op.is_previous_iterate or op.is_previous_time:
            if isinstance(op, pp.ad.MixedDimensionalVariable):
                subs = op.sub_vars
                return (np.concatenate([sv.parse(mdg) for sv in subs])
                        if subs else np.zeros(0))
            return op.parse(mdg)
        return sparsa.LocalAd(np.zeros(0), {})  # current variable with 0 dofs (empty)
    return op.parse(mdg)


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
    surr, const, baked, cache, out_regs = [], [], [], {}, []
    for o in ops:
        tag = _compile(o, rec, equation_system, var_by_key, surr, const, baked, mdg, cache, sparsa)
        if tag[0] != "reg":
            raise _CompileUnsupported("equation root is not variable-dependent")
        out_regs.append(tag[1])
    # Compiled kernels are built lazily, per matrix-structure signature, in _replay.
    return _Bundle(rec.build(out_regs), tape.layout, var_leaves, surr, const, baked)


def _build_leaves(bundle, state, mdg, sparsa):
    """Assemble the reg -> value leaf dict for one replay (refreshing variable seeds,
    re-parsed constants, and surrogate value/partials). Shared by the Python replay and the
    compiled executor, and by the compile-time warmup."""
    leaves = {}
    for reg, vid, dofs in bundle.var_leaves:
        leaves[reg] = sparsa.LocalAd(state[dofs], {vid: sparsa.DiagBlock(np.ones(dofs.size))})
    for reg, mat in bundle.baked:  # geometric projection matrices, constant across replays
        leaves[reg] = mat
    for op, reg in bundle.const_leaves:  # re-parse CURRENT values (no stale baked data)
        leaves[reg] = _parse_const_leaf(op, mdg, sparsa)
    for op, value_reg, partial_regs, dep_rows in bundle.surrogates:
        leaves[value_reg] = np.hstack([op._fetch_data(op, g, False) for g in op.domains])
        if dep_rows:  # value-only (all-constant-arg) surrogates carry no partials
            derivs = np.atleast_2d(
                np.hstack([op._fetch_data(op, g, True) for g in op.domains]))
            for preg, row in zip(partial_regs, dep_rows):
                leaves[preg] = derivs[row]
    return leaves


def _structure_signature(bundle, leaves) -> bytes:
    """A key over the nonzero PATTERN of every re-parsed sparse leaf matrix. Constant for
    fixed discretizations (TPFA/MPFA/divergence/projections); changes when an upwind matrix
    flips an upstream cell. Two replays with the same signature have identical Jacobian
    structure, so a compiled kernel built for one is exact for the other."""
    parts: list[bytes] = []
    for _op, reg in bundle.const_leaves:
        v = leaves.get(reg)
        if sps.issparse(v):
            v = v.tocsr()
            parts.append(v.indptr.tobytes())
            parts.append(v.indices.tobytes())
    return b"".join(parts)


def _compiled_for(bundle, leaves, sparsa):
    """Return a CompiledProgram matching the current matrix structure (from the per-pattern
    cache, compiling a new one if needed and within the cap), or None to use the Python
    replay. The returned kernel is structurally exact for ``leaves``."""
    if not bundle.compile_enabled:
        return None
    sig = _structure_signature(bundle, leaves)
    cp = bundle.compiled_cache.get(sig)
    if cp is not None:
        return cp
    if len(bundle.compiled_cache) >= _COMPILE_CACHE_CAP:
        if not bundle.cap_logged:
            bundle.cap_logged = True
            logger.info(
                "sparsa: matrix-structure pattern cache cap (%d) reached; new patterns "
                "use the Python replay. Raise SPARSA_COMPILE_CACHE_CAP if assembly is slow.",
                _COMPILE_CACHE_CAP)
        return None  # too many distinct patterns -> stop compiling, stay correct via replay
    try:
        cp = sparsa.compile_program(bundle.program, leaves)
        cp.compile_assembly(bundle.layout)
    except sparsa.UnsupportedProgram as err:
        bundle.compile_enabled = False  # a construct we cannot lower -> never retry
        logger.info("sparsa: program not lowerable (%s); using the Python replay.", err)
        return None
    except Exception as err:  # pragma: no cover - never let compilation break the solve
        bundle.compile_enabled = False
        logger.warning("sparsa: compilation failed (%r); using the Python replay.", err)
        return None
    bundle.compiled_cache[sig] = cp
    logger.debug("sparsa: compiled a new matrix-structure pattern (#%d).",
                 len(bundle.compiled_cache))
    return cp


def _replay(bundle, equation_system, state, mdg, sparsa, derivative):
    leaves = _build_leaves(bundle, state, mdg, sparsa)
    # Tier-2 compiled fast path: numba kernels + baked global scatter, no scipy in the loop.
    # Valid only for the exact matrix structure it was built for -> selected by signature.
    cp = _compiled_for(bundle, leaves, sparsa)
    if cp is not None:
        if derivative:
            return [pp.ad.AdArray(val, jac) for (val, jac) in cp.assemble(leaves)]
        return [np.atleast_1d(v) for v in cp.run_values(leaves)]

    outs = bundle.program.run(leaves)
    if derivative:
        return [_finalize(ad, bundle.layout, equation_system, sparsa) for ad in outs]
    # residual-only: the structure is identical, so the SAME Program serves both. We
    # replay and take only the value (skipping the global Jacobian scatter), which still
    # avoids the eager DAG re-walk -- the dominant cost of the per-iteration convergence
    # residual re-assembly.
    return [ad.val if isinstance(ad, sparsa.LocalAd)
            else np.atleast_1d(np.asarray(ad, dtype=float)) for ad in outs]


class SparsaParser:
    """Drop-in replacement for ``_ad_parser.AdParser`` backed by sparsa.

    For derivative assembles of a (structurally fixed) equation list, the DAG is compiled
    into a sparsa Program once and replayed thereafter; other calls use the eager path.
    """

    def __init__(self, mdg: pp.MixedDimensionalGrid) -> None:
        self.mdg = mdg
        self._sparsa = _require_sparsa()
        self._bundles: dict = {}
        self._stats = {"compiled_jac": 0, "compiled_res": 0, "eager_list": 0,
                       "eager_single_jac": 0, "eager_single_res": 0}

    def clear_cache(self) -> None:
        pass

    def evaluate(self, op, equation_system, derivative, state=None):
        sparsa = self._sparsa
        if state is None:
            state = equation_system.get_variable_values(iterate_index=0)

        # Compile-once fast path for an equation LIST. The structure is fixed, so the
        # SAME compiled Program serves both the Jacobian assemble (derivative=True) and
        # the residual-only assemble (derivative=False, used by the convergence check).
        if isinstance(op, list):
            key = tuple(id(o) for o in op)
            if key not in self._bundles:
                try:
                    self._bundles[key] = _build_bundle(
                        op, equation_system, state, self.mdg, sparsa)
                except _CompileUnsupported:
                    self._bundles[key] = None  # uncompilable -> eager path
            bundle = self._bundles[key]
            if bundle is not None:
                self._stats["compiled_jac" if derivative else "compiled_res"] += 1
                return _replay(bundle, equation_system, state, self.mdg, sparsa, derivative)
            self._stats["eager_list"] += 1
        else:
            self._stats["eager_single_jac" if derivative else "eager_single_res"] += 1

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
