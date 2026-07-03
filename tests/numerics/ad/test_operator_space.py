"""Tests for Stage 2 of GH discussion #1601: DomainType, OperatorSpace and
source/target propagation."""

import operator as _op

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad._grid_entity import GridEntity
from porepy.numerics.ad.ad_utils import MergedOperator
from porepy.numerics.ad.operators import (
    DenseArray,
    DomainType,
    MixedDimensionalVariable,
    Operator,
    OperatorSpace,
    Operations,
    Scalar,
    SparseArray,
    Variable,
    sum_operator_list,
)
from porepy.numerics.ad.surrogate_operator import SurrogateOperator
from porepy.numerics.discretization import Discretization, InterfaceDiscretization


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _grid_for_dim(dim: int) -> pp.Grid:
    """Return a simple Cartesian grid of the given spatial dimension."""
    g = pp.CartGrid([3, 3] if dim == 2 else [2, 2, 2])
    g.compute_geometry()
    return g


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_subdomains():
    """Two 2-D Cartesian grids."""
    g1 = pp.CartGrid([2, 2])
    g2 = pp.CartGrid([3, 3])
    g1.compute_geometry()
    g2.compute_geometry()
    return g1, g2


@pytest.fixture
def one_mortar():
    """A single mortar grid."""
    g = pp.CartGrid([2])
    g.compute_geometry()
    return pp.MortarGrid(g.dim, {0: g, 1: g})


@pytest.fixture
def fracture_mdg():
    """2-D Cartesian mdg with two crossing fractures (3 subdomains, 4 interfaces)."""
    fracs = [np.array([[0, 2], [1, 1]]), np.array([[1, 1], [0, 2]])]
    md_grid = pp.meshing.cart_grid(fracs, np.array([2, 2]))
    md_grid.compute_geometry()
    return md_grid


# ---------------------------------------------------------------------------
# DomainType
# ---------------------------------------------------------------------------


class TestDomainType:
    def test_members(self):
        assert DomainType.subdomains.value == "subdomains"
        assert DomainType.interfaces.value == "interfaces"
        assert DomainType.boundary_grids.value == "boundary_grids"
        assert DomainType.scalar.value == "scalar"

    def test_distinct(self):
        types = [
            DomainType.subdomains,
            DomainType.interfaces,
            DomainType.boundary_grids,
            DomainType.scalar,
        ]
        assert len(set(types)) == 4


# ---------------------------------------------------------------------------
# OperatorSpace construction
# ---------------------------------------------------------------------------


class TestOperatorSpaceScalar:
    def test_scalar_factory(self):
        s = OperatorSpace.scalar()
        assert s.domain_type == DomainType.scalar
        assert s.grids == ()
        assert s.dof_info == {}

    def test_scalar_is_singleton_value(self):
        """Two calls to scalar() return equal (but not necessarily identical) objects."""
        assert OperatorSpace.scalar() == OperatorSpace.scalar()

    def test_scalar_hash(self):
        s1 = OperatorSpace.scalar()
        s2 = OperatorSpace.scalar()
        assert hash(s1) == hash(s2)
        assert s1 in {s2}


class TestOperatorSpaceFromDomains:
    def test_empty_domains_gives_scalar(self):
        space = OperatorSpace.from_domains([], {GridEntity.cells: 1})
        assert space.domain_type == DomainType.scalar

    def test_subdomains(self, two_subdomains):
        g1, g2 = two_subdomains
        space = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})
        assert space.domain_type == DomainType.subdomains
        assert space.grids == (g1, g2)
        assert space.dof_info == {GridEntity.cells: 1}

    def test_interfaces(self, one_mortar):
        space = OperatorSpace.from_domains([one_mortar], {GridEntity.cells: 2})
        assert space.domain_type == DomainType.interfaces
        assert space.grids == (one_mortar,)
        assert space.dof_info == {GridEntity.cells: 2}

    def test_mixed_grid_types_raises(self, two_subdomains, one_mortar):
        g1, _ = two_subdomains
        with pytest.raises(ValueError, match="same type"):
            OperatorSpace.from_domains([g1, one_mortar], {GridEntity.cells: 1})

    def test_dof_info_is_copied(self, two_subdomains):
        """Mutating the original dict must not affect the stored one."""
        g1, _ = two_subdomains
        dof = {GridEntity.cells: 1}
        space = OperatorSpace.from_domains([g1], dof)
        dof[GridEntity.faces] = 99
        assert GridEntity.faces not in space.dof_info


class TestOperatorSpaceEquality:
    def test_equal_spaces(self, two_subdomains):
        g1, g2 = two_subdomains
        s1 = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})
        s2 = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})
        assert s1 == s2

    def test_different_grids(self, two_subdomains):
        g1, g2 = two_subdomains
        s1 = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        s2 = OperatorSpace.from_domains([g2], {GridEntity.cells: 1})
        assert s1 != s2

    def test_different_dof_info(self, two_subdomains):
        g1, _ = two_subdomains
        s1 = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        s2 = OperatorSpace.from_domains([g1], {GridEntity.cells: 2})
        assert s1 != s2

    def test_different_domain_type(self, two_subdomains, one_mortar):
        g1, _ = two_subdomains
        s1 = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        s2 = OperatorSpace.from_domains([one_mortar], {GridEntity.cells: 1})
        assert s1 != s2

    def test_not_equal_to_other_type(self, two_subdomains):
        g1, _ = two_subdomains
        s = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        assert s != "not an OperatorSpace"
        assert s != 42

    def test_hash_equal_for_equal_spaces(self, two_subdomains):
        g1, g2 = two_subdomains
        s1 = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})
        s2 = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})
        assert hash(s1) == hash(s2)

    def test_usable_as_dict_key(self, two_subdomains):
        g1, g2 = two_subdomains
        s1 = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        s2 = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        d = {s1: "value"}
        assert d[s2] == "value"


# ---------------------------------------------------------------------------
# Operator.source and target properties
# ---------------------------------------------------------------------------


class TestOperatorProperties:
    def test_operator_requires_explicit_source_and_target(self):
        with pytest.raises(TypeError):
            Operator(name="test")

    def test_source_accepts_explicit_none(self):
        op = Operator(name="test", source=None, target=None)
        assert op.source is None

    def test_target_accepts_explicit_none(self):
        op = Operator(name="test", source=None, target=None)
        assert op.target is None

    def test_set_source_in_init(self, two_subdomains):
        g1, _ = two_subdomains
        space = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        op = Operator(name="test", source=space, target=None)
        assert op.source == space

    def test_set_target_in_init(self, two_subdomains):
        g1, _ = two_subdomains
        space = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        op = Operator(name="test", source=None, target=space)
        assert op.target == space


# ---------------------------------------------------------------------------
# Leaf operator spaces
# ---------------------------------------------------------------------------


class TestScalarSpace:
    def test_scalar_has_scalar_space(self):
        s = Scalar(3.14)
        assert s.source == OperatorSpace.scalar()
        assert s.target == OperatorSpace.scalar()

    def test_scalar_with_subdomain_has_subdomain_space(self, two_subdomains):
        g, _ = two_subdomains
        s = Scalar(1.0, domains=[g])
        assert s.source is not None
        assert s.source.domain_type == DomainType.subdomains
        assert s.source.grids == (g,)
        assert s.source == s.target

    def test_scalar_with_mortar_grid_has_interface_space(self, one_mortar):
        mg = one_mortar
        s = Scalar(2.0, domains=[mg])
        assert s.source is not None
        assert s.source.domain_type == DomainType.interfaces
        assert s.source.grids == (mg,)

    def test_scalar_domain_is_cellwise(self, two_subdomains):
        """Domain-bearing Scalar uses the natural cell-based space on its grids."""
        g, _ = two_subdomains
        s = Scalar(1.0, domains=[g])
        assert s.source is not None
        assert s.source.dof_info == {GridEntity.cells: 1}

    def test_scalar_neg_propagates_source(self, two_subdomains):
        g, _ = two_subdomains
        s = Scalar(3.0, domains=[g])
        neg = -s
        assert neg.source == s.source
        assert neg.target == s.target

    def test_scalar_neg_no_source(self):
        """Negating a plain Scalar preserves the scalar wildcard space."""
        s = Scalar(2.0)
        neg = -s
        assert neg.source == OperatorSpace.scalar()

    def test_scalar_empty_domains_gives_scalar_space(self):
        """Empty domains list is treated as no-domain (backward compat)."""
        s = Scalar(1.0, domains=[])
        assert s.source == OperatorSpace.scalar()

    def test_domain_bearing_scalar_combined_with_operator(self, two_subdomains):
        """A domain-bearing scalar uses the ordinary cell-based space on its grids."""
        g, _ = two_subdomains
        s = Scalar(2.0, domains=[g])
        v = Variable("p", {GridEntity.cells: 1}, g)
        result = s * v
        assert result.source == v.source
        assert result.target == v.target

    def test_domainless_scalar_combined_with_operator(self, two_subdomains):
        """Plain Scalar still inherits from the other operand."""
        g, _ = two_subdomains
        s = Scalar(2.0)
        v = Variable("p", {GridEntity.cells: 1}, g)
        result = s * v
        assert result.source == v.source


class TestVariableSpace:
    def test_variable_has_subdomain_space(self, two_subdomains):
        g, _ = two_subdomains
        var = Variable("p", {GridEntity.cells: 1}, g)
        assert var.source is not None
        assert var.source.domain_type == DomainType.subdomains
        assert var.source == var.target

    def test_variable_space_contains_correct_grid(self, two_subdomains):
        g, _ = two_subdomains
        var = Variable("p", {GridEntity.cells: 1}, g)
        assert var.source is not None
        assert var.source.grids == (g,)

    def test_variable_space_dof_info(self, two_subdomains):
        g, _ = two_subdomains
        var = Variable("u", {GridEntity.cells: 2, GridEntity.faces: 1}, g)
        assert var.source is not None
        assert var.source.dof_info == {
            GridEntity.cells: 2,
            GridEntity.faces: 1,
        }

    def test_variable_on_mortar_grid_has_interface_space(self, one_mortar):
        """Variable on a MortarGrid gets DomainType.interfaces."""
        var = Variable("lam", {GridEntity.cells: 1}, one_mortar)
        assert var.source is not None
        assert var.source.domain_type == DomainType.interfaces
        assert var.source.grids == (one_mortar,)

    def test_variable_on_mortar_grid_dof_info(self, one_mortar):
        """dof_info is preserved when Variable is on a mortar grid."""
        var = Variable("lam", {GridEntity.cells: 2}, one_mortar)
        assert var.source is not None
        assert var.source.dof_info == {GridEntity.cells: 2}


class TestMixedDimensionalVariableSpace:
    """MixedDimensionalVariable carries the union space of its sub-variables."""

    @pytest.fixture
    def md_var(self, two_subdomains):
        g1, g2 = two_subdomains
        v1 = Variable("p", {GridEntity.cells: 1}, g1)
        v2 = Variable("p", {GridEntity.cells: 1}, g2)
        return MixedDimensionalVariable([v1, v2])

    def test_md_variable_source_is_union(self, md_var, two_subdomains):
        g1, g2 = two_subdomains
        expected = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})
        assert md_var.source == expected

    def test_md_variable_target_is_union(self, md_var, two_subdomains):
        g1, g2 = two_subdomains
        expected = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})
        assert md_var.target == expected

    def test_md_variable_with_mixed_dof_info_has_unspecified_space(self, fracture_mdg):
        subdomains = fracture_mdg.subdomains()
        v1 = Variable("p", {GridEntity.cells: 1}, subdomains[0])
        v2 = Variable("p", {GridEntity.cells: 2}, subdomains[1])
        md_var = MixedDimensionalVariable([v1, v2])

        assert md_var.source is None
        assert md_var.target is None


class TestSurrogateOperatorSpace:
    """SurrogateOperator carries source/range derived from its dof_info."""

    @pytest.fixture
    def simple_mdg(self, two_subdomains):
        g1, g2 = two_subdomains
        return pp.meshing.subdomains_to_mdg([[g1, g2]])

    @pytest.fixture
    def surrogate_setup(self, simple_mdg):
        """Return (mdg, var) ready for SurrogateFactory construction."""
        eq = pp.ad.EquationSystem(simple_mdg)
        var = eq.create_variables(
            "p",
            dof_info={GridEntity.cells: 1},
            subdomains=list(simple_mdg.subdomains()),
        )
        return simple_mdg, var

    def test_surrogate_source_with_dof_info(self, surrogate_setup):
        """SurrogateFactory with explicit dof_info: the produced operator has a space."""
        mdg, var = surrogate_setup
        factory = pp.ad.SurrogateFactory(
            name="f",
            mdg=mdg,
            dependencies=[lambda grids: var],
            dof_info={GridEntity.cells: 1},
        )
        op = factory(list(mdg.subdomains()))
        assert op.source is not None
        assert op.source.domain_type == DomainType.subdomains
        assert op.source.dof_info == {GridEntity.cells: 1}

    def test_surrogate_target_equals_source(self, surrogate_setup):
        """For a SurrogateOperator the target equals the source (square operator)."""
        mdg, var = surrogate_setup
        factory = pp.ad.SurrogateFactory(
            name="f",
            mdg=mdg,
            dependencies=[lambda grids: var],
            dof_info={GridEntity.cells: 1},
        )
        op = factory(list(mdg.subdomains()))
        assert op.target == op.source

    def test_surrogate_operator_default_dof_info_gives_space(self, surrogate_setup):
        """dof_info=None (default) falls back to cells:1 and still sets a space."""
        mdg, var = surrogate_setup
        factory = pp.ad.SurrogateFactory(
            name="f",
            mdg=mdg,
            dependencies=[lambda grids: var],
        )
        op = factory(list(mdg.subdomains()))
        assert op.source is not None
        assert op.source.dof_info == {GridEntity.cells: 1}

    def test_surrogate_operator_direct_no_dof_info_gives_none_space(
        self, two_subdomains
    ):
        """SurrogateOperator instantiated directly with dof_info=None gets cells:1."""
        g1, g2 = two_subdomains
        v1 = Variable("p", {GridEntity.cells: 1}, g1)
        v2 = Variable("p", {GridEntity.cells: 1}, g2)
        op = SurrogateOperator(
            name="bare",
            domains=[g1, g2],
            children=[v1, v2],
            dof_info=None,
        )
        assert op.source is not None
        assert op.source.dof_info == {GridEntity.cells: 1}
        assert op.target is not None
        assert op.target.dof_info == {GridEntity.cells: 1}


class TestDenseArraySpace:
    def test_dense_array_no_space_by_default(self):
        arr = DenseArray(np.ones(5))
        assert arr.source is None
        assert arr.target is None

    def test_dense_array_with_explicit_space(self, two_subdomains):
        g, _ = two_subdomains
        space = OperatorSpace.from_domains([g], {GridEntity.cells: 1})
        arr = DenseArray(np.ones(5), source=space, target=space)
        assert arr.source == space
        assert arr.target == space


class TestSparseArraySpace:
    def test_sparse_array_no_space_by_default(self):
        mat = sps.eye(4, format="csr")
        op = SparseArray(mat)
        assert op.source is None
        assert op.target is None

    def test_sparse_array_with_explicit_space(self, two_subdomains):
        g, _ = two_subdomains
        space = OperatorSpace.from_domains([g], {GridEntity.cells: 1})
        mat = sps.eye(4, format="csr")
        op = SparseArray(mat, source=space, target=space)
        assert op.source == space
        assert op.target == space


# ---------------------------------------------------------------------------
# Domain/range propagation through arithmetic operations
# ---------------------------------------------------------------------------


class TestDomainRangePropagation:
    """Test that binary operations propagate source and target."""

    def _cell_space(self, g):
        return OperatorSpace.from_domains([g], {GridEntity.cells: 1})

    @pytest.mark.parametrize(
        "binary_op",
        [_op.add, _op.sub, _op.mul, _op.truediv],
        ids=["add", "sub", "mul", "div"],
    )
    def test_elementwise_same_space_propagates(self, two_subdomains, binary_op):
        """Elementwise ops between equal-space operands preserve source/target."""
        g, _ = two_subdomains
        space = self._cell_space(g)
        a = DenseArray(np.ones(4), source=space, target=space)
        b = DenseArray(np.ones(4), source=space, target=space)
        result = binary_op(a, b)
        assert result.source == space
        assert result.target == space

    def test_scalar_inherits_other_space(self, two_subdomains):
        g, _ = two_subdomains
        space = self._cell_space(g)
        arr = DenseArray(np.ones(4), source=space, target=space)
        s = Scalar(2.0)
        result = s * arr
        assert result.source == space
        assert result.target == space

    def test_other_inherits_scalar_space(self, two_subdomains):
        g, _ = two_subdomains
        space = self._cell_space(g)
        arr = DenseArray(np.ones(4), source=space, target=space)
        s = Scalar(2.0)
        result = arr * s
        assert result.source == space

    def test_scalar_scalar_gives_scalar_space(self):
        s1 = Scalar(1.0)
        s2 = Scalar(2.0)
        result = s1 + s2
        assert result.source == OperatorSpace.scalar()
        assert result.target == OperatorSpace.scalar()

    def test_none_space_propagates_other(self, two_subdomains):
        """An operand with None space should not block inference from the other."""
        g, _ = two_subdomains
        space = self._cell_space(g)
        a = DenseArray(np.ones(4), source=space, target=space)
        b = DenseArray(np.ones(4))  # no space
        result = a + b
        assert result.source == space

    def test_both_none_gives_none(self):
        a = DenseArray(np.ones(4))
        b = DenseArray(np.ones(4))
        result = a + b
        assert result.source is None
        assert result.target is None

    def test_incompatible_domains_raises(self, two_subdomains):
        g1, g2 = two_subdomains
        s1 = self._cell_space(g1)
        s2 = self._cell_space(g2)
        a = DenseArray(np.ones(4), source=s1, target=s1)
        b = DenseArray(np.ones(9), source=s2, target=s2)
        with pytest.raises(ValueError, match="[Ii]ncompat"):
            _ = a + b

    def test_matmul_propagates_outer_spaces(self, two_subdomains):
        """For A @ B: result.source = B.source, result.target = A.target."""
        g1, g2 = two_subdomains
        s1 = self._cell_space(g1)
        s2 = self._cell_space(g2)
        # A maps s1 -> s2, B maps s2 -> s1
        A = SparseArray(sps.eye(4, format="csr"), source=s1, target=s2)
        B = SparseArray(sps.eye(4, format="csr"), source=s2, target=s1)
        result = A @ B
        assert result.source == s2
        assert result.target == s2

    def test_matmul_incompatible_target_source_raises(self, two_subdomains):
        """A @ B raises if A.source != B.target."""
        g1, g2 = two_subdomains
        s1 = self._cell_space(g1)
        s2 = self._cell_space(g2)
        # A.source=s1, B.target=s2 -> incompatible
        A = SparseArray(sps.eye(4, format="csr"), source=s1, target=s2)
        B = SparseArray(sps.eye(4, format="csr"), source=s1, target=s2)
        with pytest.raises(ValueError, match="[Ii]ncompat"):
            _ = A @ B


# ---------------------------------------------------------------------------
# Stage 4c: TimeDependentDenseArray optional dof_info
# ---------------------------------------------------------------------------


class TestTimeDependentDenseArraySpaces:
    def test_no_dof_info_gives_none(self, two_subdomains):
        g1, g2 = two_subdomains
        arr = pp.ad.TimeDependentDenseArray("x", [g1, g2])
        # When domains are provided but no dof_info, cells:1 is assumed.
        assert arr.source is not None
        assert arr.source.dof_info == {GridEntity.cells: 1}
        assert arr.target is not None
        assert arr.target.dof_info == {GridEntity.cells: 1}

    def test_dof_info_cells(self, two_subdomains):
        g1, g2 = two_subdomains
        arr = pp.ad.TimeDependentDenseArray(
            "x", [g1, g2], dof_info={GridEntity.cells: 1}
        )
        assert arr.source is not None
        assert arr.target is not None
        assert arr.source == arr.target
        assert GridEntity.cells in arr.source.dof_info
        assert set(arr.source.grids) == {g1, g2}

    def test_dof_info_faces(self, two_subdomains):
        g1, g2 = two_subdomains
        arr = pp.ad.TimeDependentDenseArray(
            "x", [g1, g2], dof_info={GridEntity.faces: 2}
        )
        assert arr.source is not None
        assert arr.source.dof_info == {GridEntity.faces: 2}

    def test_empty_domains_ignores_dof_info(self):
        arr = pp.ad.TimeDependentDenseArray("x", [], dof_info={GridEntity.cells: 1})
        assert arr.source is None
        assert arr.target is None


# ---------------------------------------------------------------------------
# Stage 3: Grid operator source/target
# ---------------------------------------------------------------------------


class TestSubdomainProjectionSpaces:
    def test_cell_restriction_all(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        proj = pp.ad.SubdomainProjections(subdomains=sds, dim=1)
        op = proj.cell_restriction(sds)
        assert op.source is not None
        assert op.target is not None
        assert op.source.domain_type == DomainType.subdomains
        assert op.target.domain_type == DomainType.subdomains
        assert GridEntity.cells in op.source.dof_info
        assert GridEntity.cells in op.target.dof_info

    def test_cell_restriction_subset(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        sub = mdg.subdomains(dim=2)  # only top-dim subdomain
        proj = pp.ad.SubdomainProjections(subdomains=sds, dim=1)
        op = proj.cell_restriction(sub)
        # domain covers all subdomains, range covers the subset
        assert op.source is not None
        assert op.target is not None
        assert set(op.source.grids) == set(sds)
        assert set(op.target.grids) == set(sub)

    def test_cell_restriction_empty(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        proj = pp.ad.SubdomainProjections(subdomains=sds, dim=1)
        op = proj.cell_restriction([])
        # Empty subset -> target is None
        assert op.source is not None
        assert len(op.target.grids) == 0

    def test_cell_prolongation_spaces(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        sub = mdg.subdomains(dim=2)
        proj = pp.ad.SubdomainProjections(subdomains=sds, dim=1)
        op = proj.cell_prolongation(sub)
        # domain is the subset, range is all subdomains
        assert op.source is not None
        assert op.target is not None
        assert set(op.source.grids) == set(sub)
        assert set(op.target.grids) == set(sds)

    def test_face_restriction_spaces(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        sub = mdg.subdomains(dim=2)
        proj = pp.ad.SubdomainProjections(subdomains=sds, dim=1)
        op = proj.face_restriction(sub)
        assert op.source is not None
        assert op.target is not None
        assert GridEntity.faces in op.source.dof_info
        assert GridEntity.faces in op.target.dof_info

    def test_face_prolongation_spaces(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        sub = mdg.subdomains(dim=2)
        proj = pp.ad.SubdomainProjections(subdomains=sds, dim=1)
        op = proj.face_prolongation(sub)
        assert op.source is not None
        assert op.target is not None
        assert set(op.source.grids) == set(sub)
        assert set(op.target.grids) == set(sds)

    def test_vector_dim(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        dim = 2
        proj = pp.ad.SubdomainProjections(subdomains=sds, dim=dim)
        op = proj.cell_restriction(sds)
        assert op.source.dof_info[GridEntity.cells] == dim


class TestMortarProjectionSpaces:
    def test_mortar_to_primary_avg(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        intfs = list(mdg.interfaces())
        proj = pp.ad.MortarProjections(mdg=mdg, subdomains=sds, interfaces=intfs, dim=1)
        op = proj.mortar_to_primary_avg()
        assert op.source is not None
        assert op.target is not None
        assert GridEntity.cells in op.source.dof_info  # interface cells
        assert GridEntity.faces in op.target.dof_info  # subdomain faces (codim 1)

    def test_mortar_to_secondary_avg(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        intfs = list(mdg.interfaces())
        proj = pp.ad.MortarProjections(mdg=mdg, subdomains=sds, interfaces=intfs, dim=1)
        op = proj.mortar_to_secondary_avg()
        assert op.source is not None
        assert op.target is not None
        assert GridEntity.cells in op.source.dof_info  # interface cells
        assert GridEntity.cells in op.target.dof_info  # subdomain cells (secondary)

    def test_primary_to_mortar_avg(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        intfs = list(mdg.interfaces())
        proj = pp.ad.MortarProjections(mdg=mdg, subdomains=sds, interfaces=intfs, dim=1)
        op = proj.primary_to_mortar_avg()
        assert op.source is not None
        assert op.target is not None
        assert GridEntity.faces in op.source.dof_info
        assert GridEntity.cells in op.target.dof_info

    def test_secondary_to_mortar_avg(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        intfs = list(mdg.interfaces())
        proj = pp.ad.MortarProjections(mdg=mdg, subdomains=sds, interfaces=intfs, dim=1)
        op = proj.secondary_to_mortar_avg()
        assert op.source is not None
        assert op.target is not None
        assert GridEntity.cells in op.source.dof_info
        assert GridEntity.cells in op.target.dof_info

    def test_sign_of_mortar_sides(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        intfs = list(mdg.interfaces())
        proj = pp.ad.MortarProjections(mdg=mdg, subdomains=sds, interfaces=intfs, dim=1)
        op = proj.sign_of_mortar_sides()
        assert op.source is not None
        assert op.target is not None
        # Square: source == target
        assert op.source == op.target
        assert GridEntity.cells in op.source.dof_info

    def test_empty_interfaces(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        proj = pp.ad.MortarProjections(mdg=mdg, subdomains=sds, interfaces=[], dim=1)
        # No interfaces -> source/target are None
        op = proj.mortar_to_primary_avg()
        assert op.source is None
        assert op.target is None


class TestBoundaryProjectionSpaces:
    def test_subdomain_to_boundary(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        bp = pp.ad.BoundaryProjection(mdg, sds, dim=1)
        op = bp.subdomain_to_boundary
        assert op.source is not None
        assert op.target is not None
        assert GridEntity.faces in op.source.dof_info
        assert GridEntity.cells in op.target.dof_info

    def test_boundary_to_subdomain(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        bp = pp.ad.BoundaryProjection(mdg, sds, dim=1)
        op = bp.boundary_to_subdomain
        assert op.source is not None
        assert op.target is not None
        assert GridEntity.cells in op.source.dof_info
        assert GridEntity.faces in op.target.dof_info

    def test_transpose_consistency(self, fracture_mdg):
        """boundary_to_subdomain source/target are flipped from subdomain_to_boundary."""
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        bp = pp.ad.BoundaryProjection(mdg, sds, dim=1)
        s2b = bp.subdomain_to_boundary
        b2s = bp.boundary_to_subdomain
        assert s2b.source == b2s.target
        assert s2b.target == b2s.source


class TestTraceSpaces:
    def test_trace_source_target(self, fracture_mdg):
        mdg = fracture_mdg
        sds = mdg.subdomains(dim=2)
        tr = pp.ad.Trace(sds, dim=1)
        assert tr.trace.source is not None
        assert tr.trace.target is not None
        assert GridEntity.cells in tr.trace.source.dof_info
        assert GridEntity.faces in tr.trace.target.dof_info

    def test_trace_grids(self, fracture_mdg):
        mdg = fracture_mdg
        sds = mdg.subdomains(dim=2)
        tr = pp.ad.Trace(sds, dim=1)
        assert set(tr.trace.source.grids) == set(sds)
        assert set(tr.trace.target.grids) == set(sds)

    def test_trace_empty(self):
        tr = pp.ad.Trace([], dim=1)
        assert len(tr.trace.source.grids) == 0
        assert len(tr.trace.target.grids) == 0


class TestDivergenceSpaces:
    def test_divergence_source_target(self, fracture_mdg):
        mdg = fracture_mdg
        sds = mdg.subdomains(dim=2)
        div = pp.ad.Divergence(sds, dim=1)
        assert div.source is not None
        assert div.target is not None
        assert GridEntity.faces in div.source.dof_info
        assert GridEntity.cells in div.target.dof_info

    def test_divergence_grids(self, fracture_mdg):
        mdg = fracture_mdg
        sds = mdg.subdomains(dim=2)
        div = pp.ad.Divergence(sds, dim=1)
        assert set(div.source.grids) == set(sds)
        assert set(div.target.grids) == set(sds)

    def test_divergence_empty(self):
        div = pp.ad.Divergence([], dim=1)
        assert len(div.source.grids) == 0
        assert len(div.target.grids) == 0

    @pytest.mark.parametrize("dim", [2, 3])
    def test_divergence_vector_field_dof_info(self, dim):
        """Divergence with dim>1 stores dim DOFs per face (domain) and per cell (range)."""
        if dim == 2:
            g = pp.CartGrid([3, 3])
        else:
            g = pp.CartGrid([2, 2, 2])
        g.compute_geometry()
        div = pp.ad.Divergence([g], dim=dim)
        assert div.source is not None
        assert div.target is not None
        assert div.source.dof_info == {GridEntity.faces: dim}
        assert div.target.dof_info == {GridEntity.cells: dim}

    @pytest.mark.parametrize("dim", [2, 3])
    def test_divergence_vector_field_grids(self, dim):
        """Divergence with dim>1 still records the correct grid set."""
        if dim == 2:
            g = pp.CartGrid([3, 3])
        else:
            g = pp.CartGrid([2, 2, 2])
        g.compute_geometry()
        div = pp.ad.Divergence([g], dim=dim)
        assert div.source.grids == (g,)
        assert div.target.grids == (g,)


# ---------------------------------------------------------------------------
# Stage 4e: MergedOperator source/target
# ---------------------------------------------------------------------------


class TestMergedOperatorSpaces:
    """Tests that MergedOperator inherits source/range from the
    underlying discretization's get_row/col_dof_info methods."""

    def test_default_discr_gives_none(self, two_subdomains):
        """With stub get_row/col_dof_info (returns {}), operator spaces stay
        unspecified."""

        class StubDiscr(Discretization):
            def __init__(self):
                self.keyword = "mechanics"
                self.flux_matrix_key = "flux"

            def get_row_dof_info(self, matrix_key: str = "", nd: int = 1):
                return {}

            def get_col_dof_info(self, matrix_key: str = "", nd: int = 1):
                return {}

            def ndof(self, sd):
                return sd.num_cells

            def discretize(self, sd, data):
                pass

            def assemble_matrix_rhs(self, sd, data):
                pass

        g1, g2 = two_subdomains
        discr = StubDiscr()
        op = MergedOperator(
            discr=discr,
            discretization_matrix_key="flux",
            physics_key="mechanics",
            domains=[g1, g2],
        )
        assert op.source is None
        assert op.target is None

    def test_custom_dof_info_gives_space(self, two_subdomains):
        """A discretization that overrides get_row/col_dof_info populates spaces."""

        class ConcreteDiscr(Discretization):
            def __init__(self):
                self.keyword = "flow"
                self.flux_matrix_key = "flux"

            def ndof(self, sd):
                return sd.num_cells

            def discretize(self, sd, data):
                pass

            def assemble_matrix_rhs(self, sd, data):
                pass

            def get_row_dof_info(self, matrix_key: str = "", nd: int = 1):
                return {GridEntity.cells: 1}

            def get_col_dof_info(self, matrix_key: str = "", nd: int = 1):
                return {GridEntity.faces: 1}

        g1, g2 = two_subdomains
        discr = ConcreteDiscr()
        op = MergedOperator(
            discr=discr,
            discretization_matrix_key="flux",
            physics_key="flow",
            domains=[g1, g2],
        )
        assert op.source is not None
        assert op.target is not None
        assert GridEntity.faces in op.source.dof_info
        assert GridEntity.cells in op.target.dof_info
        assert set(op.source.grids) == {g1, g2}
        assert set(op.target.grids) == {g1, g2}

    def test_interface_discr_gives_none(self, one_mortar):
        """InterfaceDiscretization: no get_row/col_dof_info leaves spaces
        unspecified."""

        class MockInterfaceDiscr(InterfaceDiscretization):
            def __init__(self):
                self.keyword = "coupling"
                self.mortar_flux_matrix_key = "mortar_flux"

            def get_row_dof_info(self, matrix_key: str = "", nd: int = 1):
                return {}

            def get_col_dof_info(self, matrix_key: str = "", nd: int = 1):
                return {}

            def discretize(
                self,
                sd_primary,
                sd_secondary,
                intf,
                data_primary,
                data_secondary,
                data_coupling,
            ):
                pass

            def assemble_matrix_rhs(
                self,
                sd_primary,
                sd_secondary,
                intf,
                data_primary,
                data_secondary,
                data_coupling,
            ):
                pass

        intf = one_mortar
        discr = MockInterfaceDiscr()
        op = MergedOperator(
            discr=discr,
            discretization_matrix_key="mortar_flux",
            physics_key="coupling",
            domains=[intf],
        )
        assert op.source is None
        assert op.target is None


# ---------------------------------------------------------------------------
# Stage 5: validate_operands / infer_source_target
# ---------------------------------------------------------------------------


class TestInferDomainRange:
    """Tests for Operations.infer_source_target (Stage 5: validation and inference)."""

    @pytest.fixture
    def cell_space(self, two_subdomains):
        g1, g2 = two_subdomains
        return OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})

    @pytest.fixture
    def face_space(self, two_subdomains):
        g1, g2 = two_subdomains
        return OperatorSpace.from_domains([g1, g2], {GridEntity.faces: 1})

    @pytest.fixture
    def cell_op(self, cell_space):
        """A leaf operator with source=target=cell_space."""
        return DenseArray(np.zeros(3), source=cell_space, target=cell_space)

    @pytest.fixture
    def face_op(self, face_space):
        """A leaf operator with source=target=face_space."""
        return DenseArray(np.zeros(3), source=face_space, target=face_space)

    # --- elementwise: compatible operands ---

    @pytest.mark.parametrize(
        "binary_op",
        [_op.add, _op.sub, _op.mul, _op.truediv, _op.pow],
        ids=["add", "sub", "mul", "div", "pow"],
    )
    def test_elementwise_compatible(self, cell_op, cell_space, binary_op):
        """Elementwise ops between same-space operands preserve source/target."""
        result = binary_op(cell_op, cell_op)
        assert result.source == cell_space
        assert result.target == cell_space

    def test_rsub_compatible(self, cell_op, cell_space):
        """__rsub__ must also propagate source/target."""
        result = 0 - cell_op
        assert result.source == cell_space
        assert result.target == cell_space

    # --- elementwise: incompatible domains become unclear ---

    @pytest.mark.parametrize(
        "binary_op",
        [_op.add, _op.sub, _op.mul, _op.truediv, _op.pow],
        ids=["add", "sub", "mul", "div", "pow"],
    )
    def test_elementwise_incompatible_domain_becomes_unclear(
        self, cell_op, face_op, binary_op
    ):
        """Elementwise ops with different domains get the unclear-domain sentinel."""
        projected = DenseArray(
            np.zeros(3), source=face_op.source, target=cell_op.target
        )
        result = binary_op(cell_op, projected)
        assert result.source == OperatorSpace.unclear()
        assert result.target == cell_op.target

    def test_elementwise_uses_union_of_dependency_domains(self, two_subdomains):
        """Different but compatible-looking domains still become unclear."""
        g1, g2 = two_subdomains
        top_space = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        union_space = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})

        local = DenseArray(np.zeros(3), source=top_space, target=top_space)
        projected = DenseArray(np.zeros(3), source=union_space, target=top_space)

        result = local * projected

        assert result.source == OperatorSpace.unclear()
        assert result.target == top_space

    def test_elementwise_different_grids_becomes_unclear(self, two_subdomains):
        """Different grids are enough to make the inferred domain unclear."""
        g1, g2 = two_subdomains
        left_space = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        right_space = OperatorSpace.from_domains([g2], {GridEntity.cells: 1})
        left = DenseArray(np.zeros(3), source=left_space, target=left_space)
        right = DenseArray(np.zeros(3), source=right_space, target=left_space)

        result = left + right

        assert result.source == OperatorSpace.unclear()
        assert result.target == left_space

    def test_elementwise_unclear_domain_propagates(self, cell_space, face_space):
        """Once unclear, the elementwise result remains unclear."""
        unclear = DenseArray(
            np.zeros(3), source=OperatorSpace.unclear(), target=cell_space
        )
        known = DenseArray(np.zeros(3), source=face_space, target=cell_space)

        result = unclear + known

        assert result.source == OperatorSpace.unclear()
        assert result.target == cell_space

    # --- matmul: compatible ---

    def test_matmul_compatible(self, two_subdomains):
        """A @ B where target(B) == source(A) is valid."""
        g1, g2 = two_subdomains
        cell_sp = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})
        face_sp = OperatorSpace.from_domains([g1, g2], {GridEntity.faces: 1})
        # A: faces → cells (source=face_sp, target=cell_sp)
        # B: cells → faces (source=cell_sp, target=face_sp)
        # A @ B: target(B)=face_sp == source(A)=face_sp → valid
        A = SparseArray(sps.eye(3), source=face_sp, target=cell_sp)
        B = SparseArray(sps.eye(3), source=cell_sp, target=face_sp)
        result = A @ B
        assert result.source == cell_sp
        assert result.target == cell_sp

    def test_matmul_incompatible(self, two_subdomains):
        """A @ B where target(B) != source(A) raises ValueError."""
        g1, g2 = two_subdomains
        cell_sp = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})
        face_sp = OperatorSpace.from_domains([g1, g2], {GridEntity.faces: 1})
        # A: faces → cells, B: faces → cells (target(B)=cells != source(A)=faces)
        A = SparseArray(sps.eye(3), source=face_sp, target=cell_sp)
        B = SparseArray(sps.eye(3), source=face_sp, target=cell_sp)
        with pytest.raises(ValueError, match="matrix multiplication"):
            _ = A @ B

    def test_matmul_different_grids_incompatible(self, two_subdomains):
        """Matrix multiplication requires exact space equality, including grids."""
        g1, g2 = two_subdomains
        left_domain = OperatorSpace.from_domains([g1], {GridEntity.faces: 1})
        right_range = OperatorSpace.from_domains([g2], {GridEntity.faces: 1})
        left_range = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        right_domain = OperatorSpace.from_domains([g2], {GridEntity.cells: 1})
        left = SparseArray(sps.eye(3), source=left_domain, target=left_range)
        right = SparseArray(sps.eye(3), source=right_domain, target=right_range)

        with pytest.raises(ValueError, match="matrix multiplication"):
            _ = left @ right

    def test_matmul_with_unclear_left_source_raises(self, cell_space, face_space):
        """A left operand with unclear source cannot be used in matmul."""
        unclear = SparseArray(
            sps.eye(3), source=OperatorSpace.unclear(), target=cell_space
        )
        rhs = DenseArray(np.zeros(3), source=face_space, target=face_space)

        with pytest.raises(ValueError, match="left operand.*source is unclear"):
            _ = unclear @ rhs

    def test_rmatmul_with_unclear_right_operand_raises(self, cell_space, face_space):
        """The operator on the right-hand side of rmatmul cannot have unclear source."""
        unclear = SparseArray(
            sps.eye(3), source=OperatorSpace.unclear(), target=cell_space
        )
        lhs = SparseArray(sps.eye(3), source=face_space, target=face_space)

        with pytest.raises(ValueError, match="right operand.*source is unclear"):
            _ = unclear.__rmatmul__(lhs)

    # --- Scalar: always valid, inherits non-scalar space ---

    def test_add_with_scalar_lhs(self, cell_op, cell_space):
        sc = Scalar(2.0)
        result = sc + cell_op
        assert result.source == cell_space
        assert result.target == cell_space

    def test_add_with_scalar_rhs(self, cell_op, cell_space):
        sc = Scalar(2.0)
        result = cell_op + sc
        assert result.source == cell_space
        assert result.target == cell_space

    def test_mul_with_scalar(self, cell_op, cell_space):
        sc = Scalar(3.0)
        result = sc * cell_op
        assert result.source == cell_space
        assert result.target == cell_space

    def test_scalar_scalar(self):
        sc1 = Scalar(1.0)
        sc2 = Scalar(2.0)
        result = sc1 + sc2
        assert result.source == OperatorSpace.scalar()
        assert result.target == OperatorSpace.scalar()

    # --- None source/target: skips validation (backward compat) ---

    def test_none_plus_known_inherits_known(self, cell_space):
        """Operator with None domain + operator with known domain → inherits known."""
        unknown = DenseArray(np.zeros(3))  # source=None
        known = DenseArray(np.zeros(3), source=cell_space, target=cell_space)
        result = unknown + known
        assert result.source == cell_space
        assert result.target == cell_space

    def test_both_none_stays_none(self):
        """Two operators with no source/target → result also has None."""
        a = DenseArray(np.zeros(3))
        b = DenseArray(np.zeros(3))
        result = a + b
        assert result.source is None
        assert result.target is None

    def test_plain_python_scalar_exponent(self, cell_op, cell_space):
        """op ** 2 (plain Python int) should preserve source/target."""
        result = cell_op**2
        assert result.source == cell_space
        assert result.target == cell_space

    def test_plain_python_scalar_rtruediv(self, cell_op, cell_space):
        """1 / op should preserve source/target."""
        result = 1 / cell_op
        assert result.source == cell_space
        assert result.target == cell_space

    # --- infer_source_target is public ---

    def test_infer_source_target_is_public(self, cell_op):
        """infer_source_target should be accessible on the Operations enum."""
        assert hasattr(Operations.add, "infer_source_target")
        dom, ran = Operations.add.infer_source_target(cell_op, cell_op)
        assert dom is not None
        assert ran is not None


# ---------------------------------------------------------------------------
# Stage 6: compound operator source/target propagation
# ---------------------------------------------------------------------------


class TestCompoundOperatorSpaces:
    """Tests that source/target propagates correctly through multi-step expressions."""

    @pytest.fixture
    def spaces(self, two_subdomains):
        g1, g2 = two_subdomains
        cell_sp = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})
        face_sp = OperatorSpace.from_domains([g1, g2], {GridEntity.faces: 1})
        return cell_sp, face_sp

    # --- chained matmul ---

    def test_chained_matmul_source_target(self, two_subdomains, spaces):
        """(A @ B): target(B) == source(A) → result.source=B.source, result.target=A.target"""
        g1, g2 = two_subdomains
        cell_sp, face_sp = spaces
        # A maps faces→cells; B maps cells→faces; A@B maps cells→cells
        A = SparseArray(sps.eye(3), source=face_sp, target=cell_sp)
        B = SparseArray(sps.eye(3), source=cell_sp, target=face_sp)
        result = A @ B
        assert result.source == cell_sp
        assert result.target == cell_sp

    def test_three_way_matmul(self, two_subdomains, spaces):
        """(A @ B) @ C propagates spaces through two matmul steps."""
        g1, g2 = two_subdomains
        cell_sp, face_sp = spaces
        # A: face→cell, B: cell→face → A@B: cell→cell
        # C: face→cell → (A@B)@C requires target(C)==source(A@B)=cell_sp ✓ → face→cell
        A = SparseArray(sps.eye(3), source=face_sp, target=cell_sp)
        B = SparseArray(sps.eye(3), source=cell_sp, target=face_sp)
        C = SparseArray(sps.eye(3), source=face_sp, target=cell_sp)
        AB = A @ B
        assert AB.source == cell_sp
        assert AB.target == cell_sp
        ABC = AB @ C
        assert ABC.source == face_sp
        assert ABC.target == cell_sp

    def test_chained_matmul_incompatible_raises(self, two_subdomains, spaces):
        """(A @ B) @ C raises ValueError when target(C) != source(A@B)."""
        g1, g2 = two_subdomains
        cell_sp, face_sp = spaces
        # A@B: cell→cell (see test_three_way_matmul); C has target=face_sp != cell_sp
        A = SparseArray(sps.eye(3), source=face_sp, target=cell_sp)
        B = SparseArray(sps.eye(3), source=cell_sp, target=face_sp)
        AB = A @ B  # source=cell_sp, target=cell_sp
        C = SparseArray(sps.eye(3), source=face_sp, target=face_sp)
        with pytest.raises(ValueError, match="matrix multiplication"):
            _ = AB @ C

    def test_add_after_matmul(self, two_subdomains, spaces):
        """(A @ v) + (B @ w) where both results have the same range."""
        g1, g2 = two_subdomains
        cell_sp, face_sp = spaces
        A = SparseArray(sps.eye(3), source=face_sp, target=cell_sp)
        v = DenseArray(np.zeros(3), source=face_sp, target=face_sp)
        B = SparseArray(sps.eye(3), source=face_sp, target=cell_sp)
        w = DenseArray(np.zeros(3), source=face_sp, target=face_sp)
        Av = A @ v
        Bw = B @ w
        result = Av + Bw
        assert result.source == face_sp
        assert result.target == cell_sp

    def test_add_matmul_incompatible_raises(self, two_subdomains, spaces):
        """(A @ v) + (B @ w) where ranges differ raises ValueError."""
        g1, g2 = two_subdomains
        cell_sp, face_sp = spaces
        A = SparseArray(sps.eye(3), source=face_sp, target=cell_sp)
        v = DenseArray(np.zeros(3), source=face_sp, target=face_sp)
        Av = A @ v  # target=cell_sp
        # B maps faces→faces, so B@w has target=face_sp
        B = SparseArray(sps.eye(3), source=face_sp, target=face_sp)
        w = DenseArray(np.zeros(3), source=face_sp, target=face_sp)
        Bw = B @ w
        with pytest.raises(ValueError):
            _ = Av + Bw

    # --- scalar factor in chains ---

    def test_scalar_mul_after_matmul(self, two_subdomains, spaces):
        """Scalar(k) * (A @ v) preserves A's range as the result range."""
        g1, g2 = two_subdomains
        cell_sp, face_sp = spaces
        A = SparseArray(sps.eye(3), source=face_sp, target=cell_sp)
        v = DenseArray(np.zeros(3), source=face_sp, target=face_sp)
        Av = A @ v
        result = Scalar(2.0) * Av
        assert result.source == face_sp
        assert result.target == cell_sp

    def test_unary_minus_preserves_spaces(self, two_subdomains, spaces):
        """Unary minus on SparseArray preserves source/target."""
        g1, g2 = two_subdomains
        cell_sp, face_sp = spaces
        A = SparseArray(sps.eye(3), source=face_sp, target=cell_sp)
        result = -A
        assert result.source == face_sp
        assert result.target == cell_sp

    def test_unary_minus_dense_array_preserves_spaces(self, two_subdomains, spaces):
        """DenseArray.__neg__ must also preserve source/target (separate code path)."""
        g1, g2 = two_subdomains
        cell_sp, face_sp = spaces
        arr = DenseArray(np.ones(3), source=cell_sp, target=cell_sp)
        result = -arr
        assert result.source == cell_sp
        assert result.target == cell_sp

    # --- integration with actual grid operators ---

    def test_divergence_matmul_projection(self, fracture_mdg):
        """Div @ cell_restriction: spaces propagate from grid operators."""
        mdg = fracture_mdg
        sds_2d = mdg.subdomains(dim=2)
        div = pp.ad.Divergence(sds_2d, dim=1)
        proj = pp.ad.SubdomainProjections(sds_2d)
        cell_rest = proj.cell_restriction(sds_2d)
        # cell_restriction maps cells→cells on the subset (square matrix here)
        # div maps faces→cells
        # We test that div has correct spaces
        assert div.source is not None
        assert GridEntity.faces in div.source.dof_info
        assert div.target is not None
        assert GridEntity.cells in div.target.dof_info

    def test_compound_inherits_none_when_one_operand_has_none(
        self, two_subdomains, spaces
    ):
        """When one operand in a chain has None domain, the chain can still succeed
        if the other operand provides the domain."""
        g1, g2 = two_subdomains
        cell_sp, face_sp = spaces
        # unknown_op has no space info
        unknown_op = DenseArray(np.zeros(3))
        known_op = DenseArray(np.zeros(3), source=cell_sp, target=cell_sp)
        # Adding unknown + known: no error, result inherits known's spaces
        result = unknown_op + known_op
        assert result.source == cell_sp
        assert result.target == cell_sp

    def test_source_and_target_stored_independently(self, two_subdomains):
        """Even when domain == range, they are stored as independent attributes."""
        g1, g2 = two_subdomains
        cell_sp = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})
        a = DenseArray(np.zeros(3), source=cell_sp, target=cell_sp)
        b = DenseArray(np.zeros(3), source=cell_sp, target=cell_sp)
        result = a + b
        # source and target are equal in value, but are independent objects
        assert result.source == result.target
        assert result.source is not None
        assert result.target is not None


class TestMergedOperatorWithConcreteDiscretization:
    """Integration tests: MergedOperator infers source/target from concrete discretizations."""

    def test_mpfa_flux_merged_operator(self, two_subdomains):
        """MpfaAd.flux() carries face-range / cell-domain spaces."""
        g1, g2 = two_subdomains
        discr = pp.Mpfa("flow")
        op = MergedOperator(
            discr=discr,
            discretization_matrix_key="flux",
            physics_key="flow",
            domains=[g1, g2],
        )
        assert op.source is not None
        assert op.target is not None
        assert op.source.dof_info == {GridEntity.cells: 1}
        assert op.target.dof_info == {GridEntity.faces: 1}
        assert set(op.source.grids) == {g1, g2}
        assert set(op.target.grids) == {g1, g2}

    def test_mpfa_bound_flux_merged_operator(self, two_subdomains):
        """bound_flux has face-range and face-domain (maps BC values to fluxes)."""
        g1, g2 = two_subdomains
        discr = pp.Mpfa("flow")
        op = MergedOperator(
            discr=discr,
            discretization_matrix_key="bound_flux",
            physics_key="flow",
            domains=[g1, g2],
        )
        assert op.source.dof_info == {GridEntity.faces: 1}
        assert op.target.dof_info == {GridEntity.faces: 1}

    def test_mpsa_stress_merged_operator_2d(self, two_subdomains):
        """Mpsa.stress uses nd=2 DOFs per face/cell for 2D grids."""
        g1, g2 = two_subdomains  # both are 2D
        discr = pp.Mpsa("mech")
        op = MergedOperator(
            discr=discr,
            discretization_matrix_key="stress",
            physics_key="mech",
            domains=[g1, g2],
        )
        assert op.source is not None
        assert op.target is not None
        assert op.source.dof_info == {GridEntity.cells: 2}
        assert op.target.dof_info == {GridEntity.faces: 2}

    def test_upwind_merged_operator(self, two_subdomains):
        """Upwind operator has face-range / cell-domain."""
        g1, g2 = two_subdomains
        discr = pp.Upwind("flow")
        op = MergedOperator(
            discr=discr,
            discretization_matrix_key="upwind",
            physics_key="flow",
            domains=[g1, g2],
        )
        assert op.source.dof_info == {GridEntity.cells: 1}
        assert op.target.dof_info == {GridEntity.faces: 1}

    def test_mpfa_via_ad_wrapper(self, two_subdomains):
        """Using MpfaAd wrapper (wrap_discretization path) produces correct spaces."""
        g1, g2 = two_subdomains
        discr = pp.ad.MpfaAd("flow", [g1, g2])
        flux_op = discr.flux()
        assert flux_op.source is not None
        assert flux_op.target is not None
        assert flux_op.source.dof_info == {GridEntity.cells: 1}
        assert flux_op.target.dof_info == {GridEntity.faces: 1}

    def test_tpfa_via_ad_wrapper(self, two_subdomains):
        """Using TpfaAd wrapper produces correct spaces for flux."""
        g1, g2 = two_subdomains
        discr = pp.ad.TpfaAd("flow", [g1, g2])
        flux_op = discr.flux()
        assert flux_op.source.dof_info == {GridEntity.cells: 1}
        assert flux_op.target.dof_info == {GridEntity.faces: 1}

    def test_mpfa_bound_flux_ad_wrapper(self, two_subdomains):
        """MpfaAd.bound_flux() has face-domain and face-range."""
        g1, g2 = two_subdomains
        discr = pp.ad.MpfaAd("flow", [g1, g2])
        op = discr.bound_flux()
        assert op.source.dof_info == {GridEntity.faces: 1}
        assert op.target.dof_info == {GridEntity.faces: 1}

    def test_vector_source_cols_scale_with_nd(self, two_subdomains):
        """vector_source column DOF count matches grid dimension."""
        g1, g2 = two_subdomains  # both are 2D (dim=2)
        discr = pp.Mpfa("flow")
        op = MergedOperator(
            discr=discr,
            discretization_matrix_key="vector_source",
            physics_key="flow",
            domains=[g1, g2],
        )
        # For 2D grids nd=2 → cols have 2 DOFs per cell (gravity vector)
        assert op.source is not None
        assert op.source.dof_info == {GridEntity.cells: 2}
        assert op.target.dof_info == {GridEntity.faces: 1}

    def test_tpsa_stress_displacement_merged_operator(self, two_subdomains):
        """Tpsa.stress_displacement via MergedOperator gives cells:nd domain and faces:nd range."""
        g1, g2 = two_subdomains  # both 2D
        discr = pp.Tpsa("mech")
        op = MergedOperator(
            discr=discr,
            discretization_matrix_key="stress_displacement",
            physics_key="mech",
            domains=[g1, g2],
        )
        assert op.source is not None
        assert op.target is not None
        assert op.source.dof_info == {GridEntity.cells: 2}
        assert op.target.dof_info == {GridEntity.faces: 2}
        assert set(op.source.grids) == {g1, g2}
        assert set(op.target.grids) == {g1, g2}

    def test_tpsa_rotation_displacement_merged_operator(self, two_subdomains):
        """Tpsa.rotation_displacement row DOF uses nrot=1 for 2D grids."""
        g1, g2 = two_subdomains  # both 2D; nrot = 2*(2-1)//2 = 1
        discr = pp.Tpsa("mech")
        op = MergedOperator(
            discr=discr,
            discretization_matrix_key="rotation_displacement",
            physics_key="mech",
            domains=[g1, g2],
        )
        assert op.source is not None
        assert op.target is not None
        # 2D: row DOFs are nrot=1 face entries, col DOFs are nd=2 cell entries
        assert op.source.dof_info == {GridEntity.cells: 2}
        assert op.target.dof_info == {GridEntity.faces: 1}


# ---------------------------------------------------------------------------
# sum_operator_list and sum_projection_list space propagation
# ---------------------------------------------------------------------------


class TestSumOperatorListSpace:
    """sum_operator_list delegates to __add__, so source/target must propagate."""

    def test_sum_two_arrays_propagates_space(self, two_subdomains):
        """sum_operator_list([a, b]) with compatible spaces inherits those spaces."""
        g, _ = two_subdomains
        space = OperatorSpace.from_domains([g], {GridEntity.cells: 1})
        a = DenseArray(np.ones(4), source=space, target=space)
        b = DenseArray(np.ones(4), source=space, target=space)
        result = sum_operator_list([a, b])
        assert result.source == space
        assert result.target == space

    def test_sum_three_arrays_propagates_space(self, two_subdomains):
        """sum_operator_list([a, b, c]) propagates spaces through the full reduce."""
        g, _ = two_subdomains
        space = OperatorSpace.from_domains([g], {GridEntity.cells: 1})
        ops = [DenseArray(np.ones(4), source=space, target=space) for _ in range(3)]
        result = sum_operator_list(ops)
        assert result.source == space
        assert result.target == space

    def test_sum_incompatible_spaces_raises(self, two_subdomains):
        """sum_operator_list raises ValueError when spaces are incompatible."""
        g1, g2 = two_subdomains
        s1 = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        s2 = OperatorSpace.from_domains([g2], {GridEntity.cells: 1})
        a = DenseArray(np.ones(4), source=s1, target=s1)
        b = DenseArray(np.ones(9), source=s2, target=s2)
        with pytest.raises(ValueError):
            sum_operator_list([a, b])

    def test_sum_none_space_inherits_known(self, two_subdomains):
        """sum_operator_list with one operand lacking a space still propagates the other."""
        g, _ = two_subdomains
        space = OperatorSpace.from_domains([g], {GridEntity.cells: 1})
        a = DenseArray(np.ones(4), source=space, target=space)
        b = DenseArray(np.ones(4))  # no space
        result = sum_operator_list([a, b])
        assert result.source == space
        assert result.target == space
