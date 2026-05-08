"""Tests for Stage 2 of GH discussion #1601: DomainType, OperatorSpace and
operator_domain/operator_range propagation."""

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
# Operator.operator_domain and operator_range properties
# ---------------------------------------------------------------------------


class TestOperatorProperties:
    def test_operator_requires_explicit_domain_and_range(self):
        with pytest.raises(TypeError):
            Operator(name="test")

    def test_operator_domain_accepts_explicit_none(self):
        op = Operator(name="test", domain=None, range=None)
        assert op.operator_domain is None

    def test_operator_range_accepts_explicit_none(self):
        op = Operator(name="test", domain=None, range=None)
        assert op.operator_range is None

    def test_set_domain_in_init(self, two_subdomains):
        g1, _ = two_subdomains
        space = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        op = Operator(name="test", domain=space, range=None)
        assert op.operator_domain == space

    def test_set_range_in_init(self, two_subdomains):
        g1, _ = two_subdomains
        space = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        op = Operator(name="test", domain=None, range=space)
        assert op.operator_range == space


# ---------------------------------------------------------------------------
# Leaf operator spaces
# ---------------------------------------------------------------------------


class TestScalarSpace:
    def test_scalar_has_scalar_space(self):
        s = Scalar(3.14)
        assert s.operator_domain == OperatorSpace.scalar()
        assert s.operator_range == OperatorSpace.scalar()

    def test_scalar_with_subdomain_has_subdomain_space(self, two_subdomains):
        g, _ = two_subdomains
        s = Scalar(1.0, domains=[g])
        assert s.operator_domain is not None
        assert s.operator_domain.domain_type == DomainType.subdomains
        assert s.operator_domain.grids == (g,)
        assert s.operator_domain == s.operator_range

    def test_scalar_with_mortar_grid_has_interface_space(self, one_mortar):
        mg = one_mortar
        s = Scalar(2.0, domains=[mg])
        assert s.operator_domain is not None
        assert s.operator_domain.domain_type == DomainType.interfaces
        assert s.operator_domain.grids == (mg,)

    def test_scalar_domain_is_cellwise(self, two_subdomains):
        """Domain-bearing Scalar uses the natural cell-based space on its grids."""
        g, _ = two_subdomains
        s = Scalar(1.0, domains=[g])
        assert s.operator_domain is not None
        assert s.operator_domain.dof_info == {GridEntity.cells: 1}

    def test_scalar_neg_propagates_domain(self, two_subdomains):
        g, _ = two_subdomains
        s = Scalar(3.0, domains=[g])
        neg = -s
        assert neg.operator_domain == s.operator_domain
        assert neg.operator_range == s.operator_range

    def test_scalar_neg_no_domain(self):
        """Negating a plain Scalar preserves the scalar wildcard space."""
        s = Scalar(2.0)
        neg = -s
        assert neg.operator_domain == OperatorSpace.scalar()

    def test_scalar_empty_domains_gives_scalar_space(self):
        """Empty domains list is treated as no-domain (backward compat)."""
        s = Scalar(1.0, domains=[])
        assert s.operator_domain == OperatorSpace.scalar()

    def test_domain_bearing_scalar_combined_with_operator(self, two_subdomains):
        """A domain-bearing scalar uses the ordinary cell-based space on its grids."""
        g, _ = two_subdomains
        s = Scalar(2.0, domains=[g])
        v = Variable("p", {GridEntity.cells: 1}, g)
        result = s * v
        assert result.operator_domain == v.operator_domain
        assert result.operator_range == v.operator_range

    def test_domainless_scalar_combined_with_operator(self, two_subdomains):
        """Plain Scalar still inherits from the other operand."""
        g, _ = two_subdomains
        s = Scalar(2.0)
        v = Variable("p", {GridEntity.cells: 1}, g)
        result = s * v
        assert result.operator_domain == v.operator_domain


class TestVariableSpace:
    def test_variable_has_subdomain_space(self, two_subdomains):
        g, _ = two_subdomains
        var = Variable("p", {GridEntity.cells: 1}, g)
        assert var.operator_domain is not None
        assert var.operator_domain.domain_type == DomainType.subdomains
        assert var.operator_domain == var.operator_range

    def test_variable_space_contains_correct_grid(self, two_subdomains):
        g, _ = two_subdomains
        var = Variable("p", {GridEntity.cells: 1}, g)
        assert var.operator_domain is not None
        assert var.operator_domain.grids == (g,)

    def test_variable_space_dof_info(self, two_subdomains):
        g, _ = two_subdomains
        var = Variable("u", {GridEntity.cells: 2, GridEntity.faces: 1}, g)
        assert var.operator_domain is not None
        assert var.operator_domain.dof_info == {
            GridEntity.cells: 2,
            GridEntity.faces: 1,
        }

    def test_variable_on_mortar_grid_has_interface_space(self, one_mortar):
        """Variable on a MortarGrid gets DomainType.interfaces."""
        var = Variable("lam", {GridEntity.cells: 1}, one_mortar)
        assert var.operator_domain is not None
        assert var.operator_domain.domain_type == DomainType.interfaces
        assert var.operator_domain.grids == (one_mortar,)

    def test_variable_on_mortar_grid_dof_info(self, one_mortar):
        """dof_info is preserved when Variable is on a mortar grid."""
        var = Variable("lam", {GridEntity.cells: 2}, one_mortar)
        assert var.operator_domain is not None
        assert var.operator_domain.dof_info == {GridEntity.cells: 2}


class TestMixedDimensionalVariableSpace:
    """MixedDimensionalVariable carries the union space of its sub-variables."""

    @pytest.fixture
    def md_var(self, two_subdomains):
        g1, g2 = two_subdomains
        v1 = Variable("p", {GridEntity.cells: 1}, g1)
        v2 = Variable("p", {GridEntity.cells: 1}, g2)
        return MixedDimensionalVariable([v1, v2])

    def test_md_variable_operator_domain_is_union(self, md_var, two_subdomains):
        g1, g2 = two_subdomains
        expected = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})
        assert md_var.operator_domain == expected

    def test_md_variable_operator_range_is_union(self, md_var, two_subdomains):
        g1, g2 = two_subdomains
        expected = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})
        assert md_var.operator_range == expected

    def test_md_variable_with_mixed_dof_info_has_unspecified_space(self, fracture_mdg):
        subdomains = fracture_mdg.subdomains()
        v1 = Variable("p", {GridEntity.cells: 1}, subdomains[0])
        v2 = Variable("p", {GridEntity.cells: 2}, subdomains[1])
        md_var = MixedDimensionalVariable([v1, v2])

        assert md_var.operator_domain is None
        assert md_var.operator_range is None


class TestSurrogateOperatorSpace:
    """SurrogateOperator carries operator_domain/range derived from its dof_info."""

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

    def test_surrogate_operator_domain_with_dof_info(self, surrogate_setup):
        """SurrogateFactory with explicit dof_info: the produced operator has a space."""
        mdg, var = surrogate_setup
        factory = pp.ad.SurrogateFactory(
            name="f",
            mdg=mdg,
            dependencies=[lambda grids: var],
            dof_info={GridEntity.cells: 1},
        )
        op = factory(list(mdg.subdomains()))
        assert op.operator_domain is not None
        assert op.operator_domain.domain_type == DomainType.subdomains
        assert op.operator_domain.dof_info == {GridEntity.cells: 1}

    def test_surrogate_operator_range_equals_domain(self, surrogate_setup):
        """For a SurrogateOperator the range_ equals domain (square operator)."""
        mdg, var = surrogate_setup
        factory = pp.ad.SurrogateFactory(
            name="f",
            mdg=mdg,
            dependencies=[lambda grids: var],
            dof_info={GridEntity.cells: 1},
        )
        op = factory(list(mdg.subdomains()))
        assert op.operator_range == op.operator_domain

    def test_surrogate_operator_default_dof_info_gives_space(self, surrogate_setup):
        """dof_info=None (default) falls back to cells:1 and still sets a space."""
        mdg, var = surrogate_setup
        factory = pp.ad.SurrogateFactory(
            name="f",
            mdg=mdg,
            dependencies=[lambda grids: var],
        )
        op = factory(list(mdg.subdomains()))
        assert op.operator_domain is not None
        assert op.operator_domain.dof_info == {GridEntity.cells: 1}

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
        assert op.operator_domain is not None
        assert op.operator_domain.dof_info == {GridEntity.cells: 1}
        assert op.operator_range is not None
        assert op.operator_range.dof_info == {GridEntity.cells: 1}


class TestDenseArraySpace:
    def test_dense_array_no_space_by_default(self):
        arr = DenseArray(np.ones(5))
        assert arr.operator_domain is None
        assert arr.operator_range is None

    def test_dense_array_with_explicit_space(self, two_subdomains):
        g, _ = two_subdomains
        space = OperatorSpace.from_domains([g], {GridEntity.cells: 1})
        arr = DenseArray(np.ones(5), domain=space, range=space)
        assert arr.operator_domain == space
        assert arr.operator_range == space


class TestSparseArraySpace:
    def test_sparse_array_no_space_by_default(self):
        mat = sps.eye(4, format="csr")
        op = SparseArray(mat)
        assert op.operator_domain is None
        assert op.operator_range is None

    def test_sparse_array_with_explicit_space(self, two_subdomains):
        g, _ = two_subdomains
        space = OperatorSpace.from_domains([g], {GridEntity.cells: 1})
        mat = sps.eye(4, format="csr")
        op = SparseArray(mat, domain=space, range=space)
        assert op.operator_domain == space
        assert op.operator_range == space


# ---------------------------------------------------------------------------
# Domain/range propagation through arithmetic operations
# ---------------------------------------------------------------------------


class TestDomainRangePropagation:
    """Test that binary operations propagate operator_domain and operator_range."""

    def _cell_space(self, g):
        return OperatorSpace.from_domains([g], {GridEntity.cells: 1})

    @pytest.mark.parametrize(
        "binary_op",
        [_op.add, _op.sub, _op.mul, _op.truediv],
        ids=["add", "sub", "mul", "div"],
    )
    def test_elementwise_same_space_propagates(self, two_subdomains, binary_op):
        """Elementwise ops between equal-space operands preserve domain/range."""
        g, _ = two_subdomains
        space = self._cell_space(g)
        a = DenseArray(np.ones(4), domain=space, range=space)
        b = DenseArray(np.ones(4), domain=space, range=space)
        result = binary_op(a, b)
        assert result.operator_domain == space
        assert result.operator_range == space

    def test_scalar_inherits_other_space(self, two_subdomains):
        g, _ = two_subdomains
        space = self._cell_space(g)
        arr = DenseArray(np.ones(4), domain=space, range=space)
        s = Scalar(2.0)
        result = s * arr
        assert result.operator_domain == space
        assert result.operator_range == space

    def test_other_inherits_scalar_space(self, two_subdomains):
        g, _ = two_subdomains
        space = self._cell_space(g)
        arr = DenseArray(np.ones(4), domain=space, range=space)
        s = Scalar(2.0)
        result = arr * s
        assert result.operator_domain == space

    def test_scalar_scalar_gives_scalar_space(self):
        s1 = Scalar(1.0)
        s2 = Scalar(2.0)
        result = s1 + s2
        assert result.operator_domain == OperatorSpace.scalar()
        assert result.operator_range == OperatorSpace.scalar()

    def test_none_space_propagates_other(self, two_subdomains):
        """An operand with None space should not block inference from the other."""
        g, _ = two_subdomains
        space = self._cell_space(g)
        a = DenseArray(np.ones(4), domain=space, range=space)
        b = DenseArray(np.ones(4))  # no space
        result = a + b
        assert result.operator_domain == space

    def test_both_none_gives_none(self):
        a = DenseArray(np.ones(4))
        b = DenseArray(np.ones(4))
        result = a + b
        assert result.operator_domain is None
        assert result.operator_range is None

    def test_incompatible_domains_raises(self, two_subdomains):
        g1, g2 = two_subdomains
        s1 = self._cell_space(g1)
        s2 = self._cell_space(g2)
        a = DenseArray(np.ones(4), domain=s1, range=s1)
        b = DenseArray(np.ones(9), domain=s2, range=s2)
        with pytest.raises(ValueError, match="[Ii]ncompat"):
            _ = a + b

    def test_matmul_propagates_outer_spaces(self, two_subdomains):
        """For A @ B: result.domain = B.domain, result.range = A.range."""
        g1, g2 = two_subdomains
        s1 = self._cell_space(g1)
        s2 = self._cell_space(g2)
        # A maps s1 -> s2, B maps s2 -> s1
        A = SparseArray(sps.eye(4, format="csr"), domain=s1, range=s2)
        B = SparseArray(sps.eye(4, format="csr"), domain=s2, range=s1)
        result = A @ B
        assert result.operator_domain == s2
        assert result.operator_range == s2

    def test_matmul_incompatible_range_domain_raises(self, two_subdomains):
        """A @ B raises if A.domain != B.range."""
        g1, g2 = two_subdomains
        s1 = self._cell_space(g1)
        s2 = self._cell_space(g2)
        # A.domain=s1, B.range=s2 -> incompatible
        A = SparseArray(sps.eye(4, format="csr"), domain=s1, range=s2)
        B = SparseArray(sps.eye(4, format="csr"), domain=s1, range=s2)
        with pytest.raises(ValueError, match="[Ii]ncompat"):
            _ = A @ B


# ---------------------------------------------------------------------------
# Discretization stubs
# ---------------------------------------------------------------------------


class TestDiscretizationStubs:
    def test_get_row_dof_info_default(self):
        class ConcreteDiscr(Discretization):
            def get_row_dof_info(self, matrix_key: str = "", nd: int = 1):
                return {}

            def get_col_dof_info(self, matrix_key: str = "", nd: int = 1):
                return {}

            def ndof(self, g):
                return 0

            def discretize(self, g, data):
                pass

            def assemble_matrix_rhs(self, g, data):
                pass

        discr = ConcreteDiscr("test")
        # No matrix_key: default "" always returns {}
        assert discr.get_row_dof_info() == {}
        assert discr.get_col_dof_info() == {}
        # Explicit key: still {} from base class
        assert discr.get_row_dof_info("flux") == {}
        assert discr.get_col_dof_info("flux") == {}

    def test_get_row_dof_info_overridable(self):
        class CustomDiscr(Discretization):
            def ndof(self, g):
                return 0

            def discretize(self, g, data):
                pass

            def assemble_matrix_rhs(self, g, data):
                pass

            def get_row_dof_info(self, matrix_key: str = "", nd: int = 1):
                return {GridEntity.cells: 1}

            def get_col_dof_info(self, matrix_key: str = "", nd: int = 1):
                return {GridEntity.faces: nd}

        discr = CustomDiscr("test")
        assert discr.get_row_dof_info() == {GridEntity.cells: 1}
        assert discr.get_col_dof_info("flux", nd=3) == {GridEntity.faces: 3}
        assert discr.get_col_dof_info() == {GridEntity.faces: 1}


# ---------------------------------------------------------------------------
# Stage 4c: TimeDependentDenseArray optional dof_info
# ---------------------------------------------------------------------------


class TestTimeDependentDenseArraySpaces:
    def test_no_dof_info_gives_none(self, two_subdomains):
        g1, g2 = two_subdomains
        arr = pp.ad.TimeDependentDenseArray("x", [g1, g2])
        # When domains are provided but no dof_info, cells:1 is assumed.
        assert arr.operator_domain is not None
        assert arr.operator_domain.dof_info == {GridEntity.cells: 1}
        assert arr.operator_range is not None
        assert arr.operator_range.dof_info == {GridEntity.cells: 1}

    def test_dof_info_cells(self, two_subdomains):
        g1, g2 = two_subdomains
        arr = pp.ad.TimeDependentDenseArray(
            "x", [g1, g2], dof_info={GridEntity.cells: 1}
        )
        assert arr.operator_domain is not None
        assert arr.operator_range is not None
        assert arr.operator_domain == arr.operator_range
        assert GridEntity.cells in arr.operator_domain.dof_info
        assert set(arr.operator_domain.grids) == {g1, g2}

    def test_dof_info_faces(self, two_subdomains):
        g1, g2 = two_subdomains
        arr = pp.ad.TimeDependentDenseArray(
            "x", [g1, g2], dof_info={GridEntity.faces: 2}
        )
        assert arr.operator_domain is not None
        assert arr.operator_domain.dof_info == {GridEntity.faces: 2}

    def test_empty_domains_ignores_dof_info(self):
        arr = pp.ad.TimeDependentDenseArray("x", [], dof_info={GridEntity.cells: 1})
        assert arr.operator_domain is None
        assert arr.operator_range is None


# ---------------------------------------------------------------------------
# Stage 3: Grid operator domain/range
# ---------------------------------------------------------------------------


class TestSubdomainProjectionSpaces:
    def test_cell_restriction_all(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        proj = pp.ad.SubdomainProjections(subdomains=sds, dim=1)
        op = proj.cell_restriction(sds)
        assert op.operator_domain is not None
        assert op.operator_range is not None
        assert op.operator_domain.domain_type == DomainType.subdomains
        assert op.operator_range.domain_type == DomainType.subdomains
        assert GridEntity.cells in op.operator_domain.dof_info
        assert GridEntity.cells in op.operator_range.dof_info

    def test_cell_restriction_subset(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        sub = mdg.subdomains(dim=2)  # only top-dim subdomain
        proj = pp.ad.SubdomainProjections(subdomains=sds, dim=1)
        op = proj.cell_restriction(sub)
        # domain covers all subdomains, range covers the subset
        assert op.operator_domain is not None
        assert op.operator_range is not None
        assert set(op.operator_domain.grids) == set(sds)
        assert set(op.operator_range.grids) == set(sub)

    def test_cell_restriction_empty(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        proj = pp.ad.SubdomainProjections(subdomains=sds, dim=1)
        op = proj.cell_restriction([])
        # Empty subset -> range_ is None
        assert op.operator_domain is not None
        assert op.operator_range is None

    def test_cell_prolongation_spaces(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        sub = mdg.subdomains(dim=2)
        proj = pp.ad.SubdomainProjections(subdomains=sds, dim=1)
        op = proj.cell_prolongation(sub)
        # domain is the subset, range is all subdomains
        assert op.operator_domain is not None
        assert op.operator_range is not None
        assert set(op.operator_domain.grids) == set(sub)
        assert set(op.operator_range.grids) == set(sds)

    def test_face_restriction_spaces(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        sub = mdg.subdomains(dim=2)
        proj = pp.ad.SubdomainProjections(subdomains=sds, dim=1)
        op = proj.face_restriction(sub)
        assert op.operator_domain is not None
        assert op.operator_range is not None
        assert GridEntity.faces in op.operator_domain.dof_info
        assert GridEntity.faces in op.operator_range.dof_info

    def test_face_prolongation_spaces(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        sub = mdg.subdomains(dim=2)
        proj = pp.ad.SubdomainProjections(subdomains=sds, dim=1)
        op = proj.face_prolongation(sub)
        assert op.operator_domain is not None
        assert op.operator_range is not None
        assert set(op.operator_domain.grids) == set(sub)
        assert set(op.operator_range.grids) == set(sds)

    def test_vector_dim(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        dim = 2
        proj = pp.ad.SubdomainProjections(subdomains=sds, dim=dim)
        op = proj.cell_restriction(sds)
        assert op.operator_domain.dof_info[GridEntity.cells] == dim


class TestMortarProjectionSpaces:
    def test_mortar_to_primary_avg(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        intfs = list(mdg.interfaces())
        proj = pp.ad.MortarProjections(mdg=mdg, subdomains=sds, interfaces=intfs, dim=1)
        op = proj.mortar_to_primary_avg()
        assert op.operator_domain is not None
        assert op.operator_range is not None
        assert GridEntity.cells in op.operator_domain.dof_info  # interface cells
        assert (
            GridEntity.faces in op.operator_range.dof_info
        )  # subdomain faces (codim 1)

    def test_mortar_to_secondary_avg(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        intfs = list(mdg.interfaces())
        proj = pp.ad.MortarProjections(mdg=mdg, subdomains=sds, interfaces=intfs, dim=1)
        op = proj.mortar_to_secondary_avg()
        assert op.operator_domain is not None
        assert op.operator_range is not None
        assert GridEntity.cells in op.operator_domain.dof_info  # interface cells
        assert (
            GridEntity.cells in op.operator_range.dof_info
        )  # subdomain cells (secondary)

    def test_primary_to_mortar_avg(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        intfs = list(mdg.interfaces())
        proj = pp.ad.MortarProjections(mdg=mdg, subdomains=sds, interfaces=intfs, dim=1)
        op = proj.primary_to_mortar_avg()
        assert op.operator_domain is not None
        assert op.operator_range is not None
        assert GridEntity.faces in op.operator_domain.dof_info
        assert GridEntity.cells in op.operator_range.dof_info

    def test_secondary_to_mortar_avg(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        intfs = list(mdg.interfaces())
        proj = pp.ad.MortarProjections(mdg=mdg, subdomains=sds, interfaces=intfs, dim=1)
        op = proj.secondary_to_mortar_avg()
        assert op.operator_domain is not None
        assert op.operator_range is not None
        assert GridEntity.cells in op.operator_domain.dof_info
        assert GridEntity.cells in op.operator_range.dof_info

    def test_sign_of_mortar_sides(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        intfs = list(mdg.interfaces())
        proj = pp.ad.MortarProjections(mdg=mdg, subdomains=sds, interfaces=intfs, dim=1)
        op = proj.sign_of_mortar_sides()
        assert op.operator_domain is not None
        assert op.operator_range is not None
        # Square: domain == range_
        assert op.operator_domain == op.operator_range
        assert GridEntity.cells in op.operator_domain.dof_info

    def test_empty_interfaces(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        proj = pp.ad.MortarProjections(mdg=mdg, subdomains=sds, interfaces=[], dim=1)
        # No interfaces -> domain/range are None
        op = proj.mortar_to_primary_avg()
        assert op.operator_domain is None
        assert op.operator_range is None


class TestBoundaryProjectionSpaces:
    def test_subdomain_to_boundary(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        bp = pp.ad.BoundaryProjection(mdg, sds, dim=1)
        op = bp.subdomain_to_boundary
        assert op.operator_domain is not None
        assert op.operator_range is not None
        assert GridEntity.faces in op.operator_domain.dof_info
        assert GridEntity.cells in op.operator_range.dof_info

    def test_boundary_to_subdomain(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        bp = pp.ad.BoundaryProjection(mdg, sds, dim=1)
        op = bp.boundary_to_subdomain
        assert op.operator_domain is not None
        assert op.operator_range is not None
        assert GridEntity.cells in op.operator_domain.dof_info
        assert GridEntity.faces in op.operator_range.dof_info

    def test_transpose_consistency(self, fracture_mdg):
        """boundary_to_subdomain domain/range are flipped from subdomain_to_boundary."""
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        bp = pp.ad.BoundaryProjection(mdg, sds, dim=1)
        s2b = bp.subdomain_to_boundary
        b2s = bp.boundary_to_subdomain
        assert s2b.operator_domain == b2s.operator_range
        assert s2b.operator_range == b2s.operator_domain


class TestTraceSpaces:
    def test_trace_domain_range(self, fracture_mdg):
        mdg = fracture_mdg
        sds = mdg.subdomains(dim=2)
        tr = pp.ad.Trace(sds, dim=1)
        assert tr.trace.operator_domain is not None
        assert tr.trace.operator_range is not None
        assert GridEntity.cells in tr.trace.operator_domain.dof_info
        assert GridEntity.faces in tr.trace.operator_range.dof_info

    def test_trace_grids(self, fracture_mdg):
        mdg = fracture_mdg
        sds = mdg.subdomains(dim=2)
        tr = pp.ad.Trace(sds, dim=1)
        assert set(tr.trace.operator_domain.grids) == set(sds)
        assert set(tr.trace.operator_range.grids) == set(sds)

    def test_trace_empty(self):
        tr = pp.ad.Trace([], dim=1)
        assert tr.trace.operator_domain is None
        assert tr.trace.operator_range is None


class TestDivergenceSpaces:
    def test_divergence_domain_range(self, fracture_mdg):
        mdg = fracture_mdg
        sds = mdg.subdomains(dim=2)
        div = pp.ad.Divergence(sds, dim=1)
        assert div.operator_domain is not None
        assert div.operator_range is not None
        assert GridEntity.faces in div.operator_domain.dof_info
        assert GridEntity.cells in div.operator_range.dof_info

    def test_divergence_grids(self, fracture_mdg):
        mdg = fracture_mdg
        sds = mdg.subdomains(dim=2)
        div = pp.ad.Divergence(sds, dim=1)
        assert set(div.operator_domain.grids) == set(sds)
        assert set(div.operator_range.grids) == set(sds)

    def test_divergence_empty(self):
        div = pp.ad.Divergence([], dim=1)
        assert div.operator_domain is None
        assert div.operator_range is None

    @pytest.mark.parametrize("dim", [2, 3])
    def test_divergence_vector_field_dof_info(self, dim):
        """Divergence with dim>1 stores dim DOFs per face (domain) and per cell (range)."""
        if dim == 2:
            g = pp.CartGrid([3, 3])
        else:
            g = pp.CartGrid([2, 2, 2])
        g.compute_geometry()
        div = pp.ad.Divergence([g], dim=dim)
        assert div.operator_domain is not None
        assert div.operator_range is not None
        assert div.operator_domain.dof_info == {GridEntity.faces: dim}
        assert div.operator_range.dof_info == {GridEntity.cells: dim}

    @pytest.mark.parametrize("dim", [2, 3])
    def test_divergence_vector_field_grids(self, dim):
        """Divergence with dim>1 still records the correct grid set."""
        if dim == 2:
            g = pp.CartGrid([3, 3])
        else:
            g = pp.CartGrid([2, 2, 2])
        g.compute_geometry()
        div = pp.ad.Divergence([g], dim=dim)
        assert div.operator_domain.grids == (g,)
        assert div.operator_range.grids == (g,)


# ---------------------------------------------------------------------------
# Stage 4e: MergedOperator domain/range
# ---------------------------------------------------------------------------


class TestMergedOperatorSpaces:
    """Tests that MergedOperator inherits operator_domain/range from the
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
        assert op.operator_domain is None
        assert op.operator_range is None

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
        assert op.operator_domain is not None
        assert op.operator_range is not None
        assert GridEntity.faces in op.operator_domain.dof_info
        assert GridEntity.cells in op.operator_range.dof_info
        assert set(op.operator_domain.grids) == {g1, g2}
        assert set(op.operator_range.grids) == {g1, g2}

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
        assert op.operator_domain is None
        assert op.operator_range is None


# ---------------------------------------------------------------------------
# Stage 5: validate_operands / infer_domain_range
# ---------------------------------------------------------------------------


class TestInferDomainRange:
    """Tests for Operations.infer_domain_range (Stage 5: validation and inference)."""

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
        """A leaf operator with domain=range=cell_space."""
        return DenseArray(np.zeros(3), domain=cell_space, range=cell_space)

    @pytest.fixture
    def face_op(self, face_space):
        """A leaf operator with domain=range=face_space."""
        return DenseArray(np.zeros(3), domain=face_space, range=face_space)

    # --- elementwise: compatible operands ---

    @pytest.mark.parametrize(
        "binary_op",
        [_op.add, _op.sub, _op.mul, _op.truediv, _op.pow],
        ids=["add", "sub", "mul", "div", "pow"],
    )
    def test_elementwise_compatible(self, cell_op, cell_space, binary_op):
        """Elementwise ops between same-space operands preserve domain/range."""
        result = binary_op(cell_op, cell_op)
        assert result.operator_domain == cell_space
        assert result.operator_range == cell_space

    def test_rsub_compatible(self, cell_op, cell_space):
        """__rsub__ must also propagate domain/range."""
        result = 0 - cell_op
        assert result.operator_domain == cell_space
        assert result.operator_range == cell_space

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
            np.zeros(3), domain=face_op.operator_domain, range=cell_op.operator_range
        )
        result = binary_op(cell_op, projected)
        assert result.operator_domain == OperatorSpace.unclear()
        assert result.operator_range == cell_op.operator_range

    def test_elementwise_uses_union_of_dependency_domains(self, two_subdomains):
        """Different but compatible-looking domains still become unclear."""
        g1, g2 = two_subdomains
        top_space = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        union_space = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})

        local = DenseArray(np.zeros(3), domain=top_space, range=top_space)
        projected = DenseArray(np.zeros(3), domain=union_space, range=top_space)

        result = local * projected

        assert result.operator_domain == OperatorSpace.unclear()
        assert result.operator_range == top_space

    def test_elementwise_different_grids_becomes_unclear(self, two_subdomains):
        """Different grids are enough to make the inferred domain unclear."""
        g1, g2 = two_subdomains
        left_space = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        right_space = OperatorSpace.from_domains([g2], {GridEntity.cells: 1})
        left = DenseArray(np.zeros(3), domain=left_space, range=left_space)
        right = DenseArray(np.zeros(3), domain=right_space, range=left_space)

        result = left + right

        assert result.operator_domain == OperatorSpace.unclear()
        assert result.operator_range == left_space

    def test_elementwise_unclear_domain_propagates(self, cell_space, face_space):
        """Once unclear, the elementwise result remains unclear."""
        unclear = DenseArray(
            np.zeros(3), domain=OperatorSpace.unclear(), range=cell_space
        )
        known = DenseArray(np.zeros(3), domain=face_space, range=cell_space)

        result = unclear + known

        assert result.operator_domain == OperatorSpace.unclear()
        assert result.operator_range == cell_space

    # --- matmul: compatible ---

    def test_matmul_compatible(self, two_subdomains):
        """A @ B where range(B)==domain(A) is valid."""
        g1, g2 = two_subdomains
        cell_sp = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})
        face_sp = OperatorSpace.from_domains([g1, g2], {GridEntity.faces: 1})
        # A: faces → cells (domain=face_sp, range=cell_sp)
        # B: cells → faces (domain=cell_sp, range=face_sp)
        # A @ B: range(B)=face_sp == domain(A)=face_sp → valid
        A = SparseArray(sps.eye(3), domain=face_sp, range=cell_sp)
        B = SparseArray(sps.eye(3), domain=cell_sp, range=face_sp)
        result = A @ B
        assert result.operator_domain == cell_sp
        assert result.operator_range == cell_sp

    def test_matmul_incompatible(self, two_subdomains):
        """A @ B where range(B) != domain(A) raises ValueError."""
        g1, g2 = two_subdomains
        cell_sp = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})
        face_sp = OperatorSpace.from_domains([g1, g2], {GridEntity.faces: 1})
        # A: faces → cells, B: faces → cells (range(B)=cells != domain(A)=faces)
        A = SparseArray(sps.eye(3), domain=face_sp, range=cell_sp)
        B = SparseArray(sps.eye(3), domain=face_sp, range=cell_sp)
        with pytest.raises(ValueError, match="matrix multiplication"):
            _ = A @ B

    def test_matmul_different_grids_incompatible(self, two_subdomains):
        """Matrix multiplication requires exact space equality, including grids."""
        g1, g2 = two_subdomains
        left_domain = OperatorSpace.from_domains([g1], {GridEntity.faces: 1})
        right_range = OperatorSpace.from_domains([g2], {GridEntity.faces: 1})
        left_range = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        right_domain = OperatorSpace.from_domains([g2], {GridEntity.cells: 1})
        left = SparseArray(sps.eye(3), domain=left_domain, range=left_range)
        right = SparseArray(sps.eye(3), domain=right_domain, range=right_range)

        with pytest.raises(ValueError, match="matrix multiplication"):
            _ = left @ right

    def test_matmul_with_unclear_left_domain_raises(self, cell_space, face_space):
        """A left operand with unclear domain cannot be used in matmul."""
        unclear = SparseArray(
            sps.eye(3), domain=OperatorSpace.unclear(), range=cell_space
        )
        rhs = DenseArray(np.zeros(3), domain=face_space, range=face_space)

        with pytest.raises(ValueError, match="left operand.*domain is unclear"):
            _ = unclear @ rhs

    def test_rmatmul_with_unclear_right_operand_raises(self, cell_space, face_space):
        """The operator on the right-hand side of rmatmul cannot have unclear domain."""
        unclear = SparseArray(
            sps.eye(3), domain=OperatorSpace.unclear(), range=cell_space
        )
        lhs = SparseArray(sps.eye(3), domain=face_space, range=face_space)

        with pytest.raises(ValueError, match="right operand.*domain is unclear"):
            _ = unclear.__rmatmul__(lhs)

    # --- Scalar: always valid, inherits non-scalar space ---

    def test_add_with_scalar_lhs(self, cell_op, cell_space):
        sc = Scalar(2.0)
        result = sc + cell_op
        assert result.operator_domain == cell_space
        assert result.operator_range == cell_space

    def test_add_with_scalar_rhs(self, cell_op, cell_space):
        sc = Scalar(2.0)
        result = cell_op + sc
        assert result.operator_domain == cell_space
        assert result.operator_range == cell_space

    def test_mul_with_scalar(self, cell_op, cell_space):
        sc = Scalar(3.0)
        result = sc * cell_op
        assert result.operator_domain == cell_space
        assert result.operator_range == cell_space

    def test_scalar_scalar(self):
        sc1 = Scalar(1.0)
        sc2 = Scalar(2.0)
        result = sc1 + sc2
        assert result.operator_domain == OperatorSpace.scalar()
        assert result.operator_range == OperatorSpace.scalar()

    # --- None domain/range: skips validation (backward compat) ---

    def test_none_plus_known_inherits_known(self, cell_space):
        """Operator with None domain + operator with known domain → inherits known."""
        unknown = DenseArray(np.zeros(3))  # domain=None
        known = DenseArray(np.zeros(3), domain=cell_space, range=cell_space)
        result = unknown + known
        assert result.operator_domain == cell_space
        assert result.operator_range == cell_space

    def test_both_none_stays_none(self):
        """Two operators with no domain/range → result also has None."""
        a = DenseArray(np.zeros(3))
        b = DenseArray(np.zeros(3))
        result = a + b
        assert result.operator_domain is None
        assert result.operator_range is None

    def test_plain_python_scalar_exponent(self, cell_op, cell_space):
        """op ** 2 (plain Python int) should preserve domain/range."""
        result = cell_op**2
        assert result.operator_domain == cell_space
        assert result.operator_range == cell_space

    def test_plain_python_scalar_rtruediv(self, cell_op, cell_space):
        """1 / op should preserve domain/range."""
        result = 1 / cell_op
        assert result.operator_domain == cell_space
        assert result.operator_range == cell_space

    # --- infer_domain_range is public ---

    def test_infer_domain_range_is_public(self, cell_op):
        """infer_domain_range should be accessible on the Operations enum."""
        assert hasattr(Operations.add, "infer_domain_range")
        dom, ran = Operations.add.infer_domain_range(cell_op, cell_op)
        assert dom is not None
        assert ran is not None


# ---------------------------------------------------------------------------
# Stage 6: compound operator domain/range propagation
# ---------------------------------------------------------------------------


class TestCompoundOperatorSpaces:
    """Tests that domain/range propagates correctly through multi-step expressions."""

    @pytest.fixture
    def spaces(self, two_subdomains):
        g1, g2 = two_subdomains
        cell_sp = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})
        face_sp = OperatorSpace.from_domains([g1, g2], {GridEntity.faces: 1})
        return cell_sp, face_sp

    # --- chained matmul ---

    def test_chained_matmul_domain_range(self, two_subdomains, spaces):
        """(A @ B): range(B) == domain(A) → result.domain=B.domain, result.range=A.range"""
        g1, g2 = two_subdomains
        cell_sp, face_sp = spaces
        # A maps faces→cells; B maps cells→faces; A@B maps cells→cells
        A = SparseArray(sps.eye(3), domain=face_sp, range=cell_sp)
        B = SparseArray(sps.eye(3), domain=cell_sp, range=face_sp)
        result = A @ B
        assert result.operator_domain == cell_sp
        assert result.operator_range == cell_sp

    def test_three_way_matmul(self, two_subdomains, spaces):
        """(A @ B) @ C propagates spaces through two matmul steps."""
        g1, g2 = two_subdomains
        cell_sp, face_sp = spaces
        # A: face→cell, B: cell→face → A@B: cell→cell
        # C: face→cell → (A@B)@C requires range(C)==domain(A@B)=cell_sp ✓ → face→cell
        A = SparseArray(sps.eye(3), domain=face_sp, range=cell_sp)
        B = SparseArray(sps.eye(3), domain=cell_sp, range=face_sp)
        C = SparseArray(sps.eye(3), domain=face_sp, range=cell_sp)
        AB = A @ B
        assert AB.operator_domain == cell_sp
        assert AB.operator_range == cell_sp
        ABC = AB @ C
        assert ABC.operator_domain == face_sp
        assert ABC.operator_range == cell_sp

    def test_chained_matmul_incompatible_raises(self, two_subdomains, spaces):
        """(A @ B) @ C raises ValueError when range(C) != domain(A@B)."""
        g1, g2 = two_subdomains
        cell_sp, face_sp = spaces
        # A@B: cell→cell (see test_three_way_matmul); C has range=face_sp != cell_sp
        A = SparseArray(sps.eye(3), domain=face_sp, range=cell_sp)
        B = SparseArray(sps.eye(3), domain=cell_sp, range=face_sp)
        AB = A @ B  # domain=cell_sp, range=cell_sp
        C = SparseArray(sps.eye(3), domain=face_sp, range=face_sp)
        with pytest.raises(ValueError, match="matrix multiplication"):
            _ = AB @ C

    def test_add_after_matmul(self, two_subdomains, spaces):
        """(A @ v) + (B @ w) where both results have the same range."""
        g1, g2 = two_subdomains
        cell_sp, face_sp = spaces
        A = SparseArray(sps.eye(3), domain=face_sp, range=cell_sp)
        v = DenseArray(np.zeros(3), domain=face_sp, range=face_sp)
        B = SparseArray(sps.eye(3), domain=face_sp, range=cell_sp)
        w = DenseArray(np.zeros(3), domain=face_sp, range=face_sp)
        Av = A @ v
        Bw = B @ w
        result = Av + Bw
        assert result.operator_domain == face_sp
        assert result.operator_range == cell_sp

    def test_add_matmul_incompatible_raises(self, two_subdomains, spaces):
        """(A @ v) + (B @ w) where ranges differ raises ValueError."""
        g1, g2 = two_subdomains
        cell_sp, face_sp = spaces
        A = SparseArray(sps.eye(3), domain=face_sp, range=cell_sp)
        v = DenseArray(np.zeros(3), domain=face_sp, range=face_sp)
        Av = A @ v  # range=cell_sp
        # B maps faces→faces, so B@w has range=face_sp
        B = SparseArray(sps.eye(3), domain=face_sp, range=face_sp)
        w = DenseArray(np.zeros(3), domain=face_sp, range=face_sp)
        Bw = B @ w
        with pytest.raises(ValueError):
            _ = Av + Bw

    # --- scalar factor in chains ---

    def test_scalar_mul_after_matmul(self, two_subdomains, spaces):
        """Scalar(k) * (A @ v) preserves A's range as the result range."""
        g1, g2 = two_subdomains
        cell_sp, face_sp = spaces
        A = SparseArray(sps.eye(3), domain=face_sp, range=cell_sp)
        v = DenseArray(np.zeros(3), domain=face_sp, range=face_sp)
        Av = A @ v
        result = Scalar(2.0) * Av
        assert result.operator_domain == face_sp
        assert result.operator_range == cell_sp

    def test_unary_minus_preserves_spaces(self, two_subdomains, spaces):
        """Unary minus on SparseArray preserves domain/range."""
        g1, g2 = two_subdomains
        cell_sp, face_sp = spaces
        A = SparseArray(sps.eye(3), domain=face_sp, range=cell_sp)
        result = -A
        assert result.operator_domain == face_sp
        assert result.operator_range == cell_sp

    def test_unary_minus_dense_array_preserves_spaces(self, two_subdomains, spaces):
        """DenseArray.__neg__ must also preserve domain/range (separate code path)."""
        g1, g2 = two_subdomains
        cell_sp, face_sp = spaces
        arr = DenseArray(np.ones(3), domain=cell_sp, range=cell_sp)
        result = -arr
        assert result.operator_domain == cell_sp
        assert result.operator_range == cell_sp

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
        assert div.operator_domain is not None
        assert GridEntity.faces in div.operator_domain.dof_info
        assert div.operator_range is not None
        assert GridEntity.cells in div.operator_range.dof_info

    def test_compound_inherits_none_when_one_operand_has_none(
        self, two_subdomains, spaces
    ):
        """When one operand in a chain has None domain, the chain can still succeed
        if the other operand provides the domain."""
        g1, g2 = two_subdomains
        cell_sp, face_sp = spaces
        # unknown_op has no space info
        unknown_op = DenseArray(np.zeros(3))
        known_op = DenseArray(np.zeros(3), domain=cell_sp, range=cell_sp)
        # Adding unknown + known: no error, result inherits known's spaces
        result = unknown_op + known_op
        assert result.operator_domain == cell_sp
        assert result.operator_range == cell_sp

    def test_domain_and_range_stored_independently(self, two_subdomains):
        """Even when domain == range, they are stored as independent attributes."""
        g1, g2 = two_subdomains
        cell_sp = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})
        a = DenseArray(np.zeros(3), domain=cell_sp, range=cell_sp)
        b = DenseArray(np.zeros(3), domain=cell_sp, range=cell_sp)
        result = a + b
        # domain and range are equal in value, but are independent objects
        assert result.operator_domain == result.operator_range
        assert result.operator_domain is not None
        assert result.operator_range is not None


# ---------------------------------------------------------------------------
# Stage 7: Concrete get_row/col_dof_info on standard discretizations
# ---------------------------------------------------------------------------


class TestFVEllipticDofInfo:
    """Tests for FVElliptic.get_row/col_dof_info (covers Mpfa and Tpfa)."""

    def setup_method(self):
        self.mpfa = pp.Mpfa("flow")
        self.tpfa = pp.Tpfa("flow")

    @pytest.mark.parametrize("discr_name", ["mpfa", "tpfa"])
    def test_flux_dof_info(self, discr_name):
        """Both Mpfa and Tpfa map cells → faces for the flux matrix."""
        discr = getattr(self, discr_name)
        assert discr.get_row_dof_info("flux") == {GridEntity.faces: 1}
        assert discr.get_col_dof_info("flux") == {GridEntity.cells: 1}

    def test_bound_flux_dof_info(self):
        assert self.mpfa.get_row_dof_info("bound_flux") == {GridEntity.faces: 1}
        assert self.mpfa.get_col_dof_info("bound_flux") == {GridEntity.faces: 1}

    def test_bound_pressure_cell_dof_info(self):
        assert self.mpfa.get_row_dof_info("bound_pressure_cell") == {
            GridEntity.faces: 1
        }
        assert self.mpfa.get_col_dof_info("bound_pressure_cell") == {
            GridEntity.cells: 1
        }

    def test_bound_pressure_face_dof_info(self):
        assert self.mpfa.get_row_dof_info("bound_pressure_face") == {
            GridEntity.faces: 1
        }
        assert self.mpfa.get_col_dof_info("bound_pressure_face") == {
            GridEntity.faces: 1
        }

    def test_vector_source_uses_nd(self):
        """vector_source cols are nd DOFs per cell (gravity components)."""
        assert self.mpfa.get_row_dof_info("vector_source", nd=3) == {
            GridEntity.faces: 1
        }
        assert self.mpfa.get_col_dof_info("vector_source", nd=2) == {
            GridEntity.cells: 2
        }
        assert self.mpfa.get_col_dof_info("vector_source", nd=3) == {
            GridEntity.cells: 3
        }

    def test_bound_pressure_vector_source_uses_nd(self):
        assert self.mpfa.get_col_dof_info("bound_pressure_vector_source", nd=2) == {
            GridEntity.cells: 2
        }

    @pytest.mark.parametrize("matrix_key", ["nonexistent", ""])
    def test_unknown_key_raises(self, matrix_key):
        with pytest.raises(ValueError, match="Unrecognized matrix key"):
            self.mpfa.get_row_dof_info(matrix_key)
        with pytest.raises(ValueError, match="Unrecognized matrix key"):
            self.mpfa.get_col_dof_info(matrix_key)


class TestMpsaDofInfo:
    """Tests for Mpsa.get_row/col_dof_info."""

    def setup_method(self):
        self.mpsa = pp.Mpsa("mech")

    @pytest.mark.parametrize("nd", [2, 3])
    def test_stress_dof_info(self, nd):
        """stress maps cells:nd → faces:nd."""
        assert self.mpsa.get_row_dof_info("stress", nd=nd) == {GridEntity.faces: nd}
        assert self.mpsa.get_col_dof_info("stress", nd=nd) == {GridEntity.cells: nd}

    @pytest.mark.parametrize("nd", [2, 3])
    def test_bound_stress_dof_info(self, nd):
        """bound_stress maps faces:nd → faces:nd."""
        assert self.mpsa.get_row_dof_info("bound_stress", nd=nd) == {
            GridEntity.faces: nd
        }
        assert self.mpsa.get_col_dof_info("bound_stress", nd=nd) == {
            GridEntity.faces: nd
        }

    def test_bound_displacement_cell_dof_info(self):
        assert self.mpsa.get_row_dof_info("bound_displacement_cell", nd=2) == {
            GridEntity.faces: 2
        }
        assert self.mpsa.get_col_dof_info("bound_displacement_cell", nd=2) == {
            GridEntity.cells: 2
        }

    def test_bound_displacement_face_dof_info(self):
        assert self.mpsa.get_row_dof_info("bound_displacement_face", nd=2) == {
            GridEntity.faces: 2
        }
        assert self.mpsa.get_col_dof_info("bound_displacement_face", nd=2) == {
            GridEntity.faces: 2
        }

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unrecognized matrix key"):
            self.mpsa.get_row_dof_info("nonexistent", nd=2)
        with pytest.raises(ValueError, match="Unrecognized matrix key"):
            self.mpsa.get_col_dof_info("nonexistent", nd=2)


class TestBiotDofInfo:
    """Tests for Biot.get_row/col_dof_info."""

    def setup_method(self):
        self.biot = pp.Biot("mech")

    @pytest.mark.parametrize(
        "matrix_key, nd, expected_row, expected_col",
        [
            (
                "displacement_divergence",
                2,
                {GridEntity.cells: 1},
                {GridEntity.cells: 2},
            ),
            (
                "bound_displacement_divergence",
                2,
                {GridEntity.cells: 1},
                {GridEntity.faces: 2},
            ),
            ("scalar_gradient", 2, {GridEntity.faces: 2}, {GridEntity.cells: 1}),
            ("consistency", 2, {GridEntity.cells: 1}, {GridEntity.cells: 1}),
            ("bound_pressure", 2, {GridEntity.faces: 2}, {GridEntity.cells: 1}),
            # inherited from Mpsa
            ("stress", 2, {GridEntity.faces: 2}, {GridEntity.cells: 2}),
            ("stress", 3, {GridEntity.faces: 3}, {GridEntity.cells: 3}),
            ("bound_stress", 3, {GridEntity.faces: 3}, {GridEntity.faces: 3}),
        ],
    )
    def test_dof_info(self, matrix_key, nd, expected_row, expected_col):
        assert self.biot.get_row_dof_info(matrix_key, nd=nd) == expected_row
        assert self.biot.get_col_dof_info(matrix_key, nd=nd) == expected_col

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unrecognized matrix key"):
            self.biot.get_row_dof_info("nonexistent", nd=2)
        with pytest.raises(ValueError, match="Unrecognized matrix key"):
            self.biot.get_col_dof_info("nonexistent", nd=2)


class TestUpwindDofInfo:
    """Tests for Upwind.get_row/col_dof_info."""

    def setup_method(self):
        self.upwind = pp.Upwind("flow")

    def test_upwind_dof_info(self):
        assert self.upwind.get_row_dof_info("upwind") == {GridEntity.faces: 1}
        assert self.upwind.get_col_dof_info("upwind") == {GridEntity.cells: 1}

    def test_bound_transport_dir_dof_info(self):
        assert self.upwind.get_row_dof_info("bound_transport_dir") == {
            GridEntity.faces: 1
        }
        assert self.upwind.get_col_dof_info("bound_transport_dir") == {
            GridEntity.faces: 1
        }

    def test_bound_transport_neu_dof_info(self):
        assert self.upwind.get_row_dof_info("bound_transport_neu") == {
            GridEntity.faces: 1
        }
        assert self.upwind.get_col_dof_info("bound_transport_neu") == {
            GridEntity.faces: 1
        }

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unrecognized matrix key"):
            self.upwind.get_row_dof_info("nonexistent")
        with pytest.raises(ValueError, match="Unrecognized matrix key"):
            self.upwind.get_col_dof_info("nonexistent")


class TestMergedOperatorWithConcreteDiscretization:
    """Integration tests: MergedOperator infers domain/range from concrete discretizations."""

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
        assert op.operator_domain is not None
        assert op.operator_range is not None
        assert op.operator_domain.dof_info == {GridEntity.cells: 1}
        assert op.operator_range.dof_info == {GridEntity.faces: 1}
        assert set(op.operator_domain.grids) == {g1, g2}
        assert set(op.operator_range.grids) == {g1, g2}

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
        assert op.operator_domain.dof_info == {GridEntity.faces: 1}
        assert op.operator_range.dof_info == {GridEntity.faces: 1}

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
        assert op.operator_domain is not None
        assert op.operator_range is not None
        assert op.operator_domain.dof_info == {GridEntity.cells: 2}
        assert op.operator_range.dof_info == {GridEntity.faces: 2}

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
        assert op.operator_domain.dof_info == {GridEntity.cells: 1}
        assert op.operator_range.dof_info == {GridEntity.faces: 1}

    def test_mpfa_via_ad_wrapper(self, two_subdomains):
        """Using MpfaAd wrapper (wrap_discretization path) produces correct spaces."""
        g1, g2 = two_subdomains
        discr = pp.ad.MpfaAd("flow", [g1, g2])
        flux_op = discr.flux()
        assert flux_op.operator_domain is not None
        assert flux_op.operator_range is not None
        assert flux_op.operator_domain.dof_info == {GridEntity.cells: 1}
        assert flux_op.operator_range.dof_info == {GridEntity.faces: 1}

    def test_tpfa_via_ad_wrapper(self, two_subdomains):
        """Using TpfaAd wrapper produces correct spaces for flux."""
        g1, g2 = two_subdomains
        discr = pp.ad.TpfaAd("flow", [g1, g2])
        flux_op = discr.flux()
        assert flux_op.operator_domain.dof_info == {GridEntity.cells: 1}
        assert flux_op.operator_range.dof_info == {GridEntity.faces: 1}

    def test_mpfa_bound_flux_ad_wrapper(self, two_subdomains):
        """MpfaAd.bound_flux() has face-domain and face-range."""
        g1, g2 = two_subdomains
        discr = pp.ad.MpfaAd("flow", [g1, g2])
        op = discr.bound_flux()
        assert op.operator_domain.dof_info == {GridEntity.faces: 1}
        assert op.operator_range.dof_info == {GridEntity.faces: 1}

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
        assert op.operator_domain is not None
        assert op.operator_domain.dof_info == {GridEntity.cells: 2}
        assert op.operator_range.dof_info == {GridEntity.faces: 1}

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
        assert op.operator_domain is not None
        assert op.operator_range is not None
        assert op.operator_domain.dof_info == {GridEntity.cells: 2}
        assert op.operator_range.dof_info == {GridEntity.faces: 2}
        assert set(op.operator_domain.grids) == {g1, g2}
        assert set(op.operator_range.grids) == {g1, g2}

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
        assert op.operator_domain is not None
        assert op.operator_range is not None
        # 2D: row DOFs are nrot=1 face entries, col DOFs are nd=2 cell entries
        assert op.operator_domain.dof_info == {GridEntity.cells: 2}
        assert op.operator_range.dof_info == {GridEntity.faces: 1}


# ---------------------------------------------------------------------------
# sum_operator_list and sum_projection_list space propagation
# ---------------------------------------------------------------------------


class TestSumOperatorListSpace:
    """sum_operator_list delegates to __add__, so domain/range must propagate."""

    def test_sum_two_arrays_propagates_space(self, two_subdomains):
        """sum_operator_list([a, b]) with compatible spaces inherits those spaces."""
        g, _ = two_subdomains
        space = OperatorSpace.from_domains([g], {GridEntity.cells: 1})
        a = DenseArray(np.ones(4), domain=space, range=space)
        b = DenseArray(np.ones(4), domain=space, range=space)
        result = sum_operator_list([a, b])
        assert result.operator_domain == space
        assert result.operator_range == space

    def test_sum_three_arrays_propagates_space(self, two_subdomains):
        """sum_operator_list([a, b, c]) propagates spaces through the full reduce."""
        g, _ = two_subdomains
        space = OperatorSpace.from_domains([g], {GridEntity.cells: 1})
        ops = [DenseArray(np.ones(4), domain=space, range=space) for _ in range(3)]
        result = sum_operator_list(ops)
        assert result.operator_domain == space
        assert result.operator_range == space

    def test_sum_incompatible_spaces_raises(self, two_subdomains):
        """sum_operator_list raises ValueError when spaces are incompatible."""
        g1, g2 = two_subdomains
        s1 = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        s2 = OperatorSpace.from_domains([g2], {GridEntity.cells: 1})
        a = DenseArray(np.ones(4), domain=s1, range=s1)
        b = DenseArray(np.ones(9), domain=s2, range=s2)
        with pytest.raises(ValueError):
            sum_operator_list([a, b])

    def test_sum_none_space_inherits_known(self, two_subdomains):
        """sum_operator_list with one operand lacking a space still propagates the other."""
        g, _ = two_subdomains
        space = OperatorSpace.from_domains([g], {GridEntity.cells: 1})
        a = DenseArray(np.ones(4), domain=space, range=space)
        b = DenseArray(np.ones(4))  # no space
        result = sum_operator_list([a, b])
        assert result.operator_domain == space
        assert result.operator_range == space


# ---------------------------------------------------------------------------
# Stage 7 continued: Tpsa dof_info
# ---------------------------------------------------------------------------


class TestTpsaDofInfo:
    """Tests for Tpsa.get_row/col_dof_info.

    ``nrot = nd*(nd-1)//2`` is the number of rotation DOFs per entity:
    1 in 2d, 3 in 3d.
    """

    def setup_method(self):
        self.tpsa = pp.Tpsa("mech")

    @pytest.mark.parametrize(
        "matrix_key, nd, expected_row, expected_col",
        [
            # stress_displacement: rows=faces:nd, cols=cells:nd
            ("stress_displacement", 2, {GridEntity.faces: 2}, {GridEntity.cells: 2}),
            ("stress_displacement", 3, {GridEntity.faces: 3}, {GridEntity.cells: 3}),
            # stress_rotation: rows=faces:nd, cols=cells:nrot
            ("stress_rotation", 2, {GridEntity.faces: 2}, {GridEntity.cells: 1}),
            ("stress_rotation", 3, {GridEntity.faces: 3}, {GridEntity.cells: 3}),
            # stress_total_pressure: rows=faces:nd, cols=cells:1
            ("stress_total_pressure", 2, {GridEntity.faces: 2}, {GridEntity.cells: 1}),
            # rotation_displacement: rows=faces:nrot, cols=cells:nd
            ("rotation_displacement", 2, {GridEntity.faces: 1}, {GridEntity.cells: 2}),
            ("rotation_displacement", 3, {GridEntity.faces: 3}, {GridEntity.cells: 3}),
            # rotation_rotation: rows=faces:nrot, cols=cells:nrot
            ("rotation_rotation", 2, {GridEntity.faces: 1}, {GridEntity.cells: 1}),
            ("rotation_rotation", 3, {GridEntity.faces: 3}, {GridEntity.cells: 3}),
            # mass matrices
            ("mass_total_pressure", 2, {GridEntity.faces: 1}, {GridEntity.cells: 1}),
            ("mass_displacement", 2, {GridEntity.faces: 1}, {GridEntity.cells: 2}),
            ("mass_displacement", 3, {GridEntity.faces: 1}, {GridEntity.cells: 3}),
            # boundary condition matrices
            ("bound_stress", 2, {GridEntity.faces: 2}, {GridEntity.faces: 2}),
            (
                "bound_rotation_displacement",
                2,
                {GridEntity.faces: 1},
                {GridEntity.faces: 2},
            ),
            (
                "bound_rotation_displacement",
                3,
                {GridEntity.faces: 3},
                {GridEntity.faces: 3},
            ),
            (
                "bound_mass_displacement",
                2,
                {GridEntity.faces: 1},
                {GridEntity.faces: 2},
            ),
            # displacement reconstruction matrices
            (
                "bound_displacement_cell",
                2,
                {GridEntity.faces: 2},
                {GridEntity.cells: 2},
            ),
            (
                "bound_displacement_face",
                2,
                {GridEntity.faces: 2},
                {GridEntity.faces: 2},
            ),
            (
                "bound_displacement_rotation_cell",
                2,
                {GridEntity.faces: 2},
                {GridEntity.cells: 1},
            ),
            (
                "bound_displacement_rotation_cell",
                3,
                {GridEntity.faces: 3},
                {GridEntity.cells: 3},
            ),
            (
                "bound_displacement_solid_pressure_cell",
                2,
                {GridEntity.faces: 2},
                {GridEntity.cells: 1},
            ),
        ],
    )
    def test_dof_info(self, matrix_key, nd, expected_row, expected_col):
        assert self.tpsa.get_row_dof_info(matrix_key, nd=nd) == expected_row
        assert self.tpsa.get_col_dof_info(matrix_key, nd=nd) == expected_col

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unrecognized matrix key"):
            self.tpsa.get_row_dof_info("nonexistent", nd=2)
        with pytest.raises(ValueError, match="Unrecognized matrix key"):
            self.tpsa.get_col_dof_info("nonexistent", nd=2)

    @pytest.mark.parametrize("nd", [2, 3])
    def test_nrot_formula(self, nd):
        """nrot=nd*(nd-1)//2 matches the rotation DOF count."""
        nrot = nd * (nd - 1) // 2
        assert self.tpsa.get_row_dof_info("rotation_displacement", nd=nd) == {
            GridEntity.faces: nrot
        }
        assert self.tpsa.get_col_dof_info("stress_rotation", nd=nd) == {
            GridEntity.cells: nrot
        }
