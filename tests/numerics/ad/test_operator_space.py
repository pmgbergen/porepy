"""Tests for Stage 2 of GH discussion #1601: DomainType, OperatorSpace and
operator_domain/operator_range propagation."""

import pytest
import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.operators import (
    DomainType,
    OperatorSpace,
    Operator,
    Scalar,
    DenseArray,
    SparseArray,
    Variable,
)
from porepy.numerics.ad._grid_entity import GridEntity


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
        types = [DomainType.subdomains, DomainType.interfaces,
                 DomainType.boundary_grids, DomainType.scalar]
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
    def test_operator_domain_none_by_default(self):
        op = Operator(name="test")
        assert op.operator_domain is None

    def test_operator_range_none_by_default(self):
        op = Operator(name="test")
        assert op.operator_range is None

    def test_set_domain_in_init(self, two_subdomains):
        g1, _ = two_subdomains
        space = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        op = Operator(name="test", domain=space)
        assert op.operator_domain == space

    def test_set_range_in_init(self, two_subdomains):
        g1, _ = two_subdomains
        space = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        op = Operator(name="test", range_=space)
        assert op.operator_range == space


# ---------------------------------------------------------------------------
# Leaf operator spaces
# ---------------------------------------------------------------------------


class TestScalarSpace:
    def test_scalar_has_scalar_space(self):
        s = Scalar(3.14)
        assert s.operator_domain == OperatorSpace.scalar()
        assert s.operator_range == OperatorSpace.scalar()


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
        assert var.operator_domain.dof_info == {GridEntity.cells: 2, GridEntity.faces: 1}


class TestDenseArraySpace:
    def test_dense_array_no_space_by_default(self):
        arr = DenseArray(np.ones(5))
        assert arr.operator_domain is None
        assert arr.operator_range is None

    def test_dense_array_with_explicit_space(self, two_subdomains):
        g, _ = two_subdomains
        space = OperatorSpace.from_domains([g], {GridEntity.cells: 1})
        arr = DenseArray(np.ones(5), domain=space, range_=space)
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
        op = SparseArray(mat, domain=space, range_=space)
        assert op.operator_domain == space
        assert op.operator_range == space


# ---------------------------------------------------------------------------
# Domain/range propagation through arithmetic operations
# ---------------------------------------------------------------------------


class TestDomainRangePropagation:
    """Test that binary operations propagate operator_domain and operator_range."""

    def _cell_space(self, g):
        return OperatorSpace.from_domains([g], {GridEntity.cells: 1})

    def test_add_same_space_propagates(self, two_subdomains):
        g, _ = two_subdomains
        space = self._cell_space(g)
        a = DenseArray(np.ones(4), domain=space, range_=space)
        b = DenseArray(np.ones(4), domain=space, range_=space)
        result = a + b
        assert result.operator_domain == space
        assert result.operator_range == space

    def test_sub_same_space_propagates(self, two_subdomains):
        g, _ = two_subdomains
        space = self._cell_space(g)
        a = DenseArray(np.ones(4), domain=space, range_=space)
        b = DenseArray(np.ones(4), domain=space, range_=space)
        result = a - b
        assert result.operator_domain == space
        assert result.operator_range == space

    def test_mul_same_space_propagates(self, two_subdomains):
        g, _ = two_subdomains
        space = self._cell_space(g)
        a = DenseArray(np.ones(4), domain=space, range_=space)
        b = DenseArray(np.ones(4), domain=space, range_=space)
        result = a * b
        assert result.operator_domain == space
        assert result.operator_range == space

    def test_div_same_space_propagates(self, two_subdomains):
        g, _ = two_subdomains
        space = self._cell_space(g)
        a = DenseArray(np.ones(4), domain=space, range_=space)
        b = DenseArray(np.ones(4) + 1, domain=space, range_=space)
        result = a / b
        assert result.operator_domain == space
        assert result.operator_range == space

    def test_scalar_inherits_other_space(self, two_subdomains):
        g, _ = two_subdomains
        space = self._cell_space(g)
        arr = DenseArray(np.ones(4), domain=space, range_=space)
        s = Scalar(2.0)
        result = s * arr
        assert result.operator_domain == space
        assert result.operator_range == space

    def test_other_inherits_scalar_space(self, two_subdomains):
        g, _ = two_subdomains
        space = self._cell_space(g)
        arr = DenseArray(np.ones(4), domain=space, range_=space)
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
        a = DenseArray(np.ones(4), domain=space, range_=space)
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
        a = DenseArray(np.ones(4), domain=s1, range_=s1)
        b = DenseArray(np.ones(9), domain=s2, range_=s2)
        with pytest.raises(ValueError, match="[Ii]ncompat"):
            _ = a + b

    def test_matmul_propagates_outer_spaces(self, two_subdomains):
        """For A @ B: result.domain = B.domain, result.range = A.range."""
        g1, g2 = two_subdomains
        s1 = self._cell_space(g1)
        s2 = self._cell_space(g2)
        # A maps s1 -> s2, B maps s2 -> s1
        A = SparseArray(sps.eye(4, format="csr"), domain=s1, range_=s2)
        B = SparseArray(sps.eye(4, format="csr"), domain=s2, range_=s1)
        result = A @ B
        assert result.operator_domain == s2
        assert result.operator_range == s2

    def test_matmul_incompatible_range_domain_raises(self, two_subdomains):
        """A @ B raises if A.domain != B.range."""
        g1, g2 = two_subdomains
        s1 = self._cell_space(g1)
        s2 = self._cell_space(g2)
        # A.domain=s1, B.range=s2 -> incompatible
        A = SparseArray(sps.eye(4, format="csr"), domain=s1, range_=s2)
        B = SparseArray(sps.eye(4, format="csr"), domain=s1, range_=s2)
        with pytest.raises(ValueError, match="[Ii]ncompat"):
            _ = A @ B


# ---------------------------------------------------------------------------
# Discretization stubs
# ---------------------------------------------------------------------------


class TestDiscretizationStubs:
    def test_get_row_dof_info_default(self):
        from porepy.numerics.discretization import Discretization

        class ConcreteDiscr(Discretization):
            def ndof(self, g):
                return 0

            def discretize(self, g, data):
                pass

            def assemble_matrix_rhs(self, g, data):
                pass

        discr = ConcreteDiscr("test")
        assert discr.get_row_dof_info() == {}
        assert discr.get_col_dof_info() == {}

    def test_get_row_dof_info_overridable(self):
        from porepy.numerics.discretization import Discretization

        class CustomDiscr(Discretization):
            def ndof(self, g):
                return 0

            def discretize(self, g, data):
                pass

            def assemble_matrix_rhs(self, g, data):
                pass

            def get_row_dof_info(self):
                return {GridEntity.cells: 1}

            def get_col_dof_info(self):
                return {GridEntity.faces: 2}

        discr = CustomDiscr("test")
        assert discr.get_row_dof_info() == {GridEntity.cells: 1}
        assert discr.get_col_dof_info() == {GridEntity.faces: 2}


# ---------------------------------------------------------------------------
# Stage 4c: TimeDependentDenseArray optional dof_info
# ---------------------------------------------------------------------------


class TestTimeDependentDenseArraySpaces:
    def test_no_dof_info_gives_none(self, two_subdomains):
        g1, g2 = two_subdomains
        arr = pp.ad.TimeDependentDenseArray("x", [g1, g2])
        assert arr.operator_domain is None
        assert arr.operator_range is None

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


@pytest.fixture
def fracture_mdg():
    """2-D Cartesian mdg with two crossing fractures (3 subdomains, 4 interfaces)."""
    fracs = [np.array([[0, 2], [1, 1]]), np.array([[1, 1], [0, 2]])]
    md_grid = pp.meshing.cart_grid(fracs, np.array([2, 2]))
    md_grid.compute_geometry()
    return md_grid


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
        assert GridEntity.faces in op.operator_range.dof_info  # subdomain faces (codim 1)

    def test_mortar_to_secondary_avg(self, fracture_mdg):
        mdg = fracture_mdg
        sds = list(mdg.subdomains())
        intfs = list(mdg.interfaces())
        proj = pp.ad.MortarProjections(mdg=mdg, subdomains=sds, interfaces=intfs, dim=1)
        op = proj.mortar_to_secondary_avg()
        assert op.operator_domain is not None
        assert op.operator_range is not None
        assert GridEntity.cells in op.operator_domain.dof_info  # interface cells
        assert GridEntity.cells in op.operator_range.dof_info  # subdomain cells (secondary)

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
