"""Tests for Stage 2 of GH discussion #1601: DomainType, OperatorSpace and
source/target propagation."""

import operator as _op

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.grid_entity import GridEntity
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
from porepy.applications.md_grids.mdg_library import square_with_orthogonal_fractures


def _grid_for_dim(dim: int) -> pp.Grid:
    """Return a simple Cartesian grid of the given spatial dimension."""
    g = pp.CartGrid([3, 3] if dim == 2 else [2, 2, 2])
    g.compute_geometry()
    return g


def _zeros_for(space: OperatorSpace) -> np.ndarray:
    """A zero array whose size matches ``space``'s actual DOF count.

    Used throughout this test module so that constructing a ``DenseArray``
    with a given (grid-based) space always passes the shape-consistency
    check added to :class:`DenseArray`/:class:`SparseArray`, regardless of
    the concrete grids used to build that space.
    """
    return np.zeros(space.num_dofs())


def _ones_for(space: OperatorSpace) -> np.ndarray:
    """Like :func:`_zeros_for`, but filled with ones."""
    return np.ones(space.num_dofs())


def _eye_for(target: OperatorSpace, source: OperatorSpace) -> sps.spmatrix:
    """An identity-like (possibly rectangular) sparse matrix whose shape matches
    ``(target.num_dofs(), source.num_dofs())``, for use with :class:`SparseArray`.
    """
    return sps.eye(target.num_dofs(), source.num_dofs(), format="csr")


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
    mdg, _ = square_with_orthogonal_fractures(
        grid_type="cartesian",
        meshing_args={"cell_size": 0.5},
        fracture_indices=[0],
        size=1.0,
    )
    return mdg


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


class TestOperatorSpaceScalar:
    def test_scalar_factory(self):
        s = OperatorSpace.scalar()
        assert s.domain_type == DomainType.scalar
        assert s.grids == ()
        assert s.dof_info == {}

    def test_scalar_is_singleton_value(self):
        """Two calls to scalar() return equal (but not necessarily identical)
        objects."""
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
        assert hash(s1) == hash(s2)
        d = {s1: "value"}
        assert d[s2] == "value"


class TestOperatorSpaceInequality:
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


class TestOperatorProperties:
    def test_operator_requires_explicit_source_and_target(self):
        with pytest.raises(TypeError):
            Operator(name="test")

    def test_none_source_target_raises(self, two_subdomains):
        space = OperatorSpace.from_domains([two_subdomains[0]], {GridEntity.cells: 1})
        with pytest.raises(TypeError):
            Operator(name="test", source=None, target=space)
        with pytest.raises(TypeError):
            Operator(name="test", source=space, target=None)

    def test_set_source_target_in_init(self, two_subdomains):
        space = OperatorSpace.from_domains([two_subdomains[0]], {GridEntity.cells: 1})
        op = Operator(name="test", source=space, target=space)
        assert op.source == space
        assert op.target == space


class TestScalarSpace:
    def test_scalar_has_scalar_space(self):
        s = Scalar(3.14)
        assert s.source == OperatorSpace.scalar()
        assert s.target == OperatorSpace.scalar()

    def test_scalar_scalar_gives_scalar_space(self):
        s1 = Scalar(1.0)
        s2 = Scalar(2.0)
        result = s1 + s2
        assert result.source == OperatorSpace.scalar()
        assert result.target == OperatorSpace.scalar()

    @pytest.mark.parametrize("fixture_name", ["two_subdomains", "one_mortar"])
    def test_scalar_with_subdomain_or_interface(self, request, fixture_name):
        grids = request.getfixturevalue(fixture_name)
        grids = grids if isinstance(grids, (list, tuple)) else [grids]
        grid = list(grids)[0]
        s = Scalar(1.0, domains=[grid])
        assert s.source.grids == (grid,)
        assert s.source == s.target
        expected_domain_type = (
            DomainType.subdomains
            if isinstance(grid, pp.Grid)
            else DomainType.interfaces
        )
        assert s.source.domain_type == expected_domain_type

    def test_scalar_domain_is_cellwise(self, two_subdomains):
        """Domain-bearing Scalar uses the natural cell-based space on its grids."""
        s = Scalar(1.0, domains=two_subdomains[0])
        assert s.source.dof_info == {GridEntity.cells: 1}

    def test_scalar_neg_propagates_source(self, two_subdomains):
        s = Scalar(3.0, domains=two_subdomains[0])
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
        s = Scalar(2.0, domains=two_subdomains[0])
        v = Variable("p", {GridEntity.cells: 1}, two_subdomains[0])
        result = s * v
        assert result.source == v.source
        assert result.target == v.target

    def test_domainless_scalar_combined_with_operator(self, two_subdomains):
        """Plain Scalar still inherits from the other operand."""
        s = Scalar(2.0)
        v = Variable("p", {GridEntity.cells: 1}, two_subdomains[0])
        result = s * v
        assert result.source == v.source
        assert result.target == v.target


class TestVariableSpace:
    def test_variable_has_subdomain_space(self, two_subdomains):
        var = Variable("p", {GridEntity.cells: 1}, two_subdomains[0])
        assert var.source.domain_type == DomainType.subdomains
        assert var.source == var.target

    def test_variable_space_contains_correct_grid(self, two_subdomains):
        var = Variable("p", {GridEntity.cells: 1}, two_subdomains[0])
        assert var.source.grids == (two_subdomains[0],)

    def test_variable_space_dof_info(self, two_subdomains):
        var = Variable(
            "u", {GridEntity.cells: 2, GridEntity.faces: 1}, two_subdomains[0]
        )
        assert var.source.dof_info == {GridEntity.cells: 2, GridEntity.faces: 1}

    def test_variable_on_mortar_grid_has_interface_space(self, one_mortar):
        """Variable on a MortarGrid gets DomainType.interfaces."""
        var = Variable("lam", {GridEntity.cells: 2}, one_mortar)
        assert var.source.domain_type == DomainType.interfaces
        assert var.source.grids == (one_mortar,)
        assert var.source.dof_info == {GridEntity.cells: 2}


class TestMixedDimensionalVariableSpace:
    """MixedDimensionalVariable carries the union space of its sub-variables."""

    @pytest.fixture
    def md_var(self, two_subdomains):
        g1, g2 = two_subdomains
        v1 = Variable("p", {GridEntity.cells: 1}, g1)
        v2 = Variable("p", {GridEntity.cells: 1}, g2)
        return MixedDimensionalVariable([v1, v2])

    def test_md_variable_source_target_are_union(self, md_var, two_subdomains):
        expected = OperatorSpace.from_domains(two_subdomains, {GridEntity.cells: 1})
        assert md_var.source == expected
        assert md_var.target == expected


class TestSurrogateOperatorSpace:
    """SurrogateOperator carries source/range derived from its dof_info."""

    @pytest.fixture
    def simple_mdg(self, two_subdomains):
        return pp.meshing.subdomains_to_mdg([two_subdomains])

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
        assert op.source.domain_type == DomainType.subdomains
        assert op.source.dof_info == {GridEntity.cells: 1}
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
        assert op.source.dof_info == {GridEntity.cells: 1}
        assert op.target.dof_info == {GridEntity.cells: 1}


class TestDenseArraySpace:
    def test_dense_array_with_explicit_space(self, two_subdomains):
        g, _ = two_subdomains
        space = OperatorSpace.from_domains([g], {GridEntity.cells: 1})
        arr = DenseArray(_ones_for(space), source=space, target=space)
        assert arr.source == space
        assert arr.target == space


class TestSparseArraySpace:
    def test_sparse_array_with_explicit_space(self, two_subdomains):
        g, _ = two_subdomains
        space = OperatorSpace.from_domains([g], {GridEntity.cells: 1})
        mat = sps.eye(4, format="csr")
        op = SparseArray(mat, source=space, target=space)
        assert op.source == space
        assert op.target == space


class TestShapeConsistencyCheck:
    """Tests for the shape-consistency.

    This check verifies, at construction time, that a grid-based ``source``/``target``
    space actually predicts the same number of degrees of freedom as the wrapped
    array/matrix's shape.
    """

    @pytest.fixture(autouse=True)
    def setup(self, two_subdomains):
        g, _ = two_subdomains
        self.g = g
        self.cell_space = OperatorSpace.from_domains([g], {GridEntity.cells: 1})
        self.face_space = OperatorSpace.from_domains([g], {GridEntity.faces: 1})

    def test_dense_array_consistent_shape_passes(self):
        # g has 4 cells; this must not raise.
        DenseArray(
            np.ones(self.g.num_cells), source=self.cell_space, target=self.cell_space
        )

    def test_dense_array_inconsistent_shape_raises(self):
        with pytest.raises(ValueError, match="degrees of freedom"):
            DenseArray(
                np.ones(self.g.num_cells + 1),  # g has 4 cells, not 5.
                source=self.cell_space,
                target=self.cell_space,
            )

    def test_dense_array_inconsistent_source_raises(self):
        """Source and target need not be equal, but each must independently be
        consistent with the array's size."""
        with pytest.raises(ValueError, match="degrees of freedom"):
            DenseArray(
                np.ones(self.g.num_cells),
                source=self.face_space,
                target=self.cell_space,
            )

    def test_sparse_array_consistent_shape_passes(self):
        mat = _eye_for(self.cell_space, self.face_space)
        # Rows (target) match cell_space, columns (source) match face_space.
        SparseArray(mat, source=self.face_space, target=self.cell_space)

    def test_sparse_array_inconsistent_target_raises(self):
        # Shape's row count (g.num_faces) does not match cell_space's DOF count.
        mat = _eye_for(self.face_space, self.face_space)
        with pytest.raises(ValueError, match="degrees of freedom"):
            SparseArray(mat, source=self.face_space, target=self.cell_space)

    def test_sparse_array_inconsistent_source_raises(self):
        mat = _eye_for(self.cell_space, self.cell_space)
        with pytest.raises(ValueError, match="degrees of freedom"):
            SparseArray(mat, source=self.cell_space, target=self.face_space)

    @pytest.mark.parametrize(
        "space",
        [OperatorSpace.scalar(), OperatorSpace.unclear(), OperatorSpace.waived()],
    )
    def test_spaces_skip_check(self, space):
        """Spaces that are not grid-based (scalar, unclear) skip the shape check."""
        arr = np.ones(2)
        DenseArray(arr, source=space, target=space)
        mat = sps.eye(3, format="csr")
        SparseArray(mat, source=space, target=space)


class TestTimeDependentDenseArraySpaces:
    def test_no_dof_info_gives_default(self, two_subdomains):
        g1, g2 = two_subdomains
        arr = pp.ad.TimeDependentDenseArray("x", [g1, g2])
        # When domains are provided but no dof_info, cells:1 is assumed.
        assert arr.source.dof_info == {GridEntity.cells: 1}
        assert arr.target.dof_info == {GridEntity.cells: 1}

    def test_dof_info_cells(self, two_subdomains):
        g1, g2 = two_subdomains
        arr = pp.ad.TimeDependentDenseArray(
            "x", [g1, g2], dof_info={GridEntity.cells: 1}
        )
        assert arr.source == arr.target
        assert GridEntity.cells in arr.source.dof_info
        assert set(arr.source.grids) == {g1, g2}

    def test_dof_info_faces(self, two_subdomains):
        g1, g2 = two_subdomains
        arr = pp.ad.TimeDependentDenseArray(
            "x", [g1, g2], dof_info={GridEntity.faces: 2}
        )
        assert arr.source.dof_info == {GridEntity.faces: 2}

    def test_empty_domains_and_domain_type_raises(self):
        with pytest.raises(
            ValueError, match="Either domains or domain_type must be provided"
        ):
            pp.ad.TimeDependentDenseArray("x", [], dof_info={GridEntity.cells: 1})


class TestGridOperatorSpaces:
    """Tests of operator spaces for the various grid operators.

    The general structure of the tests is to create a grid operator on a full or subset
    of {subdomain, interface, boundary} grids, and then check that the source and target
    OperatorSpace have the expected domain_type, grids, and dof_info. The
    parametrization is made to cover the relevant combinations for the different
    operators.
    """

    @pytest.mark.parametrize("restriction", [True, False])
    @pytest.mark.parametrize("cell", [True, False])
    @pytest.mark.parametrize("target_dims", [[2], [1, 2], []])
    @pytest.mark.parametrize("proj_dim", [1, 2])
    def test_subdomain_projections(
        self, fracture_mdg, restriction, cell, target_dims, proj_dim
    ):
        mdg = fracture_mdg
        sds = [sd for tg in target_dims for sd in mdg.subdomains(dim=tg)]
        proj = pp.ad.SubdomainProjections(subdomains=mdg.subdomains(), dim=proj_dim)
        if restriction:
            op = proj.cell_restriction(sds) if cell else proj.face_restriction(sds)
            assert op.source.grids == tuple(mdg.subdomains())
            assert op.target.grids == tuple(sds)
        else:
            op = proj.cell_prolongation(sds) if cell else proj.face_prolongation(sds)
            assert op.source.grids == tuple(sds)
            assert op.target.grids == tuple(mdg.subdomains())
        assert op.source.domain_type == DomainType.subdomains
        assert op.target.domain_type == DomainType.subdomains
        if cell:
            assert op.source.dof_info == {GridEntity.cells: proj_dim}
            assert op.target.dof_info == {GridEntity.cells: proj_dim}
        else:
            assert op.source.dof_info == {GridEntity.faces: proj_dim}
            assert op.target.dof_info == {GridEntity.faces: proj_dim}

    @pytest.mark.parametrize("to_interface", [True, False])
    @pytest.mark.parametrize("target_dims", [[1], [0, 1], []])
    @pytest.mark.parametrize("proj_dim", [1, 2])
    def test_mortar_projection_spaces(
        self, fracture_mdg, to_interface: bool, target_dims, proj_dim
    ):
        mdg = fracture_mdg
        interfaces = [intf for tg in target_dims for intf in mdg.interfaces(dim=tg)]
        subdomains = list(
            set(
                sd
                for intf in interfaces
                for sd in mdg.interface_to_subdomain_pair(intf)
            )
        )
        proj = pp.ad.MortarProjections(
            mdg=mdg, subdomains=subdomains, interfaces=interfaces, dim=proj_dim
        )
        if to_interface:
            op = proj.primary_to_mortar_avg()
            order = [op.source, op.target]
        else:
            op = proj.mortar_to_primary_avg()
            order = [op.target, op.source]
        # The order of source and target is so that we can use the same tests for
        # directions.
        assert order[0].grids == tuple(subdomains)
        assert order[1].grids == tuple(interfaces)
        assert order[0].domain_type == DomainType.subdomains
        assert order[1].domain_type == DomainType.interfaces
        assert order[0].dof_info == {GridEntity.faces: proj_dim}
        assert order[1].dof_info == {GridEntity.cells: proj_dim}

    @pytest.mark.parametrize("to_boundary", [True, False])
    @pytest.mark.parametrize("proj_dim", [1, 2])
    def test_boundary_projection_spaces(
        self, fracture_mdg, to_boundary: bool, proj_dim: int
    ):
        mdg = fracture_mdg
        subdomains = mdg.subdomains()
        boundary_grids = mdg.boundaries()
        bp = pp.ad.BoundaryProjection(mdg, subdomains, dim=proj_dim)

        if to_boundary:
            op = bp.subdomain_to_boundary
            order = [op.source, op.target]
        else:
            op = bp.boundary_to_subdomain
            order = [op.target, op.source]
        # The order of source and target is so that we can use the same tests for
        # directions.
        assert order[0].domain_type == DomainType.subdomains
        assert order[1].domain_type == DomainType.boundary_grids
        assert order[0].grids == tuple(subdomains)
        assert order[1].grids == tuple(boundary_grids)
        assert order[0].dof_info == {GridEntity.faces: proj_dim}
        assert order[1].dof_info == {GridEntity.cells: proj_dim}

    @pytest.mark.parametrize("target_dims", [[0, 1, 2], [2], []])
    def test_trace_operator_spaces(self, fracture_mdg, target_dims):
        mdg = fracture_mdg
        subdomains = [sd for tg in target_dims for sd in mdg.subdomains(dim=tg)]
        trace = pp.ad.Trace(subdomains)
        op = trace.trace
        for domain in [op.source, op.target]:
            assert domain.domain_type == DomainType.subdomains
            assert domain.grids == tuple(subdomains)

        assert op.source.dof_info == {GridEntity.cells: 1}
        assert op.target.dof_info == {GridEntity.faces: 1}

    @pytest.mark.parametrize("target_dims", [[0, 1, 2], [2], []])
    @pytest.mark.parametrize("proj_dims", [1, 2])
    def test_divergence_operator_spaces(self, fracture_mdg, target_dims, proj_dims):
        mdg = fracture_mdg
        subdomains = [sd for tg in target_dims for sd in mdg.subdomains(dim=tg)]
        div = pp.ad.Divergence(subdomains, dim=proj_dims)
        for domain in [div.source, div.target]:
            assert domain.domain_type == DomainType.subdomains
            assert domain.grids == tuple(subdomains)

        assert div.source.dof_info == {GridEntity.faces: proj_dims}
        assert div.target.dof_info == {GridEntity.cells: proj_dims}

    @pytest.mark.parametrize("target_dims", [[1], []])
    def test_sign_of_mortar_sides_spaces(self, fracture_mdg, target_dims):
        """MortarProjections.sign_of_mortar_sides has a typed-but-empty space (not
        None) when there are no interfaces.
        """
        mdg = fracture_mdg
        interfaces = [intf for tg in target_dims for intf in mdg.interfaces(dim=tg)]
        subdomains = list(
            set(
                sd
                for intf in interfaces
                for sd in mdg.interface_to_subdomain_pair(intf)
            )
        )
        proj = pp.ad.MortarProjections(
            mdg=mdg, subdomains=subdomains, interfaces=interfaces, dim=1
        )
        op = proj.sign_of_mortar_sides()
        for domain in [op.source, op.target]:
            assert domain.domain_type == DomainType.interfaces
            assert domain.grids == tuple(interfaces)
            assert domain.dof_info == {GridEntity.cells: 1}

    def test_boundary_projection_spaces_no_subdomains(self, fracture_mdg):
        """BoundaryProjection has a typed-but-empty space (not None) when there are no
        subdomains."""
        mdg = fracture_mdg
        bp = pp.ad.BoundaryProjection(mdg, [], dim=1)
        op = bp.subdomain_to_boundary
        assert op.source.domain_type == DomainType.subdomains
        assert op.source.grids == ()
        assert op.target.domain_type == DomainType.boundary_grids
        assert op.target.grids == ()


class TestMergedOperatorSpaces:
    """Tests that MergedOperator inherits source/range from the
    underlying discretization's get_row/col_dof_info methods."""

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
            nd=1,
            physics_key="flow",
            domains=[g1, g2],
        )
        assert GridEntity.faces in op.source.dof_info
        assert GridEntity.cells in op.target.dof_info
        assert set(op.source.grids) == {g1, g2}
        assert set(op.target.grids) == {g1, g2}


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
        return DenseArray(_zeros_for(cell_space), source=cell_space, target=cell_space)

    @pytest.fixture
    def face_op(self, face_space):
        """A leaf operator with source=target=face_space."""
        return DenseArray(_zeros_for(face_space), source=face_space, target=face_space)

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
        result = pp.ad.Scalar(42) - cell_op
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

        projected = SparseArray(
            _eye_for(cell_op.target, face_op.source),
            source=face_op.source,
            target=cell_op.target,
        )
        result = binary_op(cell_op, projected)
        assert result.source == OperatorSpace.unclear()
        assert result.target == cell_op.target

    def test_elementwise_uses_union_of_dependency_domains(self, two_subdomains):
        """Different but compatible-looking domains still become unclear."""
        g1, g2 = two_subdomains
        top_space = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        union_space = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})

        local = DenseArray(_zeros_for(top_space), source=top_space, target=top_space)
        projected = SparseArray(
            _eye_for(top_space, union_space), source=union_space, target=top_space
        )

        result = local * projected

        assert result.source == OperatorSpace.unclear()
        assert result.target == top_space

    def test_elementwise_different_grids_becomes_unclear(self, two_subdomains):
        """Different grids are enough to make the inferred domain unclear."""
        g1, g2 = two_subdomains
        left_space = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        right_space = OperatorSpace.from_domains([g2], {GridEntity.cells: 1})
        left = DenseArray(_zeros_for(left_space), source=left_space, target=left_space)
        right = SparseArray(
            _eye_for(left_space, right_space), source=right_space, target=left_space
        )

        result = left + right

        assert result.source == OperatorSpace.unclear()
        assert result.target == left_space

    def test_elementwise_unclear_domain_propagates(self, cell_space, face_space):
        """Once unclear, the elementwise result remains unclear."""
        unclear = SparseArray(
            _eye_for(cell_space, cell_space),
            source=OperatorSpace.unclear(),
            target=cell_space,
        )
        known = SparseArray(
            _eye_for(cell_space, face_space), source=face_space, target=cell_space
        )

        result = unclear + known

        assert result.source == OperatorSpace.unclear()
        assert result.target == cell_space

    # --- matmul: compatible ---

    def test_matmul_compatible(self, cell_space, face_space):
        """A @ B where target(B) == source(A) is valid."""
        A = SparseArray(
            _eye_for(cell_space, face_space), source=face_space, target=cell_space
        )
        B = SparseArray(
            _eye_for(face_space, cell_space), source=cell_space, target=face_space
        )
        result = A @ B
        assert result.source == cell_space
        assert result.target == cell_space

    def test_matmul_incompatible(self, cell_space, face_space):
        """A @ B where target(B) != source(A) raises ValueError."""
        A = SparseArray(
            _eye_for(cell_space, face_space), source=face_space, target=cell_space
        )
        B = SparseArray(
            _eye_for(cell_space, face_space), source=face_space, target=cell_space
        )
        with pytest.raises(ValueError, match="matrix multiplication"):
            _ = A @ B

    def test_matmul_different_grids_incompatible(self, two_subdomains):
        """Matrix multiplication requires exact space equality, including grids."""
        g1, g2 = two_subdomains
        left_domain = OperatorSpace.from_domains([g1], {GridEntity.faces: 1})
        right_range = OperatorSpace.from_domains([g2], {GridEntity.faces: 1})
        left_range = OperatorSpace.from_domains([g1], {GridEntity.cells: 1})
        right_domain = OperatorSpace.from_domains([g2], {GridEntity.cells: 1})
        left = SparseArray(
            _eye_for(left_range, left_domain), source=left_domain, target=left_range
        )
        right = SparseArray(
            _eye_for(right_range, right_domain),
            source=right_domain,
            target=right_range,
        )

        with pytest.raises(ValueError, match="matrix multiplication"):
            _ = left @ right

    def test_matmul_with_unclear_left_source_raises(self, cell_space, face_space):
        """A left operand with unclear source cannot be used in matmul."""
        unclear = SparseArray(
            sps.eye(cell_space.num_dofs(), format="csr"),
            source=OperatorSpace.unclear(),
            target=cell_space,
        )
        rhs = DenseArray(_zeros_for(face_space), source=face_space, target=face_space)

        with pytest.raises(ValueError, match="left operand.*source is unclear"):
            _ = unclear @ rhs

    def test_rmatmul_with_unclear_right_operand_raises(self, cell_space, face_space):
        """The operator on the right-hand side of rmatmul cannot have unclear source."""
        unclear = SparseArray(
            sps.eye(cell_space.num_dofs(), format="csr"),
            source=OperatorSpace.unclear(),
            target=cell_space,
        )
        lhs = SparseArray(
            _eye_for(face_space, face_space), source=face_space, target=face_space
        )

        with pytest.raises(ValueError, match="left operand.*source is unclear"):
            _ = lhs.__rmatmul__(unclear)

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


class TestCompoundOperatorSpaces:
    """Tests that source/target propagates correctly through multi-step expressions."""

    @pytest.fixture(autouse=True)
    def setup(self, two_subdomains, spaces):
        g1, g2 = two_subdomains
        self.cell_sp = spaces[0]
        self.face_sp = spaces[1]
        self.A = SparseArray(
            _eye_for(self.cell_sp, self.face_sp),
            source=self.face_sp,
            target=self.cell_sp,
        )
        self.B = SparseArray(
            _eye_for(self.face_sp, self.cell_sp),
            source=self.cell_sp,
            target=self.face_sp,
        )
        self.v = DenseArray(
            _zeros_for(self.face_sp), source=self.face_sp, target=self.face_sp
        )

    @pytest.fixture
    def spaces(self, two_subdomains):
        g1, g2 = two_subdomains
        cell_sp = OperatorSpace.from_domains([g1, g2], {GridEntity.cells: 1})
        face_sp = OperatorSpace.from_domains([g1, g2], {GridEntity.faces: 1})
        return cell_sp, face_sp

    def test_chained_matmul_source_target(self):
        """(A @ B): target(B) == source(A) → result.source=B.source, result.target=A.target"""
        result = self.A @ self.B
        assert result.source == self.cell_sp
        assert result.target == self.cell_sp

    def test_three_way_matmul(self):
        """(A @ B) @ C propagates spaces through two matmul steps."""
        AB = self.A @ self.B
        assert AB.source == self.cell_sp
        assert AB.target == self.cell_sp
        ABA = AB @ self.A
        assert ABA.source == self.face_sp
        assert ABA.target == self.cell_sp

    def test_chained_matmul_incompatible_raises(self):
        """(A @ B) @ C raises ValueError when target(C) != source(A@B)."""

        AB = self.A @ self.B  # source=cell_sp, target=cell_sp
        with pytest.raises(ValueError, match="matrix multiplication"):
            _ = AB @ self.B

    def test_add_after_matmul(self):
        """(A @ v) + (B @ w) where both results have the same range."""
        Av_1 = self.A @ self.v
        Av_2 = self.A @ self.v
        result = Av_1 + Av_2
        assert result.source == self.face_sp
        assert result.target == self.cell_sp

    def test_add_matmul_incompatible_raises(self):
        """(A @ v) + (B @ w) where ranges differ raises ValueError."""
        Av = self.A @ self.v
        w = DenseArray(
            _zeros_for(self.cell_sp), source=self.cell_sp, target=self.cell_sp
        )
        Bw = self.B @ w
        with pytest.raises(ValueError):
            _ = Av + Bw

    def test_scalar_mul_after_matmul(self):
        """Scalar(k) * (A @ v) preserves A's range as the result range."""
        Av = self.A @ self.v
        result = Scalar(2.0) * Av
        assert result.source == self.face_sp
        assert result.target == self.cell_sp

    def test_unary_minus_preserves_spaces(self):
        """Unary minus on SparseArray preserves source/target."""
        result = -self.A
        assert result.source == self.face_sp
        assert result.target == self.cell_sp

    def test_unary_minus_dense_array_preserves_spaces(self):
        """DenseArray.__neg__ must also preserve source/target (separate code path)."""
        arr = DenseArray(
            _ones_for(self.cell_sp), source=self.cell_sp, target=self.cell_sp
        )
        result = -arr
        assert result.source == self.cell_sp
        assert result.target == self.cell_sp


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
