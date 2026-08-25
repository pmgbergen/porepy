"""Tests for the AD representations of grid-related operators: projections between
subdomains, mortar (interface) grids and boundary grids, plus the Trace and
Divergence operators.

``TestSubdomainProjections``, ``TestMortarProjections`` and ``TestBoundaryProjection``
(and, more narrowly, ``test_trace``/``test_divergence``) each check two things about
their operator: numerical correctness of the assembled projection matrix, and that
``source``/``target`` (an ``OperatorSpace``) reports the expected ``DomainType``,
grids and ``dof_info``.

Covers, in order (see the ``# MARK:`` comments below for navigation):
  - Subdomain restriction/prolongation, for both faces and cells:
    ``TestSubdomainProjections``.
  - Mortar<->primary/secondary subdomain projections, including empty-list and
    ``sign_of_mortar_sides`` edge cases: ``TestMortarProjections``.
  - Subdomain<->boundary-grid projections and their consistency:
    ``TestBoundaryProjection``.
  - The discrete Trace and Divergence operators: ``test_trace``, ``test_divergence``.

"""

from typing import Literal

import numpy as np
import pytest
import scipy.sparse as sps

import porepy as pp


@pytest.fixture
def mdg():
    """Provide a mixed-dimensional grid for the tests."""
    fracs = [np.array([[0, 2], [1, 1]]), np.array([[1, 1], [0, 2]])]
    md_grid = pp.meshing.cart_grid(fracs, np.array([2, 2]))
    return md_grid


# MARK: Subdomain restriction/prolongation ----------------------------------------

#: (SubdomainProjections method name, GridEntity, is_restriction) for each of the four
#: cell/face x restriction/prolongation combinations tested in TestSubdomainProjections.
_SUBDOMAIN_PROJECTION_CASES = [
    pytest.param(
        "cell_restriction", pp.ad.GridEntity.cells, True, id="cell_restriction"
    ),
    pytest.param(
        "cell_prolongation", pp.ad.GridEntity.cells, False, id="cell_prolongation"
    ),
    pytest.param(
        "face_restriction", pp.ad.GridEntity.faces, True, id="face_restriction"
    ),
    pytest.param(
        "face_prolongation", pp.ad.GridEntity.faces, False, id="face_prolongation"
    ),
]


def _known_subdomain_projection_matrix(
    subdomains: list[pp.Grid],
    target_grids: list[pp.Grid],
    proj_dim: int,
    entity: Literal["cells", "faces"],
) -> sps.csr_matrix:
    """Construct the known restriction/prolongation matrix between the full
    subdomains`` list and the concatenated ``target_grids`` list, for the given
    ``entity`` and ``proj_dim``.

    Parameters:
        subdomains: List of grids to project from/to.
        target_grids: List of grids to project to/from.
        proj_dim: Number of DOFs per cell/face.
        entity: "cells" or "faces".

    Returns:
        Sparse matrix.
    """
    is_cell = entity == pp.ad.GridEntity.cells
    counts = np.array([sd.num_cells if is_cell else sd.num_faces for sd in subdomains])
    starts = np.cumsum(np.hstack((0, counts)))
    n_total = counts.sum() * proj_dim

    row_blocks, col_blocks, data_blocks = [], [], []
    offset = 0
    for sd in target_grids:
        ind = _list_ind_of_grid(subdomains, sd)
        n = sd.num_cells if is_cell else sd.num_faces
        cols = pp.array_operations.expand_indices_nd(
            np.arange(starts[ind], starts[ind + 1]), proj_dim
        )
        row_blocks.append(np.arange(n * proj_dim) + offset)
        col_blocks.append(cols)
        data_blocks.append(np.ones(n * proj_dim))
        offset += n * proj_dim

    return sps.coo_matrix(
        (np.hstack(data_blocks), (np.hstack(row_blocks), np.hstack(col_blocks))),
        shape=(offset, n_total),
    ).tocsr()


class TestSubdomainProjections:
    """Restriction and prolongation, for both faces and cells, of
    ``pp.ad.SubdomainProjections``.

    Covers three scenarios (one per test method below): projecting to/from an empty
    grid list, a single grid, and a combined list of two grids. For each scenario we
    check the shape of the projection matrix, and compare it to a known matrix assembled
    independently of the implementation under test.

    """

    @pytest.fixture(params=[True, False], ids=["scalar", "vector"])
    def proj_dim(self, request, mdg):
        return 1 if request.param else mdg.dim_max()

    @pytest.fixture
    def subdomains(self, mdg):
        return mdg.subdomains()

    @pytest.fixture
    def proj(self, subdomains, proj_dim):
        return pp.ad.SubdomainProjections(subdomains=subdomains, dim=proj_dim)

    @pytest.fixture
    def sizes(self, mdg, proj_dim):
        """(n_cells, n_faces): total DOF counts across all subdomains, at proj_dim."""
        n_cells, n_faces, _ = geometry_information(mdg, proj_dim)
        return n_cells, n_faces

    def _check_space(
        self, op, subdomains, proj_dim, entity, target_grids, is_restriction
    ):
        """Check source/target of a single restriction or prolongation operator.

        Parameters:
            op: The restriction or prolongation operator.
            subdomains: The full list of subdomains the projection was built on.
            proj_dim: DOFs per grid entity.
            entity: The GridEntity (cells or faces) the projection acts on.
            target_grids: The (sub-)list of grids passed to the restriction or
                prolongation call.
            is_restriction: True for a restriction (full -> subset), False for a
                prolongation (subset -> full).

        """
        assert op.source.domain_type == pp.ad.DomainType.subdomains
        assert op.target.domain_type == pp.ad.DomainType.subdomains
        wide, narrow = (
            (op.source, op.target) if is_restriction else (op.target, op.source)
        )
        assert wide.grids == tuple(subdomains)
        assert narrow.grids == tuple(target_grids)
        assert op.source.dof_info == {entity: proj_dim}
        assert op.target.dof_info == {entity: proj_dim}

    @pytest.mark.parametrize(
        "method_name, entity, is_restriction", _SUBDOMAIN_PROJECTION_CASES
    )
    def test_empty_subdomain_list(
        self, proj, subdomains, proj_dim, sizes, method_name, entity, is_restriction
    ):
        """Restriction/prolongation to/from an empty list of subdomains has the
        expected degenerate shape, and still reports a typed OperatorSpace (with all
        subdomains on the "wide" side, and an empty grid list on the "narrow" side).
        """
        n_cells, n_faces = sizes
        n = n_cells if entity == pp.ad.GridEntity.cells else n_faces
        expected_shape = (0, n) if is_restriction else (n, 0)

        op = getattr(proj, method_name)([])

        assert op.shape == expected_shape
        self._check_space(op, subdomains, proj_dim, entity, [], is_restriction)

    @pytest.mark.parametrize("sd_index", range(4), ids=["sd0", "sd1", "sd2", "sd3"])
    @pytest.mark.parametrize(
        "method_name, entity, is_restriction", _SUBDOMAIN_PROJECTION_CASES
    )
    def test_single_grid(
        self,
        proj,
        subdomains,
        proj_dim,
        method_name,
        entity,
        is_restriction,
        sd_index,
    ):
        """Restriction/prolongation between the full subdomain list and a single grid
        matches a manually assembled 0/1 projection matrix, and reports the grids
        ``[sd]`` on the "narrow" side of source/target.

        The ``sd_index`` parametrization assumes the ``mdg`` fixture always yields
        exactly four subdomains (one each of dim 2, 1, 1, 0).
        """
        sd = subdomains[sd_index]
        known = _known_subdomain_projection_matrix(subdomains, [sd], proj_dim, entity)
        expected = known if is_restriction else known.T

        op = getattr(proj, method_name)([sd])

        assert _compare_matrices(op, expected)
        self._check_space(op, subdomains, proj_dim, entity, [sd], is_restriction)

    @pytest.mark.parametrize(
        "method_name, entity, is_restriction", _SUBDOMAIN_PROJECTION_CASES
    )
    def test_combined_grids(
        self, proj, subdomains, mdg, proj_dim, method_name, entity, is_restriction
    ):
        """Restriction/prolongation between the full subdomain list and a list of two
        grids stacks their individual 0/1 projection blocks in the given order.
        """
        g1, g2 = mdg.subdomains(dim=1)
        known = _known_subdomain_projection_matrix(
            subdomains, [g1, g2], proj_dim, entity
        )
        expected = known if is_restriction else known.T

        op = getattr(proj, method_name)([g1, g2])

        assert _compare_matrices(op, expected)
        self._check_space(op, subdomains, proj_dim, entity, [g1, g2], is_restriction)


# MARK: Mortar<->primary/secondary subdomain projections ---------------------------

# (method_name, pair, kind, is_to_mortar) for each of the eight mortar<->primary/
# secondary projection directions tested in TestMortarProjections.test_projection.
# - method_name: the name of the SubdomainProjections method to call.
# - pair: "primary" or "secondary" (the higher- or lower-dimensional neighbor of an
# interface).
# - kind: "int" or "avg" (integration or averaging variant).
# - is_to_mortar: True if the projection is from a subdomain to the mortar, False
#   otherwise.
_MORTAR_PROJECTION_CASES = [
    pytest.param(
        "mortar_to_primary_int", "primary", "int", False, id="mortar_to_primary_int"
    ),
    pytest.param(
        "mortar_to_primary_avg", "primary", "avg", False, id="mortar_to_primary_avg"
    ),
    pytest.param(
        "primary_to_mortar_int", "primary", "int", True, id="primary_to_mortar_int"
    ),
    pytest.param(
        "primary_to_mortar_avg", "primary", "avg", True, id="primary_to_mortar_avg"
    ),
    pytest.param(
        "mortar_to_secondary_int",
        "secondary",
        "int",
        False,
        id="mortar_to_secondary_int",
    ),
    pytest.param(
        "mortar_to_secondary_avg",
        "secondary",
        "avg",
        False,
        id="mortar_to_secondary_avg",
    ),
    pytest.param(
        "secondary_to_mortar_int",
        "secondary",
        "int",
        True,
        id="secondary_to_mortar_int",
    ),
    pytest.param(
        "secondary_to_mortar_avg",
        "secondary",
        "avg",
        True,
        id="secondary_to_mortar_avg",
    ),
]

#: The four "mortar_to_X_int"/"X_to_mortar_int" directions used by the empty-list
#: tests below. These only ever exercise the integration variant, mirroring the
#: original (pre-refactor) test.
_MORTAR_INT_ONLY_CASES = [
    pytest.param("mortar_to_primary_int", "primary", False, id="mortar_to_primary"),
    pytest.param("primary_to_mortar_int", "primary", True, id="primary_to_mortar"),
    pytest.param(
        "mortar_to_secondary_int", "secondary", False, id="mortar_to_secondary"
    ),
    pytest.param(
        "secondary_to_mortar_int", "secondary", True, id="secondary_to_mortar"
    ),
]

#: Four representative (subdomain, interface) subsets, covering: the base case of all
#: grids; a subset of subdomains; a case with a non-zero projection to the primary but
#: not the secondary; and a case with a zero (but correctly shaped) projection in both
#: directions.
_MORTAR_GRID_SUBSET_CASES = [
    pytest.param(lambda mdg: mdg.subdomains(), lambda mdg: mdg.interfaces(), id="all"),
    pytest.param(
        lambda mdg: mdg.subdomains(dim=2) + mdg.subdomains(dim=1),
        lambda mdg: mdg.interfaces(dim=1),
        id="2d+1d_subdomains-1d_interfaces",
    ),
    pytest.param(
        lambda mdg: mdg.subdomains(dim=2),
        lambda mdg: mdg.interfaces(dim=1),
        id="2d_subdomain-1d_interfaces",
    ),
    pytest.param(
        lambda mdg: mdg.subdomains(dim=2),
        lambda mdg: mdg.interfaces(dim=0),
        id="2d_subdomain-0d_interfaces",
    ),
]


def _known_mortar_projection_matrices(mdg, subdomains, interfaces, proj_dim):
    """Assemble the known mortar<->primary/secondary projection matrices (both the
    integration and averaging variants) for the given lists of subdomains and
    interfaces, by placing each interface's own local mortar_to_primary/secondary_
    int/avg matrices into the right blocks of the global matrix.

    Returns:
        A dict ``{(pair, kind): matrix}`` with ``pair`` in ``{"primary",
        "secondary"}`` and ``kind`` in ``{"int", "avg"}``, each mapping mortar-cell
        DOFs (columns) to the corresponding primary (subdomain faces) or secondary
        (subdomain cells) DOFs (rows).

    """

    def _row_offset_primary(subdomains):
        return proj_dim * np.cumsum(
            np.hstack((0, np.array([sd.num_faces for sd in subdomains])))
        )

    def _row_offset_secondary(subdomains):
        return proj_dim * np.cumsum(
            np.hstack((0, np.array([sd.num_cells for sd in subdomains])))
        )

    def _col_offset(interfaces):
        return proj_dim * np.cumsum(
            np.hstack((0, np.array([m.num_cells for m in interfaces])))
        )

    def _indices_primary(intf):
        row, col, data_int = sps.find(intf.mortar_to_primary_int(nd=proj_dim))
        _, _, data_avg = sps.find(intf.mortar_to_primary_avg(nd=proj_dim))
        return row, col, data_int, data_avg

    def _indices_secondary(intf):
        row, col, data_int = sps.find(intf.mortar_to_secondary_int(nd=proj_dim))
        _, _, data_avg = sps.find(intf.mortar_to_secondary_avg(nd=proj_dim))
        return row, col, data_int, data_avg

    face_start = _row_offset_primary(subdomains)
    cell_start = _row_offset_secondary(subdomains)
    mortar_start = _col_offset(interfaces)

    row_ind_primary, col_ind_primary = [], []
    row_ind_secondary, col_ind_secondary = [], []
    data_primary_int, data_primary_avg = [], []
    data_secondary_int, data_secondary_avg = [], []

    # Loop over the interfaces and subdomains to collect the indices and data of the
    # projection matrices, provided that the subdomain is involved in the projection.
    for intf in interfaces:
        sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)
        if sd_primary in subdomains:
            r, c, data_int, data_avg = _indices_primary(intf)
            row_ind_primary.append(r + face_start[subdomains.index(sd_primary)])
            col_ind_primary.append(c + mortar_start[interfaces.index(intf)])
            data_primary_int.append(data_int)
            data_primary_avg.append(data_avg)
        if sd_secondary in subdomains:
            r, c, data_int, data_avg = _indices_secondary(intf)
            row_ind_secondary.append(r + cell_start[subdomains.index(sd_secondary)])
            col_ind_secondary.append(c + mortar_start[interfaces.index(intf)])
            data_secondary_int.append(data_int)
            data_secondary_avg.append(data_avg)

    shape_primary = (
        proj_dim * sum(sd.num_faces for sd in subdomains),
        proj_dim * sum(m.num_cells for m in interfaces),
    )
    shape_secondary = (
        proj_dim * sum(sd.num_cells for sd in subdomains),
        proj_dim * sum(m.num_cells for m in interfaces),
    )

    def _assemble(row_ind, col_ind, data, shape):
        # If no data is collected (none of the given subdomains were neighbors of the
        # interfaces), the projection matrix is zero, with the given shape.
        if len(row_ind) == 0:
            return sps.csr_matrix(shape)
        return sps.coo_matrix(
            (np.hstack(data), (np.hstack(row_ind), np.hstack(col_ind))), shape=shape
        ).tocsr()

    return {
        ("primary", "int"): _assemble(
            row_ind_primary, col_ind_primary, data_primary_int, shape_primary
        ),
        ("primary", "avg"): _assemble(
            row_ind_primary, col_ind_primary, data_primary_avg, shape_primary
        ),
        ("secondary", "int"): _assemble(
            row_ind_secondary, col_ind_secondary, data_secondary_int, shape_secondary
        ),
        ("secondary", "avg"): _assemble(
            row_ind_secondary, col_ind_secondary, data_secondary_avg, shape_secondary
        ),
    }


class TestMortarProjections:
    """Projections between mortar (interface) grids and their neighboring primary
    (higher-dimensional) and secondary (lower-dimensional) subdomain grids, via
    ``pp.ad.MortarProjections``.

    Covers, in order: projections to/from an empty list of subdomains and/or
    interfaces; ``sign_of_mortar_sides``; and the main projection scenario, checked
    both for numerical correctness (assembled independently from each interface's own
    local projection matrices) and for the OperatorSpace (source/target: DomainType,
    grids, dof_info) reported by the projection operator.

    """

    @pytest.fixture(params=[True, False], ids=["scalar", "vector"])
    def proj_dim(self, request, mdg):
        return 1 if request.param else mdg.dim_max()

    def _check_space(self, op, subdomains, interfaces, proj_dim, pair, is_to_mortar):
        entity = pp.ad.GridEntity.faces if pair == "primary" else pp.ad.GridEntity.cells
        sd_space, mortar_space = (
            (op.source, op.target) if is_to_mortar else (op.target, op.source)
        )
        assert sd_space.domain_type == pp.ad.DomainType.subdomains
        assert sd_space.grids == tuple(subdomains)
        assert sd_space.dof_info == {entity: proj_dim}
        assert mortar_space.domain_type == pp.ad.DomainType.interfaces
        assert mortar_space.grids == tuple(interfaces)
        assert mortar_space.dof_info == {pp.ad.GridEntity.cells: proj_dim}

    @pytest.mark.parametrize("method_name, pair, is_to_mortar", _MORTAR_INT_ONLY_CASES)
    def test_empty_subdomains(self, mdg, method_name, pair, is_to_mortar):
        """With an empty subdomain list, mortar<->primary/secondary projections have
        zero rows or columns (whichever side is the subdomains), but are non-zero
        along the mortar dimension, and still report a typed-but-empty (not None)
        OperatorSpace on the subdomains side.
        """
        interfaces = mdg.interfaces()
        _, _, n_mortar_cells = geometry_information(mdg, 1)
        proj = pp.ad.MortarProjections(
            subdomains=[], interfaces=interfaces, mdg=mdg, dim=1
        )
        op = getattr(proj, method_name)()

        expected_shape = (n_mortar_cells, 0) if is_to_mortar else (0, n_mortar_cells)
        assert op.shape == expected_shape
        subdomain_space = op.source if is_to_mortar else op.target
        assert subdomain_space.domain_type == pp.ad.DomainType.subdomains
        assert subdomain_space.grids == ()

    @pytest.mark.parametrize("method_name, pair, is_to_mortar", _MORTAR_INT_ONLY_CASES)
    def test_empty_interfaces(self, mdg, method_name, pair, is_to_mortar):
        """With an empty interface list, mortar<->primary/secondary projections have
        zero rows or columns (whichever side is the mortar), but are non-zero along
        the subdomain dimension, and still report a typed-but-empty (not None)
        OperatorSpace on the interfaces side.
        """
        subdomains = mdg.subdomains()
        n_cells, n_faces, _ = geometry_information(mdg, 1)
        n = n_faces if pair == "primary" else n_cells
        proj = pp.ad.MortarProjections(
            subdomains=subdomains, interfaces=[], mdg=mdg, dim=1
        )
        op = getattr(proj, method_name)()

        expected_shape = (0, n) if is_to_mortar else (n, 0)
        assert op.shape == expected_shape
        mortar_space = op.target if is_to_mortar else op.source
        assert mortar_space.domain_type == pp.ad.DomainType.interfaces
        assert mortar_space.grids == ()

    def test_empty_subdomains_and_interfaces(self, mdg):
        """With both lists empty, all four projections are trivially (0, 0)."""
        proj = pp.ad.MortarProjections(subdomains=[], interfaces=[], mdg=mdg, dim=1)
        assert proj.mortar_to_primary_int().shape == (0, 0)
        assert proj.mortar_to_secondary_int().shape == (0, 0)
        assert proj.primary_to_mortar_int().shape == (0, 0)
        assert proj.secondary_to_mortar_int().shape == (0, 0)

    def test_sign_of_mortar_sides(self, mdg):
        """sign_of_mortar_sides carries an interfaces-typed space with one DOF per
        cell, matching the (diagonal) sign matrix it wraps.
        """
        interfaces = mdg.interfaces()
        proj = pp.ad.MortarProjections(
            subdomains=mdg.subdomains(), interfaces=interfaces, mdg=mdg, dim=1
        )
        op = proj.sign_of_mortar_sides()
        assert op.source == op.target
        assert op.source.domain_type == pp.ad.DomainType.interfaces
        assert op.source.grids == tuple(interfaces)
        assert op.source.dof_info == {pp.ad.GridEntity.cells: 1}

    def test_sign_of_mortar_sides_empty_interfaces(self, mdg):
        """sign_of_mortar_sides also gets a typed-but-empty space on an empty
        interface list."""
        proj = pp.ad.MortarProjections(
            subdomains=mdg.subdomains(), interfaces=[], mdg=mdg, dim=1
        )
        op = proj.sign_of_mortar_sides()
        assert op.source.domain_type == pp.ad.DomainType.interfaces
        assert op.source.grids == ()
        assert op.source == op.target

    @pytest.mark.parametrize("non_matching", [True, False])
    @pytest.mark.parametrize(
        "int_method, avg_method",
        [
            ("mortar_to_primary_int", "mortar_to_primary_avg"),
            ("primary_to_mortar_int", "primary_to_mortar_avg"),
            ("mortar_to_secondary_int", "mortar_to_secondary_avg"),
            ("secondary_to_mortar_int", "secondary_to_mortar_avg"),
        ],
        ids=[
            "mortar_to_primary",
            "primary_to_mortar",
            "mortar_to_secondary",
            "secondary_to_mortar",
        ],
    )
    def test_projections_are_cached(self, mdg, non_matching, int_method, avg_method):
        """The projection matrices are constructed lazily, then stored and reused.

        Across a conforming interface, the integrating and the averaging variant of a
        direction are the same matrix, and share a single cached one; across a
        non-conforming interface the two differ and are cached separately. Refining
        the 1d grids makes the mortar-to-secondary mappings non-conforming, while the
        primary side stays conforming.

        Parameters:
            non_matching: If True, the 1d subdomain grids are refined, making the
                secondary side of the 1d interfaces non-conforming.

        """
        if non_matching:
            for g in mdg.subdomains(dim=1):
                g_new = pp.refinement.refine_grid_1d(g, ratio=2)
                mdg.replace_subdomains_and_interfaces({g: g_new})

        proj = pp.ad.MortarProjections(
            subdomains=mdg.subdomains(), interfaces=mdg.interfaces(), mdg=mdg, dim=1
        )
        integrated = getattr(proj, int_method)()
        # A second call returns the stored matrix rather than constructing a new one.
        assert getattr(proj, int_method)() is integrated
        # Separate "secondary" methods iff non_matching. Negate to get shared.
        shares_storage = not (non_matching and "secondary" in int_method)
        assert (getattr(proj, avg_method)() is integrated) == shares_storage

    @pytest.mark.parametrize("non_matching", [True, False])
    @pytest.mark.parametrize(
        "subdomain_selector, interface_selector", _MORTAR_GRID_SUBSET_CASES
    )
    @pytest.mark.parametrize(
        "method_name, pair, kind, is_to_mortar", _MORTAR_PROJECTION_CASES
    )
    def test_projection(
        self,
        mdg,
        proj_dim,
        non_matching,
        subdomain_selector,
        interface_selector,
        method_name,
        pair,
        kind,
        is_to_mortar,
    ):
        """Mortar<->primary/secondary subdomain projections match matrices assembled
        independently from each interface's own local mortar_to_primary/secondary_
        int/avg matrices, and report the expected OperatorSpace.

        The X_to_mortar directions are the transpose of the mortar_to_X direction,
        with integration and averaging swapped; this is a property of the underlying
        mortar projections, not an artifact of this test (see
        :func:`_known_mortar_projection_matrices`'s docstring).

        Parameters:
            non_matching: If True, the 1d subdomain grids are refined so that the
                mappings from 1d mortars to their secondary subdomains have
                non-unitary entries.
            subdomain_selector, interface_selector: Together pick one of four
                representative (subdomain, interface) subsets, see
                ``_MORTAR_GRID_SUBSET_CASES``.

        """
        if non_matching:
            # Refine the two 1d grids, such that the projection matrices have
            # non-unitary entries.
            for g in mdg.subdomains(dim=1):
                g_new = pp.refinement.refine_grid_1d(g, ratio=2)
                mdg.replace_subdomains_and_interfaces({g: g_new})

        subdomains = subdomain_selector(mdg)
        interfaces = interface_selector(mdg)

        known = _known_mortar_projection_matrices(mdg, subdomains, interfaces, proj_dim)
        other_kind = {"int": "avg", "avg": "int"}
        if is_to_mortar:
            expected = known[(pair, other_kind[kind])].T
        else:
            expected = known[(pair, kind)]

        proj = pp.ad.MortarProjections(
            subdomains=subdomains, interfaces=interfaces, mdg=mdg, dim=proj_dim
        )
        op = getattr(proj, method_name)()

        assert _compare_matrices(expected, op)
        self._check_space(op, subdomains, interfaces, proj_dim, pair, is_to_mortar)


# MARK: Subdomain<->boundary-grid projections --------------------------------------


class TestBoundaryProjection:
    """Subdomain<->boundary-grid projections, via ``pp.ad.BoundaryProjection``.

    Covers, in order: the shape of the full-subdomain-list projection; the
    per-subdomain contribution to that projection (which boundary faces each
    subdomain touches); that subdomain_to_boundary and boundary_to_subdomain are
    transposes of each other; and restricting the projection to a subset of
    subdomains (shape and per-subdomain contribution, mirroring the full-list
    checks).

    """

    @pytest.fixture(params=[True, False], ids=["scalar", "vector"])
    def proj_dim(self, request, mdg):
        return 1 if request.param else mdg.dim_max()

    @pytest.fixture
    def subdomains(self, mdg):
        # Compute geometry for the mixed-dimensional grid. This is needed for the
        # boundary projection operator.
        mdg.compute_geometry()
        return mdg.subdomains()

    @pytest.fixture
    def projection(self, mdg, subdomains, proj_dim):
        return pp.ad.BoundaryProjection(mdg, subdomains, proj_dim)

    @pytest.fixture
    def subset(self, subdomains):
        """A subset of two subdomains (the 2d grid and one of the 1d fractures),
        used to test restricting BoundaryProjection to fewer than all subdomains.
        """
        g_0 = [sd for sd in subdomains if sd.dim == 2][0]
        g_1 = [sd for sd in subdomains if sd.dim == 1][0]
        return [g_0, g_1]

    @pytest.fixture
    def subset_projection(self, mdg, subset, proj_dim):
        return pp.ad.BoundaryProjection(mdg, subset, proj_dim)

        assert s2b.source.domain_type == pp.ad.DomainType.subdomains
        assert s2b.source.grids == tuple(subdomains)
        assert s2b.source.dof_info == {pp.ad.GridEntity.faces: proj_dim}
        assert s2b.target.domain_type == pp.ad.DomainType.boundary_grids
        assert s2b.target.grids == tuple(mdg.boundaries())
        assert s2b.target.dof_info == {pp.ad.GridEntity.cells: proj_dim}
        assert b2s.source == s2b.target
        assert b2s.target == s2b.source

    @pytest.mark.parametrize(
        "sd_index, expected_sum_factor",
        [(0, 8), (1, 2), (2, 2), (3, 0)],
        ids=["top_dim", "fracture_1", "fracture_2", "intersection"],
    )
    def test_per_subdomain_contribution(
        self, mdg, subdomains, proj_dim, projection, sd_index, expected_sum_factor
    ):
        """Each subdomain's column block of subdomain_to_boundary has the expected
        number of nonzero entries: 8 boundary faces for the top-dimensional grid
        (``g_0``), 2 for each of the two fractures (each touches the boundary on two
        sides), and 0 for the 0d intersection (trivially, since it has no faces).
        """
        subdomain_to_boundary = projection.subdomain_to_boundary.parse(mdg)
        starts = np.cumsum(
            np.hstack((0, [sd.num_faces * proj_dim for sd in subdomains]))
        )
        block = subdomain_to_boundary[:, starts[sd_index] : starts[sd_index + 1]]
        assert np.sum(block) == expected_sum_factor * proj_dim

    def test_boundary_to_subdomain_is_transpose(self, mdg, projection):
        """subdomain_to_boundary and boundary_to_subdomain are transposes of each
        other, for the full list of subdomains."""
        subdomain_to_boundary = projection.subdomain_to_boundary.parse(mdg)
        boundary_to_subdomain = projection.boundary_to_subdomain.parse(mdg)
        assert np.allclose((subdomain_to_boundary - boundary_to_subdomain.T).data, 0)

    def test_subset_of_grids_shape(self, mdg, subset, proj_dim, subset_projection):
        """Restricting BoundaryProjection to a subset of subdomains gives a
        subdomain_to_boundary/boundary_to_subdomain shape consistent with that
        subset.
        """
        num_faces = proj_dim * sum(sd.num_faces for sd in subset)
        num_cells = proj_dim * sum(
            mdg.subdomain_to_boundary_grid(sd).num_cells for sd in subset
        )
        subdomain_to_boundary = subset_projection.subdomain_to_boundary.parse(mdg)
        boundary_to_subdomain = subset_projection.boundary_to_subdomain.parse(mdg)

        assert subdomain_to_boundary.shape == (num_cells, num_faces)
        assert boundary_to_subdomain.shape == (num_faces, num_cells)

    @pytest.mark.parametrize(
        "sd_index, expected_sum_factor",
        [(0, 8), (1, 2)],
        ids=["top_dim", "fracture_1"],
    )
    def test_subset_of_grids_per_subdomain_contribution(
        self,
        mdg,
        subset,
        proj_dim,
        subset_projection,
        sd_index,
        expected_sum_factor,
    ):
        """Restricting BoundaryProjection to a subset of subdomains gives the same
        per-subdomain contributions as the full-grid-list case, for the subdomains
        that remain (cf. test_per_subdomain_contribution).
        """
        subdomain_to_boundary = subset_projection.subdomain_to_boundary.parse(mdg)
        starts = np.cumsum(np.hstack((0, [sd.num_faces * proj_dim for sd in subset])))
        block = subdomain_to_boundary[:, starts[sd_index] : starts[sd_index + 1]]
        assert np.sum(block) == expected_sum_factor * proj_dim

        assert s2b_op.source.grids == tuple(subset)
        assert s2b_op.target.grids == tuple(
            mdg.subdomain_to_boundary_grid(sd) for sd in subset
        )

    def test_empty_subdomain_list(self, mdg):
        """BoundaryProjection has a typed-but-empty space (not None) when constructed
        on an empty list of subdomains.
        """
        mdg.compute_geometry()
        projection = pp.ad.BoundaryProjection(mdg, [], dim=1)
        op = projection.subdomain_to_boundary
        assert op.source.domain_type == pp.ad.DomainType.subdomains
        assert op.source.grids == ()
        assert op.target.domain_type == pp.ad.DomainType.boundary_grids
        assert op.target.grids == ()


# MARK: Trace and Divergence operators ----------------------------------------------
def test_trace(mdg: pp.MixedDimensionalGrid):
    """Test Trace operator.

    Parameters:
        mdg: Mixed-dimensional grid.

    This test is not ideal. It follows the implementation of Trace relatively closely,
    but nevertheless provides some coverage, especially if Trace is carelessly changed.
    The test constructs the expected mixed-dimensional trace matrix and compares it to
    the ones of Trace. Also checks that an error is raised if a non-scalar trace is
    constructed (not implemented), and that source/target report DomainType.subdomains
    with the trace's dof_info (one DOF per cell in the source, per face in the
    target).
    """
    # The operator should work on any subset of mdg.subdomains.
    subdomains = mdg.subdomains(dim=1)

    # Construct expected matrices.
    traces = []

    # Contruct projections to the subdomains for cell and face quantities.
    cell_projections = pp.ad.grid_operators._cell_projections(subdomains, dim=1)
    for sd in subdomains:
        local_block = np.abs(sd.cell_faces.tocsr())
        traces.append(local_block * cell_projections[sd].T)

    # Compare to operator class.
    op = pp.ad.Trace(subdomains)
    assert _compare_matrices(op.trace, sps.bmat([[m] for m in traces]))

    assert op.trace.source.domain_type == pp.ad.DomainType.subdomains
    assert op.trace.target.domain_type == pp.ad.DomainType.subdomains
    assert op.trace.source.grids == tuple(subdomains) == op.trace.target.grids
    assert op.trace.source.dof_info == {pp.ad.GridEntity.cells: 1}
    assert op.trace.target.dof_info == {pp.ad.GridEntity.faces: 1}

    # As of the writing of this test, Trace is not implemented for vector values. If it
    # is ever extended, the test should be extended accordingly (e.g. parametrized with
    # dim=[1, 2]).
    with pytest.raises(NotImplementedError):
        pp.ad.Trace(subdomains, dim=2)

    # Trace also gets a typed-but-empty space (not None) on an empty subdomain list.
    empty_op = pp.ad.Trace([])
    assert empty_op.trace.source.domain_type == pp.ad.DomainType.subdomains
    assert empty_op.trace.source.grids == ()
    assert empty_op.trace.target.domain_type == pp.ad.DomainType.subdomains
    assert empty_op.trace.target.grids == ()


@pytest.mark.parametrize("dim", [1, 4])
def test_divergence(mdg: pp.MixedDimensionalGrid, dim: int):
    """Test Divergence.

    Parameters:
        mdg: Mixed-dimensional grid.
        dim: Dimension of vector field to which Divergence is applied.

    This test is not ideal. It follows the implementation of Divergence relatively
    closely, but nevertheless provides some coverage. Frankly, there is not much more to
    do than comparing against the expected matrices, unless one wants to add more
    integration-type tests e.g. evaluating combinations with other ad entities.

    Also checks that source/target report DomainType.subdomains with one DOF per face
    in the source and per cell in the target (matching the *constructed* dim, which -
    note - is left at its default of 1 below regardless of the ``dim`` parameter; see
    the dof_info assertion).

    """
    # The operator should work on any subset of mdg.subdomains.
    subdomains = mdg.subdomains(dim=2) + mdg.subdomains(dim=0)

    # Construct expected matrix.
    divergences = list()
    for sd in subdomains:
        # Kron does no harm if dim=1
        local_block = sps.kron(sd.cell_faces.tocsr().T, sps.eye(dim))
        divergences.append(local_block)

    # Compare to operators parsed value.
    op = pp.ad.Divergence(subdomains, dim=dim)
    val = op.parse(mdg)
    assert _compare_matrices(val, sps.block_diag(divergences))

    assert op.source.domain_type == pp.ad.DomainType.subdomains
    assert op.target.domain_type == pp.ad.DomainType.subdomains
    assert op.source.grids == tuple(subdomains) == op.target.grids
    assert op.source.dof_info == {pp.ad.GridEntity.faces: 1}
    assert op.target.dof_info == {pp.ad.GridEntity.cells: 1}

    # Divergence also gets a typed-but-empty space (not None) on an empty subdomain
    # list.
    empty_op = pp.ad.Divergence([])
    assert empty_op.source.grids == () == empty_op.target.grids


def _compare_matrices(m1, m2):
    """Compare two sparse matrices.

    Parameters:
        m1: Sparse matrix or SparseArray.
        m2: Sparse matrix or SparseArray.

    Returns:
        bool: True if the matrices are equal.

    """
    # Convert ad sparse arrays to scipy sparse matrices if necessary. Then call the
    # standard comparison function for matrices.
    if isinstance(m1, pp.ad.SparseArray):
        m1 = m1._mat
    if isinstance(m2, pp.ad.SparseArray):
        m2 = m2._mat
    return pp.applications.test_utils.arrays.compare_matrices(m1, m2)


def _list_ind_of_grid(subdomains: list[pp.Grid], g: pp.Grid) -> int:
    """Get the index of a grid in a list of grids.

    Parameters:
        subdomains: List of grids.
        g: Grid.

    Returns:
        Index of grid in list.

    Raises:
        ValueError: If grid is not in list.

    """
    for i, gl in enumerate(subdomains):
        if g == gl:
            return i
    raise ValueError("grid is not in list")


def geometry_information(
    mdg: pp.MixedDimensionalGrid, dim: int
) -> tuple[int, int, int]:
    """Geometry information used in multiple test methods.

    Parameters:
        mdg: Mixed-dimensional grid.
        dim: Dimension. Each of the return values is multiplied by dim.

    Returns:
        n_cells: Number of subdomain cells.
        n_faces: Number of subdomain faces.
        n_mortar_cells: Number of interface cells.

    """
    n_cells = sum([sd.num_cells for sd in mdg.subdomains()]) * dim
    n_faces = sum([sd.num_faces for sd in mdg.subdomains()]) * dim
    n_mortar_cells = sum([intf.num_cells for intf in mdg.interfaces()]) * dim
    return n_cells, n_faces, n_mortar_cells
