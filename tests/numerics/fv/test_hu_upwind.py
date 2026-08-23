"""Tests for the two-direction hybrid upwind discretizations.

Checks that each direction of :class:`~porepy.numerics.fv.upwind.HUpwind` matches the
classic :class:`~porepy.numerics.fv.upwind.Upwind` under a mat-vec, that the sparsity
pattern survives flow reversals, and that the coupling and AD wrappers expose both
directions.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.fv.upwind import _single_point_upwind_matrices


def _base_upwind_matrix(sd, flux, bc):
    """Discretize the classic single-point :class:`Upwind` for one flux array."""
    upwind = pp.Upwind("transport")
    data = pp.initialize_data(
        {}, "transport", {"bc": bc, "darcy_flux": flux, "num_components": 1}
    )
    upwind.discretize(sd, data)
    return data[pp.DISCRETIZATION_MATRICES]["transport"][upwind.upwind_matrix_key]


def _discretize_hupwind(sd, gamma_flux, delta_flux, bc, num_components=1):
    data = pp.initialize_data(
        {},
        "hybrid_upwind",
        {
            "hybrid_gamma_flux": gamma_flux,
            "hybrid_delta_flux": delta_flux,
            "bc": bc,
            "num_components": num_components,
        },
    )
    discr = pp.HUpwind("hybrid_upwind")
    discr.discretize(sd, data)
    return data[pp.DISCRETIZATION_MATRICES]["hybrid_upwind"]


def _grid():
    sd = pp.CartGrid([4, 3])
    sd.compute_geometry()
    return sd


def test_hupwind_matches_base_upwind_per_direction():
    """Each hybrid direction reproduces the base Upwind mat-vec for that direction."""
    sd = _grid()
    bc = pp.BoundaryCondition(sd, sd.get_boundary_faces(), "dir")
    rng = np.random.default_rng(0)
    gamma_flux = rng.standard_normal(sd.num_faces)
    delta_flux = rng.standard_normal(sd.num_faces)

    md = _discretize_hupwind(sd, gamma_flux, delta_flux, bc)
    base_gamma = _base_upwind_matrix(sd, gamma_flux, bc)
    base_delta = _base_upwind_matrix(sd, delta_flux, bc)

    x = rng.standard_normal(sd.num_cells)
    # The fixed-sparsity form carries explicit zeros on the downstream cell, so the
    # matrices are not structurally identical, but the action on any vector is.
    np.testing.assert_allclose(md["transport_gamma"] @ x, base_gamma @ x)
    np.testing.assert_allclose(md["transport_delta"] @ x, base_delta @ x)


def test_hupwind_direction_independence():
    """Opposite-signed directions upwind opposite cells face-by-face."""
    sd = _grid()
    bc = pp.BoundaryCondition(sd, sd.get_boundary_faces(), "dir")
    flux = np.ones(sd.num_faces)
    md = _discretize_hupwind(sd, +flux, -flux, bc)
    # gamma upwinds with +flux, delta with -flux: on interior faces they select the two
    # different neighbour cells, so the row patterns differ.
    diff = (md["transport_gamma"] - md["transport_delta"]).tocsr()
    diff.eliminate_zeros()
    assert diff.nnz > 0


def test_hupwind_fixed_sparsity_under_sign_flip():
    """Flipping the flow direction leaves the sparsity pattern unchanged (data swaps).

    Uses Neumann boundaries so the set of dropped (boundary-handled) faces does not
    depend on the flow direction; then only the interior data may change, never the
    structural pattern.
    """
    sd = _grid()
    bc = pp.BoundaryCondition(sd, sd.get_boundary_faces(), "neu")
    flux = np.linspace(-1.0, 1.0, sd.num_faces)

    md_pos = _discretize_hupwind(sd, flux, flux, bc)
    md_neg = _discretize_hupwind(sd, -flux, -flux, bc)
    a = md_pos["transport_gamma"]
    b = md_neg["transport_gamma"]
    # Same structural pattern (indptr/indices), regardless of flow direction.
    np.testing.assert_array_equal(a.indptr, b.indptr)
    np.testing.assert_array_equal(a.indices, b.indices)


def test_single_point_upwind_matrices_zero_dim():
    """The helper returns empty matrices on a 0d grid."""
    sd = pp.PointGrid(np.zeros(3))
    sd.compute_geometry()
    bc = pp.BoundaryCondition(sd)
    up, rhs_dir, rhs_neu = _single_point_upwind_matrices(
        sd, np.zeros(sd.num_faces), bc, num_components=2
    )
    assert up.shape == (0, 2)
    assert rhs_dir.shape == (0, 0)
    assert rhs_neu.shape == (0, 0)


def test_hupwind_num_components_block_structure():
    """``num_components > 1`` produces a block-expanded upwind matrix."""
    sd = _grid()
    bc = pp.BoundaryCondition(sd, sd.get_boundary_faces(), "dir")
    flux = np.ones(sd.num_faces)
    nc = 3
    md = _discretize_hupwind(sd, flux, flux, bc, num_components=nc)
    assert md["transport_gamma"].shape == (sd.num_faces * nc, sd.num_cells * nc)


def test_hupwind_coupling_builds_both_directions():
    """HUpwindCoupling builds primary/secondary/flux matrices for gamma and delta."""
    # A simple 2d domain with a single fracture -> one codim-1 interface.
    mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "cartesian",
        {"cell_size": 0.5},
        fracture_indices=[1],
    )
    intf = mdg.interfaces(codim=1)[0]
    sd_primary, sd_secondary = mdg.interface_to_subdomain_pair(intf)
    data_primary = mdg.subdomain_data(sd_primary)
    data_secondary = mdg.subdomain_data(sd_secondary)
    data_intf = mdg.interface_data(intf)

    lf = np.ones(intf.num_cells)
    pp.initialize_data(
        data_intf,
        "hybrid_upwind",
        {"hybrid_gamma_flux": +lf, "hybrid_delta_flux": -lf},
    )
    discr = pp.HUpwindCoupling("hybrid_upwind")
    discr.discretize(
        sd_primary, sd_secondary, intf, data_primary, data_secondary, data_intf
    )
    md = data_intf[pp.DISCRETIZATION_MATRICES]["hybrid_upwind"]
    for key in (
        "upwind_primary_gamma",
        "upwind_secondary_gamma",
        "flux_gamma",
        "upwind_primary_delta",
        "upwind_secondary_delta",
        "flux_delta",
        "trace",
        "inv_trace",
        "mortar_discr",
    ):
        assert key in md, f"missing coupling matrix {key}"
    # gamma/delta ride opposite signs -> their primary selectors are complementary.
    g = md["upwind_primary_gamma"].diagonal()
    d = md["upwind_primary_delta"].diagonal()
    np.testing.assert_allclose(g + d, np.ones(intf.num_cells))


def test_hupwind_ad_wrappers_expose_directions():
    """The AD wrappers expose per-direction matrices as callable MergedOperators."""
    sd = _grid()
    discr = pp.ad.HUpwindAd("hybrid_upwind", [sd])
    for attr in (
        "upwind_gamma",
        "upwind_delta",
        "bound_transport_neu_gamma",
        "bound_transport_neu_delta",
    ):
        assert callable(getattr(discr, attr))
        assert isinstance(getattr(discr, attr)(), pp.ad.MergedOperator)
