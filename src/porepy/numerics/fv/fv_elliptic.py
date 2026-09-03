"""
Module contains superclass for mpfa and tpfa.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.discretization import Discretization

from . import _fvutils


class FVElliptic(Discretization):
    """Superclass for finite volume discretizations of the elliptic equation.

    Should not be used by itself, instead use a subclass that implements an
    actual discretization method. Known subclasses are Tpfa and Mpfa.

    """

    def __init__(self, keyword):
        # Identify which parameters to use:
        self.keyword = keyword

        # Keywords used to identify individual terms in the discretization matrix
        # dictionary:
        self.flux_matrix_key = "flux"
        """Key used to store flux discretization (transmissibility matrix) in the
        discretization matrix dictionary."""
        self.bound_flux_matrix_key = "bound_flux"
        """Key used to store discretization of boundary conditions in the discretization
        matrix dictionary."""
        self.bound_pressure_cell_matrix_key = "bound_pressure_cell"
        """Key used to store discretization of boundary conditions in the discretization
        matrix dictionary. The matrix accounts for contribution of cell center values in
        reconstruction of boundary pressures."""
        # Contribution of boundary values (Neumann or Dirichlet, depending on the
        # condition set on faces) in reconstruction of boundary pressures
        self.bound_pressure_face_matrix_key = "bound_pressure_face"
        """Key used to store discretization of boundary conditions in the discretization
        matrix dictionary. The matrix accounts for contribution of boundary values
        (Neumann or Dirichlet, depending on the condition set on faces) in
        reconstruction of boundary pressures"""
        self.vector_source_matrix_key = "vector_source"
        """Key used to store discretization of vector source terms (gravity) in the
        discretization matrix dictionary."""
        self.bound_pressure_vector_source_matrix_key = "bound_pressure_vector_source"
        """Key used to store discretization of vector source terms (gravity) in the
        discretization matrix dictionary. The matrix accounts for contribution of
        vector source terms in reconstruction of boundary pressures."""

    def ndof(self, sd: pp.Grid) -> int:
        """Return the number of degrees of freedom associated to the method.

        Parameters:
            sd: A grid.

        Returns:
            int: The number of degrees of freedom.

        """
        return sd.num_cells

    def get_row_dof_info(self, matrix_key: str = "", nd: int = 1) -> pp.ad.GridEntities:
        """Return row DOF info for the named FVElliptic matrix.

        Parameters:
            matrix_key: Attribute-name fragment (e.g. ``"flux"``).
            nd: Spatial dimension.

        Raises:
            ValueError: If the matrix_key is not recognized by this discretization.

        Returns:
            A :class:`~porepy.numerics.ad.GridEntities` with the DOFs per entity.

        """

        recognised = {
            "flux",
            "bound_flux",
            "bound_pressure_cell",
            "bound_pressure_face",
            "vector_source",
            "bound_pressure_vector_source",
        }
        if matrix_key in recognised:
            return pp.ad.GridEntities(faces=1)
        raise ValueError(
            f"Unrecognized matrix key '{matrix_key}' for FVElliptic discretization."
        )

    def get_col_dof_info(self, matrix_key: str = "", nd: int = 1) -> pp.ad.GridEntities:
        """Return column DOF info for the named FVElliptic matrix.

        Parameters:
            matrix_key: Attribute-name fragment (e.g. ``"flux"``).
            nd: Spatial dimension.

        Raises:
            ValueError: If the matrix_key is not recognized by this discretization.

        Returns:
            A :class:`~porepy.numerics.ad.GridEntities` with the DOFs per entity.

        """

        mapping: dict[str, pp.ad.GridEntities] = {
            "flux": pp.ad.GridEntities(cells=1),
            "bound_flux": pp.ad.GridEntities(faces=1),
            "bound_pressure_cell": pp.ad.GridEntities(cells=1),
            "bound_pressure_face": pp.ad.GridEntities(faces=1),
            "vector_source": pp.ad.GridEntities(cells=nd),
            "bound_pressure_vector_source": pp.ad.GridEntities(cells=nd),
        }
        if matrix_key in mapping:
            return mapping[matrix_key]
        raise ValueError(
            f"Unrecognized matrix key '{matrix_key}' for FVElliptic discretization."
        )

    def assemble_matrix_rhs(
        self, sd: pp.Grid, data: dict
    ) -> tuple[sps.spmatrix, np.ndarray]:
        """Return the matrix and right-hand side for a discretization of a second
        order elliptic equation.

        Parameters:
            sd: Computational grid, with geometry fields computed.
            data: With data stored.

        Returns:
            scipy.sparse.csr_matrix: System matrix of this discretization.
            np.ndarray: Right-hand side vector with representation of boundary
                conditions.

        """
        # Dictionaries containing discretization matrices and parameters.
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        # Extract discretization matrices.
        flux = matrix_dictionary[self.flux_matrix_key]
        bound_flux = matrix_dictionary[self.bound_flux_matrix_key]

        div = sd.divergence(dim=1)

        # Assemble matrix.
        if flux.shape[0] != sd.num_faces:
            hf2f = _fvutils.map_hf_2_f(nd=1, sd=sd)
            flux = hf2f @ flux
        matrix = div @ flux

        # Assemble right-hand side.
        if sd.dim > 0 and bound_flux.shape[0] != sd.num_faces:
            hf2f = _fvutils.map_hf_2_f(nd=1, sd=sd)
            bound_flux = hf2f @ bound_flux

        rhs = -div @ bound_flux @ parameter_dictionary["bc_values"]

        # Also assemble vector sources if discretization of the vector source term if
        # specified.
        if "vector_source" in parameter_dictionary:
            vector_source_discr = matrix_dictionary[self.vector_source_matrix_key]
            vector_source = parameter_dictionary.get("vector_source")
            rhs -= div @ vector_source_discr @ vector_source

        return matrix, rhs
