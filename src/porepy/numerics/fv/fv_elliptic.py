"""
Module contains superclass for mpfa and tpfa.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.discretization import Discretization


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

        div = pp.fvutils.scalar_divergence(sd)

        # Assemble matrix.
        if flux.shape[0] != sd.num_faces:
            hf2f = pp.fvutils.map_hf_2_f(nd=1, sd=sd)
            flux = hf2f * flux
        matrix = div * flux

        # Assemble right-hand side.
        if sd.dim > 0 and bound_flux.shape[0] != sd.num_faces:
            hf2f = pp.fvutils.map_hf_2_f(nd=1, sd=sd)
            bound_flux = hf2f * bound_flux

        rhs = -div * bound_flux * parameter_dictionary["bc_values"]

        # Also assemble vector sources if discretization of the vector source term if
        # specified.
        if "vector_source" in parameter_dictionary:
            vector_source_discr = matrix_dictionary[self.vector_source_matrix_key]
            vector_source = parameter_dictionary.get("vector_source")
            rhs -= div * vector_source_discr * vector_source

        return matrix, rhs
