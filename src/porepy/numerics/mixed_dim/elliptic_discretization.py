#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:35:21 2018

@author: eke001
"""
import porepy as pp

from porepy.numerics.mixed_dim.solver import Solver

class EllipticDiscretization(Solver):
    """ This is the parent class of all discretizations for second order elliptic
    problems. The class cannot be used itself, but should rather be seen as a
    declaration of which methods are assumed implemented for all specific
    discretization schemes.

    Subclasses are intended used both on single grids, and as components in a
    mixed-dimensional, or more generally multiple grid problem. In the latter case, this class will provide the
    discretization on individual grids; a full discretization will also need
    discretizations of the edge problems (coupling between grids) and a way to
    assemble the grids.

    The class methods should together take care of both the discretization
    (defined as construction of operators used when writing down the continuous
    form of the equations, e.g. divergence, transmissibility matrix etc), and
    assembly of (whole or part of) the system matrix. For problems on multiple
    grids, an assembly will only be a partial (considering a single grid)- the
    full system matrix for multiple grids is handled somewhere else.

    Attributes:
        keyword (str): This is used for

    """


    def __init__(self, keyword):
        """ Set the discretization, with the keyword used for storing various
        information associated with the discretization.

        Paramemeters:
            keyword (str): Identifier of all information used for this
                discretization.
        """
        self.keyword = keyword

        self.physics = keyword


    def key(self):
        """ Get

        Returns:
            String, on the form self.keyword + '_'.
        """
        return self.keyword + '_'

    def ndof(self, g):
        """ Abstract method.
        Return the number of degrees of freedom associated to the method.

        Parameter
        ---------
        g: grid, or a subclass.

        Return
        ------
        dof: the number of degrees of freedom.

        """
        raise NotImplementedError("Method not implemented")

    def extract_pressure(self, g, solution_array, d):
        """ Extract the pressure part of a solution.

        The method is trivial for finite volume methods, with the pressure
        being the only primary variable.

        Parameters:
            g (grid): To which the solution array belongs.
            solution_array (np.array): Solution for this grid obtained from
                either a mono-dimensional or a mixed-dimensional problem.
            d (dictionary): Data dictionary associated with the grid. Not used,
                but included for consistency reasons.

        Returns:
            np.array (g.num_cells): Pressure solution vector. Will be identical
                to solution_array.

        """
        raise NotImplementedError("Method not implemented")


    def extract_flux(self, g, solution_array, d):
        """ Extract the flux related to a solution.

        The flux is computed from the discretization and the given pressure solution.

        @ALL: We should incrude the boundary condition as well?

        Parameters:
            g (grid): To which the solution array belongs.
            solution_array (np.array): Solution for this grid obtained from
                either a mono-dimensional or a mixed-dimensional problem. Will
                correspond to the pressure solution.
            d (dictionary): Data dictionary associated with the grid.

        Returns:
            np.array (g.num_faces): Flux vector.

        """
        raise NotImplementedError("Method not implemented")

    def assemble_matrix_rhs(self, g, data):
        raise NotImplementedError("Method not implemented")

    def assemble_matrix(self, g, data):
        """
        Return the matrix and right-hand side for a discretization of a second
        order elliptic equation using a FV method with a multi-point flux
        approximation.

        The name of data in the input dictionary (data) are:
        k : second_order_tensor
            Permeability defined cell-wise.
        bc : boundary conditions (optional)
        bc_val : dictionary (optional)
            Values of the boundary conditions. The dictionary has at most the
            following keys: 'dir' and 'neu', for Dirichlet and Neumann boundary
            conditions, respectively.

        Parameters
        ----------
        g : grid, or a subclass, with geometry fields computed.
        data: dictionary to store the data. For details on necessary keywords,
            see method discretize()

        Return
        ------
        matrix: sparse csr (g_num_cells, g_num_cells)
            Discretization matrix.

        """
        raise NotImplementedError("Method not implemented")

    # ------------------------------------------------------------------------------#

    def assemble_rhs(self, g, data):
        """
        Return the righ-hand side for a discretization of a second order elliptic
        equation using the MPFA method. See self.matrix_rhs for a detaild
        description.
        """
        raise NotImplementedError("Method not implemented")

    def assemble_int_bound_flux(self, g, data, data_edge, grid_swap, cc, matrix, self_ind):
        raise NotImplementedError("Method not implemented")

    def assemble_int_bound_source(self, g, data, data_edge, grid_swap, cc, matrix, self_ind):
        raise NotImplementedError("Method not implemented")

    def assemble_int_bound_pressure_trace(self, g, data, data_edge, grid_swap, cc, matrix, self_ind):
        raise NotImplementedError("Method not implemented")

    def assemble_int_bound_pressure_cell(self, g, data, data_edge, grid_swap, cc, matrix, self_ind):
        raise NotImplementedError("Method not implemented")


    def enforce_neumann_int_bound(self, g_master, data_edge, matrix):
        raise NotImplementedError("Method not implemented")
