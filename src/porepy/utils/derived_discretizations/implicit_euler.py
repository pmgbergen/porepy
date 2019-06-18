"""
Module for extending the Upwind, MPFA and MassMatrix discretizations in Porepy to handle
implicit Euler time-stepping. Flux terms are multiplied by time step and the mass term
has a rhs contribution from the previous time step.
See the parent discretizations for further documentation.
"""
import porepy as pp
import numpy as np
import scipy.sparse as sps


class ImplicitMassMatrix(pp.MassMatrix):
    """
    Return rhs contribution based on the previous solution, which is stored in the
    pp.STATE field of the data dictionary.
    """

    def __init__(self, keyword="flow", variable="pressure"):
        """ Set the discretization, with the keyword used for storing various
        information associated with the discretization. The time discretisation also
        requires the previous solution, thus the variable needs to be specified.

        Paramemeters:
            keyword (str): Identifier of all information used for this
                discretization.
        """
        super().__init__(keyword)
        self.variable = variable

    def assemble_rhs(self, g, data):
        """ Overwrite MassMatrix method to return the correct rhs for an IE time
        discretization, e.g. of the Biot problem.
        """
        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        previous_solution = data[pp.STATE][self.variable]

        return matrix_dictionary["mass"] * previous_solution


class ImplicitMpfa(pp.Mpfa):
    """
    Multiply all contributions by the time step.
    """

    def assemble_matrix_rhs(self, g, data):
        """ Overwrite MPFA method to be consistent with the Biot dt convention.
        """
        a, b = super().assemble_matrix_rhs(g, data)
        dt = data[pp.PARAMETERS][self.keyword]["time_step"]
        a = a * dt
        b = b * dt
        return a, b

    def assemble_int_bound_flux(
        self, g, data, data_edge, grid_swap, cc, matrix, rhs, self_ind
    ):
        """
        Overwrite the MPFA method to be consistent with the Biot dt convention
        """
        dt = data[pp.PARAMETERS][self.keyword]["time_step"]

        div = g.cell_faces.T

        bound_flux = data[pp.DISCRETIZATION_MATRICES][self.keyword]["bound_flux"]
        # Projection operators to grid
        mg = data_edge["mortar_grid"]

        if grid_swap:
            proj = mg.mortar_to_slave_int()
        else:
            proj = mg.mortar_to_master_int()

        if g.dim > 0 and bound_flux.shape[0] != g.num_faces:
            # If bound flux is gven as sub-faces we have to map it from sub-faces
            # to faces
            hf2f = pp.fvutils.map_hf_2_f(nd=1, g=g)
            bound_flux = hf2f * bound_flux
        if g.dim > 0 and bound_flux.shape[1] != proj.shape[0]:
            raise ValueError(
                """Inconsistent shapes. Did you define a
            sub-face boundary condition but only a face-wise mortar?"""
            )

        cc[self_ind, 2] += dt * div * bound_flux * proj

    def assemble_int_bound_source(
        self, g, data, data_edge, grid_swap, cc, matrix, rhs, self_ind
    ):
        """ Abstract method. Assemble the contribution from an internal
        boundary, manifested as a source term.

        The intended use is when the internal boundary is coupled to another
        node in a mixed-dimensional method. Specific usage depends on the
        interface condition between the nodes; this method will typically be
        used to impose flux continuity on a lower-dimensional domain.

        Implementations of this method will use an interplay between the grid on
        the node and the mortar grid on the relevant edge.

        Parameters:
            g (Grid): Grid which the condition should be imposed on.
            data (dictionary): Data dictionary for the node in the
                mixed-dimensional grid.
            data_edge (dictionary): Data dictionary for the edge in the
                mixed-dimensional grid.
            grid_swap (boolean): If True, the grid g is identified with the @
                slave side of the mortar grid in data_adge.
            cc (block matrix, 3x3): Block matrix for the coupling condition.
                The first and second rows and columns are identified with the
                master and slave side; the third belongs to the edge variable.
                The discretization of the relevant term is done in-place in cc.
            matrix (block matrix 3x3): Discretization matrix for the edge and
                the two adjacent nodes.
            rhs (block_array 3x1): Right hand side contribution for the edge and
                the two adjacent nodes.
            self_ind (int): Index in cc and matrix associated with this node.
                Should be either 1 or 2.

        """
        mg = data_edge["mortar_grid"]

        if grid_swap:
            proj = mg.mortar_to_master_int()
        else:
            proj = mg.mortar_to_slave_int()
        dt = data[pp.PARAMETERS][self.keyword]["time_step"]
        cc[self_ind, 2] -= proj * dt


class ImplicitUpwind(pp.Upwind):
    """
    Multiply all contributions by the time step.
    """

    def assemble_matrix_rhs(self, g, data, d_name="darcy_flux"):
        """
        Implicit in time
        """
        if g.dim == 0:
            data["flow_faces"] = sps.csr_matrix([0.0])
            return sps.csr_matrix([0.0]), np.array([0.0])

        parameter_dictionary = data[pp.PARAMETERS][self.keyword]
        dt = parameter_dictionary["time_step"]

        a, b = super().assemble_matrix_rhs(g, data, d_name)
        a = a * dt
        b = b * dt
        return a, b
