"""
Mother class for all interface laws.
"""
import abc
from typing import Dict, Tuple, Union

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.discretization import Discretization


class AbstractInterfaceLaw(abc.ABC):
    """Partial implementation of an interface (between two grids) law. Any full
    interface law must implement the missing functions.

    Attributes:
        keyword (str): Used to identify the right parameter dictionary from the full
            data dictionary of this grid.
        intf_coupling_via_high_dim (boolean): If True, assembly will allow for a direct
            coupling between different edges. The class must then implement the function
            assemble_intf_coupling_via_high_dim().
        intf_coupling_via_low_dim (boolean): If True, assembly will allow for a direct
            coupling between different edges. The class must then implement the function
            assemble_intf_coupling_via_low_dim().

    """

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword
        self.intf_coupling_via_high_dim = False
        self.intf_coupling_via_low_dim = False

    def _key(self) -> str:
        return self.keyword + "_"

    def _discretization_key(self) -> str:
        return self._key() + pp.DISCRETIZATION

    @abc.abstractmethod
    def ndof(self, mg: pp.MortarGrid) -> int:
        """Get the number of degrees of freedom of this interface law for a
        given mortar grid.

        Parameters:
            mg (pp.MortarGrid): Mortar grid of an interface.

        Returns:
            int: Number of degrees of freedom.

        """
        pass

    @abc.abstractmethod
    def discretize(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        data_primary: Dict,
        data_secondary: Dict,
        data_intf: Dict,
    ) -> None:
        """Discretize the interface law and store the discretization in the
        interface data.

        The discretization matrix will be stored in the data dictionary of this
        interface.

        Parameters:
            sd_primary: Grid of the primary domanin.
            sd_secondary: Grid of the secondary domain.
            data_primary: Data dictionary for the primary domain.
            data_secondary: Data dictionary for the secondary domain.
            data_intf: Data dictionary for the edge between the domains.

        """
        pass

    def update_discretization(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        data_primary: Dict,
        data_secondary: Dict,
        data_intf: Dict,
    ) -> None:
        """Partial update of discretization.

        Intended use is when the discretization should be updated, e.g. because of
        changes in parameters, grid geometry or grid topology, and it is not
        desirable to recompute the discretization on the entire grid. A typical case
        will be when the discretization operation is costly, and only a minor update
        is necessary.

        The updates can generally come as a combination of two forms:
            1) The discretization on part of the grid should be recomputed.
            2) The old discretization can be used (in parts of the grid), but the
               numbering of unknowns has changed, and the discretization should be
               reorder accordingly.

        By default, this method will simply forward the call to the standard
        discretize method. Discretization methods that wants a tailored approach
        should override the standard implementation.


        Parameters:
            sd_primary: Grid of the primary domanin.
            sd_secondary: Grid of the secondary domain.
            data_primary: Data dictionary for the primary domain.
            data_secondary: Data dictionary for the secondary domain.
            data_intf: Data dictionary for the edge between the domains.

        """
        self.discretize(
            sd_primary, sd_secondary, intf, data_primary, data_secondary, data_intf
        )

    @abc.abstractmethod
    def assemble_matrix_rhs(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        data_primary: Dict,
        data_secondary: Dict,
        data_intf: Dict,
        matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble the dicretization of the interface law, and its impact on
        the neighboring domains.

        The matrix will be

        Parameters:
            sd_primary: Grid on one neighboring subdomain.
            sd_secondary: Grid on the other neighboring subdomain.
            data_primary: Data dictionary for the primary suddomain
            data_secondary: Data dictionary for the secondary subdomain.
            data_intf: Data dictionary for the edge between the subdomains
            matrix_primary: original discretization for the primary subdomain

        Returns:
            np.array: Block matrix of size 3 x 3, whwere each block represents
                coupling between variables on this interface. Index 0, 1 and 2
                represent the primary, secondary and mortar variable, respectively.
            np.array: Block matrix of size 3 x 1, representing the right hand
                side of this coupling. Index 0, 1 and 2 represent the primary,
                secondary and mortar variable, respectively.

        """
        pass

    def assemble_matrix(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        data_primary: Dict,
        data_secondary: Dict,
        data_intf: Dict,
        matrix: np.ndarray,
    ) -> np.ndarray:
        """Assemble the dicretization of the interface law, and its impact on
        the neighboring domains.

        The default implementation will assemble both the discretization matrix and the
        right hand side vector, and return only the former. This behavior is overridden
        by some discretization methods.

        Parameters:
            sd_primary: Grid on one neighboring subdomain.
            sd_secondary: Grid on the other neighboring subdomain.
            data_primary: Data dictionary for the primary suddomain
            data_secondary: Data dictionary for the secondary subdomain.
            data_intf: Data dictionary for the edge between the subdomains
            matrix_primary: original discretization for the primary subdomain

        Returns:
            np.array: Block matrix of size 3 x 3, whwere each block represents
                coupling between variables on this interface. Index 0, 1 and 2
                represent the primary, secondary and mortar variable, respectively.

        """
        A, _ = self.assemble_matrix_rhs(
            sd_primary,
            sd_secondary,
            intf,
            data_primary,
            data_secondary,
            data_intf,
            matrix,
        )
        return A

    def assemble_rhs(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        data_primary: Dict,
        data_secondary: Dict,
        data_intf: Dict,
        matrix: np.ndarray,
    ) -> np.ndarray:
        """Assemble the dicretization of the interface law, and its impact on
        the neighboring domains.

        The default implementation will assemble both the discretization matrix and the
        right hand side vector, and return only the latter. This behavior is overridden
        by some discretization methods.

        Parameters:
            sd_primary: Grid on one neighboring subdomain.
            sd_secondary: Grid on the other neighboring subdomain.
            data_primary: Data dictionary for the primary suddomain
            data_secondary: Data dictionary for the secondary subdomain.
            data_intf: Data dictionary for the edge between the subdomains
            matrix_primary: original discretization for the primary subdomain

        Returns:
            np.array: Block matrix of size 3 x 3, whwere each block represents
                coupling between variables on this interface. Index 0, 1 and 2
                represent the primary, secondary and mortar variable, respectively.

        """
        _, b = self.assemble_matrix_rhs(
            sd_primary,
            sd_secondary,
            intf,
            data_primary,
            data_secondary,
            data_intf,
            matrix,
        )
        return b

    def _define_local_block_matrix(
        self,
        sd_primary: pp.Grid,
        sd_secondary: pp.Grid,
        intf: pp.MortarGrid,
        discr_primary: Discretization,
        discr_secondary: Discretization,
        matrix: np.ndarray,
        create_matrix: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Initialize a block matrix and right hand side for the local linear
        system of the primary and secondary grid and the interface.

        The generated block matrix is 3x3, where each block is initialized as
        a sparse matrix with size corresponding to the number of dofs for
        the primary, secondary and mortar variables for this interface law.

        Parameters:
            sd_primary: Grid on one neighboring subdomain.
            sd_secondary: Grid on the other neighboring subdomain.
            data_primary: Data dictionary for the primary suddomain
            data_secondary: Data dictionary for the secondary subdomain.
            data_intf: Data dictionary for the edge between the subdomains
            matrix_primary: original discretization for the primary subdomain

        Returns:
            np.array: Block matrix of size 3 x 3, whwere each block represents
                coupling between variables on this interface. Index 0, 1 and 2
                represent the primary, secondary and mortar variable, respectively.
                Each of the blocks have an empty sparse matrix with size
                corresponding to the number of dofs of the grid and variable.
            np.array: Block matrix of size 3 x 1, representing the right hand
                side of this coupling. Index 0, 1 and 2 represent the primary,
                secondary and mortar variable, respectively.

        """
        primary_ind = 0
        secondary_ind = 1
        mortar_ind = 2

        dof_primary = discr_primary.ndof(sd_primary)
        dof_secondary = discr_secondary.ndof(sd_secondary)
        dof_mortar = self.ndof(intf)
        # We know the number of dofs from the primary and secondary side from their
        # discretizations
        dof = np.array([dof_primary, dof_secondary, dof_mortar])

        if create_matrix:
            if not dof_primary == matrix[primary_ind, primary_ind].shape[1]:
                raise ValueError(
                    """The number of dofs of the primary discretization given
                in the coupling discretization must match the number of dofs given by the
                matrix.
                """
                )
            elif not dof_secondary == matrix[primary_ind, secondary_ind].shape[1]:
                raise ValueError(
                    """The number of dofs of the secondary discretization given
                in the coupling discretization must match the number of dofs given by the
                matrix.
                """
                )
            elif not self.ndof(intf) == matrix[primary_ind, mortar_ind].shape[1]:
                raise ValueError(
                    """The number of dofs of the edge discretization given
                in the coupling discretization must match the number of dofs given by the
                matrix.
                """
                )
            cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
            cc = cc.reshape((3, 3))

        # The rhs is just zeros
        rhs = np.empty(3, dtype=object)
        rhs[primary_ind] = np.zeros(dof_primary)
        rhs[secondary_ind] = np.zeros(dof_secondary)
        rhs[mortar_ind] = np.zeros(dof_mortar)

        if not create_matrix:
            return rhs
        else:
            return cc, rhs

    def _define_local_block_matrix_intf_coupling(
        self,
        g: pp.Grid,
        discr_grid: Discretization,
        intf_primary: pp.MortarGrid,
        intf_secondary: pp.MortarGrid,
        matrix: np.ndarray,
        create_matrix: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Initialize a block matrix and right hand side for the local linear
        system of the primary and secondary grid and the interface.

        The generated block matrix is 3x3, where each block is initialized as
        a sparse matrix with size corresponding to the number of dofs for
        the primary, secondary and mortar variables for this interface law.

        Parameters:
            sd_primary: Grid on one neighboring subdomain.
            sd_secondary: Grid on the other neighboring subdomain.
            data_primary: Data dictionary for the primary suddomain
            data_secondary: Data dictionary for the secondary subdomain.
            data_intf: Data dictionary for the edge between the subdomains
            matrix_primary: original discretization for the primary subdomain

        Returns:
            np.array: Block matrix of size 3 x 3, whwere each block represents
                coupling between variables on this interface. Index 0, 1 and 2
                represent the primary, secondary and mortar variable, respectively.
                Each of the blocks have an empty sparse matrix with size
                corresponding to the number of dofs of the grid and variable.
            np.array: Block matrix of size 3 x 1, representing the right hand
                side of this coupling. Index 0, 1 and 2 represent the primary,
                secondary and mortar variable, respectively.

        """

        grid_ind = 0
        primary_ind = 1
        secondary_ind = 2

        dof_grid = discr_grid.ndof(g)
        dof_mortar_primary = self.ndof(intf_primary)
        dof_mortar_secondary = self.ndof(intf_secondary)
        dof = np.array([dof_grid, dof_mortar_primary, dof_mortar_secondary])

        if create_matrix:

            if not dof_grid == matrix[grid_ind, grid_ind].shape[1]:
                raise ValueError(
                    """The number of dofs of the primary discretization given
                in the coupling discretization must match the number of dofs given by the
                matrix.
                """
                )
            elif not dof_mortar_primary == matrix[grid_ind, primary_ind].shape[1]:
                raise ValueError(
                    """The number of dofs of the secondary discretization given
                in the coupling discretization must match the number of dofs given by the
                matrix.
                """
                )
            elif not dof_mortar_secondary == matrix[grid_ind, secondary_ind].shape[1]:
                raise ValueError(
                    """The number of dofs of the edge discretization given
                in the coupling discretization must match the number of dofs given by the
                matrix.
                """
                )
            # We know the number of dofs from the primary and secondary side from their
            # discretizations
            cc = np.array([sps.coo_matrix((i, j)) for i in dof for j in dof])
            cc = cc.reshape((3, 3))

        # The rhs is just zeros
        rhs = np.empty(3, dtype=object)
        rhs[grid_ind] = np.zeros(dof_grid)
        rhs[primary_ind] = np.zeros(dof_mortar_primary)
        rhs[secondary_ind] = np.zeros(dof_mortar_secondary)

        if create_matrix:
            return cc, rhs
        else:
            return rhs

    def assemble_intf_coupling_via_high_dim(  # type: ignore
        self,
        g_between: pp.Grid,
        data_between: Dict,
        intf_primary: pp.MortarGrid,
        sd_pair_primary: Tuple[pp.Grid, pp.Grid],
        data_intf_primary: Dict,
        intf_secondary: pp.MortarGrid,
        sd_pair_secondary: Tuple[pp.Grid, pp.Grid],
        data_intf_secondary: Dict,
        matrix: np.ndarray,
        assemble_matrix: bool = True,
        assemble_rhs: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Method to assemble the contribution from one interface to another one.

        The method must be implemented for subclasses of AbstractInterfaceLaw which has
        the attribute intf_coupling_via_high_dim set to True. For classes where the
        variable is False, there is no need for action.

        Note that the mixed-dimensional modeling framework does not allow for direct
        couplings between interfaces. However, there may be cases where an interface law
        is dependent on variables on the boundary between the higher dimensional
        grid and another interface. As we normally associate these boundary values
        with the variable on the secondary interface, this method is available
        as an alternative.

        For more details on how this function is invoked see pp.Assembler.
        Note that the coupling currently only can be invoked if the variables
        on the primary and secondary interface have the same name.

        Any discretization operation should be done as part of self.discretize().

        Parameters:
            g_between (pp.Grid): Grid of the higher dimensional neighbor to the
                main interface
            data_between (dict): Data dictionary of the intermediate grid.
            intf_primary (tuple of grids): The grids of the primary edge
            data_intf_primary (dict): Data dictionary of the primary interface.
            intf_secondary (tuple of grids): The grids of the secondary edge.
            data_intf_secondary (dict): Data dictionary of the secondary interface.
            matrix: original discretization.
            assemble_matrix (optional): If True (defalut), contributions to local matrix
                are assembled.
            assemble_rhs (optional): If True (defalut), contributions to local rhs
                are assembled.

        Returns:
            np.array: Block matrix of size 3 x 3, whwere each block represents
                coupling between variables on this interface. Index 0, 1 and 2
                represent the primary grid, the primary and secondary interface,
                respectively.
            np.array: Block matrix of size 3 x 1, representing the right hand
                side of this coupling. Index 0, 1 and 2 represent the primary grid,
                the primary and secondary interface, respectively.

        """
        if self.intf_coupling_via_high_dim:
            raise NotImplementedError(
                """Interface laws with edge couplings via the high
                                      dimensional grid must implement this model"""
            )

    def assemble_intf_coupling_via_low_dim(  # type: ignore
        self,
        g_between: pp.Grid,
        data_between: Dict,
        intf_primary: pp.MortarGrid,
        sd_pair_primary: Tuple[pp.Grid, pp.Grid],
        data_intf_primary: Dict,
        intf_secondary: pp.MortarGrid,
        sd_pair_secondary: Tuple[pp.Grid, pp.Grid],
        data_intf_secondary: Dict,
        matrix: np.ndarray,
        assemble_matrix: bool = True,
        assemble_rhs: bool = True,
    ):

        """Method to assemble the contribution from one interface to another one.

        The method must be implemented for subclasses of AbstractInterfaceLaw which has
        the attribute intf_coupling_via_low_dim set to True. For classes where the
        variable is False, there is no need for action.

        Note that the mixed-dimensional modeling framework does not allow for direct
        couplings between interfaces. However, there may be cases where an interface law
        is dependent on variables on the boundary between the lower-dimensional
        grid and another interface. As we normally associate these boundary values
        with the variable on the secondary interface, this method is available
        as an alternative.

        For more details on how this function is invoked see pp.Assembler.
        Note that the coupling currently only can be invoked if the variables
        on the primary and secondary interface have the same name.

        Any discretization operation should be done as part of self.discretize().

        Parameters:
            g_between (pp.Grid): Grid of the lower-dimensional neighbor to the
                main interface
            data_between (dict): Data dictionary of the intermediate grid.
            intf_primary (tuple of grids): The grids of the primary edge
            data_intf_primary (dict): Data dictionary of the primary interface.
            intf_secondary (tuple of grids): The grids of the secondary edge.
            data_intf_secondary (dict): Data dictionary of the secondary interface.
            matrix: original discretization.
            assemble_matrix (optional): If True (defalut), contributions to local matrix
                are assembled.
            assemble_rhs (optional): If True (defalut), contributions to local rhs
                are assembled.

        Returns:
            np.array: Block matrix of size 3 x 3, whwere each block represents
                coupling between variables on this interface. Index 0, 1 and 2
                represent the primary, secondary and mortar variable, respectively.
            np.array: Block matrix of size 3 x 1, representing the right hand
                side of this coupling. Index 0, 1 and 2 represent the primary,
                secondary and mortar variable, respectively.

        """
        if self.intf_coupling_via_low_dim:
            raise NotImplementedError(
                """Interface laws with edge couplings via the high
                                      dimensional grid must implement this model"""
            )
        else:
            pass
