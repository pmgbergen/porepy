"""
Module to store the grid hierarchy formed by a set of fractures and their
intersections along with a surrounding matrix in the form of a MixedDimensionalGrid.

"""
from __future__ import annotations

import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)

import numpy as np
from scipy import sparse as sps

import porepy as pp
from porepy.grids import mortar_grid


class MixedDimensionalGrid:
    """
    Container for the hierarchy of grids formed by fractures and their
    intersections along with a surrounding matrix.

    """

    def __init__(self) -> None:
        self._subdomain_data: Dict[pp.Grid, Dict] = {}
        self._interface_data: Dict[pp.MortarGrid, Dict] = {}
        self._interface_to_subdomains: Dict[pp.MortarGrid, Tuple[pp.Grid, pp.Grid]] = {}
        self.name = "Mixed-dimensional Grid"

    def __contains__(self, key: Any) -> bool:
        """Overload __contains__.

        Args:
            key (object): Object to be tested.

        Returns:
            True if either key is a pp.Grid, and key is among the subdomains of this
                mixed-dimensional grid, *or* key is a pp.MortarGrid and key is among the
                interfaces of this mixed-dimensional grid.

        """
        if isinstance(key, pp.Grid):
            return key in self._subdomain_data
        elif isinstance(key, pp.MortarGrid):
            return key in self._interface_data

        # Everything else is not in self.
        return False

    # --------- Iterators -------------------------
    @overload
    def subdomains(
        self, return_data: Literal[False] = False, dim: Optional[int] = None
    ) -> list[pp.Grid]:
        # Method signature intended to define type hint.
        # https://adamj.eu/tech/2021/05/29/python-type-hints-how-to-use-overload/
        # Note that the order of the overloaded methods is important, the version
        # with the default argument must come first. See
        # https://docs.python.org/3/library/typing.html#typing.overload
        # Also, it is important to have the default argument included here (e.g.,
        # include all of 'return_data: Literal[False] = False'), but not include it in
        # the below declaration, where the non-default option is handled.
        ...

    @overload
    def subdomains(
        self, return_data: Literal[True], dim: Optional[int] = None
    ) -> list[Tuple[pp.Grid, Dict]]:
        # Method signature intended to define type hint.
        # https://adamj.eu/tech/2021/05/29/python-type-hints-how-to-use-overload/
        ...

    def subdomains(
        self, return_data: bool = False, dim: Optional[int] = None
    ) -> Union[list[pp.Grid], list[Tuple[pp.Grid, Dict]]]:
        """Get a list of all subdomains in the mixed-dimensional grid.

        Optionally, the subdomains can be filtered on dimension. Also, the data
        dictionary belonging to the subdomain can be returned.

        Args:
            return_data (boolean, optional): If True, the data dictionary of the
                subdomain will be returned together with the subdomain grids. Defaults
                to False.
            dim (int, optional): If provided, only subdomains of the specified dimension
                will be returned.

        Returns:
            list: Either, list of grids associated with subdomains (default), or
                (if return_data=True) list of tuples, where the first tuple item is the
                subdomain grid, the second is the associated data dictionary.

        """
        if return_data:
            subdomains_and_data: list[tuple[pp.Grid, dict]] = []
            for sd, data in self._subdomain_data.items():
                if dim is None or dim == sd.dim:
                    subdomains_and_data.append((sd, data))
            return subdomains_and_data
        else:
            subdomains: list[pp.Grid] = []
            for sd in self._subdomain_data:
                if dim is None or dim == sd.dim:
                    subdomains.append(sd)
            return subdomains

    @overload
    def interfaces(
        self, return_data: Literal[False] = False, dim: Optional[int] = None
    ) -> list[pp.MortarGrid]:
        # Method signature intended to define type hint.
        # https://adamj.eu/tech/2021/05/29/python-type-hints-how-to-use-overload/
        ...

    @overload
    def interfaces(
        self, return_data: Literal[True], dim: Optional[int] = None
    ) -> list[Tuple[pp.MortarGrid, dict]]:
        # Method signature intended to define type hint.
        # https://adamj.eu/tech/2021/05/29/python-type-hints-how-to-use-overload/
        ...

    def interfaces(
        self, return_data: bool = False, dim: Optional[int] = None
    ) -> Union[list[pp.MortarGrid], list[Tuple[pp.MortarGrid, Dict]]]:
        """Get a list of all interfaces in the mixed-dimensional grid.

        Optionally, the interfaces can be filtered on dimension. Also, the data
        dictionary belonging to the interfaces can be returned.

        Args:
            return_data (boolean, optional): If True, the data dictionary of the
                interface will be returned together with the inerface mortar grids.
                Defaults to False.
            dim (int, optional): If provided, only interfaces of the specified dimension
                will be returned.

        Returns:
            list: Either, list of mortar grids associated with interfaces (default), or
                (if return_data=True) list of tuples, where the first tuple item is the
                interface mortar grid, the second is the associated data dictionary.

        """
        if return_data:
            interfaces_and_data: list[tuple[pp.MortarGrid, dict]] = []
            for intf, data in self._interface_data.items():
                if dim is None or dim == intf.dim:
                    interfaces_and_data.append((intf, data))
            return interfaces_and_data
        else:
            interfaces: list[pp.MortarGrid] = []
            for intf in self._interface_data:
                if dim is None or dim == intf.dim:
                    interfaces.append(intf)
            return interfaces

    # ---------- Navigate within the graph --------

    def interface_to_subdomain_pair(
        self, intf: pp.MortarGrid
    ) -> Tuple[pp.Grid, pp.Grid]:
        """Obtain the subdomains of an interface.

        The subdomains will be given in descending order with respect their
        dimension. If the interface is between grids of the same dimension, ascending subdomain
        ordering (as defined by assign_subdomain_ordering()) is used. If no ordering
        of subdomain exists, assign_subdomain_ordering() will be called by this method.

        Args:
            intf, pp.MortarGrid: The mortar grid associated with this interface.

        Returns:
            pp.Grid: The primary subdomain neighboring this interface. Normally, this
                will be the subdomain with the higher dimension.
            pp.Grid: The secondary subdomain neighboring this interface. Normally, this
                will be the subdomain with the lower dimension.

        """
        subdomains = self._interface_to_subdomains[intf]
        if subdomains[0].dim == subdomains[1].dim:
            if not (
                "node_number" in self._subdomain_data[subdomains[0]]
                and "node_number" in self._subdomain_data[subdomains[1]]
            ):
                self.assign_subdomain_ordering()

            subdomain_indexes = [self.node_number(sd) for sd in subdomains]
            if subdomain_indexes[0] < subdomain_indexes[1]:
                return subdomains[0], subdomains[1]
            else:
                return subdomains[1], subdomains[0]

        elif subdomains[0].dim > subdomains[1].dim:
            return subdomains[0], subdomains[1]
        else:
            return subdomains[1], subdomains[0]

    def subdomain_pair_to_interface(
        self, sd_pair: Tuple[pp.Grid, pp.Grid]
    ) -> pp.MortarGrid:
        """Get an interface from a pair of subdomains.

        Args:
            sd_pair (Tuple[pp.Grid, pp.Grid]): Pair of subdomains. Should exist
                in the mixed-dimensional grid.

        Returns:
            pp.MortarGrid: The interface that connects the subdomain pairs, represented
                by a MortarGrid.

        Raises:
            KeyError: If the subdomain pair is not represented in the mixed-dimensional
                grid.

        """
        reverse_dict = {v: k for k, v in self._interface_to_subdomains.items()}
        if sd_pair in reverse_dict:
            return reverse_dict[sd_pair]
        elif sd_pair[::-1] in reverse_dict:
            return reverse_dict[sd_pair[::-1]]
        else:
            raise KeyError("Unknown subdomain pair")

    def subdomain_to_interfaces(
        self, sd: pp.Grid
    ) -> Generator[pp.MortarGrid, None, None]:
        """Iterator over the interfaces of a specific subdomain.

        Args:
            sd (pp.Grid): A subdomain in the mixed-dimensional grid.

        Yields:
            pp.MortarGrid: Interfaces associated with the subdomain.

        """
        for intf in self._interface_data:
            sd_pair = self._interface_to_subdomains[intf]
            if sd_pair[0] == sd or sd_pair[1] == sd:
                yield intf

    def neighboring_subdomains(
        self, sd: pp.Grid, only_higher: bool = False, only_lower: bool = False
    ) -> List[pp.Grid]:
        """Get the subdomain neighbors of a subdomain (going through interfaces).

        Optionally, return only neighbors corresponding to higher or lower
        dimension. At most one of only_higher and only_lower can be True.

        Args:
            sd (pp.Grid): Subdomain in the mixed-dimensional grid.
            only_higher (boolean, optional): Consider only the higher-
                dimensional neighbors. Defaults to False.
            only_lower (boolean, optional): Consider only the lower-dimensional
                neighbors. Defaults to False.

        Return:
            List of pp.Grid: Subdomain neighbors of sd

        Raises:
            ValueError if both only_higher and only_lower is True.

        """

        neigh = []
        for intf, sd_pair in self._interface_to_subdomains.items():
            if sd_pair[0] == sd:
                neigh.append(sd_pair[1])
            elif sd_pair[1] == sd:
                neigh.append(sd_pair[0])

        if not only_higher and not only_lower:
            return neigh
        elif only_higher and only_lower:
            raise ValueError("Cannot return both only higher and only lower")
        elif only_higher:
            # Find the neighbours that are higher dimensional
            return [g for g in neigh if g.dim > sd.dim]
        else:
            # Find the neighbours that are higher dimensional
            return [g for g in neigh if g.dim < sd.dim]

    # ------------ Getters for subdomain and interface properties

    def subdomain_data(self, sd: pp.Grid) -> Any:
        """Getter for the subdomain data dictionary.

        Args:
            sd (pp.Grid): The subdomain grid.

        Returns:
            The data dictionary associated with the subdomain.

        """
        return self._subdomain_data[sd]

    def interface_data(self, intf: pp.MortarGrid) -> Dict:
        """Getter for the interface data dictionary.

        Args:
            intf (pp.MortarGrid): The mortar grid associated with the interface.

        Returns:
            The data dictionary associated with the interface.

        """
        return self._interface_data[intf]

    # ------------ Add new subdomains and interfaces ----------

    def add_subdomains(self, new_subdomains: Union[pp.Grid, Iterable[pp.Grid]]) -> None:
        """Add new subdomains to the mixed-dimensional grid.

        Args:
            new_subdomains (pp.Grid or Iterable of pp.Grid): The subdomains to be added. None
                of these should have been added previously.

        Raises:
            ValueError if a subdomain is already present in the mixed-dimensional grid.

        """
        if isinstance(new_subdomains, pp.Grid):
            ng = [new_subdomains]
        else:
            ng = new_subdomains  # type: ignore

        if any([i is j for i in ng for j in list(self._subdomain_data)]):
            raise ValueError("Grid already defined in MixedDimensionalGrid")

        for sd in ng:
            # Add the grid to the list of subdomains with an empty data dictionary.
            self._subdomain_data[sd] = {}

    def add_interface(
        self,
        intf: pp.MortarGrid,
        sd_pair: Tuple[pp.Grid, pp.Grid],
        primary_secondary_map: sps.spmatrix,
    ) -> None:
        """
        Add an interface to the MixedDimensionalGrid.

        Note:
            If the grids have different dimensions, the coupling will be defined with
            the higher-dimensional grid as the first ("primary") subdomain. For equal
            dimensions, the ordering of the subdomains is the same as in input grids.

        Args:
            intf (pp.MortarGrid): MortarGrid that represents this interface.
            sd_pair (Tuple, len==2). Grids to be connected. The ordering is arbitrary.
            primary_secondary_map (sps.spmatrix): Identity mapping between cells in the
                higher-dimensional grid and faces in the lower-dimensional
                grid. No assumptions are made on the type of the object at this
                stage. In the grids[0].dim = grids[1].dim case, the mapping is
                from faces of the first grid to faces of the second one. In the
                co-dimension 2 case, it maps from cells to cells.

        Raises:
            ValueError: If the interface is not specified by exactly two grids
            ValueError: If the interface already exists.
            ValueError: If the two grids are not between zero and two dimensions apart.

        """
        if len(sd_pair) != 2:
            raise ValueError("An interface should be specified by exactly two grids")

        if intf in self._interface_data:
            raise ValueError("Cannot add existing interface")

        # The data dictionary is initialized with a single field, defining the mapping
        # between grid quantities in the primary and secondary neighboring subdomains
        # (for a co-dimension 1 coupling, this will be between faces in the
        # higher-dimensional grid and cells in the lower-dimensional grid).
        data = {"face_cells": primary_secondary_map}

        # Add the interface to the list, with
        self._interface_data[intf] = data

        # The higher-dimensional grid is the first subdomain of the subdomain pair.

        # Couplings of co-dimension one()
        if sd_pair[0].dim - 1 == sd_pair[1].dim:
            sd_pair = (sd_pair[0], sd_pair[1])
        elif sd_pair[0].dim == sd_pair[1].dim - 1:
            sd_pair = (sd_pair[1], sd_pair[0])

        # Coupling of co-dimension 2
        elif sd_pair[0].dim - 2 == sd_pair[1].dim:
            sd_pair = (sd_pair[0], sd_pair[1])
        elif sd_pair[0].dim == sd_pair[1].dim - 2:
            sd_pair = (sd_pair[1], sd_pair[0])

        # Equi-dimensional coupling
        elif sd_pair[0].dim == sd_pair[1].dim:
            sd_pair = (sd_pair[0], sd_pair[1])
        else:
            raise ValueError("Can only handle subdomain coupling of co-dimension <= 2")

        self._interface_to_subdomains[intf] = sd_pair

    # --------- Remove and update subdomains

    def remove_subdomain(self, sd: pp.Grid) -> None:
        """Remove subdomain and related interfaces from the mixed-dimensional grid.

        Args:
           sd (pp.Grid): The subdomain to be removed

        """
        # Delete from the subdomain list.
        del self._subdomain_data[sd]

        # Find interfaces to be removed, store in a list.
        interfaces_to_remove: List[pp.MortarGrid] = []
        for intf in self.interfaces():
            sd_pair = self._interface_to_subdomains[intf]
            if sd_pair[0] == sd or sd_pair[1] == sd:
                interfaces_to_remove.append(intf)

        # Remove interfaces.
        for intf in interfaces_to_remove:
            del self._interface_data[intf]
            del self._interface_to_subdomains[intf]

    # ---------- Functionality related to ordering of subdomains

    def node_number(self, sd: pp.Grid) -> int:
        """Return node number of a subdomain.

        If no node ordering exists, the method calls assign_subdomain_ordering
        to create one.

        Args:
            sd:
                Subdomain to be identified.

        Returns:
            Integer node number identifying the subdomain.

        """
        data = self.subdomain_data(sd)
        if "node_number" not in data:
            self.assign_subdomain_ordering()
        return data["node_number"]

    def assign_subdomain_ordering(self, overwrite_existing: bool = True) -> None:
        """
        Assign an ordering of the subdomains in the mixed-dimensional, stored as the
        attribute 'node_number' in the data dictionary.

        The ordering starts with grids of the highest dimension. The ordering
        within each dimension is determined by an iterator over the graph, and
        can in principle change between two calls to this function.

        If an ordering covering all subdomains in the mixed-dimensional grid already
        exists, but does not coincide with the new ordering, a warning is issued. If the
        optional parameter overwrite_existing is set to False, no update is performed if
        a subdomain ordering already exists.

        Note:
            The concept of subdomain ordering is at the moment not much used, however,
            it will be brought (back) into use at a later point.

        Args:
            overwrite_existing (bool, optional): If True (default), any existing
                subdomain ordering will be overwritten.

        """

        # Check whether 'node_number' is defined for the grids already.
        ordering_exists = True
        for data in self._subdomain_data.values():
            if "node_number" not in data:
                ordering_exists = False
        if ordering_exists and not overwrite_existing:
            return

        counter = 0
        # Loop over grids in decreasing dimensions
        for dim in range(self.dim_max(), self.dim_min() - 1, -1):
            for sd, data in self.subdomains(return_data=True, dim=dim):
                # Get old value, issue warning if not equal to the new one.
                num = data.get("node_number", -1)
                if ordering_exists and num != counter:
                    warnings.warn("Order of subdomains has changed")
                # Assign new value
                data["node_number"] = counter
                counter += 1

        counter = 0
        for _, data in self.interfaces(return_data=True):
            data["edge_number"] = counter
            counter += 1

    def update_subdomain_ordering(self, removed_number: int) -> None:
        """Update an existing ordering of the subdomains, stored as the attribute
        'node_number' in the data dictionaries.

        Intended for keeping the subdomain ordering after removing a subdomain from the
        mixed-dimensional grid. In this way, the interface sorting will not be disturbed
        by the removal, but no gaps in are created.

        Args:
            removed_number (int): node_number of the removed grid.

        """

        # Loop over grids in decreasing dimensions
        for dim in range(self.dim_max(), self.dim_min() - 1, -1):
            for sd, data in self.subdomains(return_data=True, dim=dim):
                if "node_number" not in self._subdomain_data[sd]:
                    # It is not clear how severe this case is. For the moment,
                    # we give a warning, and hope the user knows what to do
                    warnings.warn(
                        "Tried to update subdomain ordering where none exists"
                    )
                    # No point in continuing with this subdomain.
                    continue

                # Obtain the old node number
                data = self._subdomain_data[sd]
                old_number = data.get("node_number", -1)
                # And replace it if it is higher than the removed one
                if old_number > removed_number:
                    data["node_number"] = old_number - 1

    def sort_subdomains(self, subdomains: List[pp.Grid]) -> List[pp.Grid]:
        """Sort subdomains according to the subdomain ordering (attribute 'node_number'
        in the data dictionary associated with the subdomain).

        Args:
            subdomains: List of subdomains.

        Returns:
            sorted_subdomain_data: The same subdomains, sorted.

        """
        assert all(["node_number" in self._subdomain_data[sd] for sd in subdomains])

        return sorted(subdomains, key=lambda n: self.node_number(n))

    # ------------- Miscellaneous functions ---------

    def compute_geometry(self) -> None:
        """Compute geometric quantities for the grids."""
        for sd in self.subdomains():
            sd.compute_geometry()

        for intf in self.interfaces():
            intf.compute_geometry()

    def copy(self) -> MixedDimensionalGrid:
        """Make a shallow copy of the mixed-dimensional grid. The underlying subdomain
        and interface grids are not copied.

        Return:
            pp.MixedDimensionalGrid: Copy of this MixedDimensionalGrid.

        """
        mdg_copy: pp.MixedDimensionalGrid = MixedDimensionalGrid()
        mdg_copy._subdomain_data = self._subdomain_data.copy()
        mdg_copy._interface_data = self._interface_data.copy()
        mdg_copy._interface_to_subdomains = self._interface_to_subdomains.copy()
        return mdg_copy

    def replace_subdomains_and_interfaces(
        self,
        sd_map: Optional[Dict[pp.Grid, pp.Grid]] = None,
        intf_map: Optional[
            Dict[
                pp.MortarGrid,
                Union[pp.MortarGrid, Dict[mortar_grid.MortarSides, pp.Grid]],
            ]
        ] = None,
        tol: float = 1e-6,
    ) -> None:
        """Replace grids and / or mortar grids in the mixed-dimensional grid.

        Args:
            sd_map (dict): Mapping between the old and new grids. Keys are
                the old grids, and values are the new grids.
            intf_map (dict): Mapping between the old mortar grid and new
                (side grids of the) mortar grid. Keys are the old mortar grids,
                and values are dictionaries with side grid number as keys, and
                side grids as values. Optionally, the mapping can be directly from old
                to new MortarGrid (sometimes it is easier to construct the new
                MortarGrid right away, before replacing it in the mixed-dimensional grid.
            tol (float): Geometric tolerance used when updating mortar projections.

        """
        # Ensure sorting of equi-dimensional subdomain pairs is possible:
        self.assign_subdomain_ordering(overwrite_existing=False)
        # The operation is carried out in three steps:
        # 1) Update the mortar grid. This will also update projections to and from the
        #    mortar grids so that they are valid for the new mortar grid, but not with
        #    respect to any changes in the subdomain grids.
        # 2) Update the subdomain grids in the MixedDimensionalGrid.
        # 3) Update the subdomain grids with respect to the interfaces in the
        #    MixedDimensionalGrid. This will update projections to and from mortar grids.
        # Thus, if both the mortar grid and an adjacent subdomain grids are updated, the
        # projections will be updated twice.

        # refine the mortar grids when specified
        if intf_map is not None:
            for intf_old, intf_new in intf_map.items():

                # Update the mortar grid. The update_mortar function in class MortarGrid
                # performs the update in place, thus the mortar grid will not be
                # modified in the mixed-dimensional grid itself (thus no changes to
                # self._interface_data or self._interface_to_subdomains)
                if isinstance(intf_new, pp.MortarGrid):
                    side_grids = intf_new.side_grids
                else:
                    side_grids = intf_new

                intf_old.update_mortar(side_grids, tol)

        # update the mixed-dimensional considering the new grids instead of the old one.
        if sd_map is not None:
            for sd_old, sd_new in sd_map.items():

                # Update the subdomain data
                data = self._subdomain_data[sd_old]
                self._subdomain_data[sd_new] = data

                # Loop over interfaces, look for ones which has the subdomain to be
                # replaced as one of its neighbors.
                for intf in self.subdomain_to_interfaces(sd_old):
                    # Get the subdomain pair of the interface - we need to replace the
                    # relevant item.
                    sd_pair = self.interface_to_subdomain_pair(intf)

                    # If we find a hit, overwrite the old interface information.
                    if sd_pair[0] == sd_old:
                        self._interface_to_subdomains[intf] = (sd_new, sd_pair[1])
                        # We know this is the primary subdomain for this interface,
                        # update mortar projections.
                        intf.update_primary(sd_new, sd_old, tol)
                    elif sd_pair[1] == sd_old:
                        self._interface_to_subdomains[intf] = (sd_pair[0], sd_new)
                        # This is the secondary subdomain for the interface.
                        intf.update_secondary(sd_new, tol)
                # This must be done after the loop, in case data is required to sort
                # subdomains by node number
                del self._subdomain_data[sd_old]

    # ----------- Apply functions to subdomains and interfaces

    def apply_function_to_subdomains(
        self, fct: Callable[[pp.Grid, Dict], Any]
    ) -> np.ndarray:
        """Loop on all subdomains and evaluate a function on each of them.

        Parameter:
            fct: function to evaluate. It takes a grid and the related data and
                returns a scalar.

        Returns:
            values: vector containing the function evaluated on each subdomain,
                ordered by 'node_number'.

        """
        values = np.empty(self.num_subdomains())
        for sd, data in self.subdomains(return_data=True):
            values[data["node_number"]] = fct(sd, data)
        return values

    def apply_function_to_interfaces(
        self, fct: Callable[[pp.Grid, pp.Grid, Dict, Dict, Dict], Any]
    ) -> sps.spmatrix:
        """Loop on all the interfaces and evaluate a function on each of them.

        Args:
            fct: function to evaluate. It returns a scalar and takes the following
                arguments (the order is strict): the primary and secondary subdomain
                grids, the primary and secondary data, the interface data.

        Returns:
            matrix: sparse strict upper triangular matrix containing the function
                evaluated on each mortar grid, ordered by their
                relative 'node_number'.

        """
        i = np.zeros(self.num_interfaces(), dtype=int)
        j = np.zeros_like(i)
        values = np.zeros(i.size)

        # Loop over the interfaces
        idx = 0
        for intf, data in self.interfaces(return_data=True):

            sd_primary, sd_secondary = self.interface_to_subdomain_pair(intf)
            data_primary = self.subdomain_data(sd_primary)
            data_secondary = self.subdomain_data(sd_secondary)

            i[idx] = data_primary["node_number"]
            j[idx] = data_secondary["node_number"]
            values[idx] = fct(
                sd_primary, sd_secondary, data_primary, data_secondary, data
            )
            idx += 1

        # Upper triangular matrix
        return sps.coo_matrix(
            (values, (i, j)),
            (self.num_interfaces(), self.num_interfaces()),
        )

    # ---- Methods for getting information on the bucket, or its components ----

    def diameter(
        self, cond: Callable[[Union[pp.Grid, pp.MortarGrid]], bool] = None
    ) -> float:
        """Compute the cell diameter (mesh size) of the mixed-dimensional grid.

        A function can be passed to filter subdomains and/or interfaces.

        Args:
            cond (Callable, optional), predicate with a subdomain or interface as input.

        Returns:
            diameter: the diameter of the mixed-dimensional grid.

        """
        if cond is None:
            cond = lambda g: True
        diam_g = [np.amax(sd.cell_diameters()) for sd in self.subdomains() if cond(sd)]

        diam_mg = [np.amax(mg.cell_diameters()) for mg in self.interfaces() if cond(mg)]

        return np.amax(np.hstack((diam_g, diam_mg)))

    #    def size(self) -> int:
    #        """
    #        Returns:
    #            int: Number of mono-dimensional grids and interfaces in the bucket.
    #
    #        """
    #        return self.num_subdomains() + self.num_interfaces()

    def dim_min(self) -> int:
        """Get minimal dimension represented in the mixed-dimensional grid.

        Returns:
            int: Minimum dimension of the grids present in the hierarchy.

        """
        return np.min([sd.dim for sd in self.subdomains()])

    def dim_max(self) -> int:
        """Get maximal dimension represented in the grid.

        Returns:
            int: Maximum dimension of the grids present in the hierarchy.

        """
        return np.max([sd.dim for sd in self.subdomains()])

    """
    def all_dims(self) -> np.ndarray:
        ""
        Returns:
            np.array: Active dimensions of the grids present in the hierarchy.

        ""
        return np.unique([grid.dim for grid in self.subdomains()])
    """

    def num_subdomain_cells(self, cond: Callable[[pp.Grid], bool] = None) -> int:
        """Compute the total number of cells of the mixed-dimensional grid.

        A function can be passed to filter subdomains and/or interfaces.

        Args:
            cond: optional, predicate with a grid as input.

        Return:
            int: the total number of cells of the grid bucket.

        """
        if cond is None:
            cond = lambda g: True
        return np.sum(  # type: ignore
            [grid.num_cells for grid in self.subdomains() if cond(grid)], dtype=int
        )

    def num_interface_cells(self, cond: Callable[[pp.MortarGrid], bool] = None) -> int:
        """
        Compute the total number of mortar cells of the mixed-dimensional grid.

        A function can be passed to filter subdomains and/or interfaces.

        Args:
            cond: optional, predicate with a grid as input.

        Return:
            num_cells (int): the total number of mortar cells of the grid bucket.

        """
        if cond is None:
            cond = lambda g: True
        return np.sum(  # type: ignore
            [mg.num_cells for mg in self.interfaces() if cond(mg)],
            dtype=int,
        )

    def num_subdomains(self):
        """Get the total number of subdomains in the mixed-dimensional grid.

        Returns:
            int: Number of subdomains in the graph.

        """
        return len(self._subdomain_data)

    def num_interfaces(self) -> int:
        """Get the total number of interfaces in the mixed-dimensional grid.

        Returns:
            int: Number of interfaces in the graph.

        """
        return len(self._interface_data)

    def __str__(self) -> str:
        max_dim = self.subdomains(dim=self.dim_max())
        num_nodes = 0
        num_cells = 0
        for sd in max_dim:
            num_nodes += sd.num_nodes
            num_cells += sd.num_cells
        s = (
            "Mixed-dimensional grid. \n"
            f"Maximum dimension present: {self.dim_max()} \n"
            f"Minimum dimension present: {self.dim_min()} \n"
        )
        s += "Size of highest dimensional grid: Cells: " + str(num_cells)
        s += ". Nodes: " + str(num_nodes) + "\n"
        s += "In lower dimensions: \n"
        for dim in range(self.dim_max() - 1, self.dim_min() - 1, -1):
            num_nodes = 0
            num_cells = 0
            for sd in self.subdomains(dim=dim):
                num_nodes += sd.num_nodes
                num_cells += sd.num_cells

            s += (
                f"{len(list(self.subdomains(dim=dim)))} grids of dimension {dim}, with in "
                f"total {num_cells} cells and {num_nodes} nodes. \n"
            )

        s += f"Total number of interfaces: {self.num_subdomains()}\n"
        for dim in range(self.dim_max(), self.dim_min(), -1):
            num_e = 0
            nc = 0
            for mg in self.interfaces(dim=dim - 1):
                num_e += 1
                nc += mg.num_cells

            s += (
                f"{num_e} interfaces between grids of dimension {dim} and {dim-1} with"
                f" in total {nc} cells\n"
            )
        return s

    def __repr__(self) -> str:
        s = (
            f"Mixed-dimensional grid containing {self.num_subdomains() } grids and "
            f"{self.num_interfaces()} interfaces.\n"
            f"Maximum dimension present: {self.dim_max()} \n"
            f"Minimum dimension present: {self.dim_min()} \n"
        )

        if self.num_subdomains() > 0:
            for dim in range(self.dim_max(), self.dim_min() - 1, -1):
                nc = 0
                num_sd = 0
                for sd in self.subdomains(dim=dim):
                    num_sd += 1
                    nc += sd.num_cells
                s += f"{num_sd} grids of dimension {dim}" f" with in total {nc} cells\n"
        if self.num_interfaces() > 0:
            for dim in range(self.dim_max(), self.dim_min(), -1):
                num_intf = 0
                num_intf_cells = 0
                for intf in self.interfaces(dim=dim - 1):
                    num_intf += 1
                    num_intf_cells += intf.num_cells

                s += (
                    f"{num_intf} interfaces between grids of dimension {dim} and {dim-1}"
                    f" with in total {num_intf_cells} mortar cells\n"
                )

        return s
