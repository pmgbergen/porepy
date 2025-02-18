"""A module to store the grid hierarchy formed by a set of fractures and their
intersections along with a surrounding matrix in the form of a mixed-dimensional grid.

"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Iterable, Literal, Optional, Union, overload

import numpy as np
from scipy import sparse as sps

import porepy as pp
from porepy.grids import mortar_grid
from porepy.utils.porepy_types import GridLike


class MixedDimensionalGrid:
    """Container for the hierarchy of grids formed by fractures and their intersections
    along with a surrounding matrix.

    """

    def __init__(self) -> None:
        self.name = "Mixed-dimensional grid"
        """Name of the md-grid, initiated as ``'Mixed-dimensional grid'``."""

        self._subdomain_data: dict[pp.Grid, dict] = {}
        """Dictionary storing data associated with subdomains."""

        self._interface_data: dict[pp.MortarGrid, dict] = {}
        """Dictionary storing data associated with interfaces."""

        self._interface_to_subdomains: dict[pp.MortarGrid, tuple[pp.Grid, pp.Grid]] = {}
        """Dictionary storing the mapping from interfaces to the adjacent subdomains."""

        self._subdomain_to_boundary_grid: dict[pp.Grid, pp.BoundaryGrid] = {}
        """Dictionary storing the boundary grids associated with subdomains.

        Note:
            We assume there is only one boundary grid for each subdomain.

        """
        self._boundary_grid_data: dict[pp.BoundaryGrid, dict] = {}
        """Dictionary storing data associated with the boundary grids."""

    def __contains__(self, key: Any) -> bool:
        """
        Parameters:
            key: Object to be tested.

        Returns:
            True if either ``key`` is a :class:`~porepy.grids.grid.Grid`,
            and key is among the subdomains of this mixed-dimensional grid,
            *or* ``key`` is a :class:`~porepy.grids.mortar_grid.MortarGrid` and among
            the interfaces of this mixed-dimensional grid, *or* ``key`` is a
            :class:`~porepy.grids.boundary_grid.BoundaryGrid` and among the boundary
            grids of this mixed-dimensional grid.

        """
        if isinstance(key, pp.Grid):
            return key in self._subdomain_data
        elif isinstance(key, pp.MortarGrid):
            return key in self._interface_data
        elif isinstance(key, pp.BoundaryGrid):
            return key in self._boundary_grid_data
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
    ) -> list[tuple[pp.Grid, dict]]:
        # Method signature intended to define type hint.
        # https://adamj.eu/tech/2021/05/29/python-type-hints-how-to-use-overload/
        ...

    def subdomains(
        self, return_data: bool = False, dim: Optional[int] = None
    ) -> Union[list[pp.Grid], list[tuple[pp.Grid, dict]]]:
        """Get a sorted list of subdomains in the mixed-dimensional grid.

        Optionally, the subdomains can be filtered by dimension. Also, the data
        dictionary belonging to the subdomain can be returned.

        Sorted by descending subdomain dimension and increasing Grid ID, see
        :meth:`argsort_grids`.

        Parameters:
            return_data: ``default=False``

                If True, the data dictionary of the subdomain will
                be returned together with the subdomain grids.
            dim: ``default=None``

                If provided, only subdomains of the specified dimension will
                be returned.

        Returns:
            Either a list of grids associated with subdomains (default), or (if
            ``return_data=True``) list of tuples, where the first tuple item is the
            subdomain grid, the second is the associated data dictionary.

        """
        subdomains: list[pp.Grid] = list()
        data_list: list[dict] = list()
        for sd, data in self._subdomain_data.items():
            if dim is None or dim == sd.dim:
                subdomains.append(sd)
                data_list.append(data)
        sort_ind = self.argsort_grids(subdomains)
        if return_data:
            return [(subdomains[i], data_list[i]) for i in sort_ind]
        else:
            return [subdomains[i] for i in sort_ind]

    @overload
    def interfaces(
        self,
        return_data: Literal[False] = False,
        dim: Optional[int] = None,
        codim: Optional[int] = None,
    ) -> list[pp.MortarGrid]:
        # Method signature intended to define type hint.
        # https://adamj.eu/tech/2021/05/29/python-type-hints-how-to-use-overload/
        ...

    @overload
    def interfaces(
        self,
        return_data: Literal[True],
        dim: Optional[int] = None,
        codim: Optional[int] = None,
    ) -> list[tuple[pp.MortarGrid, dict]]:
        # Method signature intended to define type hint.
        # https://adamj.eu/tech/2021/05/29/python-type-hints-how-to-use-overload/
        ...

    def interfaces(
        self,
        return_data: bool = False,
        dim: Optional[int] = None,
        codim: Optional[int] = None,
    ) -> Union[list[pp.MortarGrid], list[tuple[pp.MortarGrid, dict]]]:
        """Get a sorted list of interfaces in the mixed-dimensional grid.

        Optionally, the interfaces can be filtered on dimension. Also, the data
        dictionary belonging to the interfaces can be returned.

        Sorting by descending interface dimension and increasing MortarGrid ID, see
        :meth:`argsort_grids`.

        Parameters:
            return_data: ``default=False``

                If True, the data dictionary of the interface will be returned together
                with the interface mortar grids.
            dim: ``default=None``

                If provided, only interfaces of the specified dimension will
                be returned.

            codim: ``default=None``

                Codimension of the returned interfaces. Defaults to None, in which case
                no codimension filtering is performed. If provided, only interfaces with
                the specified codimension will be returned.
                If both dim and codim are provided, a combined filter will be applied.

        Returns:
            Either a list of mortar grids associated with interfaces (default), or (if
            ``return_data=True``) list of tuples, where the first tuple item is the
            interface mortar grid, the second is the associated data dictionary.

        """
        interfaces: list[pp.MortarGrid] = list()
        data_list: list[dict] = list()
        for intf, data in self._interface_data.items():
            if (dim is None or intf.dim == dim) and (
                codim is None or intf.codim == codim
            ):
                interfaces.append(intf)
                data_list.append(data)
        sort_ind = self.argsort_grids(interfaces)
        if return_data:
            return [(interfaces[i], data_list[i]) for i in sort_ind]
        else:
            return [interfaces[i] for i in sort_ind]

    @overload
    def boundaries(
        self, return_data: Literal[False] = False, dim: Optional[int] = None
    ) -> list[pp.BoundaryGrid]:
        # Method signature intended to define type hint.
        # https://adamj.eu/tech/2021/05/29/python-type-hints-how-to-use-overload/
        ...

    @overload
    def boundaries(
        self, return_data: Literal[True], dim: Optional[int] = None
    ) -> list[tuple[pp.BoundaryGrid, dict]]:
        # Method signature intended to define type hint.
        # https://adamj.eu/tech/2021/05/29/python-type-hints-how-to-use-overload/
        ...

    def boundaries(
        self, return_data: bool = False, dim: Optional[int] = None
    ) -> Union[list[pp.BoundaryGrid], list[tuple[pp.BoundaryGrid, dict]]]:
        """Get a sorted list of boundary grids in the mixed-dimensional grid.

        Optionally, the boundary grids can be filtered by dimension.

        Sorting by descending boundary dimension and increasing BoundaryGrid id,
        see :meth:`argsort_grids`.

        Parameters:
            return_data: ``default=False``

                If True, the data dictionary of the boundary grid will be returned
                together with the boundary grids.

            dim: ``default=None``

                If provided, only boundary grids of the specified
                dimension will be returned.

        Returns:
            A list of boundary grids.

        """
        if self._subdomain_data and not self._subdomain_to_boundary_grid:
            raise ValueError(
                "There are subdomains in the mixed-dimensional grid, but no boundary "
                "grids. This is probably not intended."
            )
        boundaries: list[pp.BoundaryGrid] = list()
        data_list: list[dict] = list()
        for bg, data in self._boundary_grid_data.items():
            if dim is None or dim == bg.dim:
                boundaries.append(bg)
                data_list.append(data)
        sort_ind = self.argsort_grids(boundaries)
        if return_data:
            return [(boundaries[i], data_list[i]) for i in sort_ind]
        else:
            return [boundaries[i] for i in sort_ind]

    # ---------- Navigate within the graph --------

    def interface_to_subdomain_pair(
        self, intf: pp.MortarGrid
    ) -> tuple[pp.Grid, pp.Grid]:
        """Obtain the subdomains of an interface.

        The subdomains will be given in descending order with respect to their
        dimension. If the interface is between grids of the same dimension, ascending
        subdomain ordering (as defined by :meth:`sort_subdomain_tuple`) is used.

        Parameters:
            intf: The mortar grid associated with an interface.

        Returns:
            A 2-tuple containing

                - The primary subdomain neighboring this interface. Normally, this will
                  be the subdomain with the higher dimension.
                - The secondary subdomain neighboring this interface. Normally, this
                  will be the subdomain with the lower dimension.

        """
        subdomains: tuple[pp.Grid, pp.Grid] = self._interface_to_subdomains[intf]

        return self.sort_subdomain_tuple(subdomains)

    def subdomain_pair_to_interface(
        self, sd_pair: tuple[pp.Grid, pp.Grid]
    ) -> pp.MortarGrid:
        """Get an interface from a pair of subdomains.

        Parameters:
            sd_pair: Pair of subdomains. Should exist in the mixed-dimensional grid.

        Raises:
            KeyError: If the subdomain pair is not represented in the mixed-dimensional
                grid.

        Returns:
            The interface that connects the subdomain pairs, represented by a
            MortarGrid.

        """
        reverse_dict = {v: k for k, v in self._interface_to_subdomains.items()}
        if sd_pair in reverse_dict:
            return reverse_dict[sd_pair]
        elif sd_pair[::-1] in reverse_dict:
            return reverse_dict[sd_pair[::-1]]
        else:
            raise KeyError("Unknown subdomain pair")

    def subdomain_to_interfaces(self, sd: pp.Grid) -> list[pp.MortarGrid]:
        """Iterator over the interfaces of a specific subdomain.

        Parameters:
            sd: A subdomain in the mixed-dimensional grid.

        Returns:
            Sorted list of interfaces associated with the subdomain
            (see :meth:`sort_interfaces`).

        """
        interfaces: list[pp.MortarGrid] = list()
        for intf in self._interface_data:
            sd_pair = self._interface_to_subdomains[intf]
            if sd_pair[0] == sd or sd_pair[1] == sd:
                interfaces.append(intf)
        return self.sort_interfaces(interfaces)

    def subdomain_to_boundary_grid(self, sd: pp.Grid) -> Optional[pp.BoundaryGrid]:
        """Get the boundary grid associated with a subdomain.

        Parameters:
            sd: A subdomain in the mixed-dimensional grid.

        Returns:
            The boundary grid associated with this subdomain. ``None`` is returned if
            ``sd`` is not in this mixed-dimensional grid, or the dimension of ``sd`` is
            0.

        """
        return self._subdomain_to_boundary_grid.get(sd, None)

    def neighboring_subdomains(
        self, sd: pp.Grid, only_higher: bool = False, only_lower: bool = False
    ) -> list[pp.Grid]:
        """Get the subdomain neighbors of a subdomain (going through interfaces).

        Optionally, return only neighbors corresponding to higher or lower
        dimension. At most one of only_higher and only_lower can be True.

        Parameters:
            sd: Subdomain in the mixed-dimensional grid.
            only_higher: ``default=False``

                Consider only the higher-dimensional neighbors.
            only_lower: ``default=False``

                Consider only the lower-dimensional neighbors.

        Raises:
            ValueError: If both ``only_higher`` and ``only_lower`` are True.

        Returns:
            Subdomain neighbors of ``sd``.

        """

        neigh = []
        for intf, sd_pair in self._interface_to_subdomains.items():
            if sd_pair[0] == sd:
                neigh.append(sd_pair[1])
            elif sd_pair[1] == sd:
                neigh.append(sd_pair[0])

        if only_higher and only_lower:
            raise ValueError("Cannot return both only higher and only lower")
        elif only_higher:
            # Find the neighbours that are higher dimensional
            neigh = [sd_n for sd_n in neigh if sd_n.dim > sd.dim]
        elif only_lower:
            # Find the neighbours that are higher dimensional
            neigh = [sd_n for sd_n in neigh if sd_n.dim < sd.dim]
        return self.sort_subdomains(neigh)

    # ------------ Getters for subdomain, interface and boundary grid properties

    def subdomain_data(self, sd: pp.Grid) -> dict:
        """Getter for the subdomain data dictionary.

        Parameters:
            sd: The subdomain grid.

        Returns:
            The data dictionary associated with the subdomain.

        """
        return self._subdomain_data[sd]

    def interface_data(self, intf: pp.MortarGrid) -> dict:
        """Getter for the interface data dictionary.

        Parameters:
            intf: The mortar grid associated with the interface.

        Returns:
            The data dictionary associated with the interface.

        """
        return self._interface_data[intf]

    def boundary_grid_data(self, bg: pp.BoundaryGrid) -> dict:
        """Getter for the boundary grid data dictionary.

        Parameters:
            bg: The boundary grid.

        Returns:
            The data dictionary associated with the boundary grid.

        """
        return self._boundary_grid_data[bg]

    # ------------ Add new subdomains and interfaces ----------

    def add_subdomains(self, new_subdomains: Union[pp.Grid, Iterable[pp.Grid]]) -> None:
        """Add new subdomains to the mixed-dimensional grid.

        Creates a boundary grid for the subdomain.

        Parameters:
            new_subdomains: The subdomains to be added. None of these should have been
                added previously.

        Raises:
            ValueError: If a subdomain is already present in the mixed-dimensional grid.

        """
        if isinstance(new_subdomains, pp.Grid):
            ng = [new_subdomains]
        else:
            ng = new_subdomains  # type: ignore

        if any([i is j for i in ng for j in list(self._subdomain_data)]):
            raise ValueError("Grid already defined in MixedDimensionalGrid")

        for sd in ng:
            # Add the grid to the dictionary of subdomains with an empty data dictionary.
            self._subdomain_data[sd] = {}

        # Generate boundary grids for each subdomain.
        for sd in ng:
            if sd.dim > 0:
                bg = pp.BoundaryGrid(g=sd)
                self._subdomain_to_boundary_grid[sd] = bg
                self._boundary_grid_data[bg] = {}

    def add_interface(
        self,
        intf: pp.MortarGrid,
        sd_pair: tuple[pp.Grid, pp.Grid],
        primary_secondary_map: sps.spmatrix,
    ) -> None:
        """Add an interface to the mixed-dimensional grid.

        Note:
            If the grids have different dimensions, the coupling will be defined with
            the higher-dimensional grid as the first (primary) subdomain. For equal
            dimensions, the ordering of the subdomains is the same as in input grids.

        Parameters:
            intf: MortarGrid that represents this interface.
            sd_pair: ``len==2``

                A 2-tuple containing grids to be connected. The ordering is arbitrary.
            primary_secondary_map: Identity mapping between faces in the
                higher-dimensional grid and cells in the lower-dimensional grid. No
                assumptions are made on the type of the object at this stage.

                In the ``grids[0].dim = grids[1].dim`` case,
                the mapping is from faces of the first grid to faces of the second one.
                In the co-dimension 2 case, it maps from cells to cells.

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

        # Add the interface to the dictionary, with empty data dictionary.
        self._interface_data[intf] = data

        # The higher-dimensional grid is the first subdomain of the subdomain pair.

        # Couplings of co-dimension zero to two are allowed. Sort the pair.
        if np.abs(sd_pair[0].dim - sd_pair[1].dim) < 3:
            sd_pair = self.sort_subdomain_tuple(sd_pair)
        else:
            raise ValueError("Can only handle subdomain coupling of co-dimension <= 2")

        self._interface_to_subdomains[intf] = sd_pair

    # --------- Remove and update subdomains

    def remove_subdomain(self, sd: pp.Grid) -> None:
        """Remove a subdomain, related interfaces and boundary grid from the
        mixed-dimensional grid.

        Parameters:
           sd: The subdomain to be removed.

        """
        # Delete from the subdomain list.
        del self._subdomain_data[sd]

        # Find interfaces to be removed, store in a list.
        interfaces_to_remove: list[pp.MortarGrid] = []
        for intf in self.interfaces():
            sd_pair = self._interface_to_subdomains[intf]
            if sd_pair[0] == sd or sd_pair[1] == sd:
                interfaces_to_remove.append(intf)

        # Remove interfaces.
        for intf in interfaces_to_remove:
            del self._interface_data[intf]
            del self._interface_to_subdomains[intf]

        # Find boundary grids to be removed.
        bg_to_remove = self._subdomain_to_boundary_grid[sd]
        del self._boundary_grid_data[bg_to_remove]
        del self._subdomain_to_boundary_grid[sd]

    # ---------- Functionality related to ordering of subdomains

    def sort_subdomains(self, subdomains: list[pp.Grid]) -> list[pp.Grid]:
        """Sort subdomains.

        Sorting by descending subdomain dimension and increasing node ID, see
        :meth:`argsort_grids`.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            List containing the same subdomains, sorted.

        """
        inds = self.argsort_grids(subdomains)

        sorted_subdomains = [subdomains[i] for i in inds]
        return sorted_subdomains

    def sort_subdomain_tuple(
        self, subdomains: tuple[pp.Grid, pp.Grid]
    ) -> tuple[pp.Grid, pp.Grid]:
        """Sort two subdomains.

        Sorting by descending subdomain dimension and increasing node ID, see
        :meth:`argsort_grids`.

        Parameters:
             subdomains: ``len=2``

                Tuple of two subdomains.

        Returns:
            Tuple containing the same two subdomains, sorted.

        """
        inds = self.argsort_grids(subdomains)
        return (subdomains[inds[0]], subdomains[inds[1]])

    def sort_interfaces(self, interfaces: list[pp.MortarGrid]) -> list[pp.MortarGrid]:
        """Sort interfaces.

        Sorting by descending interface dimension and increasing MortarGrid ID,
        see :meth:`argsort_grids`.

        Parameters:
            interfaces: list of interfaces.

        Returns:
            List containing the same interfaces, sorted.

        """
        inds = self.argsort_grids(interfaces)
        sorted_interfaces = [interfaces[i] for i in inds]
        return sorted_interfaces

    def argsort_grids(self, grids: Iterable[GridLike | pp.BoundaryGrid]) -> np.ndarray:
        """Return indices that would sort the subdomains or interfaces.

        Sorting is done according to two criteria:

            - Descending dimension.
            - Ascending (mortar) grid ID (for each dimension). The ID is assigned on
              instantiation of a (mortar) grid. The codimension is not considered for
              interfaces.

        Example:

            >>> inds = mdg.argsort_grids(subdomain_list)
            >>> sorted_subdomains = [subdomain_list[i] for i in inds]

        Parameters:
            grids: List of subdomain grids.

        Returns:
            Array that sorts the grids with ``shape=(len(grids),)``.

        """
        sort_inds_all: list[np.ndarray] = list()

        # Special case: No subdomains in the mixed-dimensional grid
        if len(self._subdomain_data) == 0:
            # IMPLEMENTATION NOTE (EK): Strictly speaking, there is no requirement that
            # the grids to be sorted are subdomains in this mixed-dimensional grid (note
            # there are no calls to ``self`` in the below code, except for
            # ``self.dim_max()``). While one can imagine use cases where a
            # mixed-dimensional grid is used to sort grids that are not subdomains in
            # the mixed-dimensional grid, this is not the intended use case. Therefore,
            # we raise an error if the mixed-dimensional grid is empty, but the grid
            # list is not. To remove this restriction, one could by default convert the
            # iterator of grids into a list, and pick the maximum dimension from the
            # list of grids, instead of using ``self.dim_max()``..
            assert len([g for g in grids]) == 0
            return np.array([], dtype=int)

        # Traverse dimensions in descending order. Progress to 0 in case dim_min is
        # greater than minimum dimension of interface grids.
        for dim in np.arange(self.dim_max(), -1, -1):
            # Map from this dimension to all dimensions
            inds_in_all_dims: list[int] = list()
            # Node ids of this dimension
            ids_dim: list[int] = list()
            for ind, grid in enumerate(grids):
                if grid.dim == dim:
                    inds_in_all_dims.append(ind)
                    ids_dim.append(grid.id)
            # Ascending sorting of subdomains of this dimension according node ID
            sort_inds_dim: np.ndarray = np.argsort(ids_dim)
            # Sort the map to all grids
            sorted_inds_dim: np.ndarray = np.array(inds_in_all_dims, dtype=int)[
                sort_inds_dim
            ]
            # Append to global list
            sort_inds_all.append(sorted_inds_dim)
        return np.hstack(sort_inds_all)

    # ------------- Miscellaneous functions ---------

    def compute_geometry(self) -> None:
        """Compute geometric quantities for each contained grid."""
        for g in self.subdomains() + self.interfaces() + self.boundaries():
            g.compute_geometry()

    def set_boundary_grid_projections(self) -> None:
        """Set projections to the boundary grids.

        This method must be called after having split the fracture faces, or else
        the projections will have the wrong dimension.

        """
        for bg in self.boundaries():
            bg.set_projections()

    def copy(self) -> MixedDimensionalGrid:
        """Make a shallow copy of the mixed-dimensional grid. The underlying subdomain,
        interface and boundary grids are not copied.

        Returns:
            Copy of this mixed-dimensional grid.

        """
        mdg_copy: pp.MixedDimensionalGrid = MixedDimensionalGrid()
        mdg_copy._subdomain_data = self._subdomain_data.copy()
        mdg_copy._interface_data = self._interface_data.copy()
        mdg_copy._interface_to_subdomains = self._interface_to_subdomains.copy()
        mdg_copy._subdomain_to_boundary_grid = self._subdomain_to_boundary_grid.copy()
        mdg_copy._boundary_grid_data = self._boundary_grid_data.copy()
        return mdg_copy

    def replace_subdomains_and_interfaces(
        self,
        sd_map: Optional[Mapping[pp.Grid, pp.Grid]] = None,
        intf_map: Optional[
            Mapping[
                pp.MortarGrid,
                Union[pp.MortarGrid, Mapping[mortar_grid.MortarSides, pp.Grid]],
            ]
        ] = None,
        tol: float = 1e-6,
    ) -> None:
        """Replace grids and/or mortar grids in the mixed-dimensional grid.

        Parameters:
            sd_map: ``default=None``

                Mapping between the old and new grids. Keys are the old
                grids, and values are the new grids.
            intf_map: ``default=None``

                Mapping between the old mortar grid and new (side grids of the) mortar
                grid. Keys are the old mortar grids, and values are dictionaries with
                side grid number as keys, and side grids as values. Optionally, the
                mapping can be directly from old to new mortar grid (sometimes it is
                easier to construct the new mortar grid right away, before replacing it
                in the mixed-dimensional grid).
            tol: ``default=1e-6``

                Geometric tolerance used when updating mortar projections.

        """
        # The operation is carried out in three steps:
        # 1) Update the mortar grid. This will also update projections to and from the
        #    mortar grids so that they are valid for the new mortar grid, but not with
        #    respect to any changes in the subdomain grids.
        # 2) Update the subdomain grids in the MixedDimensionalGrid.
        # 3) Update the subdomain grids with respect to the interfaces in the
        #    MixedDimensionalGrid. This will update projections to and from mortar
        #    grids. The boundary grids will be also updated.
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

                # Updating the boundary grid.
                bg_old = self._subdomain_to_boundary_grid[sd_old]
                bg_new = pp.BoundaryGrid(sd_new)
                self._boundary_grid_data[bg_new] = self._boundary_grid_data[bg_old]
                self._subdomain_to_boundary_grid[sd_new] = bg_new
                # Compute the projections to the new boundary grid.
                bg_new.compute_geometry()
                bg_new.set_projections()
                del self._boundary_grid_data[bg_old]
                del self._subdomain_to_boundary_grid[sd_old]

    # ---- Methods for getting information on the mixed dimensional grid ----

    def diameter(
        self, cond: Optional[Callable[[Union[pp.Grid, pp.MortarGrid]], bool]] = None
    ) -> float:
        """Compute the cell diameter (mesh size) of the mixed-dimensional grid.

        A function can be passed to filter subdomains and/or interfaces.

        Parameters:
            cond: ``default=None``

                Predicate with a subdomain or interface as input.

        Returns:
            The diameter of the mixed-dimensional grid.

        """
        if cond is None:
            cond = lambda g: True
        diam_g = [np.amax(sd.cell_diameters()) for sd in self.subdomains() if cond(sd)]

        diam_mg = [np.amax(mg.cell_diameters()) for mg in self.interfaces() if cond(mg)]

        return np.amax(np.hstack((diam_g, diam_mg)))

    def dim_min(self) -> int:
        """Get the minimal dimension represented in the mixed-dimensional grid.

        Raises:
            ValueError: If the mixed-dimensional grid has no subdomains.

        Returns:
            Minimum dimension of the grids present in the hierarchy.

        """
        if len(self._subdomain_data) == 0:
            raise ValueError("The mixed-dimensional grid has no subdomains.")

        # Access protected attribute instead of subdomains() to avoid sorting, which can
        # lead to a recursion error.
        return np.min([sd.dim for sd in self._subdomain_data.keys()])

    def dim_max(self) -> int:
        """Get the maximal dimension represented in the grid.

        Raises:
            ValueError: If the mixed-dimensional grid has no subdomains.

        Returns:
            Maximum dimension of the grids present in the hierarchy.

        """
        if len(self._subdomain_data) == 0:
            raise ValueError("The mixed-dimensional grid has no subdomains.")

        # Access protected attribute instead of subdomains() to avoid sorting, which can
        # lead to a recursion error.
        return max([sd.dim for sd in self._subdomain_data.keys()])

    def num_subdomain_cells(
        self, cond: Optional[Callable[[pp.Grid], bool]] = None
    ) -> int:
        """Compute the total number of subdomain cells of the mixed-dimensional grid.

        A function can be passed to filter subdomains.

        Parameters:
            cond: ``default=None``

                Predicate with a subdomain as input.

        Returns:
            The total number of subdomain cells of the grid bucket.

        """
        if cond is None:
            cond = lambda g: True
        return np.sum(  # type: ignore
            [grid.num_cells for grid in self.subdomains() if cond(grid)], dtype=int
        )

    def num_interface_cells(
        self, cond: Optional[Callable[[pp.MortarGrid], bool]] = None
    ) -> int:
        """Compute the total number of mortar cells of the mixed-dimensional grid.

        A function can be passed to filter interfaces.

        Parameters:
            cond: ``default=None``

                Predicate with an interface as input.

        Returns:
            The total number of mortar cells of the grid bucket.

        """
        if cond is None:
            cond = lambda g: True
        return np.sum(  # type: ignore
            [mg.num_cells for mg in self.interfaces() if cond(mg)],
            dtype=int,
        )

    def num_subdomains(self) -> int:
        """
        Returns:
            Total number of subdomains in the mixed-dimensional grid.

        """
        return len(self._subdomain_data)

    def num_interfaces(self) -> int:
        """
        Returns:
            Total number of interfaces in the mixed-dimensional grid.

        """
        return len(self._interface_data)

    def __str__(self) -> str:
        """A simplified string representation of the mixed-dimensional grid,
        including information about dimensions, number of subdomains and interfaces."""

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
                f"{len(list(self.subdomains(dim=dim)))} grids of dimension {dim}, with "
                f"in total {num_cells} cells and {num_nodes} nodes. \n"
            )

        s += f"Total number of interfaces: {self.num_interfaces()}\n"
        for dim in range(self.dim_max(), self.dim_min(), -1):
            num_e = 0
            nc = 0
            for mg in self.interfaces(dim=dim - 1, codim=1):
                num_e += 1
                nc += mg.num_cells

            s += (
                f"{num_e} interfaces between grids of dimension {dim} and {dim - 1}"
                f" with in total {nc} cells.\n"
            )
            for mg in self.interfaces(dim=dim - 2, codim=2):
                s += f"Interface between grids of dimension {dim} and {dim - 2} with"
                s += f" {mg.num_cells} cells.\n"
        return s

    def __repr__(self) -> str:
        """A string representation of the mixed-dimensional grid, including information
        about dimension, involved grids and their topological information."""

        s = (
            f"Mixed-dimensional grid containing {self.num_subdomains()} grids and "
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
                s += f"{num_sd} grids of dimension {dim} with in total {nc} cells\n"
        if self.num_interfaces() > 0:
            for dim in range(self.dim_max(), self.dim_min(), -1):
                num_intf = 0
                num_intf_cells = 0
                for intf in self.interfaces(dim=dim - 1, codim=1):
                    num_intf += 1
                    num_intf_cells += intf.num_cells

                s += (
                    f"{num_intf} interfaces between grids of dimension {dim} and"
                    f" {dim - 1} with in total {num_intf_cells} mortar cells.\n"
                )
                num_intf_codim_2 = 0
                num_intf_cells_codim_2 = 0
                for intf in self.interfaces(dim=dim - 2, codim=2):
                    num_intf_codim_2 += 1
                    num_intf_cells_codim_2 += intf.num_cells
                if num_intf_codim_2 > 0:
                    s += (
                        f"{num_intf_codim_2} interfaces between grids of dimension"
                        f" {dim} and {dim - 2} with in total {num_intf_cells_codim_2}"
                        f" mortar cells.\n"
                    )

        return s
