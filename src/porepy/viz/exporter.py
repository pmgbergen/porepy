"""
Module for data IO from and to vtu format via meshio.

"""
from __future__ import annotations

import os
import sys
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import meshio
import numpy as np
from deepdiff import DeepDiff

import porepy as pp

# Object type to store data to export.
Field = namedtuple("Field", ["name", "values"])

# Object for managing meshio-relevant data, as well as a container
# for its storage, taking dimensions as inputs. Since 0d grids are
# stored as 'None', allow for such values.
Meshio_Geom = namedtuple("Meshio_Geom", ["pts", "connectivity", "cell_ids"])
MD_Meshio_Geom = Dict[int, Union[None, Meshio_Geom]]

# All allowed data structures to define data for exporting
DataInput = Union[
    # Keys for states
    str,
    # Subdomain specific data types
    Tuple[List[pp.Grid], str],
    Tuple[pp.Grid, str, np.ndarray],
    Tuple[str, np.ndarray],
    # Interface specific data types
    Tuple[List[pp.MortarGrid], str],
    Tuple[pp.MortarGrid, str, np.ndarray],
]

# Data structure in which data is stored after preprocessing
SubdomainData = Dict[Tuple[pp.Grid, str], np.ndarray]
InterfaceData = Dict[Tuple[pp.MortarGrid, str], np.ndarray]


class Exporter:
    """Class for exporting and importing data and grid to and from vtu format.

    The Exporter allows for various ways to express which state variables,
    on which grids, and which extra data should be exported. A thorough
    demonstration is available as a dedicated tutorial. Check out
    tutorials/exporter.ipynb.

    In general, pvd files gather data exported in separate files, including
    data on differently dimensioned grids, constant data, and finally time
    steps. For transient simulations with multiple time steps, a single pvd
    file takes care of the ordering of all printed vtu files.

    In the case of different keywords, change the file name with
    "change_name".

    NOTE: the following names are reserved for constant data exporting
    and should not be used otherwise (otherwise data is overwritten):
    grid_dim, is_mortar, mortar_side, cell_id, subdomain_id,
    interface_id.

    Parameters:
        grid: Subdomain or mixed-dimensional grid containing all mesh information
            (and data in the latter case) to be exported.
        file_name: Basis for file names used for storing the output.
        folder_name: Name of the folder in which files are stored.
        kwargs: Optional keywords arguments:
            fixed_grid (boolean): to control whether the grid may be redefined later
                (default True).
            binary (boolean): controlling whether data is stored in binary format
                (default True).
            export_constants_separately (boolean): controlling whether
                constant data is exported in separate files, which may be of
                interest when exporting large data sets (in particular of constant
                data) for many time steps (default True); note, however, that the
                mesh is exported to each vtu file, which may also require
                significant amount of storage.

    Raises:
        TypeError: If grid has other type than :class:`~pp.grids.grid.Grid` or
            :class:`~pp.grids.md_grid.MixedDimensionalGrid`.
        TypeError: If kwargs contains unexpected keyword arguments.

    Examples:
        # Here, merely a brief demonstration of the use of Exporter is presented.

        # If you need to export the state with key "pressure" on a single grid:
        save = Exporter(g, "solution", folder_name="results")
        save.write_vtu(["pressure"])

        # In a time loop, if you need to export states with keys "pressure" and
        # "displacement" stored in a mixed-dimensional grid, do:

        save = Exporter(mdg, "solution", folder_name="results")
        while time:
            save.write_vtu(["pressure", "displacement"], time_step=i)
        save.write_pvd(times)

        # where times is a list of actual times (not time steps), associated
        # to the previously exported time steps. If times is not provided
        # the time steps will be used instead.

    """

    def __init__(
        self,
        grid: Union[pp.Grid, pp.MixedDimensionalGrid],
        file_name: str,
        folder_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Exporter is operating on mixed-dimensional grids. Convert to such.
        if isinstance(grid, pp.Grid):
            self._mdg: pp.MixedDimensionalGrid = pp.MixedDimensionalGrid()
            """Mixed-dimensional grid representation of the input grid."""

            self._mdg.add_subdomains(grid)
        elif isinstance(grid, pp.MixedDimensionalGrid):
            self._mdg = grid
        else:
            raise TypeError(
                "Exporter only supports subdomain and mixed-dimensional grids."
            )

        # Store target location for storing vtu and pvd files.
        self._file_name: str = file_name
        """Prefix for output files."""

        self._folder_name: Optional[str] = folder_name
        """Folder name for output."""

        # Check for optional keywords
        self._fixed_grid: bool = kwargs.pop("fixed_grid", True)
        """Flag controlling the grid is fixed."""

        self._binary: bool = kwargs.pop("binary", True)
        """Flag controlling whether data is stored in binary format."""

        self._export_constants_separately: bool = kwargs.pop(
            "export_constants_separately", True
        )
        """Flag controlling whether constant data is exported to a separate file."""

        if kwargs:
            msg = "Exporter() got unexpected keyword argument '{}'"
            raise TypeError(msg.format(kwargs.popitem()[0]))

        # Generate infrastructure for storing fixed-dimensional grids in
        # meshio format. Include all but the 0-d grids

        self._dims = np.unique([sd.dim for sd in self._mdg.subdomains() if sd.dim > 0])
        """Array of dimensions of all present subdomains."""

        self.meshio_geom: MD_Meshio_Geom = dict()
        """Dictionary storing the meshio representation of the subdomains."""

        # Generate infrastructure for storing fixed-dimensional mortar grids
        # in meshio format.

        self._m_dims = np.unique([intf.dim for intf in self._mdg.interfaces()])
        """Array of dimensions of the mortar grids."""

        self.m_meshio_geom: MD_Meshio_Geom = dict()
        """Dictionary storing the meshio representation of the mortar grids."""

        # Generate geometrical information in meshio format
        self._update_meshio_geom()
        self._update_constant_mesh_data()

        # Infrastructure for time management

        self._time_step_counter: int = 0
        """Counter of the time steps to be exported, to be used as appendix."""

        self._exported_timesteps: list[int] = []
        """List of exported time steps, gatherer for pvd files."""

        self._time_step_constant_data: Optional[int] = None
        """Reference to the last time step used for exporting constant data."""

        self._exported_timesteps_constants: list[int] = list()
        """Storage for file name extensions for time steps, regarding constant data."""

        self._exported_constant_data_up_to_date: bool = False
        """Identifier for whether constant data is up-to-date."""

        # Infrastructure for keeping track of assigned data

        self._has_subdomain_data = False
        """Identifier for whether subdomain data has been assigned to Exporter."""

        self._has_interface_data = False
        """Identifier for whether interface data has been assigned to Exporter."""

        self._has_constant_subdomain_data = False
        """Identifier for whether constant subdomain data has been assigned to
        Exporter."""

        self._has_constant_interface_data = False
        """Identifier for whether constant interface data has been assigned to
        Exporter."""

        # Misc

        self._padding = 6
        """Padding of zeros for creating the time step dependent appendix for output."""

    def import_from_vtu(
        self, keys: Union[str, list[str]], file_names: Union[str, list[str]], **kwargs
    ) -> None:
        """Import state variables from vtu file. It is assumed that the vtu file was
        created using PorePy, e.g., that the file names follow PorePy conventions,
        the mixed-dimensional grid is split in the usual way etc.

        Parameters:
            keys: keywords addressing cell data to be transferred.
            file_names: list of vtu files to be considered.
            kwargs:
                automatic: boolean flag controlling whether dimensionality of the grids
                    and whether it is a subdomain or interface grid is read
                    automatically from the file names; default is True.
                dims (int or list of int): compatible with file_names; list of
                    dimensions of the corresponding grids.
                are_subdomain_data (bool or list of bool): comparible with file_names;
                    list of indicators whether the corresponding grid is a subdomain
                    grid; default is True.

        Raises:
            ValueError: if some of the data is not compatible with the supposedly
                corresponding grid.

        """

        # Aux. method: Inverse of _to_vector_format, used to define vector-ranged values
        def _from_vector_format(
            value: np.ndarray, grid: Union[pp.Grid, pp.MortarGrid]
        ) -> np.ndarray:
            """Check whether the value array has the right dimension corresponding
            to the grid size. If possible, translate the value to a scalar-ranged
            object.

            This method is meant for cell data only.

            Parameters:
                value: input array to be converted.
                grid: subdomain or interface to which value is associated to.

            Raises:
                ValueError: if the value array is not compatible with the grid.

            """
            # Make some checks - note that this method handles cell data only.
            if not value.size % grid.num_cells == 0:
                # This line will raise an error if node or face data is exported.
                raise ValueError("The data array is not compatible with the grid.")

            # Flatten the data compatible with _to_vector_format.
            return np.ravel(value, "C")

        # Convert file_names to a list
        if isinstance(file_names, str):
            file_names = [file_names]

        # Consider each file separately.
        # Procedure has three steps:
        # 1. Determine dimensionality and type of grid, addressed by the vtu file.
        # 2. Check whether the vtu file is compatible with the corresponding grids.
        # 3. Transfer data from vtu to the mixed-dimensional grid.
        for i, file_name in enumerate(file_names):

            # Read all data from vtu
            vtu_data = meshio.read(file_name)

            # 1st step: Determine dimension of the grid associated to vtu file, and
            # whether the grid corresponds to a subdomain or interface.

            # Allow for automatic detection from the file name.
            if kwargs.pop("automatic", True):
                # Assuming grid_dim and is_mortar are among the stored vtu data
                # one could utilize those, but this does not have to be the case.
                # Thus, use naming convention of the Exporter for this.

                # Remove ending '.vtu' from file_name, and decompose into pieces
                # separated by '_'.
                file_name_pieces = Path(file_name).stem.split("_")

                # If the last two are numbers, the first one denotes the dimension.
                assert file_name_pieces[-1].isnumeric()
                dim_pos = -2 if file_name_pieces[-2].isnumeric() else -1
                dim = int(file_name_pieces[dim_pos])

                # If the key words before are 'mortar', or 'mortar' and 'constant',
                # the vtu file corresponds to intefaces; otherwise to subdomains.
                is_subdomain_data = not (
                    file_name_pieces[dim_pos - 1] == "mortar"
                    or file_name_pieces[dim_pos - 1] == "constant"
                    and file_name_pieces[dim_pos - 2] == "mortar"
                )

            else:
                # Read from keyword arguments

                # Dimensionality of the grid
                if "dims" not in kwargs:
                    raise ValueError(
                        """Provide dimensionality of the grids associated
                        to the vtu files."""
                    )
                dims = kwargs.pop("dims")
                assert isinstance(dims, list) or isinstance(dims, int)
                dim = dims[i] if isinstance(dims, list) else dims

                # Whether the grid is a subdomain or interface
                if "are_subdomain_data" not in kwargs:
                    is_subdomain_data = True
                else:
                    are_subdomain_data = kwargs.pop("are_subdomain_data")
                    assert isinstance(are_subdomain_data, bool) or isinstance(
                        are_subdomain_data, list
                    )
                    is_subdomain_data = (
                        are_subdomain_data[i]
                        if isinstance(are_subdomain_data, list)
                        else are_subdomain_data
                    )

            # 2nd step: Make sure that the vtu file and the current grid are compatible,
            # by comparing coordinates of nodes, connectivity of cells, and thereby
            # number of grid entities. They are identified as identical if they
            # generate the same meshio representation. NOTE: Meshes could be compatible
            # via some transformation. However, here we require identical grids.

            meshio_geometry = (
                self.meshio_geom[dim] if is_subdomain_data else self.m_meshio_geom[dim]
            )
            assert isinstance(meshio_geometry, Meshio_Geom)

            # Make sure that both grids have same nodes
            assert np.all(np.isclose(meshio_geometry.pts, vtu_data.points))

            for i, connectivity in enumerate(meshio_geometry.connectivity):

                # Make sure that the connectivity patterns are identical
                assert (
                    DeepDiff(
                        connectivity,
                        vtu_data.cells[i],
                        significant_digits=8,
                        number_format_notation="e",
                        ignore_numeric_type_changes=True,
                    )
                    == {}
                )

            # 3rd step: Transfer data. Consider each key separately.
            for key in keys:

                # Only continue if the key is present in the data
                # IMPLEMENTATION NOTE: To also consider node data, add an else below.
                if key in vtu_data.cell_data:

                    # Data is stored in a list with each element storing data for one
                    # specific cell type (most relevant for polygonal and polyhedral
                    # grids). Accumulate the data again, assuming the same
                    # dimensionality in each cell.
                    value = np.concatenate(tuple(vtu_data.cell_data[key]), axis=0)

                    # Chop data in pieces compatible with the subdomains and interfaces
                    offset = 0
                    if is_subdomain_data:
                        for sd, sd_data in self._mdg.subdomains(
                            dim=dim, return_data=True
                        ):
                            if pp.STATE not in sd_data:
                                sd_data[pp.STATE] = {}
                            sd_data[pp.STATE][key] = _from_vector_format(
                                value[offset : offset + sd.num_cells], sd
                            )
                            offset += sd.num_cells
                    else:
                        for intf, intf_data in self._mdg.interfaces(
                            dim=dim, return_data=True
                        ):
                            if pp.STATE not in intf_data:
                                intf_data[pp.STATE] = {}
                            intf_data[pp.STATE][key] = _from_vector_format(
                                value[offset : offset + intf.num_cells], intf
                            )
                            offset += intf.num_cells

    def add_constant_data(
        self, data: Optional[Union[DataInput, list[DataInput]]] = None
    ) -> None:
        """Collect user-defined constant-in-time data, associated with grids,
        and to be exported to separate files instead of the main files.

        In principle, constant data is not different from standard output
        data. It is merely printed to another file and just when any constant
        data is updated (via updates of the grid, or the use of this routine).
        Hence, the same input formats are allowed as for the usual field data,
        which is varying in time. As part of the routine, the data is converted
        to the same unified format.

        Parameters:
            data: subdomain and interface data, prescribed through strings,
                or tuples of subdomains/interfaces, keys and values. If not
                provided, only geometrical information is exported.

                NOTE: The user has to make sure that each unique key has
                associated data values for all or no grids of each specific
                dimension.

        """
        # Interpret change in constant data. Has the effect that
        # the constant data container will be exported at the
        # next application of write_vtu().
        self._exported_constant_data_up_to_date = False

        # Preprocessing of the user-defined constant data
        # Has two main goals:
        # 1. Sort wrt. whether data is associated to subdomains or interfaces.
        # 2. Unify data type.
        subdomain_data, interface_data = self._sort_and_unify_data(data)

        # Add the user-defined data to the containers for constant data.
        for key_s, value in subdomain_data.items():
            self._constant_subdomain_data[key_s] = value.copy()
        for key_i, value in interface_data.items():
            self._constant_interface_data[key_i] = value.copy()

    def write_vtu(
        self,
        data: Optional[Union[DataInput, list[DataInput]]] = None,
        time_dependent: bool = False,
        time_step: Optional[int] = None,
        grid: Optional[Union[pp.Grid, pp.MixedDimensionalGrid]] = None,
    ) -> None:
        """Interface function to export the grid and additional data with meshio.

        In 1d the cells are represented as lines, 2d the cells as polygon or
        triangle/quad, while in 3d as polyhedra/tetrahedra/hexahedra.
        In all the dimensions the geometry of the mesh needs to be computed.

        Parameters:
            data: subdomain and interface data, prescribed through strings,
                or tuples of subdomains/interfaces, keys and values. If not
                provided only geometrical infos are exported.

                NOTE: The user has to make sure that each unique key has
                associated data values for all or no grids of each specific
                dimension.
            time_dependent: If False (default), file names will not be appended
                with an index that marks the time step. Can be overwritten by
                giving a value to time_step; if not, the file names will
                subsequently be ending with 1, 2, etc.
            time_step: will be used as appendix to define the file corresponding
                to this specific time step.
            grid: subdomain or mixed-dimensional grid if it is not fixed and
                should be updated.

        Raises:
            ValueError: if a grid is provided as argument although the exporter
                has been instructed that the grid is fixed.

        """

        # Update the grid but only if allowed, i.e., the initial grid
        # has not been characterized as fixed.
        if self._fixed_grid and grid is not None:
            raise ValueError("Inconsistency in exporter setting")
        elif not self._fixed_grid and grid is not None:
            # Require a mixed-dimensional grid. Thus convert if single subdomain
            # grid provided.
            if isinstance(grid, pp.Grid):
                # Create a new mixed-dimensional solely with grid as single subdomain.
                self._mdg = pp.MixedDimensionalGrid()
                self._mdg.add_subdomains(grid)
            else:
                self._mdg = grid

            # Update geometrical info in meshio format for the updated grid
            self._update_meshio_geom()
            self._update_constant_mesh_data()

        # If the problem is time dependent, but no time step is set, we set one
        # using the updated, internal counter.
        if time_dependent and time_step is None:
            time_step = self._time_step_counter
            self._time_step_counter += 1

        # If time step is prescribed, store it.
        if time_step is not None:
            self._exported_timesteps.append(time_step)

        # Preprocessing step with two main goals:
        # 1. Sort wrt. whether data is associated to subdomains or interfaces.
        # 2. Unify data type.
        subdomain_data, interface_data = self._sort_and_unify_data(data)

        # Export constant data to separate or standard files
        if self._export_constants_separately:
            # Export constant data to vtu when outdated
            if not self._exported_constant_data_up_to_date:
                # Export constant subdomain data to vtu
                if self._constant_subdomain_data:
                    self._has_constant_subdomain_data = True
                    self._export_data_vtu(
                        self._constant_subdomain_data, time_step, constant_data=True
                    )

                # Export constant interface data to vtu
                if self._constant_interface_data:
                    self._has_constant_interface_data = True
                    self._export_data_vtu(
                        self._constant_interface_data,
                        time_step,
                        constant_data=True,
                        interface_data=True,
                    )

                # Store the time step counter for later reference
                # (required for pvd files)
                # NOTE: The counter is only updated if the constant
                # data is outdated, so if the mesh has been updated
                # or new constant data has been added in the course
                # of the simulation.
                self._time_step_constant_data = time_step

                # Identify the constant data as fixed. Has the effect
                # that the constant data won't be exported if not the
                # mesh is updated or new constant data is added.
                self._exported_constant_data_up_to_date = True

            # Store the timestep referring to the origin of the constant data
            if hasattr(self, "_time_step_constant_data"):
                if self._time_step_constant_data is not None:
                    self._exported_timesteps_constants.append(
                        self._time_step_constant_data
                    )
        else:
            # Append constant subdomain and interface data to the
            # standard containers for subdomain and interface data.
            for key_sd, value in self._constant_subdomain_data.items():
                subdomain_data[key_sd] = value.copy()
            for key_intf, value in self._constant_interface_data.items():
                interface_data[key_intf] = value.copy()

        # Export subdomain and interface data to vtu format if existing
        if subdomain_data:
            self._has_subdomain_data = True
            self._export_data_vtu(subdomain_data, time_step)
        if interface_data:
            self._has_interface_data = True
            self._export_data_vtu(interface_data, time_step, interface_data=True)

        # Export mixed-dimensional grid to pvd format
        file_name = self._make_file_name(self._file_name, time_step, extension=".pvd")
        file_name = self._append_folder_name(self._folder_name, file_name)
        self._export_mdg_pvd(file_name, time_step)

    def write_pvd(
        self,
        times: Optional[np.ndarray] = None,
        file_extension: Optional[Union[np.ndarray, list[int]]] = None,
    ) -> None:
        """Interface function to export in PVD file the time loop information.
        The user should open only this file in ParaView.

        We assume that the VTU associated files have the same name, and that.
        the VTU associated files are in the working directory.

        Parameters:
            times: array of actual times to be exported. These will be the times
                associated with individual time steps in, say, ParaView. By default,
                the times will be associated with the order in which the time steps
                were exported. This can be overridden by the file_extension argument.
                If no times are provided, the exported time steps are used.
            file_extension: End of file names used in the export of individual time
                steps, see self.write_vtu(). If provided, it should have the same
                length as time. If not provided, the file names will be picked from
                those used when writing individual time steps.

        """

        if times is None:
            times = np.array(self._exported_timesteps)

        if file_extension is None:
            file_extension = self._exported_timesteps
        elif isinstance(file_extension, np.ndarray):
            file_extension = file_extension.tolist()

            # Make sure that the inputs are consistent
            assert isinstance(file_extension, list)
            assert len(file_extension) == times.shape[0]

        # Extract the time steps related to constant data and
        # complying with file_extension. Implicitly test whether
        # file_extension is a subset of _exported_timesteps.
        # Only if constant data has been exported.
        include_constant_data = (
            self._export_constants_separately
            and len(self._exported_timesteps_constants) > 0
        )
        if include_constant_data:
            indices = [self._exported_timesteps.index(e) for e in file_extension]
            file_extension_constants = [
                self._exported_timesteps_constants[i] for i in indices
            ]

        # Perform the same procedure as in _export_mdg_pvd
        # but looping over all designated time steps.

        o_file = open(
            self._append_folder_name(self._folder_name, self._file_name) + ".pvd", "w"
        )
        b = "LittleEndian" if sys.byteorder == "little" else "BigEndian"
        c = ' compressor="vtkZLibDataCompressor"'
        header = (
            '<?xml version="1.0"?>\n'
            + '<VTKFile type="Collection" version="0.1" '
            + 'byte_order="%s"%s>\n' % (b, c)
            + "<Collection>\n"
        )
        o_file.write(header)
        fm = '\t<DataSet group="" part="" timestep="%f" file="%s"/>\n'

        # Gather all data, and assign the actual time.
        for time, fn in zip(times, file_extension):
            # Go through all possible data types, analogously to
            # _export_mdg_pvd.

            # Subdomain data
            for dim in self._dims:
                if self._has_subdomain_data and self.meshio_geom[dim] is not None:
                    o_file.write(
                        fm % (time, self._make_file_name(self._file_name, fn, dim))
                    )

            # Interface data.
            for dim in self._m_dims:
                if self._has_interface_data and self.m_meshio_geom[dim] is not None:
                    o_file.write(
                        fm
                        % (
                            time,
                            self._make_file_name(self._file_name + "_mortar", fn, dim),
                        )
                    )

        # Optionally, do the same for constant data.
        if include_constant_data:
            for time, fn_constants in zip(times, file_extension_constants):
                # Constant subdomain data.
                for dim in self._dims:
                    if (
                        self._has_constant_subdomain_data
                        and self.meshio_geom[dim] is not None
                    ):
                        o_file.write(
                            fm
                            % (
                                time,
                                self._make_file_name(
                                    self._file_name + "_constant", fn_constants, dim
                                ),
                            )
                        )

                # Constant interface data.
                for dim in self._m_dims:
                    if (
                        self._has_constant_interface_data
                        and self.m_meshio_geom[dim] is not None
                    ):
                        o_file.write(
                            fm
                            % (
                                time,
                                self._make_file_name(
                                    self._file_name + "_constant_mortar",
                                    fn_constants,
                                    dim,
                                ),
                            )
                        )

        o_file.write("</Collection>\n" + "</VTKFile>")
        o_file.close()

    # Some auxiliary routines used in write_vtu()

    def _sort_and_unify_data(
        self,
        data=None,
        # ignore type which is essentially Union[DataInput, list[DataInput]]
    ) -> tuple[SubdomainData, InterfaceData]:
        """Preprocess data.

        The routine has two goals:
        1. Splitting data into subdomain and interface data.
        2. Unify the data format. Store data in dictionaries, with keys given by
            subdomains/interfaces and names, and values given by the data arrays.

        Parameters:
            data: data provided by the user in the form of strings and/or tuples
                of subdomains/interfaces.

        Returns:
            Subdomain and interface data decomposed and brought into unified format.

        Raises:
            ValueError: if the data type provided is not supported.

        """

        # The strategy is to traverse the input data and check each data point
        # for its type and apply a corresponding conversion to a unique format.
        # For each supported input data format, a dedicated routine is implemented
        # to 1. identify whether the type is present; and 2. update the data
        # dictionaries (to be returned in this routine) in the correct format:
        # The dictionaries use (subdomain,key) and (interface,key) as key and the data
        # array as value.

        # Now a list of these subroutines follows before the definition of the
        # actual routine.

        # Aux. method: Transform scalar- to vector-ranged values.
        def _to_vector_format(
            value: np.ndarray, grid: Union[pp.Grid, pp.MortarGrid]
        ) -> np.ndarray:
            """Check whether the value array has the right dimension corresponding
            to the grid size. If possible, translate the value to a vectorial
            object, but do nothing if the data naturally can be interpreted as
            scalar data.

            Parameters:
                value: input array to be converted.
                grid: subdomain or interface to which value is associated to.

            Raises:
                ValueError: if the value array is not compatible with the grid.

            """
            # Make some checks
            if not value.size % grid.num_cells == 0:
                # This line will raise an error if node or face data is exported.
                raise ValueError("The data array is not compatible with the grid.")

            # Convert to vectorial data if more data provided than grid cells available,
            # and the value array is not already in vectorial format
            if not value.size == grid.num_cells and not (
                len(value.shape) > 1 and value.shape[1] == grid.num_cells
            ):
                value = np.reshape(value, (-1, grid.num_cells), "F")

            return value

        def add_data_from_str(
            data_pt: str, subdomain_data: dict, interface_data: dict
        ) -> tuple[dict, dict, bool]:
            """Check whether data is provided by a key of a field - could be both
            subdomain and interface data. If so, collect all data corresponding to
            subdomains and interfaces identified by the key.

            Parameters:
                data_pt: data identifier via the associated key used in pp.STATE.
                subdomain_data: container for subdomain data.
                interface_data: container for interface data.

            Returns:
                Updated data containers and flag of success.

            Raises:
                ValueError: if no data available in the mixed-dimensional grid for
                    given key.

            """

            # Only continue in case data is of type str
            if isinstance(data_pt, str):

                # Identify the key provided through the data.
                key = data_pt

                # Initialize tag storing whether there exists data associated to the key
                has_key = False

                def _add_data(
                    key: str,
                    grid: Union[pp.Grid, pp.MortarGrid],
                    grid_data: dict,
                    export_data: dict,
                ) -> bool:
                    if pp.STATE in grid_data and key in grid_data[pp.STATE]:
                        # Fetch data and convert to vectorial format if needed
                        value: np.ndarray = _to_vector_format(
                            grid_data[pp.STATE][key], grid
                        )

                        # Add data point in correct format to the collection
                        export_data[(grid, key)] = value

                        # Mark as success
                        return True
                    else:
                        return False

                # Check data associated to subdomain field data
                for sd, sd_data in self._mdg.subdomains(return_data=True):
                    if _add_data(key, sd, sd_data, subdomain_data):
                        has_key = True

                # Check data associated to interface field data
                for intf, intf_data in self._mdg.interfaces(return_data=True):
                    if _add_data(key, intf, intf_data, interface_data):
                        has_key = True

                # Make sure the key exists
                if not has_key:
                    raise ValueError(
                        f"No data with provided key {key} present in the grid."
                    )

                # Return updated dictionaries and indicate successful conversion.
                return subdomain_data, interface_data, True

            else:
                # Return original data dictionaries and indicate no modification.
                return subdomain_data, interface_data, False

        def add_data_from_tuple_subdomains_str(
            data_pt: tuple[list[pp.Grid], str],
            subdomain_data: dict,
            interface_data: dict,
        ) -> tuple[dict, dict, bool]:
            """Check whether data is provided as tuple (subdomains, key),
            where subdomains is a list of subdomains, and key is a string.
            This routine explicitly checks only for subdomain data.

            Parameters:
                data_pt: data iendtifier via the key used in pp.STATE and
                    a specific subdomain.
                subdomain_data: container for subdomain data
                interface_data: container for interface data

            Returns:
                Updated data containers and flag of success.

            Raises:
                ValueError: if there exists no state in the subdomain data with given
                    key.

            """

            # Implementation of isinstance(data_pt, tuple[list[pp.Grid], str]).
            isinstance_tuple_subdomains_str = list(map(type, data_pt)) == [
                list,
                str,
            ] and all([isinstance(sd, pp.Grid) for sd in data_pt[0]])

            # If of correct type, convert to unique format and update subdomain data.
            if isinstance_tuple_subdomains_str:

                # By construction, the components are a list of grids and a key.
                subdomains: list[pp.Grid] = data_pt[0]
                key = data_pt[1]

                # Loop over grids and fetch the states corresponding to the key
                for sd in subdomains:

                    # Fetch the data dictionary containing the data value
                    sd_data = self._mdg.subdomain_data(sd)

                    # Make sure the data exists.
                    if not (pp.STATE in sd_data and key in sd_data[pp.STATE]):
                        raise ValueError(
                            f"""No state with prescribed key {key}
                            available on selected subdomains."""
                        )

                    # Fetch data and convert to vectorial format if suitable
                    value = _to_vector_format(sd_data[pp.STATE][key], sd)

                    # Add data point in correct format to collection
                    subdomain_data[(sd, key)] = value

                # Return updated dictionaries and indicate successful conversion.
                return subdomain_data, interface_data, True

            else:
                # Return original data dictionaries and indicate no modification.
                return subdomain_data, interface_data, False

        # Aux. method: Detect and convert data of form ([interfaces], "key").
        def add_data_from_tuple_interfaces_str(
            data_pt: tuple[list[pp.MortarGrid], str],
            subdomain_data: dict,
            interface_data: dict,
        ) -> tuple[dict, dict, bool]:
            """Check whether data is provided as tuple (interfaces, key),
            where interfaces is a list of interfaces, and key is a string.
            This routine explicitly checks only for interface data.

            This routine is a translation of add_data_from_tuple_subdomains_str to
                interfaces.

            Parameters:
                data_pt: data identifier combining to be addressed interfaces and
                    a key present in pp.STATE.
                subdomain_data: container for subdomain data.
                interface_data: container for interface data.

            Returns:
                Updated data containers and flag of success.

            Raises:
                ValueError: if there exists no state in the interface data with given
                    key.

            """

            # Implementation of isinstance(t, tuple[list[pp.MortarGrid], str]).
            isinstance_tuple_interfaces_str = list(map(type, data_pt)) == [
                list,
                str,
            ] and all([isinstance(intf, pp.MortarGrid) for intf in data_pt[0]])

            # If of correct type, convert to unique format and update subdomain data.
            if isinstance_tuple_interfaces_str:

                # By construction, the components are a list of interfaces and a key.
                interfaces: list[pp.MortarGrid] = data_pt[0]
                key = data_pt[1]

                # Loop over interfaces and fetch the states corresponding to the key
                for intf in interfaces:

                    # Fetch the data dictionary containing the data value
                    intf_data = self._mdg.interface_data(intf)

                    # Make sure the data exists.
                    if not (pp.STATE in intf_data and key in intf_data[pp.STATE]):
                        raise ValueError(
                            f"""No state with prescribed key {key}
                            available on selected interfaces."""
                        )

                    # Fetch data and convert to vectorial format if suitable
                    value = _to_vector_format(intf_data[pp.STATE][key], intf)

                    # Add data point in correct format to collection
                    interface_data[(intf, key)] = value

                # Return updated dictionaries and indicate successful conversion.
                return subdomain_data, interface_data, True

            else:
                # Return original data dictionaries and indicate no modification.
                return subdomain_data, interface_data, False

        def add_data_from_tuple_subdomain_str_array(
            data_pt: tuple[pp.Grid, str, np.ndarray],
            subdomain_data: dict,
            interface_data: dict,
        ) -> tuple[dict, dict, bool]:
            """Check whether data is provided as tuple (sd, key, data),
            where sd is a single subdomain, key is a string, and data is a user-defined
            data array. This routine explicitly checks only for subdomain data.

            Parameters:
                data_pt: data tuple containing subdomain, arbitrary key, and associated
                    cell data.
                subdomain_data: container for subdomain data.
                interface_data: container for interface data.

            Returns:
                Updated data containers and flag of success.

            Raises:
                ValueError: if there exists no state in the interface data with given
                    key.

            """

            # Implementation of isinstance(t, tuple[pp.Grid, str, np.ndarray]).
            # NOTE: The type of a grid identifies the specific grid type (simplicial
            # etc.) Thus, isinstance is used here to detect whether a grid is provided.
            isinstance_tuple_subdomain_str_array = isinstance(
                data_pt[0], pp.Grid
            ) and list(map(type, data_pt))[1:] == [str, np.ndarray]

            # Convert data to unique format and update the subdomain data dictionary.
            if isinstance_tuple_subdomain_str_array:

                # Interpret (sd, key, value) = (data_pt[0], data_pt[1], data_pt[2]);
                sd = data_pt[0]
                key = data_pt[1]
                value = _to_vector_format(data_pt[2], sd)

                # Add data point in correct format to collection
                subdomain_data[(sd, key)] = value

                # Return updated dictionaries and indicate successful conversion.
                return subdomain_data, interface_data, True

            else:
                # Return original data dictionaries and indicate no modification.
                return subdomain_data, interface_data, False

        def add_data_from_tuple_interface_str_array(
            data_pt: tuple[pp.MortarGrid, str, np.ndarray],
            subdomain_data: dict,
            interface_data: dict,
        ) -> tuple[dict, dict, bool]:
            """Check whether data is provided as tuple (g, key, data),
            where e is a single interface, key is a string, and data is a user-defined
            data array. This routine explicitly checks only for interface data.

            Translation of add_data_from_tuple_subdomain_str_array to interfaces.

            Parameters:
                data_pt: data tuple containing interface, arbitrary key, and associated
                    cell data.
                subdomain_data: container for subdomain data.
                interface_data: container for interface data.

            Returns:
                Updated data containers and flag of success.

            Raises:
                ValueError: if there exists no state in the interface data with given
                    key.

            """

            # Implementation of isinstance(t, tuple[pp.MortarGrid, str, np.ndarray]).
            isinstance_tuple_interface_str_array = list(map(type, data_pt)) == [
                pp.MortarGrid,
                str,
                np.ndarray,
            ]

            # Convert data to unique format and update the interface data dictionary.
            if isinstance_tuple_interface_str_array:
                # Interpret (intf, key, value) = (data_pt[0], data_pt[1], data_pt[2]);
                intf = data_pt[0]
                key = data_pt[1]
                value = _to_vector_format(data_pt[2], intf)

                # Add data point in correct format to collection
                interface_data[(intf, key)] = value

                # Return updated dictionaries and indicate successful conversion.
                return subdomain_data, interface_data, True

            else:
                # Return original data dictionaries and indicate no modification.
                return subdomain_data, interface_data, False

        def add_data_from_tuple_str_array(
            data_pt: tuple[str, np.ndarray],
            subdomain_data: dict,
            interface_data: dict,
        ) -> tuple[dict, dict, bool]:
            """Check whether data is provided by a tuple (key, data),
            where key is a string, and data is a user-defined data array.

            This only works when the mixed-dimensional grid contains a single subdomain.

            Parameters:
                data_pt: data tuple containing arbitrary key, and associated cell data.
                subdomain_data: container for subdomain data.
                interface_data: container for interface data.

            Returns:
                Updated data containers and flag of success.

            Raises:
                ValueError: if the mixed-dimensional grid contains more than one
                    subdomain.

            """

            # Implementation if isinstance(data_pt, tuple[str, np.ndarray].
            isinstance_tuple_str_array = list(map(type, data_pt)) == [str, np.ndarray]

            # Convert data to unique format and update the interface data dictionary.
            if isinstance_tuple_str_array:

                # Fetch the correct grid. This option is only supported for
                # mixed-dimensional grids containing a single subdomain.
                subdomains = self._mdg.subdomains()
                if not len(subdomains) == 1:
                    raise ValueError(
                        f"""The data type used for {data_pt} is only
                        supported if the mixed-dimensional grid only contains a single
                        subdomain."""
                    )

                # Fetch remaining ingredients required to define subdomain data element
                sd = subdomains[0]
                key = data_pt[0]
                value = _to_vector_format(data_pt[1], sd)

                # Add data point in correct format to collection
                subdomain_data[(sd, key)] = value

                # Return updated dictionaries and indicate successful conversion.
                return subdomain_data, interface_data, True

            else:
                # Return original data dictionaries and indicate no modification.
                return subdomain_data, interface_data, False

        ################################################
        # The actual routine _sort_and_unify_data()
        ################################################

        # Convert data to list, while keeping the data structures provided,
        # or provide an empty list if no data provided
        if data is None:
            data = list()
        elif not isinstance(data, list):
            data = [data]

        # Initialize container for data associated to subdomains and interfaces
        subdomain_data: SubdomainData = dict()
        interface_data: InterfaceData = dict()

        # Define methods to be used for checking the data type and performing
        # the conversion. This list implicitly also defines which input data is
        # allowed.
        methods: list = [
            add_data_from_str,
            add_data_from_tuple_subdomains_str,
            add_data_from_tuple_interfaces_str,
            add_data_from_tuple_subdomain_str_array,
            add_data_from_tuple_interface_str_array,
            add_data_from_tuple_str_array,
        ]

        # Loop over all data points and collect them in a unified format.
        # For this, two separate dictionaries (for subdomain and interface data)
        # are used. The final format uses (subdomain/interface,key) as key of
        # the dictionaries, and the value is given by the corresponding data.
        for data_pt in data:

            # Initialize tag storing whether the conversion process for data_pt is
            # successful.
            success = False

            for method in methods:

                # Check whether data point of right type and convert to
                # the unique data type.
                subdomain_data, interface_data, success = method(
                    data_pt, subdomain_data, interface_data
                )

                # Stop, once a supported data format has been detected.
                if success:
                    break

            # Check if provided data has non-supported data format.
            if not success:
                raise ValueError(
                    f"Input data {data_pt} has wrong format and cannot be exported."
                )

        return subdomain_data, interface_data

    def _update_constant_mesh_data(self) -> None:
        """Construct/update subdomain and interface data related with geometry and
        topology.

        The data is identified as constant data. The final containers, stored as
        attributes _constant_subdomain_data and _constant_interface_data, have
        the same format as the output of _sort_and_unify_data.

        """
        # Assume a change in constant data (it is not checked whether
        # the data really has been modified). Has the effect that
        # the constant data container will be exported at the
        # next application of write_vtu().
        self._exported_constant_data_up_to_date = False

        # Define constant subdomain data related to the mesh

        # Initialize container for constant subdomain data
        if not hasattr(self, "_constant_subdomain_data"):
            self._constant_subdomain_data = dict()

        # Add mesh related, constant subdomain data by direct assignment
        for sd, sd_data in self._mdg.subdomains(return_data=True):
            ones = np.ones(sd.num_cells, dtype=int)
            self._constant_subdomain_data[(sd, "cell_id")] = np.arange(
                sd.num_cells, dtype=int
            )
            self._constant_subdomain_data[(sd, "grid_dim")] = sd.dim * ones
            self._constant_subdomain_data[(sd, "subdomain_id")] = sd.id * ones
            self._constant_subdomain_data[(sd, "is_mortar")] = 0 * ones
            self._constant_subdomain_data[(sd, "mortar_side")] = (
                pp.grids.mortar_grid.MortarSides.NONE_SIDE.value * ones
            )

        # Define constant interface data related to the mesh

        # Initialize container for constant interface data
        if not hasattr(self, "_constant_interface_data"):
            self._constant_interface_data = dict()

        # Add mesh related, constant interface data by direct assignment.
        for intf, intf_data in self._mdg.interfaces(return_data=True):

            # Construct empty arrays for all extra interface data
            self._constant_interface_data[(intf, "grid_dim")] = np.empty(0, dtype=int)
            self._constant_interface_data[(intf, "cell_id")] = np.empty(0, dtype=int)
            self._constant_interface_data[(intf, "interface_id")] = np.empty(
                0, dtype=int
            )
            self._constant_interface_data[(intf, "is_mortar")] = np.empty(0, dtype=int)
            self._constant_interface_data[(intf, "mortar_side")] = np.empty(
                0, dtype=int
            )

            # Initialize offset
            side_grid_num_cells: int = 0

            # Assign extra interface data by collecting values on both sides.
            for side, grid in intf.side_grids.items():
                ones = np.ones(grid.num_cells, dtype=int)

                # Grid dimension of the mortar grid
                self._constant_interface_data[(intf, "grid_dim")] = np.hstack(
                    (self._constant_interface_data[(intf, "grid_dim")], grid.dim * ones)
                )

                # Cell ids of the mortar grid
                self._constant_interface_data[(intf, "cell_id")] = np.hstack(
                    (
                        self._constant_interface_data[(intf, "cell_id")],
                        np.arange(grid.num_cells, dtype=int) + side_grid_num_cells,
                    )
                )

                # Grid edge number of each interface
                self._constant_interface_data[(intf, "interface_id")] = np.hstack(
                    (
                        self._constant_interface_data[(intf, "interface_id")],
                        intf.id * ones,
                    )
                )

                # Whether the interface is mortar
                self._constant_interface_data[(intf, "is_mortar")] = np.hstack(
                    (self._constant_interface_data[(intf, "is_mortar")], ones)
                )

                # Side of the mortar
                self._constant_interface_data[(intf, "mortar_side")] = np.hstack(
                    (
                        self._constant_interface_data[(intf, "mortar_side")],
                        side.value * ones,
                    )
                )

                # Update offset
                side_grid_num_cells += grid.num_cells

    def _export_data_vtu(
        self,
        data: Union[SubdomainData, InterfaceData],
        time_step: Optional[int],
        **kwargs,
    ) -> None:
        """Collect data associated to a single grid dimension
        and passing further to the final writing routine.

        For each fixed dimension, all subdomains of that dimension and the data
        related to those subdomains will be exported simultaneously.
        Analogously for interfaces.

        Parameters:
            data: Subdomain or interface data.
            time_step: time_step to be used to append the file name.
            kwargs:
                'interface_data' (boolean) indicates whether data is associated to
                    an interface, default is False.
                'constant_data' (boolean) indicates whether data is treated as
                    constant in time, default is False.

        Raises:
            TypeError: if keyword arguments contain unsupported keyword.
            ValueError: if data provided for some but not all subdomains or
                interfaces of particular dimension.

        """
        # Check for optional keywords
        self.interface_data: bool = kwargs.pop("interface_data", False)
        self.constant_data: bool = kwargs.pop("constant_data", False)
        if kwargs:
            msg = "_export_data_vtu got unexpected keyword argument '{}'"
            raise TypeError(msg.format(kwargs.popitem()[0]))

        # Fetch the dimensions to be traversed. For subdomains, fetch the dimensions
        # of the available grids, and for interfaces fetch the dimensions of the
        # available mortar grids.
        is_subdomain_data: bool = not self.interface_data
        dims = self._dims if is_subdomain_data else self._m_dims

        # Define file name base
        file_name_base: str = self._file_name
        # Extend in case of constant data
        file_name_base += "_constant" if self.constant_data else ""
        # Extend in case of interface data
        file_name_base += "_mortar" if self.interface_data else ""
        # Append folder name to file name base
        file_name_base = self._append_folder_name(self._folder_name, file_name_base)

        # Collect unique keys, and for unique sorting, sort by alphabet
        keys = list(set([key for _, key in data]))
        keys.sort()

        # Collect the data and extra data in a single stack for each dimension
        for dim in dims:
            # Define the full file name
            file_name: str = self._make_file_name(file_name_base, time_step, dim)

            # Get all geometrical entities of dimension dim:
            if is_subdomain_data:
                entities: list[Any] = self._mdg.subdomains(dim=dim)
            else:
                entities = self._mdg.interfaces(dim=dim)

            # Construct the list of fields represented on this dimension.
            fields: list[Field] = []
            for key in keys:
                # Collect the values associated to all entities
                values = []
                for e in entities:
                    if (e, key) in data:
                        values.append(data[(e, key)])

                # Require data for all or none entities of that dimension.
                if len(values) not in [0, len(entities)]:
                    raise ValueError(
                        f"""Insufficient amount of data provided for
                        key {key} on dimension {dim}."""
                    )

                # If data has been found, append data to list after stacking
                # values for different entities
                if values:
                    field = Field(key, np.hstack(values))
                    fields.append(field)

            # Print data for the particular dimension. Since geometric info
            # is required distinguish between subdomain and interface data.
            meshio_geom = (
                self.meshio_geom[dim] if is_subdomain_data else self.m_meshio_geom[dim]
            )
            if meshio_geom is not None:
                self._write(fields, file_name, meshio_geom)

    def _export_mdg_pvd(self, file_name: str, time_step: Optional[int]) -> None:
        """Routine to export to pvd format and collect all data scattered over
        several files for distinct grid dimensions.

        Parameters:
            file_name: storage path for pvd file.
            time_step: used as appendix for the file name.

        """
        # Open the file
        o_file = open(file_name, "w")

        # Write VTK header to file
        b = "LittleEndian" if sys.byteorder == "little" else "BigEndian"
        c = ' compressor="vtkZLibDataCompressor"'
        header = (
            '<?xml version="1.0"?>\n'
            + '<VTKFile type="Collection" version="0.1" '
            + 'byte_order="%s"%s>\n' % (b, c)
            + "<Collection>\n"
        )
        o_file.write(header)
        fm = '\t<DataSet group="" part="" file="%s"/>\n'

        # Subdomain data.
        for dim in self._dims:
            if self._has_subdomain_data and self.meshio_geom[dim] is not None:
                o_file.write(fm % self._make_file_name(self._file_name, time_step, dim))

        # Interface data.
        for dim in self._m_dims:
            if self._has_interface_data and self.m_meshio_geom[dim] is not None:
                o_file.write(
                    fm
                    % self._make_file_name(self._file_name + "_mortar", time_step, dim)
                )

        # If constant data is exported to separate vtu files, also include
        # these here. The procedure is similar to the above, but the file names
        # incl. the relevant time step have to be adjusted.
        if self._export_constants_separately:

            # Constant subdomain data.
            for dim in self._dims:
                if (
                    self._has_constant_subdomain_data
                    and self.meshio_geom[dim] is not None
                ):
                    o_file.write(
                        fm
                        % self._make_file_name(
                            self._file_name + "_constant",
                            self._time_step_constant_data,
                            dim,
                        )
                    )

            # Constant interface data.
            for dim in self._m_dims:
                if (
                    self._has_constant_interface_data
                    and self.m_meshio_geom[dim] is not None
                ):
                    o_file.write(
                        fm
                        % self._make_file_name(
                            self._file_name + "_constant_mortar",
                            self._time_step_constant_data,
                            dim,
                        )
                    )

        o_file.write("</Collection>\n" + "</VTKFile>")
        o_file.close()

    def _update_meshio_geom(self) -> None:
        """Manager for storing the grid information in meshio format.

        The internal variables meshio_geom and m_meshio_geom, storing all
        essential (mortar) grid information as points, connectivity and cell
        ids, are created and stored.

        """
        # Subdomains
        for dim in self._dims:
            # Get subdomains with dimension dim
            subdomains = self._mdg.subdomains(dim=dim)
            # Export and store
            self.meshio_geom[dim] = self._export_grid(subdomains, dim)

        # Interfaces
        for dim in self._m_dims:
            # Extract the mortar grids for dimension dim, unrolled by sides
            interface_side_grids = [
                grid
                for intf in self._mdg.interfaces(dim=dim)
                for _, grid in intf.side_grids.items()
            ]
            # Export and store
            self.m_meshio_geom[dim] = self._export_grid(interface_side_grids, dim)

    def _export_grid(
        self, grids: Iterable[pp.Grid], dim: int
    ) -> Union[None, Meshio_Geom]:
        """Wrapper function to export grids of dimension dim. Calls the
        appropriate dimension specific export function.

        Parameters:
            grids: Subdomains of same dimension.
            dim: Dimension of the subdomains.

        Returns:
            Meshio_Geom: Points, cells (storing the connectivity), and cell ids
            in correct meshio format.

        Raises:
            ValueError: if dim not 0, 1, 2 or 3.

        """
        if dim == 0:
            return None
        elif dim == 1:
            return self._export_grid_1d(grids)
        elif dim == 2:
            return self._export_grid_2d(grids)
        elif dim == 3:
            return self._export_grid_3d(grids)
        else:
            raise ValueError(f"Unknown dimension {dim}")

    def _export_grid_1d(self, grids: Iterable[pp.Grid]) -> Meshio_Geom:
        """Export the geometrical data (point coordinates) and connectivity
        information from the 1d PorePy grids to meshio.

        Parameters:
            grids: 1d grids.

        Returns:
            Meshio_Geom: Points, 1d cells (storing the connectivity),
            and cell ids in correct meshio format.

        """

        # In 1d each cell is a line
        cell_type = "line"

        # Dictionary storing cell->nodes connectivity information
        cell_to_nodes: dict[str, np.ndarray] = {cell_type: np.empty((0, 2), dtype=int)}

        # Dictionary collecting all cell ids for each cell type.
        # Since each cell is a line, the list of cell ids is trivial
        total_num_cells = np.sum(np.array([grid.num_cells for grid in grids]))
        cell_id: dict[str, list[int]] = {cell_type: [i for i in range(total_num_cells)]}

        # Data structure for storing node coordinates of all 1d grids.
        num_pts = np.sum([grid.num_nodes for grid in grids]).astype(int)
        meshio_pts = np.empty((num_pts, 3))  # type: ignore

        # Initialize offset. All data associated to 1d grids is stored in
        # the same vtu file, essentially using concatenation. To identify
        # each grid, keep track of number of nodes for each grid.
        nodes_offset = 0

        # Loop over all 1d grids
        for grid in grids:
            # Store node coordinates
            sl = slice(nodes_offset, nodes_offset + grid.num_nodes)
            meshio_pts[sl, :] = grid.nodes.T

            # Lines are 1-simplices, and have a trivial connectivity.
            cn_indices = self._simplex_cell_to_nodes(1, grid)

            # Add to previous connectivity information
            cell_to_nodes[cell_type] = np.vstack(
                (cell_to_nodes[cell_type], cn_indices + nodes_offset)
            )

            # Update offsets
            nodes_offset += grid.num_nodes

        # Construct the meshio data structure
        meshio_cells = list()
        meshio_cell_id = list()

        # For each cell_type store the connectivity pattern cell_to_nodes for
        # the corresponding cells with ids from cell_id.
        for (cell_type, cell_block) in cell_to_nodes.items():
            meshio_cells.append(meshio.CellBlock(cell_type, cell_block.astype(int)))
            meshio_cell_id.append(np.array(cell_id[cell_type]))

        # Return final meshio data: points, cell (connectivity), cell ids
        return Meshio_Geom(meshio_pts, meshio_cells, meshio_cell_id)

    def _simplex_cell_to_nodes(
        self, n: int, grid: pp.Grid, cells: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Determine cell to node connectivity for a general n-simplex mesh.

        Parameters:
            n: dimension of the simplices in the grid.
            grid: grid containing cells and nodes.
            cells: all n-simplex cells.

        Returns:
            np.ndarray: cell to node connectivity array, in which for each row
            (associated to cells) all associated nodes are marked.

        """
        # Determine cell-node ptr
        cn_indptr = (
            grid.cell_nodes().indptr[:-1]
            if cells is None
            else grid.cell_nodes().indptr[cells]
        )

        # Each n-simplex has n+1 nodes
        num_nodes = n + 1

        # Collect the indptr to all nodes of the cell.
        expanded_cn_indptr = np.vstack(
            [cn_indptr + i for i in range(num_nodes)]
        ).reshape(-1, order="F")

        # Detect all corresponding nodes by applying the expanded mask to the indices
        expanded_cn_indices = grid.cell_nodes().indices[expanded_cn_indptr]

        # Convert to right format.
        cn_indices = np.reshape(expanded_cn_indices, (-1, num_nodes), order="C")

        return cn_indices

    def _export_grid_2d(self, grids: Iterable[pp.Grid]) -> Meshio_Geom:
        """Export the geometrical data (point coordinates) and connectivity
        information from the 2d PorePy grids to meshio.

        Parameters:
            grids: 2d grids.

        Returns:
            Meshio_Geom: Points, 2d cells (storing the connectivity), and
            cell ids in correct meshio format.

        """

        # Use standard names for simple object types: in this routine only triangle
        # and quad cells are treated in a special manner.
        polygon_map = {"polygon3": "triangle", "polygon4": "quad"}

        # Dictionary storing cell->nodes connectivity information for all
        # cell types. For this, the nodes have to be sorted such that
        # they form a circular chain, describing the boundary of the cell.
        cell_to_nodes: dict[str, np.ndarray] = {}
        # Dictionary collecting all cell ids for each cell type.
        cell_id: dict[str, list[int]] = {}

        # Data structure for storing node coordinates of all 2d grids.
        num_pts = np.sum([grid.num_nodes for grid in grids])
        meshio_pts = np.empty((num_pts, 3))  # type: ignore

        # Initialize offsets. All data associated to 2d grids is stored in
        # the same vtu file, essentially using concatenation. To identify
        # each grid, keep track of number of nodes and cells for each grid.
        nodes_offset = 0
        cell_offset = 0

        # Loop over all 2d grids
        for grid in grids:

            # Store node coordinates
            sl = slice(nodes_offset, nodes_offset + grid.num_nodes)
            meshio_pts[sl, :] = grid.nodes.T

            # Determine cell types based on number of nodes.
            num_nodes_per_cell = grid.cell_nodes().getnnz(axis=0)

            # Loop over all available cell types and group cells of one type.
            g_cell_map = dict()
            for n in np.unique(num_nodes_per_cell):

                # Define cell type; check if it coincides with a predefined cell type
                cell_type = polygon_map.get(f"polygon{n}", f"polygon{n}")

                # Find all cells with n nodes, and store for later use
                cells = np.nonzero(num_nodes_per_cell == n)[0]
                g_cell_map[cell_type] = cells

                # Store cell ids in global container; init if entry not yet established
                if cell_type not in cell_id:
                    cell_id[cell_type] = []

                # Add offset taking into account previous grids
                cell_id[cell_type] += (cells + cell_offset).tolist()

            # Determine cell-node connectivity for each cell type and all cells.
            # Treat triangle, quad and polygonal cells differently
            # aiming for optimized performance.
            for n in np.unique(num_nodes_per_cell):

                # Define the cell type
                cell_type = polygon_map.get(f"polygon{n}", f"polygon{n}")

                # Check if cell_type is already defined in cell_to_nodes, otherwise
                # construct it
                if cell_type not in cell_to_nodes:
                    cell_to_nodes[cell_type] = np.empty((0, n), dtype=int)

                # Special case: Triangle cells, i.e., n=3.
                if cell_type == "triangle":

                    # Triangles are simplices and have a trivial connectivity.

                    # Fetch triangle cells
                    cells = g_cell_map[cell_type]
                    # Determine the trivial connectivity - triangles are 2-simplices.
                    cn_indices = self._simplex_cell_to_nodes(2, grid, cells)

                # Quad and polygon cells.
                else:
                    # For quads/polygons, grid.cell_nodes cannot be blindly used as for
                    # triangles, since the ordering of the nodes may define a cell of
                    # the type (here specific for quads)
                    # x--x and not x--x
                    #  \/          |  |
                    #  /\          |  |
                    # x--x         x--x .
                    # Therefore, use both grid.cell_faces and grid.face_nodes to make
                    # use of face information and sort those to retrieve the correct
                    # connectivity.

                    # Strategy: Collect all cell nodes including their connectivity in a
                    # matrix of double num cell size. The goal will be to gather
                    # starting and end points for all faces of each cell, sort those
                    # faces,such that they form a circular graph, and then choose the
                    # resulting starting points of all faces to define the connectivity.

                    # Fetch corresponding cells
                    cells = g_cell_map[cell_type]
                    # Determine all faces of all cells. Use an analogous approach as
                    # used to determine all cell nodes for triangle cells. And use that
                    # a polygon with n nodes has also n faces.
                    cf_indptr = grid.cell_faces.indptr[cells]
                    expanded_cf_indptr = np.vstack(
                        [cf_indptr + i for i in range(n)]
                    ).reshape(-1, order="F")
                    cf_indices = grid.cell_faces.indices[expanded_cf_indptr]

                    # Determine the associated (two) nodes of all faces for each cell.
                    fn_indptr = grid.face_nodes.indptr[cf_indices]
                    # Extract nodes for first and second node of each face; reshape such
                    # that all first nodes of all faces for each cell are stored in one
                    # row, i.e., end up with an array of size num_cells (for this cell
                    # type) x n.
                    cfn_indices = [
                        grid.face_nodes.indices[fn_indptr + i].reshape(-1, n)
                        for i in range(2)
                    ]
                    # Group first and second nodes, with alternating order of rows.
                    # By this, each cell is represented by two rows. The first and
                    # second rows contain first and second nodes of faces. And each
                    # column stands for one face.
                    cfn = np.ravel(cfn_indices, order="F").reshape(n, -1).T

                    # Sort faces for each cell such that they form a chain. Use a
                    # function compiled with Numba. This step is the bottleneck of
                    # this routine.
                    cfn = pp.utils.sort_points.sort_multiple_point_pairs(cfn).astype(
                        int
                    )

                    # For each cell pick the sorted nodes such that they form a chain
                    # and thereby define the connectivity, i.e., skip every second row.
                    cn_indices = cfn[::2, :]

                # Add offset to account for previous grids, and store
                cell_to_nodes[cell_type] = np.vstack(
                    (cell_to_nodes[cell_type], cn_indices + nodes_offset)
                )

            # Update offsets
            nodes_offset += grid.num_nodes
            cell_offset += grid.num_cells

        # Construct the meshio data structure
        meshio_cells = list()
        meshio_cell_id = list()

        # For each cell_type store the connectivity pattern cell_to_nodes for
        # the corresponding cells with ids from cell_id.
        for (cell_type, cell_block) in cell_to_nodes.items():
            # Meshio requires the keyword "polygon" for general polygons.
            # Thus, remove the number of nodes associated to polygons.
            cell_type_meshio_format = "polygon" if "polygon" in cell_type else cell_type
            meshio_cells.append(
                meshio.CellBlock(cell_type_meshio_format, cell_block.astype(int))
            )
            meshio_cell_id.append(np.array(cell_id[cell_type]))

        # Return final meshio data: points, cell (connectivity), cell ids
        return Meshio_Geom(meshio_pts, meshio_cells, meshio_cell_id)

    def _export_grid_3d(self, grids: Iterable[pp.Grid]) -> Meshio_Geom:
        """Export the geometrical data (point coordinates) and connectivity
        information from 3d PorePy grids to meshio.

        Parameters:
            grids: 3d grids.

        Returns:
            Points, 3d cells (storing the connectivity), and cell ids in
            correct meshio format.

        """

        # Three different cell types will be distinguished: Tetrahedra, hexahedra,
        # and general polyhedra. meshio does not allow for a mix of cell types
        # in 3d within a single grid (and we export all 3d grids at once). Thus,
        # if the 3d grids contains cells of varying types, all cells will be cast
        # as polyhedra.

        # Determine the cell types present among all grids.
        cell_types: set[str] = set()
        for grid in grids:

            # The number of faces per cell wil be later used to determining
            # the cell types
            num_faces_per_cell = np.unique(grid.cell_faces.getnnz(axis=0))

            if num_faces_per_cell.shape[0] == 1:
                n = num_faces_per_cell[0]
                if n == 4:
                    cell_types.add("tetra")
                elif n == 6 and isinstance(grid, pp.CartGrid):
                    cell_types.add("hexahedron")
                else:
                    cell_types.add("polyhedron")
            else:
                cell_types.add("polyhedron")

        # Use dedicated export for each grid type
        if len(cell_types) == 1 and "tetra" in cell_types:
            return self._export_simplex_3d(grids)
        elif len(cell_types) == 1 and "hexahedron" in cell_types:
            return self._export_hexahedron_3d(grids)
        else:
            return self._export_polyhedron_3d(grids)

    def _export_simplex_3d(self, grids: Iterable[pp.Grid]) -> Meshio_Geom:
        """Export the geometrical data (point coordinates) and connectivity
        information from 3d PorePy simplex grids to meshio.

        Parameters:
            grids: 3d simplex grids.

        Returns:
            Points, 3d cells (storing the connectivity), and cell ids in
            correct meshio format.

        """
        # For the special meshio geometric type "tetra", cell->nodes will
        # be used to store connectivity information. Each simplex has 4 nodes.
        cell_to_nodes = np.empty((0, 4), dtype=int)

        # Dictionary collecting all cell ids for each cell type.
        cell_id: list[int] = []

        # Data structure for storing node coordinates of all 3d grids.
        num_pts = np.sum([grid.num_nodes for grid in grids])
        meshio_pts = np.empty((num_pts, 3))  # type: ignore

        # Initialize offsets. All data associated to 3d grids is stored in
        # the same vtu file, essentially using concatenation. To identify
        # each grid, keep track of number of nodes and cells for each grid.
        nodes_offset = 0
        cell_offset = 0

        # Treat each 3d grid separately.
        for grid in grids:
            # Store node coordinates
            sl = slice(nodes_offset, nodes_offset + grid.num_nodes)
            meshio_pts[sl, :] = grid.nodes.T

            # Identify all cells as tetrahedra.
            cells = np.arange(grid.num_cells)

            # Add offset taking into account previous grids
            cell_id += (cells + cell_offset).tolist()

            # Determine local connectivity for all cells. Simplices are invariant
            # under permutations. Thus, no specific ordering of nodes is required.
            # Tetrahedra are essentially 3-simplices.
            cn_indices = self._simplex_cell_to_nodes(3, grid, cells)

            # Store cell-node connectivity, and add offset taking into account
            # previous grids
            cell_to_nodes = np.vstack((cell_to_nodes, cn_indices + nodes_offset))

            # Update offset
            nodes_offset += grid.num_nodes
            cell_offset += grid.num_cells

        # Initialize the meshio data structure for the connectivity and cell ids.
        # There is only one cell type, and meshio expects iterable data structs.
        meshio_cells = [meshio.CellBlock("tetra", cell_to_nodes.astype(int))]
        meshio_cell_id = [np.array(cell_id)]

        # Return final meshio data: points, cell (connectivity), cell ids
        return Meshio_Geom(meshio_pts, meshio_cells, meshio_cell_id)

    def _export_hexahedron_3d(self, grids: Iterable[pp.Grid]) -> Meshio_Geom:
        """Export the geometrical data (point coordinates) and connectivity
        information from 3d PorePy hexahedron grids to meshio.

        NOTE: Optimally, a StructuredGrid would be exported in this case.
        However, meshio does solely handle unstructured grids and cannot
        for instance write to *.vtr format.

        Parameters:
            grids: 3d hexahedron grids.

        Returns:
            Points, 3d cells (storing the connectivity), and cell ids in
            correct meshio format.

        """
        # For the special meshio geometric type "hexahedron", cell->nodes will
        # be used to store connectivity information. Each hexahedron has 8 nodes.
        cell_to_nodes = np.empty((0, 8), dtype=int)

        # Dictionary collecting all cell ids for each cell type.
        cell_id: list[int] = []

        # Data structure for storing node coordinates of all 3d grids.
        num_pts = np.sum([grid.num_nodes for grid in grids])
        meshio_pts = np.empty((num_pts, 3))  # type: ignore

        # Initialize offsets. Required taking into account multiple 3d grids.
        nodes_offset = 0
        cell_offset = 0

        # Treat each 3d grid separately.
        for grid in grids:
            # Store node coordinates
            sl = slice(nodes_offset, nodes_offset + grid.num_nodes)
            meshio_pts[sl, :] = grid.nodes.T

            # Identify all cells as tetrahedra.
            cells = np.arange(grid.num_cells)

            # Add offset taking into account previous grids
            cell_id += (cells + cell_offset).tolist()

            # Determine local connectivity for all cells. Simplices are invariant
            # under permutations. Thus, no specific ordering of nodes is required.

            # After all, the procedure is fairly similar to the treatment of
            # tetra cells. Most of the following code is analogous to
            # _simplex_cell_to_nodes(). However, for hexahedron cells the order
            # of the nodes is important and has to be adjusted. Based on the
            # specific definition of TensorGrids however, the node numbering
            # can be hardcoded, which results in better performance compared
            # to a modular procedure.

            # Determine cell-node ptr
            cn_indptr = (
                grid.cell_nodes().indptr[:-1]
                if cells is None
                else grid.cell_nodes().indptr[cells]
            )

            # Collect the indptr to all nodes of the cell;
            # each n-simplex cell contains n nodes
            expanded_cn_indptr = np.vstack([cn_indptr + i for i in range(8)]).reshape(
                -1, order="F"
            )

            # Detect all corresponding nodes by applying the expanded mask to indices
            expanded_cn_indices = grid.cell_nodes().indices[expanded_cn_indptr]

            # Convert to right format.
            cn_indices = np.reshape(expanded_cn_indices, (-1, 8), order="C")

            # Reorder nodes to comply with the node numbering convention of meshio
            # regarding hexahedron cells.
            cn_indices = cn_indices[:, [0, 2, 6, 4, 1, 3, 7, 5]]

            # Test whether the hard-coded numbering really defines a hexahedron.
            assert self._test_hex_meshio_format(grid.nodes, cn_indices)

            # Store cell-node connectivity, and add offset taking into account
            # previous grids
            cell_to_nodes = np.vstack((cell_to_nodes, cn_indices + nodes_offset))

            # Update offset
            nodes_offset += grid.num_nodes
            cell_offset += grid.num_cells

        # Initialize the meshio data structure for the connectivity and cell ids.
        # There is only one cell type, and meshio expects iterable data structs.
        meshio_cells = [meshio.CellBlock("hexahedron", cell_to_nodes.astype(int))]
        meshio_cell_id = [np.array(cell_id)]

        # Return final meshio data: points, cell (connectivity), cell ids
        return Meshio_Geom(meshio_pts, meshio_cells, meshio_cell_id)

    def _test_hex_meshio_format(
        self, nodes: np.ndarray, cn_indices: np.ndarray
    ) -> bool:
        """Test whether the node numbering for each cell complies
        with the hardcoded numbering used by meshio, cf. documentation
        of meshio.

        Parameters:
            nodes: grid nodes.
            cn_indices: connectivity information.

        Returns:
            Flag whether the node numbering for each cell complies with
            the numbering of meshio.

        Raises:
            ImportError: if numba is not installed.

        """

        try:
            import numba
        except ImportError:
            raise ImportError("Numba not available on the system")

        # Just in time compilation
        @numba.njit("b1(f8[:,:], i4[:,:])", cache=True, parallel=True)
        def _function_to_compile(nodes, cn_indices):
            """Test whether the node numbering for each cell complies
            with the hardcoded numbering used by meshio, cf. documentation
            of meshio.

            For this, fetch three potential sides of the hex and
            test whether they lie circular chain within a plane.
            Here, the test is successful, if the node sets [0,1,2,3],
            [4,5,6,7], and [0,1,5,4] define planes.

            Paramerers:
                nodes: grid nodes.
                cn_indices: connectivity information.

            Returns:
                Flag whether the node numbering for each cell complies with
                the numbering of meshio.

            """

            # Assume initially correct format
            correct_format = True

            # Test each cell separately
            for i in numba.prange(cn_indices.shape[0]):

                # Assume initially that the cell is a hex
                is_hex = True

                # Visit three node sets and check whether
                # they define feasible sides of a hex.
                # FIXME use shorter code of the following style - note: the order in the
                # array is not the same troubling numba
                # global_ind_0 = cn_indices[i, 0:4]
                # global_ind_1 = cn_indices[i, 4:8]
                global_ind_0 = np.array(
                    [
                        cn_indices[i, 0],
                        cn_indices[i, 1],
                        cn_indices[i, 2],
                        cn_indices[i, 3],
                    ]
                )
                global_ind_1 = np.array(
                    [
                        cn_indices[i, 4],
                        cn_indices[i, 5],
                        cn_indices[i, 6],
                        cn_indices[i, 7],
                    ]
                )
                global_ind_2 = np.array(
                    [
                        cn_indices[i, 0],
                        cn_indices[i, 1],
                        cn_indices[i, 5],
                        cn_indices[i, 4],
                    ]
                )

                # Check each side separately
                for global_ind in [global_ind_0, global_ind_1, global_ind_2]:

                    # Fetch coordinates associated to the four nodes
                    coords = nodes[:, global_ind]

                    # Check orientation by determining normalized cross products
                    # of all two consecutive connections.
                    normalized_cross = np.zeros((3, 3))
                    for j in range(3):
                        cross = np.cross(
                            coords[:, j + 1] - coords[:, j],
                            coords[:, (j + 2) % 4] - coords[:, j + 1],
                        )
                        normalized_cross[:, j] = cross / np.linalg.norm(cross)

                    # Consider the points lying in a plane when each connection
                    # produces the same normalized cross product.
                    is_plane = np.linalg.matrix_rank(normalized_cross) == 1

                    # Combine with the remaining sides
                    is_hex = is_hex and is_plane

                # Combine the results for all cells
                correct_format = correct_format and is_hex

            return correct_format

        return _function_to_compile(nodes, cn_indices)

    def _export_polyhedron_3d(self, grids: Iterable[pp.Grid]) -> Meshio_Geom:
        """Export the geometrical data (point coordinates) and connectivity
        information from 3d PorePy polyhedron grids to meshio.

        Parameters:
            grids: 3d polyhedron grids.

        Returns:
            Meshio_Geom: Points, 3d cells (storing the connectivity), and
            cell ids in correct meshio format.

        """
        # For the general geometric type "polyhedron", cell->faces->nodes will
        # be used to store connectivity information, which can become significantly
        # larger than cell->nodes information used for special type grids.
        cell_to_faces: dict[str, list[list[int]]] = {}

        # Dictionary collecting all cell ids for each cell type.
        cell_id: dict[str, list[int]] = {}

        # Data structure for storing node coordinates of all 3d grids.
        num_pts = np.sum([grid.num_nodes for grid in grids])
        meshio_pts = np.empty((num_pts, 3))  # type: ignore

        # Initialize offsets. Required taking into account multiple 3d grids.
        nodes_offset = 0
        cell_offset = 0

        # Treat each 3d grid separately.
        for grid in grids:

            # Store node coordinates
            sl = slice(nodes_offset, nodes_offset + grid.num_nodes)
            meshio_pts[sl, :] = grid.nodes.T

            # Categorize all polyhedron cells by their number of nodes.
            # Each category will be treated separately allowing for using
            # fitting datastructures.
            num_nodes_per_cell = grid.cell_nodes().getnnz(axis=0)

            g_cell_map = dict()

            for n in np.unique(num_nodes_per_cell):

                cell_type = f"polyhedron{n}"

                # Find all cells with n faces, and store for later use
                cells = np.nonzero(num_nodes_per_cell == n)[0]
                g_cell_map[cell_type] = cells

                # Store cell ids in global container; init if entry not yet established
                if cell_type not in cell_id:
                    cell_id[cell_type] = []

                # Add offset taking into account previous grids
                cell_id[cell_type] += (cells + cell_offset).tolist()

            # Determine local connectivity for all cells. Due to different
            # treatment of special and general cell types and to conform
            # with array sizes (for polyhedra), treat each cell type
            # separately.
            for cell_type in g_cell_map.keys():

                # The general strategy is to define the connectivity as cell-face-nodes
                # information, where the faces are defined by nodes. Hence, this
                # information is significantly larger than the info provided for
                # tetra cells. Here, we make use of the fact that grid.face_nodes
                # provides nodes ordered wrt. the right-hand rule.

                # Fetch cells with n faces
                cells = g_cell_map[cell_type]

                # Store shortcuts to cell-face and face-node information
                cf_indptr = grid.cell_faces.indptr
                cf_indices = grid.cell_faces.indices
                fn_indptr = grid.face_nodes.indptr
                fn_indices = grid.face_nodes.indices

                # Determine the cell-face connectivity with faces described by their
                # nodes ordered such that they form a chain and are identified by the
                # face boundary. The final data format is a list[list[np.ndarray]].
                # The outer list loops over all cells. Each cell entry contains a
                # list over faces, and each face entry is given by the face nodes.
                if cell_type not in cell_to_faces:
                    cell_to_faces[cell_type] = []
                cell_to_faces[cell_type] += [
                    [
                        fn_indices[fn_indptr[f] : fn_indptr[f + 1]]  # nodes
                        for f in cf_indices[cf_indptr[c] : cf_indptr[c + 1]]  # faces
                    ]
                    for c in cells  # cells
                ]

            # Update offset
            nodes_offset += grid.num_nodes
            cell_offset += grid.num_cells

        # Initialize the meshio data structure for the connectivity and cell ids.
        meshio_cells = list()
        meshio_cell_id = list()

        # Store the cells in meshio format
        for (cell_type, cell_block) in cell_to_faces.items():
            # Adapt the block number taking into account of previous cell types.
            meshio_cells.append(meshio.CellBlock(cell_type, cell_block))
            meshio_cell_id.append(np.array(cell_id[cell_type]))

        # Return final meshio data: points, cell (connectivity), cell ids
        return Meshio_Geom(meshio_pts, meshio_cells, meshio_cell_id)

    def _write(
        self, fields: Iterable[Field], file_name: str, meshio_geom: Meshio_Geom
    ) -> None:
        """Interface to meshio for exporting cell data.

        Parameters:
            fields: fields which shall be exported
            file_name: name of final file of export.
            meshio_geom: Namedtuple of points, connectivity information, and cell ids in
                meshio format (for a single dimension).

        Raises:
            ValueError: if some data has wrong dimension.

        """
        # Initialize empty cell data dictionary
        cell_data: dict[str, list[np.ndarray]] = {}

        # Split the data for each group of geometrically uniform cells
        # Utilize meshio_geom for this.
        for field in fields:

            # Although technically possible, as implemented, field.values should never
            # be None.
            assert field.values is not None

            # For each field create a sub-vector for each geometrically uniform group
            # of cells
            cell_data[field.name] = list()

            # Fill up the data
            for ids in meshio_geom.cell_ids:
                if field.values.ndim == 1:
                    cell_data[field.name].append(field.values[ids])
                elif field.values.ndim == 2:
                    cell_data[field.name].append(field.values[:, ids].T)
                else:
                    raise ValueError("Data values have wrong dimension")

        # Create the meshio object
        meshio_grid_to_export = meshio.Mesh(
            meshio_geom.pts, meshio_geom.connectivity, cell_data=cell_data
        )

        # Write mesh information and data to VTK format.
        meshio.write(file_name, meshio_grid_to_export, binary=self._binary)

    def _append_folder_name(
        self, folder_name: Optional[str] = None, name: str = ""
    ) -> str:
        """Auxiliary method setting up potentially non-existent folder structure and
        setting up a path for exporting.

        Parameters:
            folder_name: name of the folder.
            name: prefix of the name of the files to be written.

        Returns:
            Complete path to the files to be written.

        """

        # If no folder_name is prescribed, the files will be stored in the
        # working directory.
        if folder_name is None:
            return name

        # Set up folder structure if not existent.
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Return full path.
        return os.path.join(folder_name, name)

    def _make_file_name(
        self,
        file_name: str,
        time_step: Optional[int] = None,
        dim: Optional[int] = None,
        extension: str = ".vtu",
    ) -> str:
        """Auxiliary method to setting up file name.

        The final name is built as combination of a prescribed prefix,
        and possibly the dimension of underlying grid and time (step)
        the related data is associated to.

        Parameters:
            file_name: Prefix of the name of file to be exported.
            time_step: Time or time step (index).
            dim: Dimension of the exported grid.
            extension: Extension of the file, typically file ending defining the format.

        Returns:
            Complete name of file.

        """

        # Define non-empty time step extension including zero padding.
        time_extension = (
            "" if time_step is None else "_" + str(time_step).zfill(self._padding)
        )

        # Define non-empty dim extension
        dim_extension = "" if dim is None else "_" + str(dim)

        # Combine prefix and extensions to define the complete name
        return file_name + dim_extension + time_extension + extension
