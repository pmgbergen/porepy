"""
Module for exporting to vtu for (e.g. ParaView) visualization using the vtk module.

The Exporter class contains methods for exporting a grid or grid bucket with
associated data to the vtu format. For grid buckets with multiple grids, one
vtu file is printed for each grid. For transient simulations with multiple
time steps, a single pvd file takes care of the ordering of all printed vtu
files.
"""
import logging
import os
import sys
from collections import namedtuple
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import meshio
import numpy as np

import porepy as pp

# Module-wide logger
logger = logging.getLogger(__name__)

# Introduce types used below

# Object type to store data to export.
Field = namedtuple("Field", ["name", "values"])

# Object for managing meshio-relevant data, as well as a container
# for its storage, taking dimensions as inputs.
Meshio_Geom = namedtuple("Meshio_Geom", ["pts", "connectivity", "cell_ids"])
MD_Meshio_Geom = Dict[int, Optional[Meshio_Geom]]

# Interface between subdomains
Interface = Tuple[pp.Grid, pp.Grid]

# All allowed data structures to define data for exporting
DataInput = Union[
    # Keys for states
    str,
    # Subdomain specific data types
    Tuple[List[pp.Grid], str],
    Tuple[pp.Grid, str, np.ndarray],
    Tuple[str, np.ndarray],
    # Interface specific data types
    Tuple[List[Interface], str],
    Tuple[Interface, str, np.ndarray],
]

# Data structure in which data is stored after preprocessing
SubdomainData = Dict[Tuple[pp.Grid, str], np.ndarray]
InterfaceData = Dict[Tuple[Interface, str], np.ndarray]


class Exporter:
    """
    Class for exporting data to vtu files.

    Parameters:
    gb: grid bucket (if grid is provided, it will be converted to a grid bucket)
    file_name: the root of file name without any extension.
    folder_name: (optional) the name of the folder to save the file.
        If the folder does not exist it will be created.

    Optional arguments in kwargs:
    fixed_grid: (optional) in a time dependent simulation specify if the
        grid changes in time or not. The default is True.
    binary: (optional) export in binary format, default is True. If False,
        the output is in Ascii format, which allows better readibility and
        debugging, but requires more memory.
    export_constants_separately: (optional) export constants in separate files
        to potentially reduce memory footprint, default is True.

    How to use:

    The Exporter allows for various way to express which state variables,
    on which grids, and which extra data should be exported. A thorough
    demonstration is available as a dedicated tutorial. Check out
    tutorials/exporter.ipynb.

    Here, merely a brief demonstration of the use of Exporter is presented.

    If you need to export the state with key "pressure" on a single grid:
    save = Exporter(g, "solution", folder_name="results")
    save.write_vtu(["pressure"])

    In a time loop, if you need to export states with keys "pressure" and
    "displacement" stored in a grid bucket.
    save = Exporter(gb, "solution", folder_name="results")
    while time:
        save.write_vtu(["pressure", "displacement"], time_step=i)
    save.write_pvd(times)

    where times is a list of actual times (not time steps), associated
    to the previously exported time steps. If times is not provided
    the time steps will be used instead.

    In general, pvd files gather data exported in separate files, including
    data on differently dimensioned grids, constant data, and finally time steps.

    In the case of different keywords, change the file name with
    "change_name".

    NOTE: the following names are reserved for constant data exporting
    and should not be used otherwise (otherwise data is overwritten):
    grid_dim, is_mortar, mortar_side, cell_id, grid_node_number,
    grid_edge_number
    """

    def __init__(
        self,
        gb: Union[pp.Grid, pp.GridBucket],
        file_name: str,
        folder_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialization of Exporter.

        Parameters:
            gb (Union[pp.Grid, pp.Gridbucket]): grid or gridbucket containing all
                mesh information to be exported.
            file_name (str): basis for file names used for storing the output
            folder_name (str, optional): name of the folder in which files are stored
            kwargs (optional): Optional keywords;
                'fixed_grid' (boolean) to control whether the grid(bucket) may be redfined
                (default True);
                'binary' (boolean) controlling whether data is stored in binary format
                (default True);
                'export_constants_separately' (boolean) controlling whether
                constant data is exported in separate files, which may be of interest
                when exporting large data sets (in particular of constant data) for
                many time steps (default True); note, however, that the mesh is
                exported to each vtu file, which may also require significant amount
                of storage.
        """
        # Exporter is operating on grid buckets. Convert to grid bucket if grid is provided.
        if isinstance(gb, pp.Grid):
            self.gb = pp.GridBucket()
            self.gb.add_nodes(gb)
        elif isinstance(gb, pp.GridBucket):
            self.gb = gb
        else:
            raise TypeError("Exporter only supports single grids and grid buckets.")

        # Store target location for storing vtu and pvd files.
        self.file_name = file_name
        self.folder_name = folder_name

        # Check for optional keywords
        self.fixed_grid: bool = kwargs.pop("fixed_grid", True)
        self.binary: bool = kwargs.pop("binary", True)
        self.export_constants_separately: bool = kwargs.pop(
            "export_constants_separately", True
        )
        if kwargs:
            msg = "Exporter() got unexpected keyword argument '{}'"
            raise TypeError(msg.format(kwargs.popitem()[0]))

        # Generate infrastructure for storing fixed-dimensional grids in
        # meshio format. Include all but the 0-d grids
        self.dims = np.setdiff1d(self.gb.all_dims(), [0])
        self.meshio_geom: MD_Meshio_Geom = dict()

        # Generate infrastructure for storing fixed-dimensional mortar grids
        # in meshio format.
        self.m_dims = np.unique([d["mortar_grid"].dim for _, d in self.gb.edges()])
        self.m_meshio_geom: MD_Meshio_Geom = dict()

        # Generate geometrical information in meshio format
        self._update_meshio_geom()
        self._update_constant_mesh_data()

        # Counter for time step. Will be used to identify files of individual time step,
        # unless this is overridden by optional parameters in write
        self._time_step_counter: int = 0

        # Storage for file name extensions for time steps
        self._exported_timesteps: List[int] = []

        # Reference to the last time step used for exporting constant data.
        self._time_step_constants: int = 0
        # Storage for file name extensions for time steps, regarding constant data.
        self._exported_timesteps_constants: List[int] = list()
        # Identifier for whether constant data is up-to-date.
        self._exported_constant_data_up_to_date: bool = False

        # Parameter to be used in several occasions for adding time stamps.
        self.padding = 6

        # Require numba
        if "numba" not in sys.modules:
            raise NotImplementedError(
                "The sorting algorithm requires numba to be installed."
            )

    def add_constant_data(
        self,
        data: Optional[Union[DataInput, List[DataInput]]] = None,
    ) -> None:
        """
        Collect constant data (after unifying). And later export to own files.

        Parameters:
            data (Union[DataInput, List[DataInput]], optional): node and
                edge data, prescribed through strings, or tuples of
                grids/edges, keys and values. If not provided only
                geometical infos are exported.

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
        data: Optional[Union[DataInput, List[DataInput]]] = None,
        time_dependent: Optional[bool] = False,
        time_step: Optional[int] = None,
        gb: Optional[Union[pp.Grid, pp.GridBucket]] = None,
    ) -> None:
        """
        Interface function to export the grid and additional data with meshio.

        In 1d the cells are represented as lines, 2d the cells as polygon or
        triangle/quad, while in 3d as polyhedra/tetrahedra/hexahedra.
        In all the dimensions the geometry of the mesh needs to be computed.

        Parameters:
            data (Union[DataInput, List[DataInput]], optional): node and
                edge data, prescribed through strings, or tuples of
                grids/edges, keys and values. If not provided only
                geometrical infos are exported.

                NOTE: The user has to make sure that each unique key has
                associated data values for all or no grids of each specific
                dimension.
            time_dependent (boolean, optional): If False, file names will
                not be appended with an index that marks the time step.
                Can be overwritten by giving a value to time_step; if not,
                the file names will subsequently be ending with 1, 2, etc.
            time_step (int, optional): will be used as appendix to define
                the file corresponding to this specific time step.
            gb (Union[pp.Grid, pp.Gridbucket], optional): grid or gridbucket
                if it is not fixed and should be updated.

        """

        # Update the grid (bucket) but only if allowed, i.e., the initial grid
        # (bucket) has not been characterized as fixed.
        if self.fixed_grid and gb is not None:
            raise ValueError("Inconsistency in exporter setting")
        elif not self.fixed_grid and gb is not None:
            # Require a grid bucket. Thus, convert grid -> grid bucket.
            if isinstance(gb, pp.Grid):
                # Create a new grid bucket solely with a single grid as nodes.
                self.gb = pp.GridBucket()
                self.gb.add_nodes(gb)
            else:
                self.gb = gb

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
        if self.export_constants_separately:
            # Export constant data to vtu when outdated
            if not self._exported_constant_data_up_to_date:
                # Export constant subdomain data to vtu
                self._export_data_vtu(
                    self._constant_subdomain_data, time_step, constant_data=True
                )

                # Export constant interface data to vtu
                self._export_data_vtu(
                    self._constant_interface_data,
                    time_step,
                    constant_data=True,
                    interface_data=True,
                )

                # Store the time step counter for later reference
                # (required for pvd files)
                self._time_step_constant_data = time_step

                # Identify the constant data as fixed. Has the effect
                # that the constant data won't be exported if not the
                # mesh is updated or new constant data is added.
                self._exported_constant_data_up_to_date = True

            # Store the timestep referring to the origin of the constant data
            if self._time_step_constant_data:
                self._exported_timesteps_constants.append(self._time_step_constant_data)
        else:
            # Append constant subdomain and interface data to the
            # standard containers for subdomain and interface data.
            for key_s, value in self._constant_subdomain_data.items():
                subdomain_data[key_s] = value.copy()
            for key_i, value in self._constant_interface_data.items():
                interface_data[key_i] = value.copy()

        # Export subdomain and interface data to vtu format if existing
        if subdomain_data:
            self._export_data_vtu(subdomain_data, time_step)
        if interface_data:
            self._export_data_vtu(interface_data, time_step, interface_data=True)

        # Export grid bucket to pvd format
        file_name = self._make_file_name(self.file_name, time_step, extension=".pvd")
        file_name = self._append_folder_name(self.folder_name, file_name)
        self._export_gb_pvd(file_name, time_step)

    def write_pvd(
        self,
        times: Optional[np.ndarray] = None,
        file_extension: Optional[Union[np.ndarray, List[int]]] = None,
    ) -> None:
        """
        Interface function to export in PVD file the time loop information.
        The user should open only this file in paraview.

        We assume that the VTU associated files have the same name.
        We assume that the VTU associated files are in the working directory.

        Parameters:
        times (optional, np.ndarray): array of actual times to be exported. These will
            be the times associated with indivdiual time steps in, say, Paraview.
            By default, the times will be associated with the order in which the time
            steps were exported. This can be overridden by the file_extension argument.
            If no times are provided, the exported time steps are used.
        file_extension (np.array-like, optional): End of file names used in the export
            of individual time steps, see self.write_vtu(). If provided, it should have
            the same length as time. If not provided, the file names will be picked
            from those used when writing individual time steps.
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

        # Make mypy happy
        assert times is not None
        assert file_extension is not None

        # Extract the time steps related to constant data and
        # complying with file_extension. Implicitly test whether
        # file_extension is a subset of _exported_timesteps.
        # Only if constant data has been exported.
        include_constant_data = (
            self.export_constants_separately
            and len(self._exported_timesteps_constants) > 0
        )
        if include_constant_data:
            indices = [self._exported_timesteps.index(e) for e in file_extension]
            file_extension_constants = [
                self._exported_timesteps_constants[i] for i in indices
            ]

        # Perform the same procedure as in _export_gb_pvd
        # but looping over all designated time steps.

        o_file = open(
            self._append_folder_name(self.folder_name, self.file_name) + ".pvd", "w"
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
            # _export_gb_pvd.

            # Subdomain data
            for dim in self.dims:
                if self.meshio_geom[dim] is not None:
                    o_file.write(
                        fm % (time, self._make_file_name(self.file_name, fn, dim))
                    )

            # Interface data.
            for dim in self.m_dims:
                if self.m_meshio_geom[dim] is not None:
                    o_file.write(
                        fm
                        % (
                            time,
                            self._make_file_name(self.file_name + "_mortar", fn, dim),
                        )
                    )

        # Optionally, do the same for constant data.
        if include_constant_data:
            for time, fn_constants in zip(times, file_extension_constants):
                # Constant subdomain data.
                for dim in self.dims:
                    if self.meshio_geom[dim] is not None:
                        o_file.write(
                            fm
                            % (
                                time,
                                self._make_file_name(
                                    self.file_name + "_constant", fn_constants, dim
                                ),
                            )
                        )

                # Constant interface data.
                for dim in self.m_dims:
                    if self.m_meshio_geom[dim] is not None:
                        o_file.write(
                            fm
                            % (
                                time,
                                self._make_file_name(
                                    self.file_name + "_constant_mortar",
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
        data=None,  # ignore type which is essentially Union[DataInput, List[DataInput]]
    ) -> Tuple[SubdomainData, InterfaceData]:
        """
        Preprocess data.

        The routine has two goals:
        1. Splitting data into subdomain and interface data.
        2. Unify the data format. Store data in dictionaries, with keys given by
            grids/edges and names, and values given by the data arrays.

        Parameters:
            data (Union[DataInput, List[DataInput]], optional): data
                provided by the user in the form of strings and/or tuples
                of grids/edges.

        Returns:
            Tuple[SubdomainData, InterfaceData]: Subdomain and edge data decomposed and
                brought into unified format.
        """

        # The strategy is to traverse the input data and check each data point
        # for its type and apply a corresponding conversion to a unique format.
        # For each supported input data format, a dedicated routine is implemented
        # to 1. identify whether the type is present; and 2. update the data
        # dictionaries (to be returned in this routine) in the correct format:
        # The dictionaries use (grid,key) and (edge,key) as key and the data
        # array as value.

        # Now a list of these subroutines follows before the definition of the
        # actual routine.

        # Aux. method: Transform scalar- to vector-ranged values.
        def toVectorFormat(value: np.ndarray, g: pp.Grid) -> np.ndarray:
            """
            Check whether the value array has the right dimension corresponding
            to the grid size. If possible, translate the value to a vectorial
            object, but do nothing if the data naturally can be interpreted as
            scalar data.
            """
            # Make some checks
            if not value.size % g.num_cells == 0:
                raise ValueError("The data array is not compatible with the grid.")

            # Convert to vectorial data if more data provided than grid cells available,
            # and the value array is not already in vectorial format
            if not value.size == g.num_cells and not (
                len(value.shape) > 1 and value.shape[1] == g.num_cells
            ):
                value = np.reshape(value, (-1, g.num_cells), "F")

            return value

        # Aux. method: Check type for Interface
        def isinstance_interface(e: Interface) -> bool:
            """
            Implementation of isinstance(e, Interface).
            """
            return (
                isinstance(e, tuple)
                and len(e) == 2
                and isinstance(e[0], pp.Grid)
                and isinstance(e[1], pp.Grid)
            )

        # Aux. method: Detect and convert data of form "key".
        def add_data_from_str(pt, subdomain_data, interface_data):
            """
            Check whether data is provided by a key of a field - could be both node
            and edge data. If so, collect all data corresponding to grids and edges
            identified by the key.
            """

            # Only continue in case data is of type str
            if isinstance(pt, str):

                # Identify the key provided through the data.
                key = pt

                # Initialize tag storing whether data corrspeonding to the key has been found.
                has_key = False

                # Check data associated to subdomain field data
                for g, d in self.gb.nodes():
                    if pp.STATE in d and key in d[pp.STATE]:
                        # Mark the key as found
                        has_key = True

                        # Fetch data and convert to vectorial format if suggested by the size
                        value: np.ndarray = toVectorFormat(d[pp.STATE][key], g)

                        # Add data point in correct format to the collection
                        subdomain_data[(g, key)] = value

                # Check data associated to interface field data
                for e, d in self.gb.edges():
                    if pp.STATE in d and key in d[pp.STATE]:
                        # Mark the key as found
                        has_key = True

                        # Fetch data and convert to vectorial format if suggested by the size
                        mg = d["mortar_grid"]
                        value = toVectorFormat(d[pp.STATE][key], mg)

                        # Add data point in correct format to the collection
                        interface_data[(e, key)] = value

                # Make sure the key exists
                if not has_key:
                    raise ValueError(
                        f"No data with provided key {key} present in the grid bucket."
                    )

                # Return updated dictionaries and indicate succesful conversion.
                return subdomain_data, interface_data, True

            else:
                # Return original data dictionaries and indicate no modification.
                return subdomain_data, interface_data, False

        # Aux. method: Detect and convert data of form ([grids], "key").
        def add_data_from_Tuple_subdomains_str(
            pt: Tuple[List[pp.Grid], str], subdomain_data, interface_data
        ):
            """
            Check whether data is provided as tuple (g, key),
            where g is a list of grids, and key is a string.
            This routine explicitly checks only for subdomain data.
            """

            # Implementation of isinstance(t, Tuple[List[pp.Grid], str]).
            isinstance_tuple_subdomains_str = list(map(type, pt)) == [
                list,
                str,
            ] and all([isinstance(g, pp.Grid) for g in pt[0]])

            # If of correct type, convert to unique format and update subdomain data.
            if isinstance_tuple_subdomains_str:

                # By construction, the 1. and 2. components are a list of grids and a key.
                grids: List[pp.Grid] = pt[0]
                key = pt[1]

                # Loop over grids and fetch the states corresponding to the key
                for g in grids:

                    # Fetch the data dictionary containing the data value
                    d = self.gb.node_props(g)

                    # Make sure the data exists.
                    if not (pp.STATE in d and key in d[pp.STATE]):
                        raise ValueError(
                            f"""No state with prescribed key {key}
                            available on selected subdomains."""
                        )

                    # Fetch data and convert to vectorial format if suitable
                    value = toVectorFormat(d[pp.STATE][key], g)

                    # Add data point in correct format to collection
                    subdomain_data[(g, key)] = value

                # Return updated dictionaries and indicate succesful conversion.
                return subdomain_data, interface_data, True

            else:
                # Return original data dictionaries and indicate no modification.
                return subdomain_data, interface_data, False

        # Aux. method: Detect and convert data of form ([interfaces], "key").
        def add_data_from_Tuple_interfaces_str(pt, subdomain_data, interface_data):
            """
            Check whether data is provided as tuple (e, key),
            where e is a list of interfaces, and key is a string.
            This routine explicitly checks only for interface data.
            """

            # Implementation of isinstance(t, Tuple[List[Interface], str]).
            isinstance_tuple_interfaces_str = list(map(type, pt)) == [
                list,
                str,
            ] and all([isinstance_interface(g) for g in pt[0]])

            # If of correct type, convert to unique format and update subdomain data.
            if isinstance_tuple_interfaces_str:

                # By construction, the 1. and 2. components are a list of interfaces and a key.
                edges: List[Interface] = pt[0]
                key = pt[1]

                # Loop over edges and fetch the states corresponding to the key
                for e in edges:

                    # Fetch the data dictionary containing the data value
                    d = self.gb.edge_props(e)

                    # Make sure the data exists.
                    if not (pp.STATE in d and key in d[pp.STATE]):
                        raise ValueError(
                            f"""No state with prescribed key {key}
                            available on selected interfaces."""
                        )

                    # Fetch data and convert to vectorial format if suitable
                    mg = d["mortar_grid"]
                    value = toVectorFormat(d[pp.STATE][key], mg)

                    # Add data point in correct format to collection
                    interface_data[(e, key)] = value

                # Return updated dictionaries and indicate succesful conversion.
                return subdomain_data, interface_data, True

            else:
                # Return original data dictionaries and indicate no modification.
                return subdomain_data, interface_data, False

        # Aux. method: Detect and convert data of form (g, "key", value).
        def add_data_from_Tuple_subdomain_str_array(pt, subdomain_data, interface_data):
            """
            Check whether data is provided as tuple (g, key, data),
            where g is a single grid, key is a string, and data is a user-defined data array.
            This routine explicitly checks only for subdomain data.
            """

            # Implementation of isinstance(t, Tuple[pp.Grid, str, np.ndarray]).
            # NOTE: The type of a grid is identifying the specific grid type (simplicial
            # etc.) Thus, isinstance is used here to detect whether a grid is provided.
            types = list(map(type, pt))
            isinstance_Tuple_subdomain_str_array = isinstance(pt[0], pp.Grid) and types[
                1:
            ] == [str, np.ndarray]

            # Convert data to unique format and update the subdomain data dictionary.
            if isinstance_Tuple_subdomain_str_array:

                # Interpret (g, key, value) = (pt[0], pt[1], pt[2]);
                g = pt[0]
                key = pt[1]
                value = toVectorFormat(pt[2], g)

                # Add data point in correct format to collection
                subdomain_data[(g, key)] = value

                # Return updated dictionaries and indicate succesful conversion.
                return subdomain_data, interface_data, True

            else:
                # Return original data dictionaries and indicate no modification.
                return subdomain_data, interface_data, False

        # Aux. method: Detect and convert data of form (e, "key", value).
        def add_data_from_Tuple_interface_str_array(pt, subdomain_data, interface_data):
            """
            Check whether data is provided as tuple (g, key, data),
            where e is a single interface, key is a string, and data is a user-defined
            data array. This routine explicitly checks only for interface data.
            """

            # Implementation of isinstance(t, Tuple[Interface, str, np.ndarray]).
            isinstance_Tuple_interface_str_array = list(map(type, pt)) == [
                tuple,
                str,
                np.ndarray,
            ] and isinstance_interface(pt[0])

            # Convert data to unique format and update the interface data dictionary.
            if isinstance_Tuple_interface_str_array:
                # Interpret (e, key, value) = (pt[0], pt[1], pt[2]);
                e = pt[0]
                key = pt[1]

                # Fetch grid data for converting the value to right format;
                d = self.gb.edge_props(e)
                mg = d["mortar_grid"]
                value = toVectorFormat(pt[2], mg)

                # Add data point in correct format to collection
                interface_data[(e, key)] = value

                # Return updated dictionaries and indicate succesful conversion.
                return subdomain_data, interface_data, True

            else:
                # Return original data dictionaries and indicate no modification.
                return subdomain_data, interface_data, False

        # Aux. method: Detect and convert data of form ("key", value).
        def add_data_from_Tuple_str_array(pt, subdomain_data, interface_data):
            """
            Check whether data is provided by a tuple (key, data),
            where key is a string, and data is a user-defined data array.
            This only works when the grid bucket contains a single grid.
            """

            # Implementation if isinstance(pt, Tuple[str, np.ndarray].
            isinstance_Tuple_str_array = list(map(type, pt)) == [str, np.ndarray]

            # Convert data to unique format and update the interface data dictionary.
            if isinstance_Tuple_str_array:

                # Fetch the correct grid. This option is only supported for grid
                # buckets containing a single grid.
                grids = [g for g, _ in self.gb.nodes()]
                if not len(grids) == 1:
                    raise ValueError(
                        f"""The data type used for {pt} is only
                        supported if the grid bucket only contains a single grid."""
                    )

                # Fetch remaining ingredients required to define node data element
                g = grids[0]
                key = pt[0]
                value = toVectorFormat(pt[1], g)

                # Add data point in correct format to collection
                subdomain_data[(g, key)] = value

                # Return updated dictionaries and indicate succesful conversion.
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

        # Initialize container for data associated to nodes
        subdomain_data: SubdomainData = dict()
        interface_data: InterfaceData = dict()

        # Define methods to be used for checking the data type and performing
        # the conversion. This list implicitly also defines which input data is
        # allowed.
        methods = [
            add_data_from_str,
            add_data_from_Tuple_subdomains_str,
            add_data_from_Tuple_interfaces_str,
            add_data_from_Tuple_subdomain_str_array,
            add_data_from_Tuple_interface_str_array,
            add_data_from_Tuple_str_array,
        ]

        # Loop over all data points and collect them in a unified format.
        # For this, two separate dictionaries (for subdomain and interface data)
        # are used. The final format uses (subdomain/interface,key) as key of
        # the dictionaries, and the value is given by the corresponding data.
        for pt in data:

            # Initialize tag storing whether the conversion process for pt is successful.
            success = False

            for method in methods:

                # Check whether data point of right type and convert to
                # the unique data type.
                subdomain_data, interface_data, success = method(
                    pt, subdomain_data, interface_data
                )

                # Stop, once a supported data format has been detected.
                if success:
                    break

            # Check if provided data has non-supported data format.
            if not success:
                raise ValueError(
                    f"Input data {pt} has wrong format and cannot be exported."
                )

        return subdomain_data, interface_data

    def _update_constant_mesh_data(self) -> None:
        """
        Construct/update subdomain and interface data related with geometry and topology.

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
        for g, d in self.gb.nodes():
            ones = np.ones(g.num_cells, dtype=int)
            self._constant_subdomain_data[(g, "cell_id")] = np.arange(
                g.num_cells, dtype=int
            )
            self._constant_subdomain_data[(g, "grid_dim")] = g.dim * ones
            if "node_number" in d:
                self._constant_subdomain_data[(g, "grid_node_number")] = (
                    d["node_number"] * ones
                )
            self._constant_subdomain_data[(g, "is_mortar")] = 0 * ones
            self._constant_subdomain_data[(g, "mortar_side")] = (
                pp.grids.mortar_grid.MortarSides.NONE_SIDE.value * ones
            )

        # Define constant interface data related to the mesh

        # Initialize container for constant interface data
        if not hasattr(self, "_constant_interface_data"):
            self._constant_interface_data = dict()

        # Add mesh related, constant interface data by direct assignment.
        for e, d in self.gb.edges():

            # Fetch mortar grid
            mg = d["mortar_grid"]

            # Construct empty arrays for all extra edge data
            self._constant_interface_data[(e, "grid_dim")] = np.empty(0, dtype=int)
            self._constant_interface_data[(e, "cell_id")] = np.empty(0, dtype=int)
            self._constant_interface_data[(e, "grid_edge_number")] = np.empty(
                0, dtype=int
            )
            self._constant_interface_data[(e, "is_mortar")] = np.empty(0, dtype=int)
            self._constant_interface_data[(e, "mortar_side")] = np.empty(0, dtype=int)

            # Initialize offset
            mg_num_cells: int = 0

            # Assign extra edge data by collecting values on both sides.
            for side, g in mg.side_grids.items():
                ones = np.ones(g.num_cells, dtype=int)

                # Grid dimension of the mortar grid
                self._constant_interface_data[(e, "grid_dim")] = np.hstack(
                    (self._constant_interface_data[(e, "grid_dim")], g.dim * ones)
                )

                # Cell ids of the mortar grid
                self._constant_interface_data[(e, "cell_id")] = np.hstack(
                    (
                        self._constant_interface_data[(e, "cell_id")],
                        np.arange(g.num_cells, dtype=int) + mg_num_cells,
                    )
                )

                # Grid edge number of each edge
                self._constant_interface_data[(e, "grid_edge_number")] = np.hstack(
                    (
                        self._constant_interface_data[(e, "grid_edge_number")],
                        d["edge_number"] * ones,
                    )
                )

                # Whether the edge is mortar
                self._constant_interface_data[(e, "is_mortar")] = np.hstack(
                    (self._constant_interface_data[(e, "is_mortar")], ones)
                )

                # Side of the mortar
                self._constant_interface_data[(e, "mortar_side")] = np.hstack(
                    (
                        self._constant_interface_data[(e, "mortar_side")],
                        side.value * ones,
                    )
                )

                # Update offset
                mg_num_cells += g.num_cells

    def _export_data_vtu(
        self,
        data: Union[SubdomainData, InterfaceData],
        time_step: Optional[int],
        **kwargs,
    ) -> None:
        """
        Collect data associated to a single grid dimension
        and passing further to the final writing routine.

        For each fixed dimension, all subdomains of that dimension and the data
        related to those subdomains will be exported simultaneously.
        Analogously for interfaces.

        Parameters:
            data (Union[SubdomainData, InterfaceData]): Subdomain or interface data.
            time_step (int): time_step to be used to append the file name
            kwargs (optional): Optional keywords;
                'interface_data' (boolean) indicates whether data is associated to
                an interface, default is False;
                'constant_data' (boolean) indicates whether data is treated as
                constant in time, default is False.
        """
        # Check for optional keywords
        self.interface_data: bool = kwargs.pop("interface_data", False)
        self.constant_data: bool = kwargs.pop("constant_data", False)
        if kwargs:
            msg = "_export_data_vtu got unexpected keyword argument '{}'"
            raise TypeError(msg.format(kwargs.popitem()[0]))

        # Fetch the dimensions to be traversed. For subdomains, fetch the dimensions
        # of the available grids, and for interfaces fetch the dimensions of the available
        # mortar grids.
        is_subdomain_data: bool = not self.interface_data
        dims = self.dims if is_subdomain_data else self.m_dims

        # Define file name base
        file_name_base: str = self.file_name
        # Extend in case of constant data
        file_name_base += "_constant" if self.constant_data else ""
        # Extend in case of interface data
        file_name_base += "_mortar" if self.interface_data else ""
        # Append folder name to file name base
        file_name_base = self._append_folder_name(self.folder_name, file_name_base)

        # Collect unique keys, and for unique sorting, sort by alphabet
        keys = list(set([key for _, key in data]))
        keys.sort()

        # Collect the data and extra data in a single stack for each dimension
        for dim in dims:
            # Define the full file name
            file_name: str = self._make_file_name(file_name_base, time_step, dim)

            # Get all geometrical entities of dimension dim:
            # subdomains with correct grid dimension for subdomain data, and
            # interfaces with correct mortar grid dimension for interface data.
            # TODO update and simplify when new GridTree structure is in place.
            if is_subdomain_data:
                entities = self.gb.grids_of_dimension(dim).tolist()
            else:
                entities = [
                    e for e, d in self.gb.edges() if d["mortar_grid"].dim == dim
                ]

            # Construct the list of fields represented on this dimension.
            fields: List[Field] = []
            for key in keys:
                # Collect the values associated to all entities
                values = []
                for e in entities:
                    if (e, key) in data:
                        values.append(data[(e, key)])

                # Require data for all or none entities of that dimension.
                # NOTE: As implemented now, for any grid dimension and any prescribed
                # key, one has to provide data for all or none of the grids of that
                # dimension.
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

    def _export_gb_pvd(self, file_name: str, time_step: Optional[int]) -> None:
        """
        Routine to export to pvd format and collect all data scattered over
        several files for distinct grid dimensions.

        Parameters:
            file_name (str): storage path for pvd file
            time_step (int): used as appendix for the file name
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
        for dim in self.dims:
            if self.meshio_geom[dim] is not None:
                o_file.write(fm % self._make_file_name(self.file_name, time_step, dim))

        # Interface data.
        for dim in self.m_dims:
            if self.m_meshio_geom[dim] is not None:
                o_file.write(
                    fm
                    % self._make_file_name(self.file_name + "_mortar", time_step, dim)
                )

        # If constant data is exported to separate vtu files, also include
        # these here. The procedure is similar to the above, but the file names
        # incl. the relevant time step have to be adjusted.
        if self.export_constants_separately:
            # Constant subdomain data.
            for dim in self.dims:
                if self.meshio_geom[dim] is not None:
                    o_file.write(
                        fm
                        % self._make_file_name(
                            self.file_name + "_constant", self._time_step_constants, dim
                        )
                    )

            # Constant interface data.
            for dim in self.m_dims:
                if self.m_meshio_geom[dim] is not None:
                    o_file.write(
                        fm
                        % self._make_file_name(
                            self.file_name + "_constant_mortar",
                            self._time_step_constants,
                            dim,
                        )
                    )

        o_file.write("</Collection>\n" + "</VTKFile>")
        o_file.close()

    def _update_meshio_geom(self) -> None:
        """
        Manager for storing the grid information in meshio format.

        The internal variables meshio_geom and m_meshio_geom, storing all
        essential (mortar) grid information as points, connectivity and cell
        ids, are created and stored.
        """
        # Subdomains
        for dim in self.dims:
            # Get grids with dimension dim
            g = self.gb.get_grids(lambda g: g.dim == dim)
            # Export and store
            self.meshio_geom[dim] = self._export_grid(g, dim)

        # Interfaces
        for dim in self.m_dims:
            # Extract the mortar grids for dimension dim
            mgs = self.gb.get_mortar_grids(lambda g: g.dim == dim)
            # It contains the mortar grids "unrolled" by sides
            mg = np.array([g for m in mgs for _, g in m.side_grids.items()])
            # Export and store
            self.m_meshio_geom[dim] = self._export_grid(mg, dim)

    def _export_grid(self, gs: Iterable[pp.Grid], dim: int) -> Union[None, Meshio_Geom]:
        """
        Wrapper function to export grids of dimension dim. Calls the
        appropriate dimension specific export function.

        Parameters:
            gs (Iterable[pp.Grid]): Subdomains of same dimension.
            dim (int): Dimension of the subdomains.

        Returns:
            Meshio_Geom: Points, cells (storing the connectivity), and cell ids
            in correct meshio format.
        """
        if dim == 0:
            return None
        elif dim == 1:
            return self._export_1d(gs)
        elif dim == 2:
            return self._export_2d(gs)
        elif dim == 3:
            return self._export_3d(gs)
        else:
            raise ValueError(f"Unknown dimension {dim}")

    def _export_1d(self, gs: Iterable[pp.Grid]) -> Meshio_Geom:
        """
        Export the geometrical data (point coordinates) and connectivity
        information from the 1d PorePy grids to meshio.

        Parameters:
            gs (Iterable[pp.Grid]): 1d grids.

        Returns:
            Meshio_Geom: Points, 1d cells (storing the connectivity),
            and cell ids in correct meshio format.
        """

        # In 1d each cell is a line
        cell_type = "line"

        # Dictionary storing cell->nodes connectivity information
        cell_to_nodes: Dict[str, np.ndarray] = {cell_type: np.empty((0, 2), dtype=int)}

        # Dictionary collecting all cell ids for each cell type.
        # Since each cell is a line, the list of cell ids is trivial
        total_num_cells = np.sum(np.array([g.num_cells for g in gs]))
        cell_id: Dict[str, List[int]] = {
            cell_type: np.arange(total_num_cells, dtype=int).tolist()
        }

        # Data structure for storing node coordinates of all 1d grids.
        num_pts = np.sum([g.num_nodes for g in gs])
        meshio_pts = np.empty((num_pts, 3))  # type: ignore

        # Initialize offset. All data associated to 1d grids is stored in
        # the same vtu file, essentially using concatenation. To identify
        # each grid, keep track of number of nodes for each grid.
        nodes_offset = 0

        # Loop over all 1d grids
        for g in gs:

            # Store node coordinates
            sl = slice(nodes_offset, nodes_offset + g.num_nodes)
            meshio_pts[sl, :] = g.nodes.T

            # Lines are 1-simplices, and have a trivial connectvity.
            cn_indices = self._simplex_cell_to_nodes(1, g)

            # Add to previous connectivity information
            cell_to_nodes[cell_type] = np.vstack(
                (cell_to_nodes[cell_type], cn_indices + nodes_offset)
            )

            # Update offsets
            nodes_offset += g.num_nodes

        # Construct the meshio data structure
        num_blocks = len(cell_to_nodes)
        meshio_cells = np.empty(num_blocks, dtype=object)
        meshio_cell_id = np.empty(num_blocks, dtype=object)

        # For each cell_type store the connectivity pattern cell_to_nodes for
        # the corresponding cells with ids from cell_id.
        for block, (cell_type, cell_block) in enumerate(cell_to_nodes.items()):
            meshio_cells[block] = meshio.CellBlock(cell_type, cell_block.astype(int))
            meshio_cell_id[block] = np.array(cell_id[cell_type])

        # Return final meshio data: points, cell (connectivity), cell ids
        return Meshio_Geom(meshio_pts, meshio_cells, meshio_cell_id)

    def _simplex_cell_to_nodes(
        self, n: int, g: pp.Grid, cells: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Determine cell to node connectivity for a general n-simplex mesh.

        Parameters:
            n (int): dimension of the simplices in the grid.
            g (pp.Grid): grid containing cells and nodes.
            cells (np.ndarray, optional): all n-simplex cells

        Returns:
            np.ndarray: cell to node connectivity array, in which for each row
                (associated to cells) all associated nodes are marked.
        """
        # Determine cell-node ptr
        cn_indptr = (
            g.cell_nodes().indptr[:-1]
            if cells is None
            else g.cell_nodes().indptr[cells]
        )

        # Each n-simplex has n+1 nodes
        num_nodes = n + 1

        # Collect the indptr to all nodes of the cell.
        expanded_cn_indptr = np.vstack(
            [cn_indptr + i for i in range(num_nodes)]
        ).reshape(-1, order="F")

        # Detect all corresponding nodes by applying the expanded mask to the indices
        expanded_cn_indices = g.cell_nodes().indices[expanded_cn_indptr]

        # Convert to right format.
        cn_indices = np.reshape(expanded_cn_indices, (-1, num_nodes), order="C")

        return cn_indices

    def _export_2d(self, gs: Iterable[pp.Grid]) -> Meshio_Geom:
        """
        Export the geometrical data (point coordinates) and connectivity
        information from the 2d PorePy grids to meshio.

        Parameters:
            gs (Iterable[pp.Grid]): 2d grids.

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
        cell_to_nodes: Dict[str, np.ndarray] = {}
        # Dictionary collecting all cell ids for each cell type.
        cell_id: Dict[str, List[int]] = {}

        # Data structure for storing node coordinates of all 2d grids.
        num_pts = np.sum([g.num_nodes for g in gs])
        meshio_pts = np.empty((num_pts, 3))  # type: ignore

        # Initialize offsets. All data associated to 2d grids is stored in
        # the same vtu file, essentially using concatenation. To identify
        # each grid, keep track of number of nodes and cells for each grid.
        nodes_offset = 0
        cell_offset = 0

        # Loop over all 2d grids
        for g in gs:

            # Store node coordinates
            sl = slice(nodes_offset, nodes_offset + g.num_nodes)
            meshio_pts[sl, :] = g.nodes.T

            # Determine cell types based on number of faces=nodes per cell.
            num_faces_per_cell = g.cell_faces.getnnz(axis=0)

            # Loop over all available cell types and group cells of one type.
            g_cell_map = dict()
            for n in np.unique(num_faces_per_cell):

                # Define cell type; check if it coincides with a predefined cell type
                cell_type = polygon_map.get(f"polygon{n}", f"polygon{n}")

                # Find all cells with n faces, and store for later use
                cells = np.nonzero(num_faces_per_cell == n)[0]
                g_cell_map[cell_type] = cells

                # Store cell ids in global container; init if entry not yet established
                if cell_type not in cell_id:
                    cell_id[cell_type] = []

                # Add offset taking into account previous grids
                cell_id[cell_type] += (cells + cell_offset).tolist()

            # Determine cell-node connectivity for each cell type and all cells.
            # Treat triangle, quad and polygonal cells differently
            # aiming for optimized performance.
            for n in np.unique(num_faces_per_cell):

                # Define the cell type
                cell_type = polygon_map.get(f"polygon{n}", f"polygon{n}")

                # Check if cell_type already defined in cell_to_nodes, otherwise construct
                if cell_type not in cell_to_nodes:
                    cell_to_nodes[cell_type] = np.empty((0, n), dtype=int)

                # Special case: Triangle cells, i.e., n=3.
                if cell_type == "triangle":

                    # Triangles are simplices and have a trivial connectivity.

                    # Fetch triangle cells
                    cells = g_cell_map[cell_type]
                    # Determine the trivial connectivity - triangles are 2-simplices.
                    cn_indices = self._simplex_cell_to_nodes(2, g, cells)

                # Quad and polygon cells.
                else:
                    # For quads/polygons, g.cell_nodes cannot be blindly used as for
                    # triangles, since the ordering of the nodes may define a cell of
                    # the type (here specific for quads)
                    # x--x and not x--x
                    #  \/          |  |
                    #  /\          |  |
                    # x--x         x--x .
                    # Therefore, use both g.cell_faces and g.face_nodes to make use of face
                    # information and sort those to retrieve the correct connectivity.

                    # Strategy: Collect all cell nodes including their connectivity in a
                    # matrix of double num cell size. The goal will be to gather starting
                    # and end points for all faces of each cell, sort those faces, such that
                    # they form a circlular graph, and then choose the resulting starting
                    # points of all faces to define the connectivity.

                    # Fetch corresponding cells
                    cells = g_cell_map[cell_type]
                    # Determine all faces of all cells. Use an analogous approach as used to
                    # determine all cell nodes for triangle cells. And use that a polygon with
                    # n nodes has also n faces.
                    cf_indptr = g.cell_faces.indptr[cells]
                    expanded_cf_indptr = np.vstack(
                        [cf_indptr + i for i in range(n)]
                    ).reshape(-1, order="F")
                    cf_indices = g.cell_faces.indices[expanded_cf_indptr]

                    # Determine the associated (two) nodes of all faces for each cell.
                    fn_indptr = g.face_nodes.indptr[cf_indices]
                    # Extract nodes for first and second node of each face; reshape such
                    # that all first nodes of all faces for each cell are stored in one row,
                    # i.e., end up with an array of size num_cells (for this cell type) x n.
                    cfn_indices = [
                        g.face_nodes.indices[fn_indptr + i].reshape(-1, n)
                        for i in range(2)
                    ]
                    # Group first and second nodes, with alternating order of rows.
                    # By this, each cell is respresented by two rows. The first and second
                    # rows contain first and second nodes of faces. And each column stands
                    # for one face.
                    cfn = np.ravel(cfn_indices, order="F").reshape(n, -1).T

                    # Sort faces for each cell such that they form a chain. Use a function
                    # compiled with Numba. This step is the bottleneck of this routine.
                    cfn = self._sort_point_pairs_numba(cfn).astype(int)

                    # For each cell pick the sorted nodes such that they form a chain
                    # and thereby define the connectivity, i.e., skip every second row.
                    cn_indices = cfn[::2, :]

                # Add offset to account for previous grids, and store
                cell_to_nodes[cell_type] = np.vstack(
                    (cell_to_nodes[cell_type], cn_indices + nodes_offset)
                )

            # Update offsets
            nodes_offset += g.num_nodes
            cell_offset += g.num_cells

        # Construct the meshio data structure
        num_blocks = len(cell_to_nodes)
        meshio_cells = np.empty(num_blocks, dtype=object)
        meshio_cell_id = np.empty(num_blocks, dtype=object)

        # For each cell_type store the connectivity pattern cell_to_nodes for
        # the corresponding cells with ids from cell_id.
        for block, (cell_type, cell_block) in enumerate(cell_to_nodes.items()):

            # Meshio requires the keyword "polygon" for general polygons.
            # Thus, remove the number of nodes associated to polygons.
            cell_type_meshio_format = "polygon" if "polygon" in cell_type else cell_type
            meshio_cells[block] = meshio.CellBlock(
                cell_type_meshio_format, cell_block.astype(int)
            )
            meshio_cell_id[block] = np.array(cell_id[cell_type])

        # Return final meshio data: points, cell (connectivity), cell ids
        return Meshio_Geom(meshio_pts, meshio_cells, meshio_cell_id)

    def _sort_point_pairs_numba(self, lines: np.ndarray) -> np.ndarray:
        """
        This a simplified copy of pp.utils.sort_points.sort_point_pairs.

        Essentially the same functionality as sort_point_pairs, but stripped down
        to the special case of circular chains. Differently to sort_point_pairs, this
        variant sorts an arbitrary amount of independent point pairs. The chains
        are restricted by the assumption that each contains equally many line segments.
        Finally, this routine uses numba.

        Parameters:
            lines (np.ndarray): Array of size 2 * num_chains x num_lines_per_chain,
                containing node indices. For each pair of two rows, each column
                represents a line segment connectng the two nodes in the two entries
                of this column.

        Returns:
            np.ndarray: Sorted version of lines, where for each chain, the collection
                of line segments has been potentially flipped and sorted.
        """

        import numba

        @numba.jit(nopython=True)
        def _function_to_compile(lines):
            """
            Copy of pp.utils.sort_points.sort_point_pairs. This version is extended
            to multiple chains. Each chain is implicitly assumed to be circular.
            """

            # Retrieve number of chains and lines per chain from the shape.
            # Implicitly expect that all chains have the same length
            num_chains, chain_length = lines.shape
            # Since for each chain lines includes two rows, divide by two
            num_chains = int(num_chains / 2)

            # Initialize array of sorted lines to be the final output
            sorted_lines = np.zeros((2 * num_chains, chain_length))
            # Fix the first line segment for each chain and identify
            # it as in place regarding the sorting.
            sorted_lines[:, 0] = lines[:, 0]
            # Keep track of which lines have been fixed and which are still candidates
            found = np.zeros(chain_length)
            found[0] = 1

            # Loop over chains and consider each chain separately.
            for c in range(num_chains):
                # Initialize found making any line segment aside of the first a candidate
                found[1:] = 0

                # Define the end point of the previous and starting point for the next
                # line segment
                prev = sorted_lines[2 * c + 1, 0]

                # The sorting algorithm: Loop over all positions in the chain to be set next.
                # Find the right candidate to be moved to this position and possibly flipped
                # if needed. A candidate is identified as fitting if it contains one point
                # equal to the current starting point. This algorithm uses a double loop,
                # which is the most naive approach. However, assume chain_length is in
                # general small.
                for i in range(
                    1, chain_length
                ):  # The first line has already been found
                    for j in range(
                        1, chain_length
                    ):  # The first line has already been found
                        # A candidate line segment with matching start and end point
                        # in the first component of the point pair.
                        if np.abs(found[j]) < 1e-6 and lines[2 * c, j] == prev:
                            # Copy the segment to the right place
                            sorted_lines[2 * c : 2 * c + 2, i] = lines[
                                2 * c : 2 * c + 2, j
                            ]
                            # Mark as used
                            found[j] = 1
                            # Define the starting point for the next line segment
                            prev = lines[2 * c + 1, j]
                            break
                        # A candidate line segment with matching start and end point
                        # in the second component of the point pair.
                        elif np.abs(found[j]) < 1e-6 and lines[2 * c + 1, j] == prev:
                            # Flip and copy the segment to the right place
                            sorted_lines[2 * c, i] = lines[2 * c + 1, j]
                            sorted_lines[2 * c + 1, i] = lines[2 * c, j]
                            # Mark as used
                            found[j] = 1
                            # Define the starting point for the next line segment
                            prev = lines[2 * c, j]
                            break

            # Return the sorted lines defining chains.
            return sorted_lines

        # Run numba compiled function
        return _function_to_compile(lines)

    def _export_3d(self, gs: Iterable[pp.Grid]) -> Meshio_Geom:
        """
        Export the geometrical data (point coordinates) and connectivity
        information from 3d PorePy grids to meshio.

        Parameters:
            gs (Iterable[pp.Grid]): 3d grids.

        Returns:
            Meshio_Geom: Points, 3d cells (storing the connectivity), and
            cell ids in correct meshio format.
        """

        # Three different cell types will be distinguished: Tetrahedra, hexahedra,
        # and general polyhedra. meshio does not allow for a mix of cell types
        # in 3d within a single grid (and we export all 3d grids at once). Thus,
        # if the 3d grids contains cells of varying types, all cells will be cast
        # as polyhedra.

        # Determine the cell types present among all grids.
        cell_types: Set[str] = set()
        for g in gs:

            # The number of faces per cell wil be later used to determining
            # the cell types
            num_faces_per_cell = np.unique(g.cell_faces.getnnz(axis=0))

            if num_faces_per_cell.shape[0] == 1:
                n = num_faces_per_cell[0]
                if n == 4:
                    cell_types.add("tetra")
                elif n == 6 and isinstance(g, pp.CartGrid):
                    cell_types.add("hexahedron")
                else:
                    cell_types.add("polyhedron")
            else:
                cell_types.add("polyhedron")

        # Use dedicated export for each grid type
        if len(cell_types) == 1 and "tetra" in cell_types:
            return self._export_simplex_3d(gs)
        elif len(cell_types) == 1 and "hexahedron" in cell_types:
            return self._export_hexahedron_3d(gs)
        else:
            return self._export_polyhedron_3d(gs)

    def _export_simplex_3d(self, gs: Iterable[pp.Grid]) -> Meshio_Geom:
        """
        Export the geometrical data (point coordinates) and connectivity
        information from 3d PorePy simplex grids to meshio.

        Parameters:
            gs (Iterable[pp.Grid]): 3d simplex grids.

        Returns:
            Meshio_Geom: Points, 3d cells (storing the connectivity), and
            cell ids in correct meshio format.
        """
        # For the special meshio geometric type "tetra", cell->nodes will
        # be used to store connectivity information. Each simplex has 4 nodes.
        cell_to_nodes = np.empty((0, 4), dtype=int)

        # Dictionary collecting all cell ids for each cell type.
        cell_id: List[int] = []

        # Data structure for storing node coordinates of all 3d grids.
        num_pts = np.sum([g.num_nodes for g in gs])
        meshio_pts = np.empty((num_pts, 3))  # type: ignore

        # Initialize offsets. All data associated to 3d grids is stored in
        # the same vtu file, essentially using concatenation. To identify
        # each grid, keep track of number of nodes and cells for each grid.
        nodes_offset = 0
        cell_offset = 0

        # Treat each 3d grid separately.
        for g in gs:

            # Store node coordinates
            sl = slice(nodes_offset, nodes_offset + g.num_nodes)
            meshio_pts[sl, :] = g.nodes.T

            # Identify all cells as tetrahedra.
            cells = np.arange(g.num_cells)

            # Add offset taking into account previous grids
            cell_id += (cells + cell_offset).tolist()

            # Determine local connectivity for all cells. Simplices are invariant
            # under permuatations. Thus, no specific ordering of nodes is required.
            # Tetrahedra are essentially 3-simplices.
            cn_indices = self._simplex_cell_to_nodes(3, g, cells)

            # Store cell-node connectivity, and add offset taking into account
            # previous grids
            cell_to_nodes = np.vstack((cell_to_nodes, cn_indices + nodes_offset))

            # Update offset
            nodes_offset += g.num_nodes
            cell_offset += g.num_cells

        # Initialize the meshio data structure for the connectivity and cell ids.
        # There is only one cell type.
        meshio_cells = np.empty(1, dtype=object)
        meshio_cell_id = np.empty(1, dtype=object)
        meshio_cells[0] = meshio.CellBlock("tetra", cell_to_nodes.astype(int))
        meshio_cell_id[0] = np.array(cell_id)

        # Return final meshio data: points, cell (connectivity), cell ids
        return Meshio_Geom(meshio_pts, meshio_cells, meshio_cell_id)

    def _export_hexahedron_3d(self, gs: Iterable[pp.Grid]) -> Meshio_Geom:
        """
        Export the geometrical data (point coordinates) and connectivity
        information from 3d PorePy hexahedron grids to meshio.

        NOTE: Optimally, a StructuredGrid would be exported in this case.
        However, meshio does solely handle unstructured grids and cannot
        for instance write to *.vtr format.

        Parameters:
            gs (Iterable[pp.Grid]): 3d hexahedron grids.

        Returns:
            Meshio_Geom: Points, 3d cells (storing the connectivity), and
            cell ids in correct meshio format.
        """
        # For the special meshio geometric type "hexahedron", cell->nodes will
        # be used to store connectivity information. Each hexahedron has 8 nodes.
        cell_to_nodes = np.empty((0, 8), dtype=int)

        # Dictionary collecting all cell ids for each cell type.
        cell_id: List[int] = []

        # Data structure for storing node coordinates of all 3d grids.
        num_pts = np.sum([g.num_nodes for g in gs])
        meshio_pts = np.empty((num_pts, 3))  # type: ignore

        # Initialize offsets. Required taking into account multiple 3d grids.
        nodes_offset = 0
        cell_offset = 0

        # Treat each 3d grid separately.
        for g in gs:

            # Store node coordinates
            sl = slice(nodes_offset, nodes_offset + g.num_nodes)
            meshio_pts[sl, :] = g.nodes.T

            # Identify all cells as tetrahedra.
            cells = np.arange(g.num_cells)

            # Add offset taking into account previous grids
            cell_id += (cells + cell_offset).tolist()

            # Determine local connectivity for all cells. Simplices are invariant
            # under permuatations. Thus, no specific ordering of nodes is required.

            # After all, the procedure is fairly similar to the treatment of
            # tetra cells. Most of the following code is analogous to
            # _simplex_cell_to_nodes(). However, for hexaedron cells the order
            # of the nodes is important and has to be adjusted. Based on the
            # specific definition of TensorGrids however, the node numbering
            # can be hardcoded, which results in better performance compared
            # to a modular procedure.

            # Determine cell-node ptr
            cn_indptr = (
                g.cell_nodes().indptr[:-1]
                if cells is None
                else g.cell_nodes().indptr[cells]
            )

            # Collect the indptr to all nodes of the cell;
            # each n-simplex cell contains n nodes
            expanded_cn_indptr = np.vstack([cn_indptr + i for i in range(8)]).reshape(
                -1, order="F"
            )

            # Detect all corresponding nodes by applying the expanded mask to indices
            expanded_cn_indices = g.cell_nodes().indices[expanded_cn_indptr]

            # Convert to right format.
            cn_indices = np.reshape(expanded_cn_indices, (-1, 8), order="C")

            # Reorder nodes to comply with the node numbering convention of meshio
            # regarding hexahedron cells.
            cn_indices = cn_indices[:, [0, 2, 6, 4, 1, 3, 7, 5]]

            # Test whether the hard-coded numbering really defines a hexahedron.
            assert self._test_hex_meshio_format(g.nodes, cn_indices)

            # Store cell-node connectivity, and add offset taking into account
            # previous grids
            cell_to_nodes = np.vstack((cell_to_nodes, cn_indices + nodes_offset))

            # Update offset
            nodes_offset += g.num_nodes
            cell_offset += g.num_cells

        # Initialize the meshio data structure for the connectivity and cell ids.
        # There is only one cell type.
        meshio_cells = np.empty(1, dtype=object)
        meshio_cell_id = np.empty(1, dtype=object)
        meshio_cells[0] = meshio.CellBlock("hexahedron", cell_to_nodes.astype(int))
        meshio_cell_id[0] = np.array(cell_id)

        # Return final meshio data: points, cell (connectivity), cell ids
        return Meshio_Geom(meshio_pts, meshio_cells, meshio_cell_id)

    def _test_hex_meshio_format(self, nodes, cn_indices) -> bool:

        import numba

        @numba.jit(nopython=True)
        def _function_to_compile(nodes, cn_indices) -> bool:
            """
            Test whether the node numbering for each cell complies
            with the hardcoded numbering used by meshio, cf. documentation
            of meshio.

            For this, fetch three potential sides of the hex and
            test whether they lie circular chain within a plane.
            Here, the test is successful, if the node sets [0,1,2,3],
            [4,5,6,7], and [0,1,5,4] define planes.
            """

            # Assume initially correct format
            correct_format = True

            # Test each cell separately
            for i in range(cn_indices.shape[0]):

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

    def _export_polyhedron_3d(self, gs: Iterable[pp.Grid]) -> Meshio_Geom:
        """
        Export the geometrical data (point coordinates) and connectivity
        information from 3d PorePy polyhedron grids to meshio.

        Parameters:
            gs (Iterable[pp.Grid]): 3d polyhedron grids.

        Returns:
            Meshio_Geom: Points, 3d cells (storing the connectivity), and
            cell ids in correct meshio format.
        """
        # For the general geometric type "polyhedron", cell->faces->nodes will
        # be used to store connectivity information, which can become significantly
        # larger than cell->nodes information used for special type grids.
        cell_to_faces: Dict[str, List[List[int]]] = {}

        # Dictionary collecting all cell ids for each cell type.
        cell_id: Dict[str, List[int]] = {}

        # Data structure for storing node coordinates of all 3d grids.
        num_pts = np.sum([g.num_nodes for g in gs])
        meshio_pts = np.empty((num_pts, 3))  # type: ignore

        # Initialize offsets. Required taking into account multiple 3d grids.
        nodes_offset = 0
        cell_offset = 0

        # Treat each 3d grid separately.
        for g in gs:

            # Store node coordinates
            sl = slice(nodes_offset, nodes_offset + g.num_nodes)
            meshio_pts[sl, :] = g.nodes.T

            # The number of faces per cell wil be later used to determining
            # the cell types
            num_faces_per_cell = g.cell_faces.getnnz(axis=0)

            # Categorize all polyhedron cells by their number of faces.
            # Each category will be treated separatrely allowing for using
            # fitting datastructures.
            g_cell_map = dict()
            for n in np.unique(num_faces_per_cell):

                cell_type = f"polyhedron{n}"

                # Find all cells with n faces, and store for later use
                cells = np.nonzero(num_faces_per_cell == n)[0]
                g_cell_map[cell_type] = cells

                # Store cell ids in global container; init if entry not yet established
                if cell_type not in cell_id:
                    cell_id[cell_type] = []

                # Add offset taking into account previous grids
                cell_id[cell_type] += (cells + cell_offset).tolist()

            # Determine local connectivity for all cells. Due to differnt
            # treatment of special and general cell types and to conform
            # with array sizes (for polyhedra), treat each cell type
            # separately.
            for cell_type in g_cell_map.keys():

                # The general strategy is to define the connectivity as cell-face-nodes
                # information, where the faces are defined by nodes. Hence, this
                # information is significantly larger than the info provided for
                # tetra cells. Here, we make use of the fact that g.face_nodes
                # provides nodes ordered wrt. the right-hand rule.

                # Fetch cells with n faces
                cells = g_cell_map[cell_type]

                # Store short cuts to cell-face and face-node information
                cf_indptr = g.cell_faces.indptr
                cf_indices = g.cell_faces.indices
                fn_indptr = g.face_nodes.indptr
                fn_indices = g.face_nodes.indices

                # Determine the cell-face connectivity (with faces described by their
                # nodes ordered such that they form a chain and are identified by the
                # face boundary. The final data format is a List[List[np.ndarray]].
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
            nodes_offset += g.num_nodes
            cell_offset += g.num_cells

        # Determine the total number of blocks / cell types.
        num_blocks = len(cell_to_faces)

        # Initialize the meshio data structure for the connectivity and cell ids.
        meshio_cells = np.empty(num_blocks, dtype=object)
        meshio_cell_id = np.empty(num_blocks, dtype=object)

        # Store the cells in meshio format
        for block, (cell_type, cell_block) in enumerate(cell_to_faces.items()):
            # Adapt the block number taking into account of previous cell types.
            meshio_cells[block] = meshio.CellBlock(cell_type, cell_block)
            meshio_cell_id[block] = np.array(cell_id[cell_type])

        # Return final meshio data: points, cell (connectivity), cell ids
        return Meshio_Geom(meshio_pts, meshio_cells, meshio_cell_id)

    def _write(
        self,
        fields: Iterable[Field],
        file_name: str,
        meshio_geom: Meshio_Geom,
    ) -> None:
        """
        Interface to meshio for exporting cell data.

        Parameters:
            fields (iterable, Field): fields which shall be exported
            file_name (str): name of final file of export
            meshio_geom (Meshio_Geom): Namedtuple of points,
                connectivity information, and cell ids in
                meshio format (for a single dimension).
        """
        # Initialize empty cell data dictinary
        cell_data = {}

        # Split the data for each group of geometrically uniform cells
        # Utilize meshio_geom for this.
        for field in fields:

            # Although technically possible, as implemented, field.values should never be None.
            assert field.values is not None

            # For each field create a sub-vector for each geometrically uniform group of cells
            num_block = meshio_geom.cell_ids.size
            cell_data[field.name] = np.empty(num_block, dtype=object)

            # Fill up the data
            for block, ids in enumerate(meshio_geom.cell_ids):
                if field.values.ndim == 1:
                    cell_data[field.name][block] = field.values[ids]
                elif field.values.ndim == 2:
                    cell_data[field.name][block] = field.values[:, ids].T
                else:
                    raise ValueError

        # Create the meshio object
        meshio_grid_to_export = meshio.Mesh(
            meshio_geom.pts, meshio_geom.connectivity, cell_data=cell_data
        )

        # Write mesh information and data to VTK format.
        meshio.write(file_name, meshio_grid_to_export, binary=self.binary)

    def _append_folder_name(
        self, folder_name: Optional[str] = None, name: str = ""
    ) -> str:
        """
        Auxiliary method setting up potentially non-existent folder structure and
        setting up a path for exporting.

        Parameters:
            folder_name (str, optional): name of the folder
            name (str): prefix of the name of the files to be written

        Returns:
            str: complete path to the files to be written.
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
        """
        Auxiliary method to setting up file name.

        The final name is built as combination of a prescribed prefix,
        and possibly the dimension of underlying grid and time (step)
        the related data is associated to.

        Parameters:
            file_name (str): prefix of the name of file to be exported
            time_step (Union[float int], optional): time or time step (index)
            dim (int, optional): dimension of the exported grid
            extension (str): extension of the file, typically file ending
                defining the format

        Returns:
            str: complete name of file
        """

        # Define non-empty time step extension including zero padding.
        time_extension = (
            "" if time_step is None else "_" + str(time_step).zfill(self.padding)
        )

        # Define non-empty dim extension
        dim_extension = "" if dim is None else "_" + str(dim)

        # Combine prefix and extensions to define the complete name
        return file_name + dim_extension + time_extension + extension
