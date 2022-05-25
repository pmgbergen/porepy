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
from typing import Dict, Iterable, List, Optional, Tuple, Union

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

# Altogether allowed data structures to define data for exporting
DataInput = Union[
    # Keys for states
    str,
    # Subdomain specific data types
    Tuple[Union[pp.Grid, List[pp.Grid]], str],
    Tuple[pp.Grid, str, np.ndarray],
    Tuple[str, np.ndarray],
    # Interface specific data types
    Tuple[Union[Interface, List[Interface]], str],
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
    binary: (optional) export in binary format, default is True.
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

    where times, is a list of actual times (not time steps), associated
    to the previously exported time steps.

    In general, pvd files gather data exported in separate files, including
    data on differently dimensioned grids, constant data, and finally time steps.


    deltaT is a scalar, denoting the time step size, and steps

    In the case of different keywords, change the file name with
    "change_name".

    NOTE: the following names are reserved for data exporting: grid_dim,
    is_mortar, mortar_side, cell_id, grid_node_number, grid_edge_number
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
            folder_name (str, optional): folder name, all files are stored in
            kwargs (optional): Optional keywords;
                'fixed_grid' (boolean) to control whether the grid(bucket) may be redfined
                (default True);
                'binary' (boolean) controling whether data is stored in binary format
                (default True);
                'export_constants_separately' (boolean) controling whether
                constant data is exported in separate files (default True).
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

    # TODO in use by anyone?
    def change_name(self, file_name: str) -> None:
        """
        Change the root name of the files, useful when different keywords are
        considered but on the same grid.

        Parameters:
            file_name (str): the new root name of the files.

        """
        self.file_name = file_name

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
        """
        # Identify change in constant data. Has the effect that
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
                geometical infos are exported.
            time_dependent (boolean, optional): If False, file names will
                not be appended with an index that markes the time step.
                Can be overwritten by giving a value to time_step; if not,
                the file names will subsequently be ending with 1, 2, etc.
            time_step (int, optional): will be used ass eppendic to define
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

            # Store the timestep refering to the origin of the constant data
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
        times: np.ndarray,
        file_extension: Optional[Union[np.ndarray, List[int]]] = None,
    ) -> None:
        """
        Interface function to export in PVD file the time loop information.
        The user should open only this file in paraview.

        We assume that the VTU associated files have the same name.
        We assume that the VTU associated files are in the same folder.

        Parameters:
        times (np.ndarray): array of times to be exported. These will be the times
            associated with indivdiual time steps in, say, Paraview. By default,
            the times will be associated with the order in which the time steps were
            exported. This can be overridden by the file_extension argument.
        file_extension (np.array-like, optional): End of file names used in the export
            of individual time steps, see self.write_vtu(). If provided, it should have
            the same length as time. If not provided, the file names will be picked
            from those used when writing individual time steps.
        """
        if file_extension is None:
            file_extension = self._exported_timesteps
            file_extension_constants = self._exported_timesteps_constants
            indices = [self._exported_timesteps.index(e) for e in file_extension]
            file_extension_constants = [
                self._exported_timesteps_constants[i] for i in indices
            ]
        elif isinstance(file_extension, np.ndarray):
            file_extension = file_extension.tolist()

            # Make sure that the inputs are consistent
            assert isinstance(file_extension, list)
            assert len(file_extension) == times.shape[0]

            # Extract the time steps related to constant data and
            # complying with file_extension. Implicitly test wheter
            # file_extension is a subset of _exported_timesteps.
            indices = [self._exported_timesteps.index(e) for e in file_extension]
            file_extension_constants = [
                self._exported_timesteps_constants[i] for i in indices
            ]

        # Make mypy happy
        assert file_extension is not None

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

        for time, fn, fn_constants in zip(
            times, file_extension, file_extension_constants
        ):
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

            # Constant data.
            if self.export_constants_separately:
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

        # Convert data to list, while keeping the data structures provided,
        # or provide an empty list if no data provided
        if data is None:
            data = list()
        elif not isinstance(data, list):
            data = [data]

        # Initialize container for data associated to nodes
        subdomain_data: SubdomainData = dict()
        interface_data: InterfaceData = dict()

        # Auxiliary function transforming scalar ranged values
        # to vector ranged values when suitable.
        def toVectorFormat(value: np.ndarray, g: pp.Grid) -> np.ndarray:
            """
            Check whether the value array has the right dimension corresponding
            to the grid size. If possible, translate the value to a vectorial
            object, But do nothing if the data naturally can be interpreted as
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

        # Auxiliary functions for type checking

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

        def isinstance_Tuple_subdomains_str(t: Tuple[List[pp.Grid], str]) -> bool:
            """
            Implementation of isinstance(t, Tuple[List[pp.Grid], str]).

            Detects data input of type ([g1, g2, ...], "name"),
            where g1, g2, ... are subdomains.
            """
            # Implementation of isinstance(Tuple[List[pp.Grid], str], t)
            return list(map(type, t)) == [list, str] and all(
                [isinstance(g, pp.Grid) for g in t[0]]
            )

        def isinstance_Tuple_interfaces_str(t: Tuple[List[Interface], str]) -> bool:
            """
            Implementation of isinstance(t, Tuple[List[Interface], str]).

            Detects data input of type ([e1, e2, ...], "name"),
            where e1, e2, ... are interfaces.
            """
            # Implementation of isinstance(t, Tuple[List[Tuple[pp.Grid, pp.Grid]], str])
            return list(map(type, t)) == [list, str] and all(
                [isinstance_interface(g) for g in t[0]]
            )

        def isinstance_Tuple_subdomain_str_array(
            t: Tuple[pp.Grid, str, np.ndarray]
        ) -> bool:
            """
            Implementation of isinstance(t, Tuple[pp.Grid, str, np.ndarray]).
            """
            types = list(map(type, t))
            return isinstance(t[0], pp.Grid) and types[1:] == [str, np.ndarray]

        def isinstance_Tuple_interface_str_array(
            t: Tuple[Interface, str, np.ndarray]
        ) -> bool:
            """
            Implementation of isinstance(t, Tuple[Interface, str, np.ndarray]).
            """
            return list(map(type, t)) == [
                tuple,
                str,
                np.ndarray,
            ] and isinstance_interface(t[0])

        def isinstance_Tuple_str_array(pt) -> bool:
            """
            Implementation if isinstance(pt, Tuple[str, np.ndarray].
            Detects data input of type ("name", value).
            """
            # Check whether the tuple is of length 2 and has the right types.
            return list(map(type, pt)) == [str, np.ndarray]

        # Loop over all data points and convert them collect them in the format
        # (grid/edge, key, data) for single grids etc. and store them in two
        # dictionaries with keys (grid/egde, key) and value given by data.
        # Distinguish here between subdomain and interface data and store
        # accordingly.
        for pt in data:

            # Allow for different cases. Distinguish each case separately.

            # Case 1: Data provided by the key of a field only - could be both node
            # and edge data. Idea: Collect all data corresponding to grids and edges
            # related to the key.
            if isinstance(pt, str):

                # The data point is simply a string addressing a state using a key
                key = pt

                # Fetch grids and interfaces as well as data associated to the key
                has_key = False

                # Try to find the key in the data dictionary associated to subdomains
                for g, d in self.gb.nodes():
                    if pp.STATE in d and key in d[pp.STATE]:
                        # Mark the key as found
                        has_key = True

                        # Fetch data and convert to vectorial format if suggested by the size
                        value: np.ndarray = toVectorFormat(d[pp.STATE][key], g)

                        # Add data point in correct format to the collection
                        subdomain_data[(g, key)] = value

                # Try to find the key in the data dictionary associated to interfaces
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

            # Case 2a: Data provided by a tuple (g, key).
            # Here, g is a single grid or a list of grids.
            # Idea: Collect the data only among the grids specified.
            elif isinstance_Tuple_subdomains_str(pt):

                # By construction, the first component contains is a list of grids.
                grids: List[pp.Grid] = pt[0]

                # By construction, the second component contains a key.
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

            # Case 2b: Data provided by a tuple (e, key)
            # Here, e is a list of edges.
            # Idea: Collect the data only among the grids specified.
            elif isinstance_Tuple_interfaces_str(pt):

                # By construction, the first component contains a list of interfaces.
                edges: List[Interface] = pt[0]

                # By construction, the second component contains a key.
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

            # Case 3: Data provided by a tuple (key, data).
            # This case is only well-defined, if the grid bucket contains
            # only a single grid.
            # Idea: Extract the unique grid and continue as in Case 2.
            elif isinstance_Tuple_str_array(pt):

                # Fetch the correct grid. This option is only supported for grid
                # buckets containing a single grid.
                grids = [g for g, _ in self.gb.nodes()]
                g = grids[0]
                if not len(grids) == 1:
                    raise ValueError(
                        f"""The data type used for {pt} is only
                        supported if the grid bucket only contains a single grid."""
                    )

                # Fetch remaining ingredients required to define node data element
                key = pt[0]
                value = toVectorFormat(pt[1], g)

                # Add data point in correct format to collection
                subdomain_data[(g, key)] = value

            # Case 4a: Data provided as tuple (g, key, data).
            # Idea: Since this is already the desired format, process naturally.
            elif isinstance_Tuple_subdomain_str_array(pt):
                # Data point in correct format
                g = pt[0]
                key = pt[1]
                value = toVectorFormat(pt[2], g)

                # Add data point in correct format to collection
                subdomain_data[(g, key)] = value

            # Case 4b: Data provided as tuple (e, key, data).
            # Idea: Since this is already the desired format, process naturally.
            elif isinstance_Tuple_interface_str_array(pt):
                # Data point in correct format
                e = pt[0]
                key = pt[1]
                d = self.gb.edge_props(e)
                mg = d["mortar_grid"]
                value = toVectorFormat(pt[2], mg)

                # Add data point in correct format to collection
                interface_data[(e, key)] = value

            else:
                raise ValueError(
                    f"The provided data type used for {pt} is not supported."
                )

        return subdomain_data, interface_data

    def _update_constant_mesh_data(self) -> None:
        """
        Construct/update subdomain and interface data related with geometry and topology.

        The data is identified as constant data. The final containers, stored as
        attributes _constant_subdomain_data and _constant_interface_data, have
        the same format as the output of _sort_and_unify_data.
        """
        # Identify change in constant data. Has the effect that
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
        related to that subdomains will be exported simultaneously.
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
                # FIXME: By changing the export strategy, this could be fixed.
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
        # incl. the relevant time step has to be adjusted.
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
        Auxiliary method upating internal copies of the grids. Merely asks for
        exporting grids.
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

        # Initialize offset. Required taking into account multiple 1d grids.
        nodes_offset = 0

        # Loop over all 1d grids
        for g in gs:

            # Store node coordinates
            sl = slice(nodes_offset, nodes_offset + g.num_nodes)
            meshio_pts[sl, :] = g.nodes.T

            # Lines are simplices, and have a trivial connectvity.
            cn_indices = self._simplex_cell_to_nodes(2, g)

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
            n (int): order of the simplices in the grid.
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

        # Collect the indptr to all nodes of the cell; each n-simplex cell contains n nodes
        expanded_cn_indptr = np.vstack([cn_indptr + i for i in range(n)]).reshape(
            -1, order="F"
        )

        # Detect all corresponding nodes by applying the expanded mask to the indices
        expanded_cn_indices = g.cell_nodes().indices[expanded_cn_indptr]

        # Convert to right format.
        cn_indices = np.reshape(expanded_cn_indices, (-1, n), order="C")

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

        # Initialize offsets. Required taking into account multiple 2d grids.
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
                    # Determine the trivial connectivity.
                    cn_indices = self._simplex_cell_to_nodes(3, g, cells)

                # Quad and polygon cells.
                else:
                    # For quads/polygons, g.cell_nodes cannot be blindly used as for
                    # triangles, since the ordering of the nodes may define a cell of
                    # the type (here specific for quads) x--x and not x--x
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

            # Meshio requires the keyword "polgon" for general polygons.
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
            # Since for each chain lines includes two rows, take the half
            num_chains = int(num_chains / 2)

            # Initialize array of sorted lines to be the final output
            sorted_lines = np.zeros((2 * num_chains, chain_length))
            # Fix the first line segment for each chain and identify
            # it as in place regarding the sorting.
            sorted_lines[:, 0] = lines[:, 0]
            # Keep track of which lines have been fixed and which are still candidates
            found = np.zeros(chain_length)
            found[0] = 1

            # Loop over chains and Consider each chain separately.
            for c in range(num_chains):
                # Initialize found making any line segment aside of the first a candidate
                found[1:] = 0

                # Define the end point of the previous and starting point for the next
                # line segment
                prev = sorted_lines[2 * c + 1, 0]

                # The sorting algorithm: Loop over all position in the chain to be set next.
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

        # FIXME The current implementation on first sight could suggest that
        # grids with varying cell types are allowed. However, this is not the
        # case. Rewriting the code makeing this distinction more clear would
        # allow for better overview.

        # Three different cell types will be distinguished: Tetrahedra, hexahedra,
        # and general polyhedra. meshio does not allow for a mix of cell types
        # in 3d within a single grid. Thus, if a grid contains cells of varying
        # types, all cells will be casted as polyhedra.

        # Dictionaries storing cell->faces and cell->nodes connectivity
        # information for all cell types. For the special meshio geometric
        # types "tetra" and "hexahedron", cell->nodes will be used; the
        # local node ordering for each cell has to comply with the convention
        # in meshio. For the general "polyhedron" type, instead the connectivity
        # will be described by prescribing cell->faces->nodes connectivity, i.e.,
        # for each cell, all faces are listed, and for all these faces, its
        # nodes are identified, ordered such that they describe a circular chain.
        cell_to_faces: Dict[str, List[List[int]]] = {}
        cell_to_nodes: Dict[str, np.ndarray] = {}

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

            # Loop over all available cell types and group cells of one type.
            g_cell_map = dict()
            for n in np.unique(num_faces_per_cell):

                # Define the cell type.
                # Make sure that either all cells are identified as "tetra",
                # "hexahedron", or "polyehdron". "tetra" grids have the unique
                # property that all cells have 4 faces; "hexahdron" grids in
                # PorePy are solely of CartGrid type, and all cells have 6
                # faces (NOTE that coarsening a grid does not change its type,
                # hence checking solely the type is not sufficient); all other
                # grids will be identified as "polyehdron" grids, even if they
                # contain single "tetra" cells.
                if n == 4 and len(set(num_faces_per_cell)) == 1:
                    cell_type = "tetra"
                elif (
                    isinstance(g, pp.CartGrid)
                    and n == 6
                    and len(set(num_faces_per_cell)) == 1
                ):
                    cell_type = "hexahedron"
                else:
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
            for n in np.unique(num_faces_per_cell):

                # Define the cell type as above
                if n == 4 and len(set(num_faces_per_cell)) == 1:
                    cell_type = "tetra"
                elif (
                    isinstance(g, pp.CartGrid)
                    and n == 6
                    and len(set(num_faces_per_cell)) == 1
                ):
                    cell_type = "hexahedron"
                else:
                    cell_type = f"polyhedron{n}"

                # Special case: Tetrahedra
                if cell_type == "tetra":
                    # Since tetra is a meshio-known cell type, cell_to_nodes connectivity
                    # information is provided, i.e., for each cell the nodes in the
                    # meshio-defined local numbering of nodes is generated. Since tetra
                    # is a simplex, the nodes do not need to be ordered in any specific
                    # type, as the geometrical object is invariant under permutations.

                    # Fetch tetra cells
                    cells = g_cell_map[cell_type]

                    # Tetrahedra are simplices and have a trivial connectivity.
                    cn_indices = self._simplex_cell_to_nodes(4, g, cells)

                    # Initialize data structure if not available yet
                    if cell_type not in cell_to_nodes:
                        cell_to_nodes[cell_type] = np.empty((0, 4), dtype=int)

                    # Store cell-node connectivity, and add offset taking into account
                    # previous grids
                    cell_to_nodes[cell_type] = np.vstack(
                        (cell_to_nodes[cell_type], cn_indices + nodes_offset)
                    )

                elif cell_type == "hexahedron":
                    # NOTE: Optimally, a StructuredGrid would be exported in this case.
                    # However, meshio does solely handle unstructured grids and cannot
                    # for instance write to *.vtr format.

                    # Similar to "tetra", "hexahedron" is a meshio-known cell type.
                    # Thus, the local connectivity is prescribed by cell-nodes
                    # information.

                    # After all, the procedure is fairly similar to the treatment of
                    # tetra cells. Most of the following code is analogous to
                    # _simplex_cell_to_nodes(). However, for hexaedron cells the order
                    # of the nodes is important and has to be adjusted. Based on the
                    # specific definition of TensorGrids however, the node numbering
                    # can be hardcoded, which results in better performance compared
                    # to a modular procedure.

                    # Need to sort the nodes according to the convention within meshio.
                    # Fetch hexahedron cells
                    cells = g_cell_map[cell_type]

                    # Determine cell-node ptr
                    cn_indptr = (
                        g.cell_nodes().indptr[:-1]
                        if cells is None
                        else g.cell_nodes().indptr[cells]
                    )

                    # Collect the indptr to all nodes of the cell;
                    # each n-simplex cell contains n nodes
                    expanded_cn_indptr = np.vstack(
                        [cn_indptr + i for i in range(8)]
                    ).reshape(-1, order="F")

                    # Detect all corresponding nodes by applying the expanded mask to indices
                    expanded_cn_indices = g.cell_nodes().indices[expanded_cn_indptr]

                    # Convert to right format.
                    cn_indices = np.reshape(expanded_cn_indices, (-1, 8), order="C")

                    # Reorder nodes to comply with the node numbering convention of meshio
                    # regarding hexahedron cells.
                    cn_indices = cn_indices[:, [0, 2, 6, 4, 1, 3, 7, 5]]

                    # Initialize data structure if not available yet
                    if cell_type not in cell_to_nodes:
                        cell_to_nodes[cell_type] = np.empty((0, 8), dtype=int)

                    # Store cell-node connectivity, and add offset taking into account
                    # previous grids
                    cell_to_nodes[cell_type] = np.vstack(
                        (cell_to_nodes[cell_type], cn_indices + nodes_offset)
                    )

                else:
                    # Identify remaining cells as polyhedra

                    # The general strategy is to define the connectivity as cell-face
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
                    # The outer list, loops over all cells. Each cell entry contains a
                    # list over faces, and each face entry is given by the face nodes.
                    if cell_type not in cell_to_faces:
                        cell_to_faces[cell_type] = []
                    cell_to_faces[cell_type] += [
                        [
                            fn_indices[fn_indptr[f] : fn_indptr[f + 1]]  # nodes
                            for f in cf_indices[
                                cf_indptr[c] : cf_indptr[c + 1]
                            ]  # faces
                        ]
                        for c in cells  # cells
                    ]

            # Update offset
            nodes_offset += g.num_nodes
            cell_offset += g.num_cells

        # Determine the total number of blocks. Recall that special cell types
        # and general polyhedron{n} cells are stored differently.
        num_special_blocks = len(cell_to_nodes)
        num_polyhedron_blocks = len(cell_to_faces)
        num_blocks = num_special_blocks + num_polyhedron_blocks

        # Initialize the meshio data structure for the connectivity and cell ids.
        meshio_cells = np.empty(num_blocks, dtype=object)
        meshio_cell_id = np.empty(num_blocks, dtype=object)

        # Store cells with special cell types.
        for block_s, (cell_type_s, cell_block_s) in enumerate(cell_to_nodes.items()):
            meshio_cells[block_s] = meshio.CellBlock(
                cell_type_s, cell_block_s.astype(int)
            )
            meshio_cell_id[block_s] = np.array(cell_id[cell_type_s])

        # Store general polyhedra cells.
        for block, (cell_type, cell_block) in enumerate(cell_to_faces.items()):
            # Adapt the block number taking into account of previous cell types.
            block += num_special_blocks
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

            # TODO RM? as implemented now, field.values should never be None.
            if field.values is None:
                continue

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
        # same folder.
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

        The final name is build as combination of a prescribed prefix,
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
