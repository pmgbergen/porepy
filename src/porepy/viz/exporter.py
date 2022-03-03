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
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union

import meshio
import numpy as np
import scipy.sparse as sps

import porepy as pp

# Module-wide logger
logger = logging.getLogger(__name__)

import time

class Field:
    """
    Internal class to store information for the data to export.
    """

    def __init__(self, name: str, values: Optional[np.ndarray] = None) -> None:
        # name of the field
        self.name = name
        self.values = values

    def __repr__(self) -> str:
        """
        Repr function
        """
        return self.name + " - values: " + str(self.values)

    def check(self, values: Optional[np.ndarray], g: pp.Grid) -> None:
        """
        Consistency checks making sure the field self.name is filled and has
        the right dimension.
        """
        if values is None:
            raise ValueError(
                "Field " + str(self.name) + " must be filled. It can not be None"
            )
        if np.atleast_2d(values).shape[1] != g.num_cells:
            raise ValueError("Field " + str(self.name) + " has wrong dimension.")

class Exporter:
    def __init__(
        self,
        gb: Union[pp.Grid, pp.GridBucket],
        file_name: str,
        folder_name: Optional[str] = None,
        **kwargs,
    ) -> None:
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
        binary: export in binary format, default is True.

        How to use:
        If you need to export a single grid:
        save = Exporter(g, "solution", folder_name="results")
        save.write_vtu({"cells_id": cells_id, "pressure": pressure})

        In a time loop:
        save = Exporter(gb, "solution", folder_name="results")
        while time:
            save.write_vtu({"conc": conc}, time_step=i)
        save.write_pvd(steps*deltaT)

        if you need to export the state of variables as stored in the GridBucket:
        save = Exporter(gb, "solution", folder_name="results")
        # export the field stored in data[pp.STATE]["pressure"]
        save.write_vtu(gb, ["pressure"])

        In a time loop:
        while time:
            save.write_vtu(["conc"], time_step=i)
        save.write_pvd(steps*deltaT)

        In the case of different keywords, change the file name with
        "change_name".

        NOTE: the following names are reserved for data exporting: grid_dim,
        is_mortar, mortar_side, cell_id

        """

        # Exporter is operating on grid buckets. If a grid is provided, convert to a grid bucket.
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
        if kwargs:
            msg = "Exporter() got unexpected keyword argument '{}'"
            raise TypeError(msg.format(kwargs.popitem()[0]))

        # Hard code keywords
        self.cell_id_key = "cell_id"

        # Generate infrastructure for storing fixed-dimensional grids in
        # meshio format. Include all but the 0-d grids
        self.dims = np.setdiff1d(self.gb.all_dims(), [0])
        num_dims = self.dims.size
        self.meshio_geom = dict(zip(self.dims, [tuple()] * num_dims))  # type: ignore

        # Generate infrastructure for storing fixed-dimensional mortar grids
        # in meshio format.
        self.m_dims = np.unique([d["mortar_grid"].dim for _, d in self.gb.edges()])
        num_m_dims = self.m_dims.size
        self.m_meshio_geom = dict(zip(self.m_dims, [tuple()] * num_m_dims))  # type: ignore

        # Check if numba is installed
        self._has_numba: bool = "numba" in sys.modules

        # Generate geometrical information in meshio format
        self._update_meshio_geom()

        # Counter for time step. Will be used to identify files of individual time step,
        # unless this is overridden by optional parameters in write
        self._time_step_counter: int = 0
        # Storage for file name extensions for time steps
        self._exported_time_step_file_names: List[int] = []

    def change_name(self, file_name: str) -> None:
        """
        Change the root name of the files, useful when different keywords are
        considered but on the same grid.

        Parameters:
        file_name: the new root name of the files.
        """
        self.file_name = file_name

    # TODO do we need dicts here? do we want to support both for grid buckets (this does not make sense)
    def write_vtu(
        self,
        data: Optional[Union[Dict, List[str]]] = None,
        time_dependent: bool = False,
        time_step: int = None,
        gb: Optional[Union[pp.Grid, pp.GridBucket]] = None,
    ) -> None:
        """
        Interface function to export the grid and additional data with meshio.

        In 1d the cells are represented as lines, 2d the cells as polygon or triangle/quad,
        while in 3d as polyhedra/tetrahedra/hexahedra.
        In all the dimensions the geometry of the mesh needs to be computed.

        Parameters:
        data: if g is a single grid then data is a dictionary (see example)
              if g is a grid bucket then list of names for optional data,
              they are the keys in the grid bucket (see example).
        time_dependent: (bolean, optional) If False, file names will not be appended with
                        an index that markes the time step. Can be overridden by giving
                        a value to time_step.
        time_step: (optional) in a time dependent problem defines the part of the file
                   name associated with this time step. If not provided, subsequent
                   time steps will have file names ending with 0, 1, etc.
        grid: (optional) in case of changing grid set a new one.

        """
        if self.fixed_grid and gb is not None:
            raise ValueError("Inconsistency in exporter setting")
        elif not self.fixed_grid and gb is not None:
            # Convert to grid bucket if a grid is provided.
            if isinstance(gb, pp.Grid):
                self.gb = pp.GridBucket()
                self.gb.add_nodes(gb) 
            else:
                self.gb = gb

            # Update geometrical info in meshio format
            self._update_meshio_geom()

        # If the problem is time dependent, but no time step is set, we set one
        # using the updated, internal counter.
        if time_dependent and time_step is None:
            time_step = self._time_step_counter
            self._time_step_counter += 1

        # If the problem is time dependent (with specified or automatic time step index)
        # add the time step to the exported files
        if time_step is not None:
            self._exported_time_step_file_names.append(time_step)

        # NOTE grid buckets are exported with dict data, while single
        # grids (converted to grid buckets) are exported with list data.
        if isinstance(data, list):
            self._export_list_data(data, time_step) # type: ignore
        elif isinstance(data, dict):
            self._export_dict_data(data, time_step)
        else:
            raise NotImplemented("No other data type than list and dict supported.")

    # TODO does not contain all keywords as in write_vtu. make consistent?
    def write_pvd(
        self,
        timestep: np.ndarray,
        file_extension: Optional[Union[np.ndarray, List[int]]] = None,
    ) -> None:
        """
        Interface function to export in PVD file the time loop information.
        The user should open only this file in paraview.

        We assume that the VTU associated files have the same name.
        We assume that the VTU associated files are in the same folder.

        Parameters:
        timestep: numpy of times to be exported. These will be the time associated with
            indivdiual time steps in, say, Paraview. By default, the times will be
            associated with the order in which the time steps were exported. This can
            be overridden by the file_extension argument.
        file_extension (np.array-like, optional): End of file names used in the export
            of individual time steps, see self.write_vtu(). If provided, it should have
            the same length as time. If not provided, the file names will be picked
            from those used when writing individual time steps.

        """
        if file_extension is None:
            file_extension = self._exported_time_step_file_names
        elif isinstance(file_extension, np.ndarray):
            file_extension = file_extension.tolist()

        assert file_extension is not None  # make mypy happy

        o_file = open(self._make_folder(self.folder_name, self.file_name) + ".pvd", "w")
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

        for time, fn in zip(timestep, file_extension):
            for dim in self.dims:
                o_file.write(
                    fm % (time, self._make_file_name(self.file_name, fn, dim))
                )

        o_file.write("</Collection>\n" + "</VTKFile>")
        o_file.close()

    def _export_dict_data(self, data: Dict[str, np.ndarray], time_step):
        """
        Export single grid to a vtu file with dictionary data.

        Parameters:
            data (Dict[str, np.ndarray]): Data to be exported.
            time_step (float) : Time step, to be appended at the vtu output file.
        """
        # Currently this method does only support single grids.
        if self.dims.shape[0] > 1:
            raise NotImplementedError("Grid buckets with more than one grid are not supported.")

        # No need of special naming, create the folder
        name = self._make_folder(self.folder_name, self.file_name)
        name = self._make_file_name(name, time_step)

        # Provide an empty dict if data is None
        if data is None:
            data = dict()

        # Store the provided data as list of Fields
        fields: List[Field] = []
        if len(data) > 0:
            fields.extend([Field(n, v) for n, v in data.items()])

        # Extract grid (it is assumed it is the only one)
        grid = [g for g, _ in self.gb.nodes()][0]

        # Add grid dimension to the data
        grid_dim = grid.dim * np.ones(grid.num_cells, dtype=int)
        fields.extend([Field("grid_dim", grid_dim)])

        # Write data to file
        self._write(fields, name, self.meshio_geom[grid.dim])

    def _export_list_data(self, data: List[str], time_step: float) -> None:
        """Export the entire GridBucket and additional list data to vtu.

        Parameters:
            data (List[str]): Data to be exported in addition to default GridBucket data.
            time_step (float) : Time step, to be appended at the vtu output file.
        """
        # Convert data to list, or provide an empty list
        if data is not None:
            data = np.atleast_1d(data).tolist()
        else:
            data = list()

        # Extract data which is contained in nodes (and not edges).
        # IMPLEMENTATION NOTE: We need a unique set of keywords for node_data. The simpler
        # option would have been to  gather all keys and uniquify by converting to a set,
        # and then back to a list. However, this will make the ordering of the keys random,
        # and it turned out that this complicates testing (see tests/unit/test_vtk).
        # It was therefore considered better to use a more complex loop which
        # (seems to) guarantee a deterministic ordering of the keys.
        node_data = list()
        # For each element in data, apply a brute force approach and check whether there
        # exists a data dictionary associated to a node which contains a state variable
        # with same key. If so, add the key and move on to the next key.
        for key in data:
            for _, d in self.gb.nodes():
                if pp.STATE in d and key in d[pp.STATE]:
                    node_data.append(key)
                    # After successfully identifying data contained in nodes, break the loop
                    # over nodes to avoid any unintended repeated listing of key.
                    break

        # Transfer data to fields.
        node_fields: List[Field] = []
        if len(node_data) > 0:
            node_fields.extend([Field(d) for d in node_data])

        # consider the grid_bucket node data
        extra_node_names = ["grid_dim", "grid_node_number", "is_mortar", "mortar_side"]
        extra_node_fields = [Field(name) for name in extra_node_names]
        node_fields.extend(extra_node_fields)

        self.gb.assign_node_ordering(overwrite_existing=False)
        self.gb.add_node_props(extra_node_names)
        # fill the extra data
        for g, d in self.gb:
            ones = np.ones(g.num_cells, dtype=int)
            d["grid_dim"] = g.dim * ones
            d["grid_node_number"] = d["node_number"] * ones
            d["is_mortar"] = 0 * ones
            d["mortar_side"] = pp.grids.mortar_grid.MortarSides.NONE_SIDE.value * ones

        # collect the data and extra data in a single stack for each dimension
        for dim in self.dims:
            file_name = self._make_file_name(self.file_name, time_step, dim)
            file_name = self._make_folder(self.folder_name, file_name)
            for field in node_fields:
                grids = self.gb.get_grids(lambda g: g.dim == dim)
                values = []
                for g in grids:
                    if field.name in data:
                        values.append(self.gb.node_props(g, pp.STATE)[field.name])
                    else:
                        values.append(self.gb.node_props(g, field.name))
                    field.check(values[-1], g)
                field.values = np.hstack(values)

            if self.meshio_geom[dim] is not None:
                self._write(node_fields, file_name, self.meshio_geom[dim])

        self.gb.remove_node_props(extra_node_names)

        # Extract data which is contained in edges (and not nodes).
        # IMPLEMENTATION NOTE: See the above loop to construct node_data for an explanation
        # of this elaborate construction of `edge_data`
        edge_data = list()
        for key in data:
            for _, d in self.gb.edges():
                if pp.STATE in d and key in d[pp.STATE]:
                    edge_data.append(key)
                    # After successfully identifying data contained in edges, break the loop
                    # over edges to avoid any unintended repeated listing of key.
                    break

        # Transfer data to fields.
        edge_fields: List[Field] = []
        if len(edge_data) > 0:
            edge_fields.extend([Field(d) for d in edge_data])

        # consider the grid_bucket edge data
        extra_edge_names = ["grid_dim", "grid_edge_number", "is_mortar", "mortar_side"]
        extra_edge_fields = [Field(name) for name in extra_edge_names]
        edge_fields.extend(extra_edge_fields)

        self.gb.add_edge_props(extra_edge_names)
        # fill the extra data
        for _, d in self.gb.edges():
            d["grid_dim"] = {}
            d["cell_id"] = {}
            d["grid_edge_number"] = {}
            d["is_mortar"] = {}
            d["mortar_side"] = {}
            mg = d["mortar_grid"]
            mg_num_cells = 0
            for side, g in mg.side_grids.items():
                ones = np.ones(g.num_cells, dtype=int)
                d["grid_dim"][side] = g.dim * ones
                d["is_mortar"][side] = ones
                d["mortar_side"][side] = side.value * ones
                d["cell_id"][side] = np.arange(g.num_cells, dtype=int) + mg_num_cells
                mg_num_cells += g.num_cells
                d["grid_edge_number"][side] = d["edge_number"] * ones

        # collect the data and extra data in a single stack for each dimension
        for dim in self.m_dims:
            file_name = self._make_file_name_mortar(
                self.file_name, time_step=time_step, dim=dim
            )
            file_name = self._make_folder(self.folder_name, file_name)

            mgs = self.gb.get_mortar_grids(lambda g: g.dim == dim)
            cond = lambda d: d["mortar_grid"].dim == dim
            edges: List[Tuple[pp.Grid, pp.Grid]] = [
                e for e, d in self.gb.edges() if cond(d)
            ]

            for field in edge_fields:
                values = []
                for mg, edge in zip(mgs, edges):
                    if field.name in data:
                        values.append(self.gb.edge_props(edge, pp.STATE)[field.name])
                    else:
                        for side, _ in mg.side_grids.items():
                            # Convert edge to tuple to be compatible with GridBucket
                            # data structure
                            values.append(self.gb.edge_props(edge, field.name)[side])

                field.values = np.hstack(values)

            if self.m_meshio_geom[dim] is not None:
                self._write(edge_fields, file_name, self.m_meshio_geom[dim])

        file_name = self._make_file_name(self.file_name, time_step, extension=".pvd")
        file_name = self._make_folder(self.folder_name, file_name)
        self._export_pvd_gb(file_name, time_step)

        self.gb.remove_edge_props(extra_edge_names)

    def _export_pvd_gb(self, file_name: str, time_step: float) -> None:
        o_file = open(file_name, "w")
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

        # Include the grids and mortar grids of all dimensions, but only if the
        # grids (or mortar grids) of this dimension are included in the vtk export
        for dim in self.dims:
            if self.meshio_geom[dim] is not None:
                o_file.write(
                    fm % self._make_file_name(self.file_name, time_step, dim=dim)
                )
        for dim in self.m_dims:
            if self.m_meshio_geom[dim] is not None:
                o_file.write(
                    fm % self._make_file_name_mortar(self.file_name, dim, time_step)
                )

        o_file.write("</Collection>\n" + "</VTKFile>")
        o_file.close()

    def _export_grid(
        self, gs: Iterable[pp.Grid], dim: int
    ) -> Union[None, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Wrapper function to export grids of dimension dim. Calls the
        appropriate dimension specific export function.
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

    def _export_1d(
        self, gs: Iterable[pp.Grid]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Export the geometrical data (point coordinates) and connectivity
        information from the 1d PorePy grids to meshio.
        """

        # in 1d we have only one cell type
        cell_type = "line"

        # cell connectivity information
        num_cells = np.sum(np.array([g.num_cells for g in gs]))
        cell_to_nodes = {cell_type: np.empty((num_cells, 2))}  # type: ignore
        # cell id map
        cell_id = {cell_type: np.empty(num_cells, dtype=int)}  # type: ignore
        cell_offset = 0

        # points
        num_pts = np.sum([g.num_nodes for g in gs])
        meshio_pts = np.empty((num_pts, 3))  # type: ignore
        nodes_offset = 0

        # loop on all the 1d grids
        for g in gs:
            # save the points information
            sl = slice(nodes_offset, nodes_offset + g.num_nodes)
            meshio_pts[sl, :] = g.nodes.T

            # TODO vectorize, and use directly indices and indptr without sorting
            # Cell-node relations
            g_cell_nodes = g.cell_nodes()
            g_nodes_cells, g_cells, _ = sps.find(g_cell_nodes)
            # Ensure ordering of the cells
            g_nodes_cells = g_nodes_cells[np.argsort(g_cells)]

            # loop on all the grid cells
            for c in np.arange(g.num_cells):
                loc = slice(g_cell_nodes.indptr[c], g_cell_nodes.indptr[c + 1])
                # get the local nodes and save them
                cell_to_nodes[cell_type][cell_offset, :] = g_nodes_cells[loc] + nodes_offset
                cell_id[cell_type][cell_offset] = cell_offset
                cell_offset += 1

            nodes_offset += g.num_nodes

        # construct the meshio data structure
        num_blocks = len(cell_to_nodes)
        meshio_cells = np.empty(num_blocks, dtype=object)
        meshio_cell_id = np.empty(num_blocks, dtype=object)

        for block, (cell_type, cell_block) in enumerate(cell_to_nodes.items()):
            meshio_cells[block] = meshio.CellBlock(cell_type, cell_block.astype(int))
            meshio_cell_id[block] = np.array(cell_id[cell_type])

        return meshio_pts, meshio_cells, meshio_cell_id

    def _export_2d(
        self, gs: Iterable[pp.Grid]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Export the geometrical data (point coordinates) and connectivity
        information from the 2d PorePy grids to meshio.

        Parameters:
            gs (Iterable[pp.Grid]): Subdomains.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Points, 2d cells (storing the
                connectivity), and cell ids in correct meshio format.
        """

        # Use standard names for simple object types: in this routine only triangle
        # and quad cells are treated in a special manner.
        polygon_map  = {"polygon3": "triangle", "polygon4": "quad"}

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
                    cell_to_nodes[cell_type] = np.empty((0,n), dtype=int)

                # Special case: Triangle cells, i.e., n=3.
                if cell_type == "triangle":

                    # Triangles have a trivial connectivity since all nodes are connected.

                    # Fetch triangle cells
                    cells = g_cell_map[cell_type]
                    # Extract indptr for triangle cells. Here, we in general loose information about
                    # the next cell, i.e., the end of the indices corresponding to the selected cells.
                    cn_indptr = g.cell_nodes().indptr[cells]
                    # Expand indptr, explicitly specifying the location of all nodes for each triangle cell.
                    # Integrate the explicit knowledge that each triangle cell consists of 3 nodes.
                    expanded_cn_indptr = np.vstack([cn_indptr + i for i in range(n)]).reshape(-1, order="F")
                    # Detect all corresponding nodes by applying the expanded mask to the indices
                    expanded_cn_indices = g.cell_nodes().indices[expanded_cn_indptr]
                    # Bring to number of triangle cells x 3 format by reshaping
                    cn = np.reshape(expanded_cn_indices, (-1,n), order="C")

                # Quad and polygon cells.
                else:
                    # For quads/polygons, g.cell_nodes cannot be blindly used as for triangles, since the
                    # ordering of the nodes may define a cell of the type (here specific for quads)
                    # x--x and not x--x
                    #  \/          |  |
                    #  /\          |  |
                    # x--x         x--x .
                    # Therefore, use both g.cell_faces and g.face_nodes to make use of face information
                    # and sort those correctly to retrieve the correct connectivity.,

                    # Strategy: Collect all cell nodes including their connectivity in a matrix of
                    # double num cell size. The goal will be to gather starting and end points for
                    # all faces of each cell, sort those faces, such that they form a circlular graph,
                    # and then choose the resulting starting points of all faces to define the connectivity.

                    # Fetch corresponding cells
                    cells = g_cell_map[cell_type]
                    # Determine all faces of all cells. Use an analogous approach as used to determine
                    # all cell nodes for triangle cells. And use that a polygon with n nodes has also
                    # n faces.
                    cf_indptr = g.cell_faces.indptr[cells]
                    expanded_cf_indptr = np.vstack([cf_indptr + i for i in range(n)]).reshape(-1, order="F")
                    cf_indices = g.cell_faces.indices[expanded_cf_indptr]

                    # Determine the associated (two) nodes of all faces for each cell.
                    fn_indptr = g.face_nodes.indptr[cf_indices]
                    # Extract nodes for first and second node of each face; reshape such
                    # that all first nodes of all faces for each cell are stored in one row,
                    # i.e., end up with an array of size num_cells (for this cell type) x n.
                    cfn_indices = [g.face_nodes.indices[fn_indptr + i].reshape(-1,n) for i in range(2)]
                    # Group first and second nodes, with alternating order of rows.
                    # By this, each cell is respresented by two rows. The first and second
                    # rows contain first and second nodes of faces. And each column stands
                    # for one face.
                    cfn = np.ravel(cfn_indices, order="F").reshape(n,-1).T

                    # Sort faces for each cell such that they form a chain. Use a function
                    # compiled with Numba. Currently, no alternative is provided when Numba
                    # is not installed. This step is the bottleneck of this routine.
                    if not self._has_numba:
                        raise NotImplementedError("The sorting algorithm requires numba to be installed.")
                    cfn = self._sort_point_pairs_numba(cfn).astype(int)

                    # For each cell pick the sorted nodes such that they form a chain and thereby
                    # define the connectivity, i.e., skip every second row.
                    cn = cfn[::2,:]

                # Add offset to account for previous grids, and store
                cell_to_nodes[cell_type] = np.vstack(
                    (cell_to_nodes[cell_type], cn + nodes_offset)
                )

            # Update offset
            nodes_offset += g.num_nodes
            cell_offset += g.num_cells

        # Construct the meshio data structure
        num_blocks = len(cell_to_nodes)
        meshio_cells = np.empty(num_blocks, dtype=object)
        meshio_cell_id = np.empty(num_blocks, dtype=object)

        # For each cell_type store the connectivity pattern cell_to_nodes for
        # the corresponding cells with ids from cell_id.
        for block, (cell_type, cell_block) in enumerate(cell_to_nodes.items()):
            # Meshio does not accept polygon{n} cell types. Remove {n} if
            # cell type not special geometric object.
            cell_type_meshio_format = polygon_map.get(f"polygon{n}", "polygon")
            meshio_cells[block] = meshio.CellBlock(
                cell_type_meshio_format,
                cell_block.astype(int)
            )
            meshio_cell_id[block] = np.array(cell_id[cell_type])

        # Return final meshio data: points, cell (connectivity), cell ids
        return meshio_pts, meshio_cells, meshio_cell_id

    def _sort_point_pairs_numba(self, lines):
        """This a simplified copy of pp.utils.sort_points.sort_point_pairs.

        Essentially the same functionality as sort_point_pairs, but stripped down
        to the special case of circular chains. Different to sort_point_pairs, this
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
            """Copy of pp.utils.sort_points.sort_point_pairs. This version is extended
            to multiple chains. Each chain is implicitly assumed to be circular."""

            # Retrieve number of chains and lines per chain from the shape.
            # Implicitly expect that all chains have the same length
            num_chains, chain_length = lines.shape
            # Since for each chain lines includes two rows, take the half
            num_chains = int(num_chains/2)

            # Initialize array of sorted lines to be the final output
            sorted_lines = np.zeros((2*num_chains,chain_length))
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
    
                # Define the end point of the previous and starting point for the next line segment
                prev = sorted_lines[2*c+1, 0]
    
                # The sorting algorithm: Loop over all position in the chain to be set next.
                # Find the right candidate to be moved to this position and possibly flipped
                # if needed. A candidate is identified as fitting if it contains one point
                # equal to the current starting point. This algorithm uses a double loop,
                # which is the most naive approach. However, assume chain_length is in
                # general small.
                for i in range(1, chain_length):  # The first line has already been found
                    for j in range(1, chain_length): # The first line has already been found
                        # A candidate line segment with matching start and end point
                        # in the first component of the point pair.
                        if np.abs(found[j]) < 1e-6 and lines[2*c, j] == prev:
                            # Copy the segment to the right place
                            sorted_lines[2*c:2*c+2, i] = lines[2*c:2*c+2, j]
                            # Mark as used
                            found[j] = 1
                            # Define the starting point for the next line segment
                            prev = lines[2*c+1, j]
                            break
                        # A candidate line segment with matching start and end point
                        # in the second component of the point pair.
                        elif np.abs(found[j])<1e-6 and lines[2*c+1, j] == prev:
                            # Flip and copy the segment to the right place
                            sorted_lines[2*c, i] = lines[2*c+1, j]
                            sorted_lines[2*c+1, i] = lines[2*c, j]
                            # Mark as used
                            found[j] = 1
                            # Define the starting point for the next line segment
                            prev = lines[2*c, j]
                            break
   
            # Return the sorted lines defining chains.
            return sorted_lines

        # Run numba compiled function
        return _function_to_compile(lines)

    def _export_3d(
        self, gs: Iterable[pp.Grid]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Export the geometrical data (point coordinates) and connectivity
        information from the 3d PorePy grids to meshio.

        Parameters:
            gs (Iterable[pp.Grid]): Subdomains.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Points, 3d cells (storing the
                connectivity), and cell ids in correct meshio format.
        """

        # Use standard name for simple object types: in this routine only tetra cells
        # are treated in a special manner.
        polyhedron_map  = {"polyhedron4": "tetra"}

        # Dictionaries storing cell->faces and cell->nodes connectivity information
        # for all cell types. For this, the nodes have to be sorted such that
        # they form a circular chain, describing the boundary of the cell.
        cell_to_faces: Dict[str, List[List[int]]] = {}
        cell_to_nodes: Dict[str, np.ndarray] = {}
        # Dictionary collecting all cell ids for each cell type.
        cell_id: Dict[str, List[int]] = {}

        # Data structure for storing node coordinates of all 2d grids.
        num_pts = np.sum([g.num_nodes for g in gs])
        meshio_pts = np.empty((num_pts, 3))  # type: ignore

        # Initialize offsets. Required taking into account multiple 2d grids.
        nodes_offset = 0
        cell_offset = 0

        # Loop over all 3d grids
        for g in gs:

            # Store node coordinates
            sl = slice(nodes_offset, nodes_offset + g.num_nodes)
            meshio_pts[sl, :] = g.nodes.T

            # Determine cell types based on number of nodes per cell.
            num_faces_per_cell = g.cell_faces.getnnz(axis=0)

            # Loop over all available cell types and group cells of one type.
            g_cell_map = dict()
            for n in np.unique(num_faces_per_cell):
                # Define cell type; check if it coincides with a predefined cell type
                cell_type = polyhedron_map.get(f"polyhedron{n}", f"polyhedron{n}")
                # Find all cells with n faces, and store for later use
                cells = np.nonzero(num_faces_per_cell == n)[0]
                g_cell_map[cell_type] = cells
                # Store cell ids in global container; init if entry not yet established
                if cell_type not in cell_id:
                    cell_id[cell_type] = []
                # Add offset taking into account previous grids
                cell_id[cell_type] += (cells + cell_offset).tolist()

            # Determine connectivity. Loop over available cell types, and treat tetrahedra
            # and general polyhedra differently aiming for optimized performance. 

            for n in np.unique(num_faces_per_cell):

                # Define the cell type
                cell_type = polyhedron_map.get(f"polyhedron{n}", f"polyhedron{n}")

                # Special case: Tetrahedra
                if cell_type == "tetra":
                    # Since tetra is a meshio-known cell type, cell_to_nodes connectivity information
                    # is provided, i.e., for each cell the nodes in the meshio-defined local numbering
                    # of nodes is generated. Since tetra is a simplex, the nodes do not need to be
                    # ordered in any specific type, as the geometrical object is invariant under
                    # permutations.

                    # Fetch tetra cells
                    cells = g_cell_map[cell_type]
                    # Extract indptr for all tetra cells.
                    cn_indptr = g.cell_nodes().indptr[cells]
                    # Expand indptr, explicitly specifying the location of all nodes for each tetra cell.
                    # Integrate the explicit knowledge that each tetra cell consists of 4 nodes.
                    expanded_cn_indptr = np.vstack([cn_indptr + i for i in range(4)]).reshape(-1, order="F")
                    # Detect all corresponding nodes by applying the expanded mask to the indices
                    cn_indices = g.cell_nodes().indices[expanded_cn_indptr]
                    # Bring to number of num tetra cells x 4 format by reshaping
                    cn_indices = np.reshape(cn_indices, (-1,4), order="C")
                    # Initialize data structure if not available yet
                    if cell_type not in cell_to_nodes:
                        cell_to_nodes[cell_type] = np.empty((0,4), dtype=int)
                    # Store cell-node connectivity, and add offset taking into account previous grids
                    cell_to_nodes[cell_type] = np.vstack(
                        (cell_to_nodes[cell_type], cn_indices + nodes_offset)
                    )

                # Identify remaining cells as polyhedra
                else:
                    # The general strategy is to define the connectivity as cell-face information,
                    # where the faces are defined by nodes. Hence, this information is significantly
                    # larger than the info provided for tetra cells. Here, we make use of the fact
                    # that g.face_nodes provides nodes ordered wrt. the right-hand rule.
                    
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
                            fn_indices[fn_indptr[f] : fn_indptr[f+1]]           # nodes
                            for f in cf_indices[cf_indptr[c]:cf_indptr[c+1]]    # faces
                        ] for c in cells                                        # cells
                    ]

            # Update offset
            nodes_offset += g.num_nodes
            cell_offset += g.num_cells

        # Determine the total number of blocks. Recall that tetra and general
        # polyhedron{n} cells are stored differently.
        num_tetra_blocks = len(cell_to_nodes)
        num_polyhedron_blocks = len(cell_to_faces)
        num_blocks = num_tetra_blocks + num_polyhedron_blocks

        # Initialize the meshio data structure for the connectivity and cell ids.
        meshio_cells = np.empty(num_blocks, dtype=object)
        meshio_cell_id = np.empty(num_blocks, dtype=object)

        # Store tetra cells. Use cell_to_nodes to store the connectivity info.
        for block, (cell_type, cell_block) in enumerate(cell_to_nodes.items()):
            meshio_cells[block] = meshio.CellBlock(cell_type, cell_block.astype(int))
            meshio_cell_id[block] = np.array(cell_id[cell_type])

        # Store tetra cells first. Use the more general cell_to_faces to store
        # the connectivity info.
        for block, (cell_type, cell_block) in enumerate(cell_to_faces.items()):
            # Adapt the block number taking into account of previous cell types.
            block += num_tetra_blocks
            meshio_cells[block] = meshio.CellBlock(cell_type, cell_block)
            meshio_cell_id[block] = np.array(cell_id[cell_type])

        # Return final meshio data: points, cell (connectivity), cell ids
        return meshio_pts, meshio_cells, meshio_cell_id

    def _write(self, fields: Iterable[Field], file_name: str, meshio_geom) -> None:

        cell_data = {}

        # we need to split the data for each group of geometrically uniform cells
        cell_id = meshio_geom[2]
        num_block = cell_id.size

        for field in fields:
            if field.values is None:
                continue

            # for each field create a sub-vector for each geometrically uniform group of cells
            cell_data[field.name] = np.empty(num_block, dtype=object)
            # fill up the data
            for block, ids in enumerate(cell_id):
                if field.values.ndim == 1:
                    cell_data[field.name][block] = field.values[ids]
                elif field.values.ndim == 2:
                    cell_data[field.name][block] = field.values[:, ids].T
                else:
                    raise ValueError

        # add also the cells ids
        cell_data.update({self.cell_id_key: cell_id})

        # create the meshio object
        meshio_grid_to_export = meshio.Mesh(
            meshio_geom[0], meshio_geom[1], cell_data=cell_data
        )
        meshio.write(file_name, meshio_grid_to_export, binary=self.binary)

    def _update_meshio_geom(self):
        for dim in self.dims:
            g = self.gb.get_grids(lambda g: g.dim == dim)
            self.meshio_geom[dim] = self._export_grid(g, dim)

        for dim in self.m_dims:
            # extract the mortar grids for dimension dim
            mgs = self.gb.get_mortar_grids(lambda g: g.dim == dim)
            # it contains the mortar grids "unrolled" by sides
            mg = np.array([g for m in mgs for _, g in m.side_grids.items()])
            self.m_meshio_geom[dim] = self._export_grid(mg, dim)

    def _make_folder(self, folder_name: Optional[str] = None, name: str = "") -> str:
        if folder_name is None:
            return name

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        return os.path.join(folder_name, name)

    def _make_file_name(
        self,
        file_name: str,
        time_step: Optional[float] = None,
        dim: Optional[int] = None,
        extension: str = ".vtu",
    ) -> str:

        padding = 6
        if dim is None:  # normal grid
            if time_step is None:
                return file_name + extension
            else:
                time = str(time_step).zfill(padding)
                return file_name + "_" + time + extension
        else:  # part of a grid bucket
            if time_step is None:
                return file_name + "_" + str(dim) + extension
            else:
                time = str(time_step).zfill(padding)
                return file_name + "_" + str(dim) + "_" + time + extension

    def _make_file_name_mortar(
        self, file_name: str, dim: int, time_step: Optional[float] = None
    ) -> str:

        # we keep the order as in _make_file_name
        assert dim is not None

        extension = ".vtu"
        file_name = file_name + "_mortar_"
        padding = 6
        if time_step is None:
            return file_name + str(dim) + extension
        else:
            time = str(time_step).zfill(padding)
            return file_name + str(dim) + "_" + time + extension
