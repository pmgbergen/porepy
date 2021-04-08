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

module_sections = ["visualization"]
# ------------------------------------------------------------------------------#


class Field:
    """
    Internal class to store information for the data to export.
    """

    @pp.time_logger(sections=module_sections)
    def __init__(self, name: str, values: Optional[np.ndarray] = None) -> None:
        # name of the field
        self.name = name
        self.values = values

    @pp.time_logger(sections=module_sections)
    def __repr__(self) -> str:
        """
        Repr function
        """
        return self.name + " - values: " + str(self.values)

    @pp.time_logger(sections=module_sections)
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

    @pp.time_logger(sections=module_sections)
    def _check_values(self) -> None:
        if self.values is None:
            raise ValueError("Field " + str(self.name) + " values not valid")


# ------------------------------------------------------------------------------#


class Fields:
    """
    Internal class to store a list of field.
    """

    @pp.time_logger(sections=module_sections)
    def __init__(self) -> None:
        self._fields: List[Field] = []

    @pp.time_logger(sections=module_sections)
    def __iter__(self) -> Generator[Field, None, None]:
        """
        Iterator on all the fields.
        """
        for f in self._fields:
            yield f

    @pp.time_logger(sections=module_sections)
    def __repr__(self) -> str:
        """
        Repr function
        """
        return "\n".join([repr(f) for f in self])

    @pp.time_logger(sections=module_sections)
    def extend(self, new_fields: List[Field]) -> None:
        """
        Extend the list of fields with additional fields.
        """
        if isinstance(new_fields, list):
            self._fields.extend(new_fields)
        else:
            raise ValueError

    @pp.time_logger(sections=module_sections)
    def names(self) -> List[str]:
        """
        Return the list of name of the fields.
        """
        return [f.name for f in self]


# ------------------------------------------------------------------------------#


class Exporter:
    @pp.time_logger(sections=module_sections)
    def __init__(
        self,
        grid: Union[pp.Grid, pp.GridBucket],
        file_name: str,
        folder_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Class for exporting data to vtu files.

        Parameters:
        grid: the grid or grid bucket
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

        if isinstance(grid, pp.GridBucket):
            self.is_GridBucket = True
            self.gb: pp.GridBucket = grid

        else:
            self.is_GridBucket = False
            self.grid: pp.Grid = grid  # type: ignore

        self.file_name = file_name
        self.folder_name = folder_name
        self.fixed_grid: bool = kwargs.pop("fixed_grid", True)
        self.binary: bool = kwargs.pop("binary", True)
        if kwargs:
            msg = "Exporter() got unexpected keyword argument '{}'"
            raise TypeError(msg.format(kwargs.popitem()[0]))

        self.cell_id_key = "cell_id"

        if self.is_GridBucket:
            # Fixed-dimensional grids to be included in the export. We include
            # all but the 0-d grids
            self.dims = np.setdiff1d(self.gb.all_dims(), [0])
            num_dims = self.dims.size
            self.meshio_geom = dict(zip(self.dims, [tuple()] * num_dims))  # type: ignore

            # mortar grid variables
            self.m_dims = np.unique([d["mortar_grid"].dim for _, d in self.gb.edges()])
            num_m_dims = self.m_dims.size
            self.m_meshio_geom = dict(zip(self.m_dims, [tuple()] * num_m_dims))  # type: ignore
        else:
            self.meshio_geom = tuple()  # type: ignore

        # Assume numba is available
        self.has_numba: bool = True

        self._update_meshio_geom()

        # Counter for time step. Will be used to identify files of individual time step,
        # unless this is overridden by optional parameters in write
        self._time_step_counter: int = 0
        # Storage for file name extensions for time steps
        self._exported_time_step_file_names: List[int] = []

    # ------------------------------------------------------------------------------#

    @pp.time_logger(sections=module_sections)
    def change_name(self, file_name: str) -> None:
        """
        Change the root name of the files, useful when different keywords are
        considered but on the same grid.

        Parameters:
        file_name: the new root name of the files.
        """
        self.file_name = file_name

    # ------------------------------------------------------------------------------#

    @pp.time_logger(sections=module_sections)
    def write_vtu(
        self,
        data: Optional[Union[Dict, List[str]]] = None,
        time_dependent: bool = False,
        time_step: int = None,
        grid: Optional[Union[pp.Grid, pp.GridBucket]] = None,
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
        if self.fixed_grid and grid is not None:
            raise ValueError("Inconsistency in exporter setting")
        elif not self.fixed_grid and grid is not None:
            if self.is_GridBucket:
                self.gb = grid  # type: ignore
            else:
                self.grid = grid  # type: ignore

            self._update_meshio_geom()

        # If the problem is time dependent, but no time step is set, we set one
        if time_dependent and time_step is None:
            time_step = self._time_step_counter
            self._time_step_counter += 1

        # If the problem is time dependent (with specified or automatic time step index)
        # add the time step to the exported files
        if time_step is not None:
            self._exported_time_step_file_names.append(time_step)

        if self.is_GridBucket:
            self._export_gb(data, time_step)  # type: ignore
        else:
            self._export_g(data, time_step)  # type: ignore

    # ------------------------------------------------------------------------------#

    @pp.time_logger(sections=module_sections)
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

        if self.is_GridBucket:
            for time, fn in zip(timestep, file_extension):
                for dim in self.dims:
                    o_file.write(
                        fm % (time, self._make_file_name(self.file_name, fn, dim))
                    )
        else:
            for time, fn in zip(timestep, file_extension):
                o_file.write(fm % (time, self._make_file_name(self.file_name, fn)))

        o_file.write("</Collection>\n" + "</VTKFile>")
        o_file.close()

    # ------------------------------------------------------------------------------#

    @pp.time_logger(sections=module_sections)
    def _export_g(self, data: Dict[str, np.ndarray], time_step):
        """
        Export a single grid (not grid bucket) to a vtu file.
        """
        # No need of special naming, create the folder
        name = self._make_folder(self.folder_name, self.file_name)
        name = self._make_file_name(name, time_step)

        # Provide an empty dict if data is None
        if data is None:
            data = dict()

        fields = Fields()
        if len(data) > 0:
            fields.extend([Field(n, v) for n, v in data.items()])

        grid_dim = self.grid.dim * np.ones(self.grid.num_cells, dtype=int)

        fields.extend(
            [
                Field(
                    "grid_dim",
                    grid_dim,
                )
            ]
        )

        self._write(fields, name, self.meshio_geom)

    # ------------------------------------------------------------------------------#

    @pp.time_logger(sections=module_sections)
    def _export_gb(self, data: List[str], time_step: float):
        # Convert data to list, or provide an empty list
        if data is not None:
            data = np.atleast_1d(data).tolist()
        else:
            data = list()

        fields = Fields()
        if len(data) > 0:
            fields.extend([Field(d) for d in data])

        # consider the grid_bucket node data
        extra_node_names = ["grid_dim", "grid_node_number", "is_mortar", "mortar_side"]
        extra_node_fields = [Field(name) for name in extra_node_names]
        fields.extend(extra_node_fields)

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
            for field in fields:
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
                self._write(fields, file_name, self.meshio_geom[dim])

        self.gb.remove_node_props(extra_node_names)

        # consider the grid_bucket edge data
        extra_edge_fields = Fields()
        extra_edge_names = ["grid_dim", "grid_edge_number", "is_mortar", "mortar_side"]
        extra_edge_fields.extend([Field(name) for name in extra_edge_names])
        self.gb.add_edge_props(extra_edge_names)
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

            for field in extra_edge_fields:
                values = []
                for mg, edge in zip(mgs, edges):
                    for side, _ in mg.side_grids.items():
                        # Convert edge to tuple to be compatible with GridBucket
                        # data structure
                        values.append(self.gb.edge_props(edge, field.name)[side])

                field.values = np.hstack(values)

            if self.m_meshio_geom[dim] is not None:
                self._write(extra_edge_fields, file_name, self.m_meshio_geom[dim])

        file_name = self._make_file_name(self.file_name, time_step, extension=".pvd")
        file_name = self._make_folder(self.folder_name, file_name)
        self._export_pvd_gb(file_name, time_step)

        self.gb.remove_edge_props(extra_edge_names)

    # ------------------------------------------------------------------------------#

    @pp.time_logger(sections=module_sections)
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

    # ------------------------------------------------------------------------------#

    @pp.time_logger(sections=module_sections)
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

    # ------------------------------------------------------------------------------#

    @pp.time_logger(sections=module_sections)
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
        cell_pos = 0

        # points
        num_pts = np.sum([g.num_nodes for g in gs])
        meshio_pts = np.empty((num_pts, 3))  # type: ignore
        pts_pos = 0

        # loop on all the 1d grids
        for g in gs:
            # save the points information
            sl = slice(pts_pos, pts_pos + g.num_nodes)
            meshio_pts[sl, :] = g.nodes.T

            # Cell-node relations
            g_cell_nodes = g.cell_nodes()
            g_nodes_cells, g_cells, _ = sps.find(g_cell_nodes)
            # Ensure ordering of the cells
            g_nodes_cells = g_nodes_cells[np.argsort(g_cells)]

            # loop on all the grid cells
            for c in np.arange(g.num_cells):
                loc = slice(g_cell_nodes.indptr[c], g_cell_nodes.indptr[c + 1])
                # get the local nodes and save them
                cell_to_nodes[cell_type][cell_pos, :] = g_nodes_cells[loc] + pts_pos
                cell_id[cell_type][cell_pos] = cell_pos
                cell_pos += 1

            pts_pos += g.num_nodes

        # construct the meshio data structure
        num_block = len(cell_to_nodes)
        meshio_cells = np.empty(num_block, dtype=object)
        meshio_cell_id = np.empty(num_block, dtype=object)

        for block, (cell_type, cell_block) in enumerate(cell_to_nodes.items()):
            meshio_cells[block] = meshio.CellBlock(cell_type, cell_block.astype(int))
            meshio_cell_id[block] = np.array(cell_id[cell_type])

        return meshio_pts, meshio_cells, meshio_cell_id

    # ------------------------------------------------------------------------------#

    @pp.time_logger(sections=module_sections)
    def _export_2d(
        self, gs: Iterable[pp.Grid]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Export the geometrical data (point coordinates) and connectivity
        information from the 2d PorePy grids to meshio.
        """

        # use standard name for simple object type
        polygon_map = {"polygon3": "triangle", "polygon4": "quad"}

        # cell->nodes connectivity information
        cell_to_nodes: Dict[str, np.ndarray] = {}
        # cell id map
        cell_id: Dict[str, List[int]] = {}

        # points
        num_pts = np.sum([g.num_nodes for g in gs])
        meshio_pts = np.empty((num_pts, 3))  # type: ignore
        pts_pos = 0
        cell_pos = 0

        # loop on all the 2d grids
        for g in gs:
            # save the points information
            sl = slice(pts_pos, pts_pos + g.num_nodes)
            meshio_pts[sl, :] = g.nodes.T

            # Cell-face and face-node relations
            g_faces_cells, g_cells, _ = sps.find(g.cell_faces)
            # Ensure ordering of the cells
            g_faces_cells = g_faces_cells[np.argsort(g_cells)]
            g_nodes_faces, _, _ = sps.find(g.face_nodes)

            # loop on all the grid cells
            for c in np.arange(g.num_cells):
                loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
                # get the nodes for the current cell
                nodes = np.array(
                    [
                        g_nodes_faces[
                            g.face_nodes.indptr[f] : g.face_nodes.indptr[f + 1]
                        ]
                        for f in g_faces_cells[loc]
                    ]
                ).T
                # sort the nodes
                nodes_loc, *_ = pp.utils.sort_points.sort_point_pairs(nodes)

                # define the type of cell we are currently saving
                cell_type = "polygon" + str(nodes_loc.shape[1])
                cell_type = polygon_map.get(cell_type, cell_type)

                # if the cell type is not present, then add it
                if cell_type not in cell_to_nodes:
                    cell_to_nodes[cell_type] = np.atleast_2d(nodes_loc[0] + pts_pos)
                    cell_id[cell_type] = [cell_pos]
                else:
                    cell_to_nodes[cell_type] = np.vstack(
                        (cell_to_nodes[cell_type], nodes_loc[0] + pts_pos)
                    )
                    cell_id[cell_type] += [cell_pos]
                cell_pos += 1

            pts_pos += g.num_nodes

        # construct the meshio data structure
        num_block = len(cell_to_nodes)
        meshio_cells = np.empty(num_block, dtype=object)
        meshio_cell_id = np.empty(num_block, dtype=object)

        for block, (cell_type, cell_block) in enumerate(cell_to_nodes.items()):
            meshio_cells[block] = meshio.CellBlock(cell_type, cell_block.astype(int))
            meshio_cell_id[block] = np.array(cell_id[cell_type])

        return meshio_pts, meshio_cells, meshio_cell_id

    # ------------------------------------------------------------------------------#

    @pp.time_logger(sections=module_sections)
    def _export_3d(
        self, gs: Iterable[pp.Grid]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Export the geometrical data (point coordinates) and connectivity
        information from the 3d PorePy grids to meshio.
        """

        # use standard name for simple object type
        # NOTE: this part is not developed
        # polygon_map  = {"polyhedron4": "tetra", "polyhedron8": "hexahedron"}

        # cell-faces and cell nodes connectivity information
        cell_to_faces: Dict[str, List[List[int]]] = {}
        cell_to_nodes: Dict[str, np.ndarray] = {}
        # cell id map
        cell_id: Dict[str, List[int]] = {}

        # points
        num_pts = np.sum([g.num_nodes for g in gs])
        meshio_pts = np.empty((num_pts, 3))  # type: ignore
        pts_pos = 0
        cell_pos = 0

        # loop on all the 3d grids
        for g in gs:
            # save the points information
            sl = slice(pts_pos, pts_pos + g.num_nodes)
            meshio_pts[sl, :] = g.nodes.T

            # Cell-face and face-node relations
            g_faces_cells, g_cells, _ = sps.find(g.cell_faces)
            # Ensure ordering of the cells
            g_faces_cells = g_faces_cells[np.argsort(g_cells)]

            g_nodes_faces, _, _ = sps.find(g.face_nodes)

            cptr = g.cell_faces.indptr
            fptr = g.face_nodes.indptr
            face_per_cell = np.diff(cptr)
            nodes_per_face = np.diff(fptr)

            # Total number of nodes to be written in the face-node relation
            num_cell_nodes = np.array([nodes_per_face[i] for i in g.cell_faces.indices])

            n = g.nodes
            fc = g.face_centers
            normal_vec = g.face_normals / g.face_areas

            # Use numba if available, unless the problem is very small, in which
            # case the pure python version probably is faster than combined compile
            # and runtime for numba
            # The number 1000 here is somewhat random.
            if self.has_numba and g.num_cells > 1000:
                logger.info("Construct 3d grid information using numba")
                cell_nodes = self._point_ind_numba(
                    cptr,
                    fptr,
                    g_faces_cells,
                    g_nodes_faces,
                    n,
                    fc,
                    normal_vec,
                    num_cell_nodes,
                )
            else:
                logger.info("Construct 3d grid information using pure python")
                cell_nodes = self._point_ind(
                    cptr,
                    fptr,
                    g_faces_cells,
                    g_nodes_faces,
                    n,
                    fc,
                    normal_vec,
                    num_cell_nodes,
                )

            # implementation note: I did not even try feeding this to numba, my
            # guess is that it will not like the vtk specific stuff.
            nc = 0
            f_counter = 0

            # loop on all the grid cells
            for c in np.arange(g.num_cells):
                faces_loc: List[int] = []
                # loop on all the cell faces
                for _ in np.arange(face_per_cell[c]):
                    fi = g.cell_faces.indices[f_counter]
                    faces_loc += [cell_nodes[nc : (nc + nodes_per_face[fi])]]
                    nc += nodes_per_face[fi]
                    f_counter += 1

                # collect all the nodes for the cell
                nodes_loc = np.unique(faces_loc).astype(int)

                # define the type of cell we are currently saving
                cell_type = "polyhedron" + str(nodes_loc.size)
                # cell_type = polygon_map.get(cell_type, cell_type)

                # if the cell type is not present, then add it
                if cell_type not in cell_to_nodes:
                    cell_to_nodes[cell_type] = np.atleast_2d(nodes_loc + pts_pos)
                    cell_to_faces[cell_type] = [faces_loc]
                    cell_id[cell_type] = [cell_pos]
                else:
                    cell_to_nodes[cell_type] = np.vstack(
                        (cell_to_nodes[cell_type], nodes_loc + pts_pos)
                    )
                    cell_to_faces[cell_type] += [faces_loc]
                    cell_id[cell_type] += [cell_pos]
                cell_pos += 1

            pts_pos += g.num_nodes

        # NOTE: only cell->faces relation will be kept, so far we export only polyhedron
        # otherwise in the next lines we should consider also the cell->nodes map

        # construct the meshio data structure
        num_block = len(cell_to_nodes)
        meshio_cells = np.empty(num_block, dtype=object)
        meshio_cell_id = np.empty(num_block, dtype=object)

        for block, (cell_type, cell_block) in enumerate(cell_to_faces.items()):
            meshio_cells[block] = meshio.CellBlock(cell_type, cell_block)
            meshio_cell_id[block] = np.array(cell_id[cell_type])

        return meshio_pts, meshio_cells, meshio_cell_id

    # ------------------------------------------------------------------------------#

    @pp.time_logger(sections=module_sections)
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

    # ------------------------------------------------------------------------------#

    @pp.time_logger(sections=module_sections)
    def _update_meshio_geom(self):
        if self.is_GridBucket:
            for dim in self.dims:
                g = self.gb.get_grids(lambda g: g.dim == dim)
                self.meshio_geom[dim] = self._export_grid(g, dim)

            for dim in self.m_dims:
                # extract the mortar grids for dimension dim
                mgs = self.gb.get_mortar_grids(lambda g: g.dim == dim)
                # it contains the mortar grids "unrolled" by sides
                mg = np.array([g for m in mgs for _, g in m.side_grids.items()])
                self.m_meshio_geom[dim] = self._export_grid(mg, dim)
        else:
            self.meshio_geom = self._export_grid(
                np.atleast_1d(self.grid), self.grid.dim
            )

    # ------------------------------------------------------------------------------#

    @pp.time_logger(sections=module_sections)
    def _make_folder(self, folder_name: Optional[str] = None, name: str = "") -> str:
        if folder_name is None:
            return name

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        return os.path.join(folder_name, name)

    # ------------------------------------------------------------------------------#

    @pp.time_logger(sections=module_sections)
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

    # ------------------------------------------------------------------------------#

    @pp.time_logger(sections=module_sections)
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

    # ------------------------------------------------------------------------------#

    ### Below follows utility functions for sorting points in 3d

    @pp.time_logger(sections=module_sections)
    def _point_ind(
        self,
        cell_ptr,
        face_ptr,
        face_cells,
        nodes_faces,
        nodes,
        fc,
        normals,
        num_cell_nodes,
    ):
        cell_nodes = np.zeros(num_cell_nodes.sum(), dtype=int)
        counter = 0
        for ci in range(cell_ptr.size - 1):
            loc_c = slice(cell_ptr[ci], cell_ptr[ci + 1])

            for fi in face_cells[loc_c]:
                loc_f = slice(face_ptr[fi], face_ptr[fi + 1])
                ptsId = nodes_faces[loc_f]
                num_p_loc = ptsId.size
                nodes_loc = nodes[:, ptsId]
                # Sort points. Cut-down version of
                # sort_points.sort_points_plane() and subfunctions
                reference = np.array([0.0, 0.0, 1])
                angle = np.arccos(np.dot(normals[:, fi], reference))
                vect = np.cross(normals[:, fi], reference)
                # Cut-down version of cg.rot()
                W = np.array(
                    [
                        [0.0, -vect[2], vect[1]],
                        [vect[2], 0.0, -vect[0]],
                        [-vect[1], vect[0], 0.0],
                    ]
                )
                R = (
                    np.identity(3)
                    + np.sin(angle) * W
                    + (1.0 - np.cos(angle)) * np.linalg.matrix_power(W, 2)
                )
                # pts is now a npt x 3 matrix
                pts = np.array(
                    [R.dot(nodes_loc[:, i]) for i in range(nodes_loc.shape[1])]
                )
                center = R.dot(fc[:, fi])
                # Distance from projected points to center
                delta = np.array([pts[i] - center for i in range(pts.shape[0])])[:, :2]
                nrm = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2)
                delta = delta / nrm[:, np.newaxis]

                argsort = np.argsort(np.arctan2(delta[:, 0], delta[:, 1]))
                cell_nodes[counter : (counter + num_p_loc)] = ptsId[argsort]
                counter += num_p_loc

        return cell_nodes

    @pp.time_logger(sections=module_sections)
    def _point_ind_numba(
        self,
        cell_ptr,
        face_ptr,
        faces_cells,
        nodes_faces,
        nodes,
        fc,
        normals,
        num_cell_nodes,
    ):
        import numba

        @numba.jit(
            "i4[:](i4[:],i4[:],i4[:],i4[:],f8[:,:],f8[:,:],f8[:,:],i4[:])",
            nopython=True,
            nogil=False,
        )
        def _function_to_compile(
            cell_ptr,
            face_ptr,
            faces_cells,
            nodes_faces,
            nodes,
            fc,
            normals,
            num_cell_nodes,
        ):
            """Implementation note: This turned out to be less than pretty, and quite
            a bit more explicit than the corresponding pure python implementation.
            The process was basically to circumvent whatever statements numba did not
            like. Not sure about why this ended so, but there you go.
            """
            cell_nodes = np.zeros(num_cell_nodes.sum(), dtype=np.int32)
            counter = 0
            fc.astype(np.float64)
            for ci in range(cell_ptr.size - 1):
                loc_c = slice(cell_ptr[ci], cell_ptr[ci + 1])
                for fi in faces_cells[loc_c]:
                    loc_f = np.arange(face_ptr[fi], face_ptr[fi + 1])
                    ptsId = nodes_faces[loc_f]
                    num_p_loc = ptsId.size
                    nodes_loc = np.zeros((3, num_p_loc))
                    for iter1 in range(num_p_loc):
                        nodes_loc[:, iter1] = nodes[:, ptsId[iter1]]
                    #            # Sort points. Cut-down version of
                    #            # sort_points.sort_points_plane() and subfunctions
                    reference = np.array([0.0, 0.0, 1])
                    angle = np.arccos(np.dot(normals[:, fi], reference))
                    # Hand code cross product, not supported by current numba version
                    vect = np.array(
                        [
                            normals[1, fi] * reference[2]
                            - normals[2, fi] * reference[1],
                            normals[2, fi] * reference[0]
                            - normals[0, fi] * reference[2],
                            normals[0, fi] * reference[1]
                            - normals[1, fi] * reference[0],
                        ],
                        dtype=np.float64,
                    )
                    ##            # Cut-down version of cg.rot()
                    W = np.array(
                        [
                            0.0,
                            -vect[2],
                            vect[1],
                            vect[2],
                            0.0,
                            -vect[0],
                            -vect[1],
                            vect[0],
                            0.0,
                        ]
                    ).reshape((3, 3))
                    R = (
                        np.identity(3)
                        + np.sin(angle) * W.reshape((3, 3))
                        + (
                            (1.0 - np.cos(angle)) * np.linalg.matrix_power(W, 2).ravel()
                        ).reshape((3, 3))
                    )
                    ##            # pts is now a npt x 3 matrix
                    num_p = nodes_loc.shape[1]
                    pts = np.zeros((3, num_p))
                    fc_loc = fc[:, fi]
                    center = np.zeros(3)
                    for i in range(3):
                        center[i] = (
                            R[i, 0] * fc_loc[0]
                            + R[i, 1] * fc_loc[1]
                            + R[i, 2] * fc_loc[2]
                        )
                    for i in range(num_p):
                        for j in range(3):
                            pts[j, i] = (
                                R[j, 0] * nodes_loc[0, i]
                                + R[j, 1] * nodes_loc[1, i]
                                + R[j, 2] * nodes_loc[2, i]
                            )
                    ##            # Distance from projected points to center
                    delta = 0 * pts
                    for i in range(num_p):
                        delta[:, i] = pts[:, i] - center
                    nrm = np.sqrt(delta[0] ** 2 + delta[1] ** 2)
                    for i in range(num_p):
                        delta[:, i] = delta[:, i] / nrm[i]
                    ##
                    argsort = np.argsort(np.arctan2(delta[0], delta[1]))
                    cell_nodes[counter : (counter + num_p_loc)] = ptsId[argsort]
                    counter += num_p_loc

            return cell_nodes

        return _function_to_compile(
            cell_ptr,
            face_ptr,
            faces_cells,
            nodes_faces,
            nodes,
            fc,
            normals,
            num_cell_nodes,
        )
