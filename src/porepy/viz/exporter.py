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
import warnings

import numpy as np
import scipy.sparse as sps

import porepy as pp

try:
    import vtk
    import vtk.util.numpy_support as ns
except ImportError:
    warnings.warn("No vtk module loaded. Export with pp.Exporter will not work.")

# Module-wide logger
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------#


class Field(object):
    """
    Internal class to store information for the data to export.
    """

    def __init__(self, name, cell_data=False, point_data=False, values=None):
        assert np.logical_xor(cell_data, point_data)
        # name of the field
        self.name = name
        # if the field is vtk cell data
        self.cell_data = cell_data
        # if the field is vtk point data
        self.point_data = point_data
        # number of components of the field
        self.num_components = None
        # values of the field, use set_values to set it
        if values is not None:
            self.set_values(values)
        else:
            self.values = None

    def __repr__(self):
        """
        Repr function
        """
        s = self.name + " - data type: "
        if self.cell_data:
            s += "cell"
        else:
            s += "point"
        return s + " - values: " + str(self.values)

    def dtype(self):
        """
        Return the vtk type of the field
        """
        self._check_values()

        # map numpy to vtk types
        map_type = {
            np.dtype("bool"): vtk.VTK_CHAR,
            np.dtype("int64"): vtk.VTK_INT,
            np.dtype("float64"): vtk.VTK_DOUBLE,
            np.dtype("int32"): vtk.VTK_INT,
        }

        return map_type[self.values.dtype]

    def check(self, values, g):
        """
        Consistency checks making sure the field self.name is filled and has
        the right dimension.
        """
        if values is None:
            raise ValueError(
                "Field " + str(self.name) + " must be filled. It can not be None"
            )
        num_elem = g.num_cells if self.cell_data else g.num_nodes
        if np.atleast_2d(values).shape[1] != num_elem:
            raise ValueError("Field " + str(self.name) + " has wrong dimension.")

    def set_values(self, values):
        """
        Function useful to set the values
        """
        self.num_components = 1 if values.ndim == 1 else 3
        self.values = values.ravel(order="F")

    def _check_values(self):
        if self.values is None:
            raise ValueError("Field " + str(self.name) + " values not valid")


# ------------------------------------------------------------------------------#


class Fields(object):
    """
    Internal class to store a list of field.
    """

    def __init__(self):
        self.fields = None

    def __iter__(self):
        """
        Iterator on all the fields.
        """
        for f in self.fields:
            yield f

    def __repr__(self):
        """
        Repr function
        """
        return "\n".join([repr(f) for f in self])

    def extend(self, fields):
        """
        Extend the list of fields with additional fields.
        """
        if self.fields is None:
            if isinstance(fields, list):
                self.fields = fields
            elif isinstance(fields, Fields):
                self.fields = fields.fields
            else:
                raise ValueError
        else:
            if isinstance(fields, list):
                self.fields.extend(fields)
            elif isinstance(fields, Fields):
                self.fields.extend(fields.fields)
            else:
                raise ValueError

    def names(self):
        """
        Return the list of name of the fields.
        """
        return [f.name for f in self]


# ------------------------------------------------------------------------------#


class Exporter:
    def __init__(self, grid, file_name, folder_name=None, **kwargs):
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
        simplicial: consider only simplicial elements (triangles and tetra)

        How to use:
        If you need to export a single grid:
        save = Exporter(g, "solution", folder_name="results")
        save.write_vtk({"cells_id": cells_id, "pressure": pressure})

        In a time loop:
        save = Exporter(gb, "solution", folder_name="results")
        while time:
            save.write_vtk({"conc": conc}, time_step=i)
        save.write_pvd(steps*deltaT)

        if you need to export the state of variables as stored in the GridBucket:
        save = Exporter(gb, "solution", folder_name="results")
        # export the field stored in data[pp.STATE]["pressure"]
        save.write_vtk(gb, ["pressure"])

        In a time loop:
        while time:
            save.write_vtk(["conc"], time_step=i)
        save.write_pvd(steps*deltaT)

        In the case of different keywords, change the file name with
        "change_name".

        NOTE: the following names are reserved for data exporting: grid_dim,
        is_mortar, mortar_side, cell_id

        Raises:
        ImportError if the module vtk is not available

        """

        self.gb = grid
        self.file_name = file_name
        self.folder_name = folder_name
        self.fixed_grid = kwargs.pop("fixed_grid", True)
        self.binary = kwargs.pop("binary", True)
        self.simplicial = kwargs.pop("simplicial", False)
        if kwargs:
            msg = "Exporter() got unexpected keyword argument '{}'"
            raise TypeError(msg.format(kwargs.popitem()[0]))

        self.is_GridBucket = isinstance(self.gb, pp.GridBucket)
        self.is_not_vtk = "vtk" not in sys.modules

        if self.is_not_vtk:
            raise ImportError("Could not load vtk module")

        if self.is_GridBucket:
            # Fixed-dimensional grids to be included in the export. We include
            # all but the 0-d grids
            self.dims = np.setdiff1d(self.gb.all_dims(), [0])
            num_dims = self.dims.size
            self.gb_VTK = dict(zip(self.dims, [None] * num_dims))

            # mortar grid variables
            self.m_dims = np.unique([d["mortar_grid"].dim for _, d in self.gb.edges()])
            num_m_dims = self.m_dims.size
            self.m_gb_VTK = dict(zip(self.m_dims, [None] * num_m_dims))
        else:
            self.gb_VTK = None

        # Assume numba is available
        self.has_numba = True

        self._update_gb_VTK()

        # Counter for time step. Will be used to identify files of individual time step,
        # unless this is overridden by optional parameters in write_vtk
        self._time_step_counter = 0
        # Storage for file name extensions for time steps
        self._exported_time_step_file_names = []

    # ------------------------------------------------------------------------------#

    def change_name(self, file_name):
        """
        Change the root name of the files, useful when different keywords are
        considered but on the same grid.

        Parameters:
        file_name: the new root name of the files.
        """
        self.file_name = file_name

    # ------------------------------------------------------------------------------#

    def write_vtk(
        self,
        data=None,
        time_dependent=False,
        time_step=None,
        grid=None,
        point_data=False,
    ):
        """
        Interface function to export the grid and additional data in VTK.

        In 2d the cells are represented as polygon, while in 3d as polyhedra.
        The VTK module needs to be installed.
        In 3d the geometry of the mesh needs to be computed.

        To work with python3, the package vtk should be installed in version 7
        or higher.

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
        point_data: ***

        """
        if self.is_not_vtk:
            return

        if self.fixed_grid and grid is not None:
            raise ValueError("Inconsistency in exporter setting")
        elif not self.fixed_grid and grid is not None:
            self.gb = grid
            self.is_GridBucket = isinstance(self.gb, pp.GridBucket)
            self._update_gb_VTK()

        # If the problem is time dependent, but no time step is set, we set one
        if time_dependent and time_step is None:
            time_step = self._time_step_counter
            self._time_step_counter += 1

        # If the problem is time dependent (with specified or automatic time step index)
        # add the time step to the exported files
        if time_step is not None:
            self._exported_time_step_file_names.append(time_step)

        if self.is_GridBucket:
            self._export_vtk_gb(data, time_step, point_data)
        else:
            self._export_vtk_single(data, time_step, point_data)

    # ------------------------------------------------------------------------------#

    def write_pvd(self, timestep, file_extension=None):
        """
        Interface function to export in PVD file the time loop information.
        The user should open only this file in paraview.

        We assume that the VTU associated files have the same name.
        We assume that the VTU associated files are in the same folder.

        Parameters:
        timestep: vector of times to be exported. These will be the time associated with
            indivdiual time steps in, say, Paraview. By default, the times will be
            associated with the order in which the time steps were exported. This can
            be overridden by the file_extension argument.
        file_extension (np.array-like, optional): End of file names used in the export
            of individual time steps, see self.write_vtk(). If provided, it should have
            the same length as time. If not provided, the file names will be picked
            from those used when writing individual time steps.

        """
        if self.is_not_vtk:
            return

        if file_extension is None:
            file_extension = self._exported_time_step_file_names

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

    def _export_vtk_single(self, data, time_step, point_data):
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
            if point_data:
                fields.extend(
                    [Field(n, point_data=True, values=v) for n, v in data.items()]
                )
            else:
                fields.extend(
                    [Field(n, cell_data=True, values=v) for n, v in data.items()]
                )

        fields.extend(
            [
                Field(
                    "grid_dim",
                    cell_data=True,
                    values=self.gb.dim * np.ones(self.gb.num_cells),
                ),
                Field("cell_id", cell_data=True, values=np.arange(self.gb.num_cells)),
            ]
        )

        self._write_vtk(fields, name, self.gb_VTK)

    # ------------------------------------------------------------------------------#

    def _export_vtk_gb(self, data, time_step, point_data):
        # Convert data to list, or provide an empty list
        if data is not None:
            data = np.atleast_1d(data).tolist()
        else:
            data = list()

        fields = Fields()
        if len(data) > 0:
            if point_data:
                fields.extend([Field(d, point_data=True) for d in data])
            else:
                fields.extend([Field(d, cell_data=True) for d in data])

        # consider the grid_bucket node data
        extra_fields = Fields()
        extra_fields.extend(
            [
                Field("grid_dim", cell_data=True),
                Field("cell_id", cell_data=True),
                Field("grid_node_number", cell_data=True),
                Field("is_mortar", cell_data=True),
                Field("mortar_side", cell_data=True),
            ]
        )
        fields.extend(extra_fields)

        self.gb.assign_node_ordering(overwrite_existing=False)
        self.gb.add_node_props(extra_fields.names())
        # fill the extra data
        for g, d in self.gb:
            ones = np.ones(g.num_cells, dtype=np.int)
            d["grid_dim"] = g.dim * ones
            d["cell_id"] = np.arange(g.num_cells, dtype=np.int)
            d["grid_node_number"] = d["node_number"] * ones
            d["is_mortar"] = np.zeros(g.num_cells, dtype=np.bool)
            d["mortar_side"] = int(pp.grids.mortar_grid.NONE_SIDE) * ones

        # collect the data and extra data in a single stack for each dimension
        for dim in self.dims:
            file_name = self._make_file_name(self.file_name, time_step, dim)
            file_name = self._make_folder(self.folder_name, file_name)
            for field in fields:
                grids = self.gb.get_grids(lambda g: g.dim == dim)
                values = np.empty(grids.size, dtype=np.object)
                for i, g in enumerate(grids):
                    if field.name in data:
                        values[i] = self.gb._nodes[g][pp.STATE][field.name]
                    else:
                        values[i] = self.gb._nodes[g][field.name]
                    field.check(values[i], g)
                field.set_values(np.hstack(values))

            if self.gb_VTK[dim] is not None:
                self._write_vtk(fields, file_name, self.gb_VTK[dim])

        self.gb.remove_node_props(extra_fields.names())

        # consider the grid_bucket edge data
        extra_fields = Fields()
        extra_fields.extend(
            [
                Field("grid_dim", cell_data=True),
                Field("cell_id", cell_data=True),
                Field("grid_edge_number", cell_data=True),
                Field("is_mortar", cell_data=True),
                Field("mortar_side", cell_data=True),
            ]
        )
        self.gb.add_edge_props(extra_fields.names())
        for _, d in self.gb.edges():
            d["grid_dim"] = {}
            d["cell_id"] = {}
            d["grid_edge_number"] = {}
            d["is_mortar"] = {}
            d["mortar_side"] = {}
            mg = d["mortar_grid"]
            mg_num_cells = 0
            for side, g in mg.side_grids.items():
                ones = np.ones(g.num_cells, dtype=np.int)
                d["grid_dim"][side] = g.dim * ones
                d["is_mortar"][side] = ones.astype(np.bool)
                d["mortar_side"][side] = int(side) * ones
                d["cell_id"][side] = np.arange(g.num_cells, dtype=np.int) + mg_num_cells
                mg_num_cells += g.num_cells
                d["grid_edge_number"][side] = d["edge_number"] * ones

        # collect the data and extra data in a single stack for each dimension
        for dim in self.m_dims:
            file_name = self._make_file_name_mortar(self.file_name, time_step, dim)
            file_name = self._make_folder(self.folder_name, file_name)

            mgs = self.gb.get_mortar_grids(lambda g: g.dim == dim)
            cond = lambda d: d["mortar_grid"].dim == dim
            edges = np.array([e for e, d in self.gb.edges() if cond(d)])
            num_grids = np.sum([m.num_sides() for m in mgs])

            for field in extra_fields:
                values = np.empty(num_grids, dtype=np.object)
                i = 0
                for mg, edge in zip(mgs, edges):
                    for side, _ in mg.side_grids.items():
                        # Convert edge to tuple to be compatible with GridBucket
                        # data structure
                        values[i] = self.gb.edge_props(tuple(edge), field.name)[side]
                        i += 1

                field.set_values(np.hstack(values))

            if self.m_gb_VTK[dim] is not None:
                self._write_vtk(extra_fields, file_name, self.m_gb_VTK[dim])

        file_name = self._make_file_name(self.file_name, time_step, extension=".pvd")
        file_name = self._make_folder(self.folder_name, file_name)
        self._export_pvd_gb(file_name, time_step)

        self.gb.remove_edge_props(extra_fields.names())

    # ------------------------------------------------------------------------------#

    def _export_pvd_gb(self, file_name, time_step):
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
            if self.gb_VTK[dim] is not None:
                o_file.write(
                    fm % self._make_file_name(self.file_name, time_step, dim=dim)
                )
        for dim in self.m_dims:
            if self.m_gb_VTK[dim] is not None:
                o_file.write(
                    fm % self._make_file_name_mortar(self.file_name, time_step, dim=dim)
                )

        o_file.write("</Collection>\n" + "</VTKFile>")
        o_file.close()

    # ------------------------------------------------------------------------------#

    def _export_vtk_grid(self, gs, dim):
        """
        Wrapper function to export grids of dimension dim. Calls the
        appropriate dimension specific export function.
        """
        if dim == 0:
            return
        elif dim == 1:
            return self._export_vtk_1d(gs)
        elif dim == 2:
            return self._export_vtk_2d(gs)
        elif dim == 3:
            return self._export_vtk_3d(gs)

    # ------------------------------------------------------------------------------#

    def _export_vtk_1d(self, gs):
        """
        Export the geometrical data (point coordinates) and connectivity
        information from the 1d PorePy grids to vtk.
        """
        gVTK = vtk.vtkUnstructuredGrid()
        ptsVTK = vtk.vtkPoints()

        ptsId_global = 0
        for g in gs:
            cell_nodes = g.cell_nodes()
            nodes, cells, _ = sps.find(cell_nodes)

            for c in np.arange(g.num_cells):
                loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c + 1])
                ptsId = nodes[loc] + ptsId_global
                fsVTK = vtk.vtkIdList()
                [fsVTK.InsertNextId(p) for p in ptsId]
                gVTK.InsertNextCell(vtk.VTK_LINE, fsVTK)

            ptsId_global += g.num_nodes
            [ptsVTK.InsertNextPoint(*node) for node in g.nodes.T]

        gVTK.SetPoints(ptsVTK)

        return gVTK

    # ------------------------------------------------------------------------------#

    def _export_vtk_2d(self, gs):
        """
        Export the geometrical data (point coordinates) and connectivity
        information from the 2d PorePy grids to vtk.
        """
        gVTK = vtk.vtkUnstructuredGrid()
        ptsVTK = vtk.vtkPoints()

        # Offset for the assignment of point indices. Needed to unify the counting of
        # points for all grids in gs
        ptsId_global = 0
        for g in gs:
            # Cell-face and face-node relations
            faces_cells, _, _ = sps.find(g.cell_faces)
            nodes_faces, _, _ = sps.find(g.face_nodes)

            # For each cell, find its boundary edges, as a list of start and endpoints,
            # sorted so that the edges form a closed loop around the cell
            for c in np.arange(g.num_cells):
                # Faces of this cell
                loc_faces = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
                # Points of the faces
                ptsId = np.array(
                    [
                        nodes_faces[g.face_nodes.indptr[f] : g.face_nodes.indptr[f + 1]]
                        for f in faces_cells[loc_faces]
                    ]
                ).T
                # Sort the points.
                # Note that the indices assigned here also adjusts to the global point
                # numbering
                # Split computation in two to catch extra return argument
                ptsId_paired, _ = pp.utils.sort_points.sort_point_pairs(ptsId)
                ptsId_paired = ptsId_paired[0, :] + ptsId_global

                # Create a list of Ids, add all point pairs.
                fsVTK = vtk.vtkIdList()
                for p in ptsId_paired:
                    fsVTK.InsertNextId(p)
                # Add a new polygon
                gVTK.InsertNextCell(vtk.VTK_POLYGON, fsVTK)

            # Adjust the offset with the number of nodes in this grid
            ptsId_global += g.num_nodes
            # Add the new nodes to the point list
            for node in g.nodes.T:
                ptsVTK.InsertNextPoint(*node)

        gVTK.SetPoints(ptsVTK)

        return gVTK

    # ------------------------------------------------------------------------------#

    def _export_vtk_3d(self, gs):
        """
        Export the geometrical data (point coordinates) and connectivity
        information from the 3d PorePy grids to vtk.
        """
        # This functionality became rather complex, with possible use of numba.
        # Decided to dump this to a separate file.
        return self._define_gvtk_3d(gs)

    # ------------------------------------------------------------------------------#

    def _write_vtk(self, fields, file_name, g_VTK):
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputData(g_VTK)
        writer.SetFileName(file_name)

        if fields is not None:
            for field in fields:
                if field.values is None:
                    continue
                dataVTK = ns.numpy_to_vtk(
                    field.values, deep=True, array_type=field.dtype()
                )
                dataVTK.SetName(field.name)
                dataVTK.SetNumberOfComponents(field.num_components)

                if field.cell_data:
                    g_VTK.GetCellData().AddArray(dataVTK)
                elif field.point_data:
                    g_VTK.GetPointData().AddArray(dataVTK)

        if not self.binary:
            writer.SetDataModeToAscii()
        writer.Update()

        if fields is not None:
            for field in fields:
                if field.cell_data:
                    g_VTK.GetCellData().RemoveArray(field.name)
                elif field.point_data:
                    g_VTK.GetPointData().RemoveArray(field.name)

    # ------------------------------------------------------------------------------#

    def _update_gb_VTK(self):
        if self.is_GridBucket:
            for dim in self.dims:
                g = self.gb.get_grids(lambda g: g.dim == dim)
                self.gb_VTK[dim] = self._export_vtk_grid(g, dim)

            for dim in self.m_dims:
                # extract the mortar grids for dimension dim
                mgs = self.gb.get_mortar_grids(lambda g: g.dim == dim)
                # it contains the mortar grids "unrolled" by sides
                mg = np.array([g for m in mgs for _, g in m.side_grids.items()])
                self.m_gb_VTK[dim] = self._export_vtk_grid(mg, dim)
        else:
            self.gb_VTK = self._export_vtk_grid([self.gb], self.gb.dim)

    # ------------------------------------------------------------------------------#

    def _make_folder(self, folder_name, name=None):
        if folder_name is None:
            return name

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        return os.path.join(folder_name, name)

    # ------------------------------------------------------------------------------#

    def _make_file_name(self, file_name, time_step=None, dim=None, extension=".vtu"):

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

    def _make_file_name_mortar(self, file_name, time_step=None, dim=None):

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

    def _define_gvtk_3d(self, gs):
        # NOTE: we are assuming only one 3d grid
        gVTK = vtk.vtkUnstructuredGrid()
        ptsVTK = vtk.vtkPoints()

        for g in gs:
            faces_cells, cells, _ = sps.find(g.cell_faces)
            nodes_faces, faces, _ = sps.find(g.face_nodes)

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
                    faces_cells,
                    nodes_faces,
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
                    faces_cells,
                    nodes_faces,
                    n,
                    fc,
                    normal_vec,
                    num_cell_nodes,
                )
            # implementation note: I did not even try feeding this to numba, my
            # guess is that it will not like the vtk specific stuff.
            node_counter = 0
            face_counter = 0
            for c in np.arange(g.num_cells):
                if self.simplicial:
                    loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c + 1])
                    ptsId = np.array(
                        [
                            nodes_faces[
                                g.face_nodes.indptr[f] : g.face_nodes.indptr[f + 1]
                            ]
                            for f in faces_cells[loc]
                        ]
                    ).T
                    ptsId = np.unique(ptsId)
                    fsVTK = vtk.vtkIdList()
                    [fsVTK.InsertNextId(p) for p in ptsId]

                    gVTK.InsertNextCell(vtk.VTK_TETRA, fsVTK)
                else:
                    fsVTK = vtk.vtkIdList()
                    # Number faces that make up the cell
                    fsVTK.InsertNextId(face_per_cell[c])
                    for f in range(face_per_cell[c]):
                        fi = g.cell_faces.indices[face_counter]
                        fsVTK.InsertNextId(
                            nodes_per_face[fi]
                        )  # Number of points in face
                        for ni in range(nodes_per_face[fi]):
                            fsVTK.InsertNextId(cell_nodes[node_counter])
                            node_counter += 1
                        face_counter += 1

                    gVTK.InsertNextCell(vtk.VTK_POLYHEDRON, fsVTK)

            [ptsVTK.InsertNextPoint(*node) for node in g.nodes.T]

        gVTK.SetPoints(ptsVTK)

        return gVTK

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
        cell_nodes = np.zeros(num_cell_nodes.sum(), dtype=np.int)
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
