import sys, os
import numpy as np
import scipy.sparse as sps
import logging

try:
    import vtk
    import vtk.util.numpy_support as ns
except ImportError:
    import warnings
    warnings.warn("No vtk module loaded.")
try:
    import numba
except ImportError:
    warnings.warn('Numba not available. Export may be slow for large grids')

from porepy.grids import grid_bucket
from porepy.utils import sort_points


# Module-wide logger
logger = logging.getLogger(__name__)


class Exporter():
    def __init__(self, grid, name, folder=None, **kwargs):
        """
        Parameters:
        grid: the grid or grid bucket
        name: the root of file name without any extension.
        folder: (optional) the folder to save the file. If the folder does not
            exist it will be created.

        Optional arguments in kwargs:
        fixed_grid: (optional) in a time dependent simulation specify if the
            grid changes in time or not. The default is True.
        binary: export in binary format, default is True.

        How to use:
        If you need to export a single grid:
        save = Exporter(g, "solution", folder="results")
        save.write_vtk({"cells_id": cells_id, "pressure": pressure})

        In a time loop:
        save = Exporter(gb, "solution", folder="results")
        while time:
            save.write_vtk({"conc": conc}, time_step=i)
        save.write_pvd(steps*deltaT)

        if you need to export the grid bucket
        save = Exporter(gb, "solution", folder="results")
        save.write_vtk(gb, ["cells_id", "pressure"])

        In a time loop:
        while time:
            save.write_vtk(["conc"], time_step=i)
        save.write_pvd(steps*deltaT)

        In the case of different physics, change the file name with
        "change_name".

        """

        self.gb = grid
        self.name = name
        self.folder = folder
        self.fixed_grid = kwargs.get('fixed_grid', True)
        self.binary = kwargs.get('binary', True)

        self.is_GridBucket = isinstance(self.gb, grid_bucket.GridBucket)
        self.is_not_vtk = 'vtk' not in sys.modules

        if self.is_not_vtk:
            return

        if self.is_GridBucket:
            self.gb_VTK = np.empty(self.gb.size(), dtype=np.object)
        else:
            self.gb_VTK = None

        self.has_numba = 'numba' in sys.modules

        if self.fixed_grid:
            self._update_gb_VTK()

#------------------------------------------------------------------------------#

    def change_name(self, name):
        """
        Change the root name of the files, useful when different physics are
        considered but on the same grid.

        Parameters:
        name: the new root name of the files.
        """
        self.name = name

#------------------------------------------------------------------------------#

    def write_vtk(self, data=None, time_step=None, grid=None):
        """ Interface function to export in VTK the grid and additional data.

        In 2d the cells are represented as polygon, while in 3d as polyhedra.
        VTK module need to be installed.
        In 3d the geometry of the mesh needs to be computed.

        To work with python3, the package vtk should be installed in version 7
        or higher.

        Parameters:
        data: if g is a single grid then data is a dictionary (see example)
              if g is a grid bucket then list of names for optional data,
              they are the keys in the grid bucket (see example).
        time_step: (optional) in a time dependent problem defines the full name of
            the file.
        grid: (optional) in case of changing grid set a new one.

        """
        if self.is_not_vtk:
            return

        if self.fixed_grid and grid is not None:
            raise ValueError("Inconsistency in exporter setting")
        elif not self.fixed_grid and grid is not None:
            self.gb = grid
            self.is_GridBucket = isinstance(self.gb, grid_bucket.GridBucket)
            self._update_gVTK()

        if self.is_GridBucket:
            self._export_vtk_gb(data, time_step)
        else:
            # No need of special naming, create the folder
            name = self._make_folder(self.folder, self.name)
            self._export_vtk_single(data, time_step, self.gb, name)

#------------------------------------------------------------------------------#

    def write_pvd(self, time):
        """ Interface function to export in PVD file the time loop informations.
        The user should open only this file in paraview.

        We assume that the VTU associated files have the same name.
        We assume that the VTU associated files are in the same folder.

        Parameters:
        time: vector of times.

        """
        if self.is_not_vtk:
            return

        o_file = open(self._make_folder(self.folder, self.name)+".pvd", 'w')
        b = 'LittleEndian' if sys.byteorder == 'little' else 'BigEndian'
        c = ' compressor="vtkZLibDataCompressor"'
        header = '<?xml version="1.0"?>\n'+ \
                 '<VTKFile type="Collection" version="0.1" ' + \
                                              'byte_order="%s"%s>\n' % (b,c) + \
                 '<Collection>\n'
        o_file.write(header)
        fm = '\t<DataSet group="" part="" timestep="%f" file="%s"/>\n'

        time_step = np.arange(time.size)

        if self.is_GridBucket:
            [o_file.write(fm%(time[t],
                        self._make_file_name(self.name, t, d['node_number']))) \
                                         for t in time_step for _, d in self.gb]
        else:
            [o_file.write(fm%(time[t], self._make_file_name(self.name, t))) \
                                                             for t in time_step]

        o_file.write('</Collection>\n'+'</VTKFile>')
        o_file.close()

#------------------------------------------------------------------------------#

    def _export_vtk_single(self, data, time_step, g, name):
        name = self._make_file_name(name, time_step)
        self._write_vtk(data, name, self.gb_VTK)

#------------------------------------------------------------------------------#

    def _export_vtk_gb(self, data, time_step):
        if data is not None:
            data = np.atleast_1d(data).tolist()
        assert isinstance(data, list) or data is None
        data = list() if data is None else data
        data.append('grid_dim')

        self.gb.assign_node_ordering(overwrite_existing=False)
        self.gb.add_node_props(['grid_dim', 'file_name'])

        for g, d in self.gb:
            if g.dim > 0:
                d['file_name'] = self._make_file_name(self.name, time_step,
                                                               d['node_number'])
                file_name = self._make_folder(self.folder, d['file_name'])
                d['grid_dim'] = np.tile(g.dim, g.num_cells)
                dic_data = self.gb.node_props_of_keys(g, data)
                g_VTK = self.gb_VTK[d['node_number']]
                self._write_vtk(dic_data, file_name, g_VTK)

        name = self._make_folder(self.folder, self.name)+".pvd"
        self._export_pvd_gb(name)

#------------------------------------------------------------------------------#

    def _export_pvd_gb(self, name):
        o_file = open(name, 'w')
        b = 'LittleEndian' if sys.byteorder == 'little' else 'BigEndian'
        c = ' compressor="vtkZLibDataCompressor"'
        header = '<?xml version="1.0"?>\n'+ \
                 '<VTKFile type="Collection" version="0.1" ' + \
                                              'byte_order="%s"%s>\n' % (b,c) + \
                 '<Collection>\n'
        o_file.write(header)
        fm = '\t<DataSet group="" part="" file="%s"/>\n'
        [o_file.write( fm % d['file_name'] ) for g, d in self.gb if g.dim!=0]
        o_file.write('</Collection>\n'+'</VTKFile>')
        o_file.close()

#------------------------------------------------------------------------------#

    def _export_vtk_grid(self, g):
        if g.dim == 0:
            return
        elif g.dim == 1:
            return self._export_vtk_1d(g)
        elif g.dim == 2:
            return self._export_vtk_2d(g)
        elif g.dim == 3:
            return self._export_vtk_3d(g)

#------------------------------------------------------------------------------#

    def _export_vtk_1d(self, g):
        cell_nodes = g.cell_nodes()
        nodes, cells, _  = sps.find(cell_nodes)
        gVTK = vtk.vtkUnstructuredGrid()

        for c in np.arange(g.num_cells):
            loc = slice(cell_nodes.indptr[c], cell_nodes.indptr[c+1])
            ptsId = nodes[loc]
            fsVTK = vtk.vtkIdList()
            [fsVTK.InsertNextId(p) for p in ptsId]

            gVTK.InsertNextCell(vtk.VTK_LINE, fsVTK)

        ptsVTK = vtk.vtkPoints()
        if g.nodes.shape[0] == 1:
            [ptsVTK.InsertNextPoint(node, 0., 0.) for node in g.nodes.T]
        if g.nodes.shape[0] == 2:
            [ptsVTK.InsertNextPoint(node[0], node[1], 0.) for node in g.nodes.T]
        else:
            [ptsVTK.InsertNextPoint(*node) for node in g.nodes.T]
        gVTK.SetPoints(ptsVTK)

        return gVTK

#------------------------------------------------------------------------------#

    def _export_vtk_2d(self, g):
        faces_cells, _, _ = sps.find(g.cell_faces)
        nodes_faces, _, _ = sps.find(g.face_nodes)

        gVTK = vtk.vtkUnstructuredGrid()

        for c in np.arange(g.num_cells):
            loc = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
            ptsId = np.array([nodes_faces[g.face_nodes.indptr[f]:\
                                          g.face_nodes.indptr[f+1]]
                              for f in faces_cells[loc]]).T
            ptsId = sort_points.sort_point_pairs(ptsId)[0,:]

            fsVTK = vtk.vtkIdList()
            [fsVTK.InsertNextId(p) for p in ptsId]

            gVTK.InsertNextCell(vtk.VTK_POLYGON, fsVTK)

        ptsVTK = vtk.vtkPoints()
        if g.nodes.shape[0] == 2:
            [ptsVTK.InsertNextPoint(node[0], node[1], 0.) for node in g.nodes.T]
        else:
            [ptsVTK.InsertNextPoint(*node ) for node in g.nodes.T]
        gVTK.SetPoints(ptsVTK)

        return gVTK

#------------------------------------------------------------------------------#

    def _export_vtk_3d(self, g):
        # This functionality became rather complex, with possible use of numba.
        # Decided to dump this to a separate file.
        return self._define_gvtk_3d(g)

#------------------------------------------------------------------------------#

    def _write_vtk(self, data, name, g_VTK):
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetInputData(g_VTK)
        writer.SetFileName(name)

        if data is not None:
            for name_field, values_field in data.items():
                dataVTK = ns.numpy_to_vtk(values_field.ravel(order='F'),
                                          deep=True, array_type=vtk.VTK_DOUBLE)
                dataVTK.SetName(str(name_field))
                dataVTK.SetNumberOfComponents(1 if values_field.ndim == 1 else 3)
                g_VTK.GetCellData().AddArray(dataVTK)

        if not self.binary:
            writer.SetDataModeToAscii()
        writer.Update()

        if data is not None:
            for name_field, _ in data.items():
                cell_data = g_VTK.GetCellData().RemoveArray(str(name_field))

#------------------------------------------------------------------------------#

    def _update_gb_VTK(self):
        if self.is_GridBucket:
            for g, d in self.gb:
                self.gb_VTK[d['node_number']] = self._export_vtk_grid(g)
        else:
            self.gb_VTK = self._export_vtk_grid(self.gb)

#------------------------------------------------------------------------------#

    def _make_folder(self, folder, name=None):
        if folder is None: return name

        if not os.path.exists(folder):
            os.makedirs(folder)
        return os.path.join(folder, name)

#------------------------------------------------------------------------------#

    def _make_file_name(self, name, time_step=None, node_number=None):

        extension = ".vtu"
        padding = 6
        if node_number is None: # normal grid
            if time_step is None:
                return name + extension
            else:
                time = str(time_step).zfill(padding)
                return name + "_" + time + extension
        else: # part of a grid bucket
            grid = str(node_number).zfill(padding)
            if time_step is None:
                return name + "_" + grid + extension
            else:
                time = str(time_step).zfill(padding)
                return name + "_" + grid + "_" + time + extension

#------------------------------------------------------------------------------#

    def _define_gvtk_3d(self, g):
        gVTK = vtk.vtkUnstructuredGrid()

        faces_cells, cells, _ = sps.find(g.cell_faces)
        nodes_faces, faces, _ = sps.find(g.face_nodes)

        cptr = g.cell_faces.indptr
        fptr = g.face_nodes.indptr
        face_per_cell = np.diff(cptr)
        nodes_per_face = np.diff(fptr)

        # Total number of nodes to be written in the face-node relation
        num_cell_nodes = np.array([nodes_per_face[i] \
                                   for i in g.cell_faces.indices])

        n = g.nodes
        fc = g.face_centers
        normal_vec = g.face_normals / g.face_areas

        # Use numba if available, unless the problem is very small, in which
        # case the pure python version probably is faster than combined compile
        # and runtime for numba
        # The number 1000 here is somewhat random.
        if self.has_numba and g.num_cells > 1000:
            logger.info('Construct 3d grid information using numba')
            cell_nodes = _point_ind_numba(cptr, fptr, faces_cells, nodes_faces,
                                          n, fc, normal_vec, num_cell_nodes)
        else:
            logger.info('Construct 3d grid information using pure python')
            cell_nodes = _point_ind(cptr, fptr, faces_cells, nodes_faces, n,
                                    fc, normal_vec, num_cell_nodes)
        # implementation note: I did not even try feeding this to numba, my
        # guess is that it will not like the vtk specific stuff.
        node_counter = 0
        face_counter = 0
        for c in np.arange(g.num_cells):
            fsVTK = vtk.vtkIdList()
            fsVTK.InsertNextId(face_per_cell[c]) # Number faces that make up the cell
            for f in range(face_per_cell[c]):
                fi = g.cell_faces.indices[face_counter]
                fsVTK.InsertNextId(nodes_per_face[fi]) # Number of points in face
                for ni in range(nodes_per_face[fi]):
                    fsVTK.InsertNextId(cell_nodes[node_counter])
                    node_counter += 1
                face_counter += 1

            gVTK.InsertNextCell(vtk.VTK_POLYHEDRON, fsVTK)

        ptsVTK = vtk.vtkPoints()
        [ptsVTK.InsertNextPoint(*node) for node in g.nodes.T]
        gVTK.SetPoints(ptsVTK)

        return gVTK


def _point_ind(cell_ptr, face_ptr, face_cells, nodes_faces, nodes,
               fc, normals, num_cell_nodes):
    cell_nodes = np.zeros(num_cell_nodes.sum(), dtype=np.int)
    counter = 0
    for ci in range(cell_ptr.size - 1):
        loc_c = slice(cell_ptr[ci], cell_ptr[ci + 1])

        for fi in face_cells[loc_c]:
            loc_f = slice(face_ptr[fi], face_ptr[fi+1])
            ptsId = nodes_faces[loc_f]
            num_p_loc = ptsId.size
            nodes_loc = nodes[:, ptsId]
            # Sort points. Cut-down version of
            # sort_points.sort_points_plane() and subfunctions
            reference = np.array([0., 0., 1])
            angle = np.arccos(np.dot(normals[:, fi], reference))
            vect = np.cross(normals[:, fi], reference)
            # Cut-down version of cg.rot()
            W = np.array( [[       0., -vect[2],  vect[1]],
                           [  vect[2],       0., -vect[0]],
                           [ -vect[1],  vect[0],       0. ]])
            R = np.identity(3) + np.sin(angle)*W + \
                  (1.-np.cos(angle)) * np.linalg.matrix_power(W, 2)
            # pts is now a npt x 3 matrix
            pts = np.array([R.dot(nodes_loc[:, i])\
                                  for i in range(nodes_loc.shape[1])])
            center = R.dot(fc[:, fi])
            # Distance from projected points to center
            delta = np.array([pts[i] - center\
                              for i in range(pts.shape[0])])[:, :2]
            nrm = np.sqrt(delta[:, 0]**2 + delta[:, 1]**2)
            delta = delta / nrm[:, np.newaxis]

            argsort = np.argsort(np.arctan2(delta[:, 0], delta[:, 1]))
            cell_nodes[counter:(counter+num_p_loc)] = ptsId[argsort]
            counter += num_p_loc

    return cell_nodes

if 'numba' in sys.modules:
    @numba.jit("i4[:](i4[:],i4[:],i4[:],i4[:],f8[:,:],f8[:,:],f8[:,:],i4[:])",
               nopython=True, nogil=False)
    def _point_ind_numba(cell_ptr, face_ptr, faces_cells, nodes_faces,
                         nodes, fc, normals, num_cell_nodes):
        """ Implementation note: This turned out to be less than pretty, and quite
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
                loc_f = np.arange(face_ptr[fi], face_ptr[fi+1])
                ptsId = nodes_faces[loc_f]
                num_p_loc = ptsId.size
                nodes_loc = np.zeros((3, num_p_loc))
                for iter1 in range(num_p_loc):
                    nodes_loc[:, iter1] = nodes[:, ptsId[iter1]]
    #            # Sort points. Cut-down version of
    #            # sort_points.sort_points_plane() and subfunctions
                reference = np.array([0., 0., 1])
                angle = np.arccos(np.dot(normals[:, fi], reference))
                # Hand code cross product, not supported by current numba version
                vect = np.array([  normals[1, fi] * reference[2]\
                                 - normals[2, fi] * reference[1],
                                   normals[2, fi] * reference[0]\
                                 - normals[0, fi] * reference[2],
                                   normals[0, fi] * reference[1]\
                                 - normals[1, fi] * reference[0]
                                 ], dtype=np.float64)
    ##            # Cut-down version of cg.rot()
                W = np.array( [       0., -vect[2],  vect[1],
                                 vect[2],       0., -vect[0],
                                -vect[1],  vect[0],       0. ]).reshape((3, 3))
                R = np.identity(3) + np.sin(angle)*W.reshape((3, 3)) + \
                ((1.-np.cos(angle)) * np.linalg.matrix_power(W, 2).ravel()).reshape((3, 3))
    ##            # pts is now a npt x 3 matrix
                num_p = nodes_loc.shape[1]
                pts = np.zeros((3, num_p))
                fc_loc = fc[:, fi]
                center = np.zeros(3)
                for i in range(3):
                    center[i] = R[i, 0] * fc_loc[0] + R[i, 1] * fc_loc[1] + \
                                R[i, 2] * fc_loc[2]
                for i in range(num_p):
                    for j in range(3):
                        pts[j, i] = R[j, 0] * nodes_loc[0, i]\
                                  + R[j, 1] * nodes_loc[1, i] + \
                                  + R[j, 2] * nodes_loc[2, i]
    ##            # Distance from projected points to center
                delta = 0*pts
                for i in range(num_p):
                    delta[:, i] = pts[:, i] - center
                nrm = np.sqrt(delta[0]**2 + delta[1]**2)
                for i in range(num_p):
                    delta[:, i] = delta[:, i] / nrm[i]
    ##
                argsort = np.argsort(np.arctan2(delta[0], delta[1]))
                cell_nodes[counter:(counter+num_p_loc)] = ptsId[argsort]
                counter += num_p_loc

        return cell_nodes


#------------------------------------------------------------------------------#
