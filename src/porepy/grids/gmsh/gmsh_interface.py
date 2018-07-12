# Methods to work directly with the gmsh format

import numpy as np
import sys
import os
from meshio import gmsh_io

from porepy.utils import sort_points, read_config
import porepy.grids.constants as gridding_constants


class GmshWriter(object):
    """
     Write a gmsh.geo file for a fractured 2D domains, possibly including
     compartments
    """

    def __init__(
        self,
        pts,
        lines,
        polygons=None,
        domain=None,
        nd=None,
        mesh_size=None,
        mesh_size_bound=None,
        line_type=None,
        intersection_points=None,
        tolerance=None,
        edges_2_frac=None,
        meshing_algorithm=None,
        fracture_tags=None,
    ):
        """

        :param pts: np.ndarary, Points
        :param lines: np.ndarray. Non-intersecting lines in the geometry.
        :param nd: Dimension. Inferred from points if not provided
        """
        self.pts = pts
        self.lines = lines
        self.polygons = polygons
        if nd is None:
            if pts.shape[0] == 2:
                self.nd = 2
            elif pts.shape[0] == 3:
                self.nd = 3
        else:
            self.nd = nd

        self.domain = domain

        if fracture_tags is not None:
            self.polygon_tags = fracture_tags

        self.mesh_size = mesh_size
        self.mesh_size_bound = mesh_size_bound

        # Points that should be decleared physical (intersections between 3
        # fractures)
        self.intersection_points = intersection_points
        self.tolerance = tolerance
        self.e2f = edges_2_frac

        self.meshing_algorithm = meshing_algorithm

    def write_geo(self, file_name):

        if self.tolerance is not None:
            s = "Geometry.Tolerance = " + str(self.tolerance) + ";\n"
        else:
            s = "\n"
        s += self.__write_points()

        if self.nd == 2:
            if self.domain is not None:
                s += self.__write_boundary_2d()
            s += self.__write_fractures_compartments_2d()
        elif self.nd == 3:
            s += self.__write_lines()
            if self.domain is not None:
                s += self.__write_boundary_3d()
            s += self.__write_polygons()

        s += self.__write_physical_points()
        s += self.__write_meshing_algorithm()

        with open(file_name, "w") as f:
            f.write(s)

    def __write_fractures_compartments_2d(self):
        # Both fractures and compartments are
        constants = gridding_constants.GmshConstants()

        frac_ind = np.argwhere(
            np.logical_or(
                self.lines[2] == constants.COMPARTMENT_BOUNDARY_TAG,
                self.lines[2] == constants.FRACTURE_TAG,
            )
        ).ravel()
        frac_lines = self.lines[:, frac_ind]

        frac_id = frac_lines[3, :]
        if frac_id.size == 0:
            return str()
        range_id = np.arange(np.amin(frac_id), np.amax(frac_id) + 1)

        s = "// Start specification of fractures\n"
        seg_id = 0
        for i in range_id:
            local_seg_id = str()
            for mask in np.flatnonzero(frac_id == i):
                s += (
                    "frac_line_"
                    + str(seg_id)
                    + " = newl; "
                    + "Line(frac_line_"
                    + str(seg_id)
                    + ") = {"
                    + "p"
                    + str(int(frac_lines[0, mask]))
                    + ", p"
                    + str(int(frac_lines[1, mask]))
                    + "};\n"
                    + "Line{ frac_line_"
                    + str(seg_id)
                    + "} In Surface{domain_surf};\n"
                )
                local_seg_id += "frac_line_" + str(seg_id) + ", "
                seg_id += 1

            local_seg_id = local_seg_id[:-2]
            s += (
                'Physical Line("'
                + constants.PHYSICAL_NAME_FRACTURES
                + str(i)
                + '") = { '
                + local_seg_id
                + " };\n"
            )
            s += "\n"

        s += "// End of fracture specification\n\n"
        return s

    def __write_boundary_2d(self):
        constants = gridding_constants.GmshConstants()
        bound_line_ind = np.argwhere(
            self.lines[2] == constants.DOMAIN_BOUNDARY_TAG
        ).ravel()
        bound_line = self.lines[:2, bound_line_ind]
        bound_line = sort_points.sort_point_pairs(bound_line, check_circular=True)

        s = "// Start of specification of domain"
        s += "// Define lines that make up the domain boundary\n"

        loop_str = "{"
        for i in range(bound_line.shape[1]):
            s += "bound_line_" + str(i) + " = newl; Line(bound_line_" + str(i) + ") ={"
            s += (
                "p"
                + str(int(bound_line[0, i]))
                + ", p"
                + str(int(bound_line[1, i]))
                + "};\n"
            )
            loop_str += "bound_line_" + str(i) + ", "

        s += "\n"
        loop_str = loop_str[:-2]  # Remove last comma
        loop_str += "};\n"
        s += "// Line loop that makes the domain boundary\n"
        s += "Domain_loop = newll;\n"
        s += "Line Loop(Domain_loop) = " + loop_str
        s += "domain_surf = news;\n"
        s += "Plane Surface(domain_surf) = {Domain_loop};\n"
        s += (
            'Physical Surface("'
            + constants.PHYSICAL_NAME_DOMAIN
            + '") = {domain_surf};\n'
        )
        s += "// End of domain specification\n\n"
        return s

    def __write_boundary_3d(self):
        ls = "\n"
        s = "// Start domain specification" + ls
        # Write surfaces:
        s += self.__write_polygons(boundary=True)

        # Make a box out of them
        s += "domain_loop = newsl;" + ls
        s += "Surface Loop(domain_loop) = {"
        for pi in range(len(self.polygons[0])):
            if self.polygon_tags["boundary"][pi]:
                s += "auxiliary_" + str(pi) + ","
        s = s[:-1]
        s += "};" + ls
        s += "Volume(1) = {domain_loop};" + ls
        s += (
            'Physical Volume("'
            + gridding_constants.GmshConstants().PHYSICAL_NAME_DOMAIN
            + '") = {1};'
            + ls
        )

        s += "// End of domain specification\n\n"
        return s

    def __write_points(self, boundary=False):
        p = self.pts
        num_p = p.shape[1]
        if p.shape[0] == 2:
            p = np.vstack((p, np.zeros(num_p)))
        s = "// Define points\n"
        for i in range(self.pts.shape[1]):
            s += "p" + str(i) + " = newp; Point(p" + str(i) + ") = "
            s += "{" + str(p[0, i]) + ", " + str(p[1, i]) + ", " + str(p[2, i])
            if self.mesh_size is not None:
                s += ", " + str(self.mesh_size[i]) + " };\n"
            else:
                s += "};\n"

        s += "// End of point specification\n\n"
        return s

    def __write_lines(self, embed_in=None):
        l = self.lines
        num_lines = l.shape[1]
        ls = "\n"
        s = "// Define lines " + ls
        constants = gridding_constants.GmshConstants()
        if l.shape[0] > 2:
            lt = l[2]
            has_tags = True
        else:
            has_tags = False
        for i in range(num_lines):
            si = str(i)
            s += (
                "frac_line_"
                + si
                + "= newl; Line(frac_line_"
                + si
                + ") = {p"
                + str(l[0, i])
                + ", p"
                + str(l[1, i])
                + "};"
                + ls
            )
            if has_tags:
                s += 'Physical Line("'
                if l[2, i] == constants.FRACTURE_TIP_TAG:
                    s += constants.PHYSICAL_NAME_FRACTURE_TIP
                elif l[2, i] == constants.FRACTURE_INTERSECTION_LINE_TAG:
                    s += constants.PHYSICAL_NAME_FRACTURE_LINE
                else:
                    # This is a line that need not be physical (recognized by
                    # the parser of output from gmsh). Applies to boundary and
                    # subdomain boundary lines.
                    s += constants.PHYSICAL_NAME_AUXILIARY_LINE

                s += si + '") = {frac_line_' + si + "};" + ls

            s += ls
        s += "// End of line specification " + ls + ls
        return s

    def __write_polygons(self, boundary=False):
        """
        Writes either all fractures or all boundary planes.
        """
        constants = gridding_constants.GmshConstants()
        bound_tags = self.polygon_tags.get("boundary", [False] * len(self.polygons[0]))
        subd_tags = self.polygon_tags.get("subdomain", [False] * len(self.polygons[0]))

        ls = "\n"
        # Name boundary or fracture
        f_or_b = "auxiliary" if boundary else "fracture"
        if not boundary:
            s = "// Start fracture specification" + ls
        else:
            s = ""
        for pi in range(len(self.polygons[0])):
            if bound_tags[pi] != boundary:
                continue
            # Check if the polygon is a subdomain boundary, i.e., auxiliary
            # polygon.
            auxiliary = subd_tags[pi]
            if auxiliary:
                # Keep track of "fake fractures", i.e., subdomain
                # boundaries.
                f_or_b = "auxiliary"
            p = self.polygons[0][pi].astype("int")
            reverse = self.polygons[1][pi]
            # First define line loop
            s += "frac_loop_" + str(pi) + " = newll; " + ls
            s += "Line Loop(frac_loop_" + str(pi) + ") = { "
            for i, li in enumerate(p):
                if reverse[i]:
                    s += "-"
                s += "frac_line_" + str(li)
                if i < p.size - 1:
                    s += ", "

            s += "};" + ls

            n = f_or_b + "_"
            # Then the surface
            s += n + str(pi) + " = news; "
            s += (
                "Plane Surface(" + n + str(pi) + ") = {frac_loop_" + str(pi) + "};" + ls
            )

            if bound_tags[pi] or auxiliary:
                # Domain boundary or "fake fracture" = subdomain boundary
                s += (
                    'Physical Surface("'
                    + constants.PHYSICAL_NAME_AUXILIARY
                    + str(pi)
                    + '") = {auxiliary_'
                    + str(pi)
                    + "};"
                    + ls
                )

            else:
                # Normal fracture
                s += (
                    'Physical Surface("'
                    + constants.PHYSICAL_NAME_FRACTURES
                    + str(pi)
                    + '") = {fracture_'
                    + str(pi)
                    + "};"
                    + ls
                )
                if self.domain is not None:
                    s += "Surface{" + n + str(pi) + "} In Volume{1};" + ls + ls

            for li in self.e2f[pi]:
                s += "Line{frac_line_" + str(li) + "} In Surface{" + n
                s += str(pi) + "};" + ls
            s += ls

        if not boundary:
            s += "// End of fracture specification" + ls + ls

        return s

    def __write_physical_points(self):
        ls = "\n"
        s = "// Start physical point specification" + ls

        constants = gridding_constants.GmshConstants()

        for i, p in enumerate(self.intersection_points):
            s += (
                'Physical Point("'
                + constants.PHYSICAL_NAME_FRACTURE_POINT
                + str(i)
                + '") = {p'
                + str(p)
                + "};"
                + ls
            )
        s += "// End of physical point specification" + ls + ls
        return s

    def __write_meshing_algorithm(self):
        # See: http://www.manpagez.com/info/gmsh/gmsh-2.4.0/gmsh_76.php
        if self.meshing_algorithm is None:
            return ""
        else:
            return "\nMesh.Algorithm = " + str(self.meshing_algorithm) + ";"


# ----------- end of GmshWriter ----------------------------------------------


class GmshGridBucketWriter(object):
    """
    Dump a grid bucket to a gmsh .msh file, to be read by other software.

    The function assumes that the grid consists of simplices, and error
    messages will be raised if otherwise. The extension should not be
    difficult, but the need has not been there yet.

    All grids in all dimensions will have a separate physical name (in the gmsh
    sense), on the format GRID_#ID_DIM_#DIMENSION. Here #ID is the index of
    the corresponding node in the grid bucket, as defined by
    gb.assign_node_ordering. #DIMENSION is the dimension of the grid.

    """

    def __init__(self, gb):
        """
        Parameters:
            gb (gridding.grid_bucket): Grid bucket to be dumped.

        """
        self.gb = gb

        # Assign ordering of the nodes in gb - used for unique identification
        # of each grid
        gb.assign_node_ordering()

        # Compute number of grids in each dimension of the gb
        self._num_grids()

    def write(self, file_name):
        """
        Write the whole bucket to a .msh file.

        Parameters:
            file_name (str): Name of dump file.

        """
        s = self._preamble()
        s += self._physical_names()
        s += self._points()
        s += self._elements()

        with open(file_name, "w") as f:
            f.write(s)

    def _preamble(self):
        # Write the preamble (mesh Format) section
        s_preamble = "$MeshFormat\n"
        s_preamble += "2.2 0 8\n"
        s_preamble += "$EndMeshFormat\n"
        return s_preamble

    def _num_grids(self):
        # Find number of grids in each dimension
        max_dim = 3
        num_grids = np.zeros(max_dim + 1, dtype="int")
        for dim in range(max_dim + 1):
            num_grids[dim] = len(self.gb.grids_of_dimension(dim))

            # Identify the highest dimension
        while num_grids[-1] == 0:
            num_grids = num_grids[:-1]

            # We will pick the global point set from the highest dimensional
            # grid. The current implementation assumes there is a single grid
            # in that dimension. Taking care of multiple grids should not be
            # difficult, but it has not been necessary up to know.
            if num_grids[-1] != 1:
                raise NotImplementedError(
                    "Have not considered several grids\
                                          in the highest dimension"
                )
        self.num_grids = num_grids

    def _points(self):
        # The global point set
        p = self.gb.grids_of_dimension(len(self.num_grids) - 1)[0].nodes

        ls = "\n"
        s = "$Nodes" + ls
        s += str(p.shape[1]) + ls
        for i in range(p.shape[1]):
            s += (
                str(i + 1)
                + " "
                + str(p[0, i])
                + " "
                + str(p[1, i])
                + " "
                + str(p[2, i])
                + ls
            )
        s += "$EndNodes" + ls
        return s

    def _physical_names(self):
        ls = "\n"
        s = "$PhysicalNames" + ls

        # Use one physical name for each grid (including highest dimensional
        # one)
        s += str(self.gb.size()) + ls
        for i, g in enumerate(self.gb):
            dim = g[0].dim
            s += (
                str(dim)
                + " "
                + str(i + 1)
                + " "
                + "GRID_"
                + str(g[1]["node_number"])
                + "_DIM_"
                + str(dim)
                + ls
            )
        s += "$EndPhysicalNames" + ls
        return s

    def _elements(self):
        ls = "\n"
        s = "$Elements" + ls

        num_cells = 0
        for g, _ in self.gb:
            num_cells += g.num_cells
        s += str(num_cells) + ls

        # Element types (as specified by the gmsh .msh format), index by
        # dimensions. This assumes all cells are simplices.
        elem_type = [15, 1, 2, 4]
        for i, gr in enumerate(self.gb):
            g = gr[0]
            gn = str(gr[1]["node_number"])

            # Sanity check - all cells should be simplices
            assert np.all(np.diff(g.cell_nodes().indptr) == g.dim + 1)

            # Write the cell-node relation as an num_cells x dim+1 array
            cn = g.cell_nodes().indices.reshape((g.num_cells, g.dim + 1))

            et = str(elem_type[g.dim])
            for ci in range(g.num_cells):
                s += str(ci + 1) + " " + et + " " + str(1) + " " + gn + " "
                # There may be underlaying assumptions in gmsh on the ordering
                # of nodes.
                for d in range(g.dim + 1):
                    # Increase vertex offset by 1
                    s += str(cn[ci, d] + 1) + " "
                s += ls

        s += "$EndElements" + ls
        return s


# ------------------ End of GmshGridBucketWriter------------------------------


def run_gmsh(in_file, out_file, dims, **kwargs):
    """
    Convenience function to run gmsh.

    Parameters:
        in_file (str): Name of gmsh configuration file (.geo)
        out_file (str): Name of output file for gmsh (.msh)
        dims (int): Number of dimensions gmsh should grid. If dims is less than
            the geometry dimensions, gmsh will grid all lower-dimensional
            objcets described in in_file (e.g. all surfaces embeded in a 3D
            geometry).
        **kwargs: Options passed on to gmsh. See gmsh documentation for
            possible values.

    Returns:
        double: Status of the generation, as returned by os.system. 0 means the
            simulation completed successfully, >0 signifies problems.

    """
    # Import config file to get location of gmsh executable.
    config = read_config.read()
    path_to_gmsh = config["gmsh_path"]

    opts = " "
    for key, val in kwargs.items():
        # Gmsh keywords are specified with prefix '-'
        if key[0] != "-":
            key = "-" + key
        opts += key + " " + str(val) + " "

    if dims == 2:
        cmd = path_to_gmsh + " -2 " + in_file + " -o " + out_file + opts
    else:
        cmd = path_to_gmsh + " -3 " + in_file + " -o " + out_file + opts

    status = os.system(cmd)

    return status
