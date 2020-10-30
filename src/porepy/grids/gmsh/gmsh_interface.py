# Methods to work directly with the gmsh format

from pathlib import Path
from typing import List, Union

import numpy as np

import porepy.grids.constants as gridding_constants
from porepy.utils import sort_points

try:
    import gmsh
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "To run gmsh python api on your system, "
        "download the relevant gmsh*-sdk.* from http://gmsh.info/bin/. "
        "Then, Add the 'lib' directory from the SDK to PYTHONPATH: \n"
        "export PYTHONPATH=${PYTHONPATH}:path/to/gmsh*-sdk.*/lib"
    )


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
        fracture_tags=None,
        domain_boundary_points=None,
        fracture_and_boundary_points=None,
        fracture_constraint_intersection_points=None,
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

        self.domain_boundary_points = domain_boundary_points
        self.fracture_and_boundary_points = fracture_and_boundary_points
        self.fracture_constraint_intersection_points = (
            fracture_constraint_intersection_points
        )

    def write_geo(self, file_name):

        if self.tolerance is not None:
            s = "Geometry.Tolerance = " + str(self.tolerance) + ";\n"
        else:
            s = "\n"

        # Define the points
        s += self._write_points()

        if self.nd == 1:
            s += self._write_fractures_1d()

        elif self.nd == 2:
            if self.domain is not None:
                s += self._write_boundary_2d()
            s += self._write_fractures_2d()

        elif self.nd == 3:
            # First, define the edges that are combined into polygons
            s += self._write_lines()
            # Write domain boundary if relevant. This is a combination of the edges
            if self.domain is not None:
                s += self._write_boundary_3d()
            # Finally, write the fractures
            s += self._write_polygons()

        s += self._write_physical_points()

        with open(file_name, "w") as f:
            f.write(s)

    def _write_fractures_1d(self):
        # Both fractures and compartments are
        constants = gridding_constants.GmshConstants()

        # We consider fractures, boundary tag, an auxiliary tag (fake fractures/mesh
        # constraints)
        ind = np.argwhere(
            np.logical_or.reduce(
                (
                    self.lines[2] == constants.COMPARTMENT_BOUNDARY_TAG,
                    self.lines[2] == constants.FRACTURE_TAG,
                    self.lines[2] == constants.AUXILIARY_TAG,
                )
            )
        ).ravel()
        lines = self.lines[:, ind]
        tag = self.lines[2, ind]

        lines_id = lines[3, :]
        if lines_id.size == 0:
            return str()
        range_id = np.arange(np.amin(lines_id), np.amax(lines_id) + 1)

        s = "// Start specification of fractures/compartment boundary/auxiliary elements\n"
        seg_id = 0
        for i in range_id:
            local_seg_id = str()
            for mask in np.flatnonzero(lines_id == i):

                # give different name for fractures/boundary and auxiliary
                if tag[mask] != constants.AUXILIARY_TAG:
                    name = "frac_line_"
                    physical_name = constants.PHYSICAL_NAME_FRACTURES
                else:
                    name = "seg_line_"
                    physical_name = constants.PHYSICAL_NAME_AUXILIARY

                s += (
                    name
                    + str(seg_id)
                    + " = newl; "
                    + "Line("
                    + name
                    + str(seg_id)
                    + ") = {"
                    + "p"
                    + str(int(lines[0, mask]))
                    + ", p"
                    + str(int(lines[1, mask]))
                    + "};\n"
                )
                local_seg_id += name + str(seg_id) + ", "
                seg_id += 1

            local_seg_id = local_seg_id[:-2]
            s += (
                'Physical Line("'
                + physical_name
                + str(i)
                + '") = { '
                + local_seg_id
                + " };\n"
            )
            s += "\n"

        s += "// End of /compartment boundary/auxiliary elements specification\n\n"
        return s

    def _write_fractures_2d(self):
        # Both fractures and compartments are
        constants = gridding_constants.GmshConstants()

        # We consider fractures and boundary tag
        ind = np.argwhere(
            np.logical_or.reduce(
                (
                    self.lines[2] == constants.FRACTURE_TAG,
                    self.lines[2] == constants.AUXILIARY_TAG,
                )
            )
        ).ravel()
        lines = self.lines[:, ind]
        tag = self.lines[2, ind]

        lines_id = lines[3, :]
        if lines_id.size == 0:
            return str()
        range_id = np.arange(np.amin(lines_id), np.amax(lines_id) + 1)

        s = "// Start specification of fractures/compartment boundary/auxiliary elements\n"
        seg_id = 0
        for i in range_id:
            local_seg_id = str()
            for mask in np.flatnonzero(lines_id == i):

                # give different name for fractures/boundary and auxiliary
                if tag[mask] != constants.AUXILIARY_TAG:
                    name = "frac_line_"
                    physical_name = constants.PHYSICAL_NAME_FRACTURES
                else:
                    name = "seg_line_"
                    physical_name = constants.PHYSICAL_NAME_AUXILIARY

                s += (
                    name
                    + str(seg_id)
                    + " = newl; "
                    + "Line("
                    + name
                    + str(seg_id)
                    + ") = {"
                    + "p"
                    + str(int(lines[0, mask]))
                    + ", p"
                    + str(int(lines[1, mask]))
                    + "};\n"
                    + "Line{"
                    + name
                    + str(seg_id)
                    + "} In Surface{domain_surf};\n"
                )
                local_seg_id += name + str(seg_id) + ", "
                seg_id += 1

            local_seg_id = local_seg_id[:-2]
            s += (
                'Physical Line("'
                + physical_name
                + str(i)
                + '") = { '
                + local_seg_id
                + " };\n"
            )
            s += "\n"

        s += "// End of /compartment boundary/auxiliary elements specification\n\n"
        return s

    def _write_boundary_2d(self):
        constants = gridding_constants.GmshConstants()
        bound_line_ind = np.argwhere(
            self.lines[2] == constants.DOMAIN_BOUNDARY_TAG
        ).ravel()
        bound_line = self.lines[:, bound_line_ind]
        bound_line, _ = sort_points.sort_point_pairs(bound_line, check_circular=True)

        s = "// Start of specification of domain"
        s += "// Define lines that make up the domain boundary\n"

        bound_id = bound_line[3, :]
        range_id = np.arange(np.amin(bound_id), np.amax(bound_id) + 1)
        seg_id = 0

        loop_str = "{"
        for i in range_id:
            local_bound_id = str()
            for mask in np.flatnonzero(bound_id == i):
                s += "bound_line_" + str(seg_id) + " = newl;\n"
                s += "Line(bound_line_" + str(seg_id) + ") ={"
                s += (
                    "p"
                    + str(int(bound_line[0, mask]))
                    + ", p"
                    + str(int(bound_line[1, mask]))
                    + "};\n"
                )

                loop_str += "bound_line_" + str(seg_id) + ", "
                local_bound_id += "bound_line_" + str(seg_id) + ", "
                seg_id += 1

            local_bound_id = local_bound_id[:-2]
            s += (
                'Physical Line("'
                + constants.PHYSICAL_NAME_DOMAIN_BOUNDARY
                + str(i)
                + '") = { '
                + local_bound_id
                + " };\n"
            )

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

    def _write_boundary_3d(self):
        ls = "\n"
        s = "// Start domain specification" + ls
        # Write surfaces:
        s += self._write_polygons(boundary=True)

        # Make a box out of them
        s += "domain_loop = newsl;" + ls
        s += "Surface Loop(domain_loop) = {"
        for pi in range(len(self.polygons[0])):
            if self.polygon_tags["boundary"][pi]:
                s += " boundary_surface_" + str(pi) + ","
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

    def _write_points(self, boundary=False):
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

    def _write_lines(self, embed_in=None):
        lines = self.lines
        num_lines = lines.shape[1]
        ls = "\n"
        s = "// Define lines " + ls
        constants = gridding_constants.GmshConstants()
        if lines.shape[0] > 2:
            has_tags = True
        else:
            has_tags = False
        for i in range(num_lines):
            si = str(i)
            s += (
                "frac_line_"
                + si
                + " = newl; Line(frac_line_"
                + si
                + ") = {p"
                + str(lines[0, i])
                + ", p"
                + str(lines[1, i])
                + "};"
                + ls
            )
            if has_tags:
                s += 'Physical Line("'
                if lines[2, i] == constants.FRACTURE_TIP_TAG:
                    s += constants.PHYSICAL_NAME_FRACTURE_TIP
                elif lines[2, i] == constants.FRACTURE_INTERSECTION_LINE_TAG:
                    s += constants.PHYSICAL_NAME_FRACTURE_LINE
                elif lines[2, i] == constants.DOMAIN_BOUNDARY_TAG:
                    s += constants.PHYSICAL_NAME_DOMAIN_BOUNDARY
                elif lines[2, i] == constants.FRACTURE_LINE_ON_DOMAIN_BOUNDARY_TAG:
                    s += constants.PHYSICAL_NAME_FRACTURE_BOUNDARY_LINE
                else:
                    # This is a line that need not be physical (recognized by
                    # the parser of output from gmsh). Applies to boundary and
                    # subdomain boundary lines.
                    s += constants.PHYSICAL_NAME_AUXILIARY_LINE

                s += si + '") = {frac_line_' + si + "};" + ls

            s += ls
        s += "// End of line specification " + ls + ls
        return s

    def _write_polygons(self, boundary=False):
        """
        Writes either all fractures or all boundary planes.
        """
        constants = gridding_constants.GmshConstants()
        bound_tags = self.polygon_tags.get("boundary", [False] * len(self.polygons[0]))
        constraint_tags = self.polygon_tags["constraint"]

        ls = "\n"
        # Name boundary or fracture
        if not boundary:
            s = "// Start fracture specification" + ls
        else:
            s = "// Start boundary surface specification" + ls
        for pi in range(len(self.polygons[0])):
            if bound_tags[pi] != boundary:
                continue
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

            if boundary:
                surf_stem = "boundary_surface_"
            else:
                if constraint_tags[pi]:
                    surf_stem = "auxiliary_surface_"
                else:
                    surf_stem = "fracture_"

            # Then the surface
            s += surf_stem + str(pi) + " = news; "
            s += (
                "Plane Surface("
                + surf_stem
                + str(pi)
                + ") = {frac_loop_"
                + str(pi)
                + "};"
                + ls
            )

            if bound_tags[pi]:
                # Domain boundary
                s += (
                    'Physical Surface("'
                    + constants.PHYSICAL_NAME_DOMAIN_BOUNDARY_SURFACE
                    + str(pi)
                    + '") = {'
                    + surf_stem
                    + str(pi)
                    + "};"
                    + ls
                )
            elif constraint_tags[pi]:
                s += (
                    'Physical Surface("'
                    + constants.PHYSICAL_NAME_AUXILIARY
                    + str(pi)
                    + '") = {'
                    + surf_stem
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
                    + '") = {'
                    + surf_stem
                    + str(pi)
                    + "};"
                    + ls
                )
            if not bound_tags[pi] and self.domain is not None:
                s += "Surface{" + surf_stem + str(pi) + "} In Volume{1};" + ls + ls

            for li in self.e2f[pi]:
                s += "Line{frac_line_" + str(li) + "} In Surface{" + surf_stem
                s += str(pi) + "};" + ls
            s += ls

        if not boundary:
            s += "// End of fracture specification" + ls + ls

        return s

    def _write_physical_points(self):
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

        if self.domain_boundary_points is not None:
            for i, p in enumerate(self.domain_boundary_points):
                s += (
                    'Physical Point("'
                    + constants.PHYSICAL_NAME_BOUNDARY_POINT
                    + str(i)
                    + '") = {p'
                    + str(p)
                    + "};"
                    + ls
                )

        if self.fracture_and_boundary_points is not None:
            for i, p in enumerate(self.fracture_and_boundary_points):
                s += (
                    'Physical Point("'
                    + constants.PHYSICAL_NAME_FRACTURE_BOUNDARY_POINT
                    + str(i)
                    + '") = {p'
                    + str(p)
                    + "};"
                    + ls
                )

        if self.fracture_constraint_intersection_points is not None:
            for i, p in enumerate(self.fracture_constraint_intersection_points):
                s += (
                    'Physical Point("'
                    + constants.PHYSICAL_NAME_FRACTURE_CONSTRAINT_INTERSECTION_POINT
                    + str(i)
                    + '") = {p'
                    + str(p)
                    + "};"
                    + ls
                )

        s += "// End of physical point specification" + ls + ls
        return s


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
        for gr in self.gb:
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


def run_gmsh(in_file: Union[str, Path], out_file: Union[str, Path], dim: int) -> None:
    """Convenience function to run gmsh.

    Parameters:
        in_file : str or pathlib.Path
            Name of (or path to) gmsh configuration file (.geo)
        out_file : str or pathlib.Path
            Name of (or path to) output file for gmsh (.msh)
        dim : int
            Number of dimensions gmsh should grid. If dims is less than
            the geometry dimensions, gmsh will grid all lower-dimensional
            objects described in in_file (e.g. all surfaces embedded in a 3D
            geometry).

    """
    # Helper functions

    def _dump_gmsh_log(_log: List[str], in_file_name: Path) -> Path:
        """Write a gmsh log to file.

        Takes in the entire log and path to the in_file (from outer scope)
        Return name of the log file
        """
        debug_file_name = in_file_name.with_name(f"gmsh_log_{in_file_name.stem}.dbg")
        with debug_file_name.open(mode="w") as f:
            for _line in _log:
                f.write(_line + "\n")

        return debug_file_name

    # Ensure that in_file has extension .geo, out_file extension .msh
    in_file = Path(in_file).with_suffix(".geo")
    out_file = Path(out_file).with_suffix(".msh")

    if not in_file.is_file():
        raise FileNotFoundError(f"file {in_file!r} not found.")

    gmsh.initialize()

    # Experimentation indicated that the gmsh api failed to raise error values when
    # passed corrupted .geo files. To catch errors we therefore read the gmsh log, and
    # look for error messages.
    gmsh.logger.start()
    gmsh.open(str(in_file))

    # Look for errors
    log = gmsh.logger.get()
    for line in log:
        if "Error:" in line:
            fn = _dump_gmsh_log(log, in_file)
            raise ValueError(
                f"""Error when reading gmsh file {in_file}.
                        Gmsh log written to file {fn}"""
            )

    # Generate the mesh
    gmsh.model.mesh.generate(dim=dim)

    # Look for errors
    log = gmsh.logger.get()
    for line in log:
        if "Error:" in line:
            fn = _dump_gmsh_log(log, in_file)
            raise ValueError(
                f"Error in gmsh when generating mesh for {in_file}\n"
                f"Gmsh log written to file {fn}"
            )

    # The gmsh write should be safe for errors
    gmsh.write(str(out_file))
    # Done
    gmsh.finalize()
