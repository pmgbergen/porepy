# Methods to work directly with the gmsh format

import numpy as np
from compgeom import sort_points
# from ..fractured import grid_2d
from gridding.gmsh import mesh_io
import sys
import os
import gridding.constants as gridding_constants

class GmshWriter(object):
    """
     Write a gmsh.geo file for a fractured 2D domains, possibly including
     compartments
    """

    def __init__(self, pts, lines, domain=None, nd=None, lchar=None,
                 lchar_bound=None):
        """

        :param pts: np.ndarary, Points
        :param lines: np.ndarray. Non-intersecting lines in the geometry.
        :param nd: Dimension. Inferred from points if not provided
        """
        self.pts = pts
        self.lines = lines
        if nd is None:
            if pts.shape[0] == 2:
                self.nd = 2
            elif pts.shape[0] == 3:
                self.nd = 3
        else:
            self.nd = nd

        if lchar is None:
            self.lchar = 1 * np.ones(self.pts.shape[1])
        else:
            self.lchar = lchar

        if domain is not None:
            self.domain = domain

        if lchar_bound is None:
            self.lchar_bound = 1


    def write_geo(self, file_name):
        s = self.__write_points()

        if self.nd == 2:
            s += self.__write_boundary_2d()
            s += self.__write_fractures_compartments_2d()
        elif self.nd == 3:
            s += self.__write_boundary_3d()

        with open(file_name, 'w') as f:
            f.write(s)

    def __write_fractures_compartments_2d(self):
        # Both fractures and compartments are
        constants = gridding_constants.GmshConstants()

        frac_ind = np.argwhere(np.logical_or(
            self.lines[2] == constants.COMPARTMENT_BOUNDARY_TAG,
            self.lines[2] == constants.FRACTURE_TAG)).ravel()
        frac_lines = self.lines[:, frac_ind]

        s = '// Start specification of fractures \n'
        for i in range(frac_ind.size):
            s += 'frac_line_' + str(i) + ' = newl; Line(frac_line_' + str(i) \
                 + ') ={'
            s += 'p' + str(int(frac_lines[0, i])) + ', p' \
                 + str(int(frac_lines[1, i])) + '}; \n'
            s += 'Physical Line(\"' + constants.PHYSICAL_NAME_FRACTURES \
                 + str(i) + '\") = { frac_line_' + str(i) + ' };\n'
            s += 'Line{ frac_line_' + str(i) + '} In Surface{domain_surf}; \n'
            s += '\n'

        s += '// End of fracture specification \n\n'
        return s

    def __write_boundary_2d(self):
        constants = gridding_constants.GmshConstants()
        bound_line_ind = np.argwhere(self.lines[2] ==
                                     constants.DOMAIN_BOUNDARY_TAG).ravel()
        bound_line = self.lines[:2, bound_line_ind]
        bound_line = sort_points.sort_point_pairs(bound_line,
                                                  check_circular=True)

        s = '// Start of specification of domain'
        s += '// Define lines that make up the domain boundary \n'

        loop_str = '{'
        for i in range(bound_line.shape[1]):
            s += 'bound_line_' + str(i) + ' = newl; Line(bound_line_'\
                 + str(i) + ') ={'
            s += 'p' + str(int(bound_line[0, i])) + ', p' + \
                 str(int(bound_line[1, i])) + '}; \n'
            loop_str += 'bound_line_' + str(i) + ', '

        s += '\n'
        loop_str = loop_str[:-2]  # Remove last comma
        loop_str += '}; \n'
        s += '// Line loop that makes the domain boundary \n'
        s += 'Domain_loop = newll; \n'
        s += 'Line Loop(Domain_loop) = ' + loop_str
        s += 'domain_surf = news; \n'
        s += 'Plane Surface(domain_surf) = {Domain_loop}; \n'
        s += 'Physical Surface(\"' + constants.PHYSICAL_NAME_DOMAIN + \
             '\") = {domain_surf}; \n'
        s += '// End of domain specification \n \n'
        return s

    def __write_boundary_3d(self):
        # Write the bounding box in 3D
        # Pull out bounding coordinates
        xmin = str(self.domain['xmin']) + ', '
        xmax = str(self.domain['xmax']) + ', '
        ymin = str(self.domain['ymin']) + ', '
        ymax = str(self.domain['ymax']) + ', '
        zmin = str(self.domain['zmin']) + ', '
        zmax = str(self.domain['zmax']) + ', '

        h = str(self.lchar_bound) + '};'
        ls = '\n'

        s = '// Define bounding box \n'

        # Points in bottom of box
        s += 'p_bound_000 = newp; Point(p_bound_000) = {'
        s += xmin + ymin + zmin + h + ls
        s += 'p_bound_100 = newp; Point(p_bound_100) = {'
        s += xmax+ ymin + zmin + h + ls
        s += 'p_bound_110 = newp; Point(p_bound_110) = {'
        s += xmax + ymax + zmin + h + ls
        s += 'p_bound_010 = newp; Point(p_bound_010) = {'
        s += xmin + ymax + zmin + h + ls
        s += ls

        # Lines connecting points
        s += 'bound_line_1 = newl; Line(bound_line_1) = { p_bound_000, \
            p_bound_100};' + ls
        s += 'bound_line_2 = newl; Line(bound_line_2) = { p_bound_100, \
            p_bound_110};' + ls
        s += 'bound_line_3 = newl; Line(bound_line_3) = { p_bound_110, \
            p_bound_010};' + ls
        s += 'bound_line_4 = newl; Line(bound_line_4) = { p_bound_010, \
            p_bound_000};' +ls
        s += 'bottom_loop = newll;' + ls
        s += 'Line Loop(bottom_loop) = {bound_line_1, bound_line_2, \
                bound_line_3, bound_line_4};' + ls
        s += 'bottom_surf = news;' + ls
        s += 'Plane Surface(bottom_surf) = {bottom_loop};' + ls

        dz = self.domain['zmax'] - self.domain['zmin']
        s += 'Extrude {0, 0, ' + str(dz) + '} {Surface{bottom_surf}; }' + ls
        s += '// End of domain specification ' + ls + ls

        return s

    def __write_points(self):
        p = self.pts
        num_p = p.shape[1]
        if p.shape[0] == 2:
            p = np.vstack((p, np.zeros(num_p)))
        s = '// Define points \n'
        for i in range(self.pts.shape[1]):
            s += 'p' + str(i) + ' = newp; Point(p' + str(i) + ') = '
            s += '{' + str(p[0, i]) + ', ' + str(p[1, i]) + ', '\
                 + str(p[2, i]) + ', ' + str(self.lchar[i]) + ' };\n'
        s += '// End of point specification \n \n'
        return s



def read_gmsh(out_file):
    points, cells, phys_names, cell_info = mesh_io.read(out_file)
    return points, cells, phys_names, cell_info


def run_gmsh(path_to_gmsh, in_file, out_file, dims):
    """
    Convenience function to run gmsh.

    TODO: Add possibility of including options for gmsh.

    Parameters:
        path_to_gmsh (str): Path to the location of the gmsh binary
        in_file (str): Name of gmsh configuration file (.geo)
        out_file (str): Name of output file for gmsh (.msh)
        dims (int): Number of dimensions gmsh should grid. If dims is less than
            the geometry dimensions, gmsh will grid all lower-dimensional
            objcets described in in_file (e.g. all surfaces embeded in a 3D
            geometry).

    Returns:
        double: Status of the generation, as returned by os.system. 0 means the
            simulation completed successfully, >0 signifies problems.

    """

    if dims == 2:
        cmd = path_to_gmsh + ' -2 ' + in_file + ' -o ' + out_file
    else:
        cmd = path_to_gmsh + ' -3 ' + in_file + ' -o ' + out_file
    status = os.system(cmd)

    return status

