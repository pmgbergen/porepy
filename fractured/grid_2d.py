import numpy as np


class GriddingComponentsConstants(object):
    """
    This class is a container for storing constant values that are used in
    the meshing algorithm. The intention is to make them available to all
    functions and modules.

    This may not be the most pythonic way of doing this, but it works.
    """
    def __init__(self):
        self.DOMAIN_BOUNDARY_TAG = 1
        self.COMPARTMENT_BOUNDARY_TAG = 2
        self.FRACTURE_TAG = 3

        self.PHYSICAL_NAME_DOMAIN = 'DOMAIN'
        self.PHYSICAL_NAME_FRACTURES = 'FRACTURE_'


def __merge_fracs_compartments_domain_2d(dom, frac_p, frac_l, comp):
    """
    Merge fractures, domain boundaries and lines for compartments.
    The unified description is ready for feeding into meshing tools such as
    gmsh

    Parameters:
    dom: dictionary defining domain. fields xmin, xmax, ymin, ymax
    frac_p: np.ndarray. Points used in fracture definition. 2 x num_points.
    frac_l: np.ndarray. Connection between fracture points. 2 x num_fracs
    comp: Compartment lines

    returns:
    p: np.ndarary. Merged list of points for fractures, compartments and domain
        boundaries.
    l: np.ndarray. Merged list of line connections (first two rows), tag
        identifying which type of line this is (third row), and a running index
        for all lines (fourth row)
    tag_description: list describing the tags, that is, row 3 and 4 in l
    """

    # Use constants set outside. If we ever
    constants = GriddingComponentsConstants()

    # First create lines that define the domain
    x_min = dom['xmin']
    x_max = dom['xmax']
    y_min = dom['ymin']
    y_max = dom['ymax']
    dom_p = np.array([[x_min, x_max, x_max, x_min],
                      [y_min, y_min, y_max, y_max]])
    dom_lines = np.array([[0, 1], [1, 2], [2, 3], [3, 0]]).T

    num_dom_lines = dom_lines.shape[1]  # Should be 4

    # Then the compartments
    # First ignore compartment lines that have no y-component
    comp = np.squeeze(comp[:, np.argwhere(np.abs(comp[1]) > 1e-5)])

    num_comps = comp.shape[1]
    # Assume that the compartments are not intersecting at the boundary.
    # This may need revisiting when we allow for general lines
    # (not only y=const)
    comp_p = np.zeros((2, num_comps * 2))

    for i in range(num_comps):
        # The line is defined as by=1, thus y = 1/b
        comp_p[:, 2 * i: 2 * (i + 1)] = np.array([[x_min, x_max],
                                                  [1. / comp[1, i],
                                                   1. / comp[1, i]]])

    # Define compartment lines
    comp_lines = np.arange(2 * num_comps, dtype='int').reshape((2, -1),
                                                               order='F')

    # The  lines will have all fracture-related tags set to zero.
    # The plan is to ignore these tags for the boundary and compartments,
    # so it should not matter
    dom_tags = constants.DOMAIN_BOUNDARY_TAG * np.ones((1, num_dom_lines))
    dom_l = np.vstack((dom_lines, dom_tags))
    comp_tags = constants.COMPARTMENT_BOUNDARY_TAG * np.ones((1, num_comps))
    comp_l = np.vstack((comp_lines, comp_tags))

    # Also add a tag to the fractures, signifying that these are fractures
    frac_l = np.vstack((frac_l,
                        constants.FRACTURE_TAG * np.ones(frac_l.shape[1])))

    # Merge the point arrays, compartment points first
    p = np.hstack((dom_p, comp_p, frac_p))

    # Adjust index of compartment points to account for domain points coming
    # before them in the list
    comp_l[:2] += dom_p.shape[1]

    # Adjust index of fracture points to account for the compart points
    frac_l[:2] += dom_p.shape[1] + comp_p.shape[1]

    l = np.hstack((dom_l, comp_l, frac_l)).astype('int')

    # Add a second tag as an identifier of each line.
    l = np.vstack((l, np.arange(l.shape[1])))

    tag_description = ['Line type', 'Line number']

    return p, l, tag_description

