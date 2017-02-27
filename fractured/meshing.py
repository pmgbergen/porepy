"""

Main module for grid generation in fractured domains in 2d and 3d.

The module serves as the only neccessary entry point to create the grid. It
will therefore wrap interface to different mesh generators, pass options to the
generators etc.

"""
import numpy as np

import grid_2d, grid_3d


def create_grid(fracs, domain, **kwargs):
    """
    Main function for grid generation.

    Parameters:
        fracs (list of np.ndarray): One list item for each fracture. Each item
            consist of a (nd x n) array describing fracture vertices. The
            fractures may be intersecting.
        domain (dict): Domain specification, determined by
        **kwargs: May contain fracture tags, options for gridding, etc.

    """

    ndim = fracs.shape[0]

    # Call relevant method, depending on grid dimensions
    # Note: If we ever develop interfaces to grid generators other than gmsh,
    # this should not be visible here, but rather in the respective
    # nd.create_grid methods.
    if ndim == 2:
        # This will fail, either change method parameters, or process data.
        grid_2d.create_grid(fracs)
    elif ndim == 3:
        grid_3d.create_grid(fracs, domain)

    # Somehow take care of the output, and return appropriate values.


