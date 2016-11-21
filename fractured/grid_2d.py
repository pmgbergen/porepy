
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

