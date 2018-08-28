class GmshConstants(object):
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
        # Tag for objects on the fracture tip
        self.FRACTURE_TIP_TAG = 4
        # Tag for objcets on the intersection between two fractures
        # (co-dimension n-2)
        self.FRACTURE_INTERSECTION_LINE_TAG = 5
        # Tag for objects on the intersection between three fractures
        # (co-dimension n-3)
        self.FRACTURE_INTERSECTION_POINT_TAG = 6
        # General auxiliary tag
        self.AUXILIARY_TAG = 7

        self.PHYSICAL_NAME_DOMAIN = "DOMAIN"
        self.PHYSICAL_NAME_FRACTURES = "FRACTURE_"
        self.PHYSICAL_NAME_AUXILIARY = "AUXILIARY_"

        # Physical name for fracture tips
        self.PHYSICAL_NAME_FRACTURE_TIP = "FRACTURE_TIP_"
        self.PHYSICAL_NAME_FRACTURE_LINE = "FRACTURE_LINE_"
        self.PHYSICAL_NAME_AUXILIARY_LINE = "AUXILIARY_LINE_"
        self.PHYSICAL_NAME_FRACTURE_POINT = "FRACTURE_POINT_"
