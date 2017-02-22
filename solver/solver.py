class Solver(object):

#------------------------------------------------------------------------------#

    def matrix_rhs(self, g, data):
        raise NotImplementedError("Method not implemented")

#------------------------------------------------------------------------------#

    def ndof(self, g):
        raise NotImplementedError("Method not implemented")

#------------------------------------------------------------------------------#
