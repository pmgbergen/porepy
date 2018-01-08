import numpy as np

from porepy.numerics import elliptic
from porepy.numerics import parabolic
from porepy.params import bc, tensor

from porepy.numerics import darcy_and_transport

#------------------------------------------------------------------------------#

def assign_data(gb, data_problem, data_class, data_key):
    """
    Loop over grids to assign flow problem data.
    Darcy data_key: problem
    Transport data_key: transport_data
    """
    gb.add_node_props([data_key])
    for g, d in gb:
        d[data_key] = data_class(g, d, **data_problem)

#------------------------------------------------------------------------------#

def get_dir_boundary(problem):
    bound_faces = problem.grid().get_domain_boundary_faces()
    bound_face_centers = problem.grid().face_centers[:, bound_faces]

    dir_boundary = {}
    if bound_faces.size == 0:
        return dir_boundary

    if 'left' in problem.dirichlet_boundary:
        left = bound_face_centers[0, :] < problem.domain['xmin'] + problem.tol
        dir_boundary['left'] = left
    if 'right' in problem.dirichlet_boundary:
        right = bound_face_centers[0, :] > problem.domain['xmax'] - problem.tol
        dir_boundary['right'] = right
    if 'top' in problem.dirichlet_boundary:
        top = bound_face_centers[1, :] > problem.domain['ymax'] - problem.tol
        dir_boundary['top'] = top
    if 'bottom' in problem.dirichlet_boundary:
        bottom = bound_face_centers[1, :] < problem.domain['ymin'] + problem.tol
        dir_boundary['bottom'] = bottom

    return dir_boundary

#------------------------------------------------------------------------------#

class FlowData(elliptic.EllipticDataAssigner):
    """
    Assign flow problem data to a given grid.
    """

    def __init__(self, g, d, **kwargs):

        self.tol = kwargs.get('tol', 1e-6)
        self.dim_max = kwargs.get('dim_max', 2)
        self.domain = kwargs['domain']

        self.kf = kwargs['kf']
        self.km = kwargs['km']

        self.reference_aperture = kwargs['aperture']

        self.dirichlet_boundary = kwargs['dirichlet_boundary']

        elliptic.EllipticDataAssigner.__init__(self, g, d)

    def aperture(self):
        a = np.power(self.reference_aperture, self.dim_max - self.grid().dim)
        return np.ones(self.grid().num_cells) * a

    def permeability(self):
        ones = np.ones(self.grid().num_cells)
        if self.grid().dim == self.dim_max:
            return tensor.SecondOrder(3, self.km*ones)
        else:
            return tensor.SecondOrder(3, self.kf*ones)

    def bc(self):
        bound_faces = self.grid().get_domain_boundary_faces()
        dir_boundary = get_dir_boundary(self)

        labels = np.array(['neu']*bound_faces.size)
        for _, face_ids in dir_boundary.items():
            labels[face_ids] = 'dir'

        return bc.BoundaryCondition(self.grid(), bound_faces, labels)

    def bc_val(self):
        bound_faces = self.grid().get_domain_boundary_faces()
        dir_boundary = get_dir_boundary(self)

        bc_val = np.zeros(self.grid().num_faces)
        for boundary, face_ids in dir_boundary.items():
            bc_val[bound_faces[face_ids]] = self.dirichlet_boundary[boundary]

        return bc_val

#------------------------------------------------------------------------------#

class TransportData(parabolic.ParabolicDataAssigner):
    """
    Assign transport problem data to given grid.
    """

    def __init__(self, g, d, **kwargs):

        self.tol = kwargs.get('tol', 1e-6)
        self.dim_max = kwargs.get('dim_max', 2)
        self.domain = kwargs['domain']

        self.initial = kwargs.get('initial_condition', 0)
        self.reference_aperture = kwargs['aperture']

        self.dirichlet_boundary = kwargs['dirichlet_boundary']

        parabolic.ParabolicDataAssigner.__init__(self, g, d)

    def bc(self):
        bound_faces = self.grid().get_domain_boundary_faces()
        dir_boundary = get_dir_boundary(self)

        labels = np.array(['neu']*bound_faces.size)
        for _, face_ids in dir_boundary.items():
            labels[face_ids] = 'dir'

        return bc.BoundaryCondition(self.grid(), bound_faces, labels)

    def bc_val(self, t):
        bound_faces = self.grid().get_domain_boundary_faces()
        dir_boundary = get_dir_boundary(self)

        bc_val = np.zeros(self.grid().num_faces)
        for boundary, face_ids in dir_boundary.items():
            bc_val[bound_faces[face_ids]] = 1

        return bc_val

    def initial_condition(self):
        return self.initial*np.ones(self.grid().num_cells)

    def aperture(self):
        a = np.power(self.reference_aperture, self.dim_max - self.grid().dim)
        return np.ones(self.grid().num_cells) * a

#------------------------------------------------------------------------------#

class TransportSolver(parabolic.ParabolicModel):
    """
    Make a ParabolicModel for the transport problem with specified parameters.
    """

    def __init__(self, gb, **data_problem):

        time_step = data_problem['end_time']/data_problem['number_time_steps']
        parabolic.ParabolicModel.__init__(self, gb, time_step=time_step,
                                          **data_problem)
        self._solver.parameters['store_results'] = True

    def space_disc(self):
        return self.advective_disc()

#------------------------------------------------------------------------------#

class Simulation(darcy_and_transport.DarcyAndTransport):
    """
    Combine the two problems for convenience.
    """

    def __init__(self, gb, data_problem):

        # Darcy problem
        assign_data(gb, data_problem, FlowData, 'problem')
        flow = elliptic.EllipticModel(gb, **data_problem)

        # transport problem
        assign_data(gb, data_problem, TransportData, 'transport_data')
        transport = TransportSolver(gb, **data_problem)

        darcy_and_transport.DarcyAndTransport.__init__(self, flow, transport)

#------------------------------------------------------------------------------#
