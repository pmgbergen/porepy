import numpy as np

from porepy.numerics import elliptic
from porepy.numerics import parabolic
from porepy.params import bc, tensor
from porepy.fracs import importer

from porepy.numerics import darcy_and_transport

#------------------------------------------------------------------------------#

def create_grid(file_name, domain, mesh_size, tol):

    # pack the data for the grid generation
    mesh_kwargs = {'tol': tol}
    mesh_kwargs['mesh_size'] = {'mode': 'weighted', 'value': mesh_size,
                                'bound_value': mesh_size}

    # import the fractures and create the grid
    gb = importer.dfm_2d_from_csv(file_name, mesh_kwargs, domain)
    gb.compute_geometry()
    gb.assign_node_ordering()

    return gb

#------------------------------------------------------------------------------#

def assign_data(gb, data_problem, data_class, data_key):
    """
    Loop over grids to assign flow problem data.
    Darcy data_key: problem
    Transport data_key: transport_data
    """
    gb.add_node_props([data_key])

    cells = np.nan*np.ones(data_problem['wells'].shape[1])
    distance = np.inf*np.ones(cells.size)
    for g, _ in gb:
        cells_loc, distance_loc = assign_well(g, data_problem)
        print(cells_loc, distance_loc)
    fix_wells()

    for g, d in gb:
        d[data_key] = data_class(g, d, **data_problem)

#------------------------------------------------------------------------------#

def assign_well(g, data_problem):

    wells = data_problem['wells']
    if g.dim == 1 and not ('well' in g.tags):
        g.tags['well'] = -np.ones(g.num_cells)
        g.tags['well_distance'] = np.inf*np.ones(g.num_cells)

        cells, distance = g.closest_cell(wells, return_distance=True)
        g.tags['well'][cells] = np.arange(cells.size)
        g.tags['well_distance'][cells] = distance
        return cells, distance
    else:
        return np.nan*np.ones(wells.shape[1]), np.inf*np.ones(wells.shape[1])

#------------------------------------------------------------------------------#

def fix_wells():
    pass

#------------------------------------------------------------------------------#

class FlowData(elliptic.EllipticDataAssigner):
    """
    Assign flow problem data to a given grid.
    """

    def __init__(self, g, d, **kwargs):

        self.tol = kwargs.get('tol', 1e-6)
        self.domain = kwargs['domain']

        self.kf = kwargs['kf']
        self.km = kwargs['km']

        self.reference_aperture = kwargs['aperture']
        self.bc_pressure = kwargs['bc_pressure']

        elliptic.EllipticDataAssigner.__init__(self, g, d)

    def aperture(self):
        a = np.power(self.reference_aperture, 2-self.grid().dim)
        return np.ones(self.grid().num_cells) * a

    def permeability(self):
        ones = np.ones(self.grid().num_cells)
        if self.grid().dim == 2:
            return tensor.SecondOrder(3, self.km*ones)
        else:
            return tensor.SecondOrder(3, self.kf*ones)

    def bc(self):
        bound_faces = self.grid().tags['domain_boundary_faces'].nonzero()[0]
        labels = np.array(['dir']*bound_faces.size)
        return bc.BoundaryCondition(self.grid(), bound_faces, labels)

    def bc_val(self):
        bound_faces = self.grid().tags['domain_boundary_faces'].nonzero()[0]
        bc_val = np.zeros(self.grid().num_faces)
        bc_val[bound_faces] = self.bc_pressure
        return bc_val

    def source(self):
        return np.zeros(self.grid().num_cells)

#------------------------------------------------------------------------------#

class TransportData(parabolic.ParabolicDataAssigner):
    """
    Assign transport problem data to given grid.
    """

    def __init__(self, g, d, **kwargs):

        self.tol = kwargs.get('tol', 1e-6)
        self.domain = kwargs['domain']

        self.reference_aperture = kwargs['aperture']
        self.bc_temperature = kwargs['bc_temperature']

        parabolic.ParabolicDataAssigner.__init__(self, g, d)

    def bc(self):
        bound_faces = self.grid().tags['domain_boundary_faces'].nonzero()[0]
        labels = np.array(['dir']*bound_faces.size)
        return bc.BoundaryCondition(self.grid(), bound_faces, labels)

    def bc_val(self, t):
        bound_faces = self.grid().tags['domain_boundary_faces'].nonzero()[0]
        bc_val = np.zeros(self.grid().num_faces)
        bc_val[bound_faces] = self.bc_temperature
        return bc_val

    def initial_condition(self):
        return self.bc_temperature*np.ones(self.grid().num_cells)

    def aperture(self):
        a = np.power(self.reference_aperture, 2-self.grid().dim)
        return np.ones(self.grid().num_cells) * a

#------------------------------------------------------------------------------#

class TransportSolver(parabolic.ParabolicModel):
    """
    Make a ParabolicModel for the transport problem with specified parameters.
    """

    def __init__(self, gb, **data_problem):

        time_step = data_problem['end_time'] /\
                    float(data_problem['number_time_steps'])
        parabolic.ParabolicModel.__init__(self, gb, time_step=time_step,
                                          **data_problem)

    def space_disc(self):
        return self.advective_disc()

#------------------------------------------------------------------------------#

class Simulation(darcy_and_transport.DarcyAndTransport):
    """
    Combine the two problems for convenience.
    """

    def __init__(self, gb, data_problem):

        # find the wells

        # Darcy problem
        assign_data(gb, data_problem, FlowData, 'problem')
        data_problem['file_name'] = 'pressure'
        flow = elliptic.EllipticModel(gb, **data_problem)

        # transport problem
        assign_data(gb, data_problem, TransportData, 'transport_data')
        data_problem['file_name'] = 'temperature'
        transport = TransportSolver(gb, **data_problem)

        darcy_and_transport.DarcyAndTransport.__init__(self, flow, transport)

#------------------------------------------------------------------------------#
