import numpy as np
import porepy as pp

from porepy.numerics import darcy_and_transport

#------------------------------------------------------------------------------#

def create_grid(file_name, domain, mesh_size, tol):

    # pack the data for the grid generation
    mesh_kwargs = {'tol': tol}
    mesh_kwargs['mesh_size_frac'] = mesh_size
    kwargs = {'assign_fracture_id': True}

    # import the fractures and create the grid
    gb = pp.importer.dfm_2d_from_csv(file_name, mesh_kwargs, domain, **kwargs)
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

    wells_fracture = data_problem['wells_fracture']
    wells = data_problem['wells']
    for g, _ in gb:
        if g.dim == 1:
            well_id = np.isin(wells_fracture, g.tags['fracture_id'])
            if np.any(well_id):
                wells_id = np.where(well_id)[0]
                wells_cell = g.closest_cell(wells[:, wells_id])
            else:
                wells_cell, wells_id = [np.nan]*2
        else:
            wells_cell, wells_id = [np.nan]*2

        g.tags['wells_cell'] = wells_cell
        g.tags['wells_id'] = wells_id

    for g, d in gb:
        d[data_key] = data_class(g, d, **data_problem)

#------------------------------------------------------------------------------#

def detect_fractures(wells, pts, edges):
    start, end = pts[:, edges[0, :]], pts[:, edges[1, :]]
    dist, _ = pp.cg.dist_points_segments(wells, start, end)
    wells_fracture = np.argmin(dist, axis=1)

    color = np.inf*np.ones(edges.shape[1])
    color[wells_fracture] = wells_fracture
    return wells_fracture, color

#------------------------------------------------------------------------------#

class FlowData(pp.EllipticDataAssigner):
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
        self.wells_flow_rate = kwargs['wells_flow_rate']

        pp.EllipticDataAssigner.__init__(self, g, d)

    def aperture(self):
        a = np.power(self.reference_aperture, 2-self.grid().dim)
        return np.ones(self.grid().num_cells) * a

    def permeability(self):
        ones = np.ones(self.grid().num_cells)
        if self.grid().dim == 2:
            return pp.SecondOrderTensor(3, self.km*ones)
        else:
            return pp.SecondOrderTensor(3, self.kf*ones)

    def bc(self):
        bound_faces = self.grid().tags['domain_boundary_faces'].nonzero()[0]
        labels = np.array(['dir']*bound_faces.size)
        return pp.BoundaryCondition(self.grid(), bound_faces, labels)

    def bc_val(self):
        bound_faces = self.grid().tags['domain_boundary_faces'].nonzero()[0]
        bc_val = np.zeros(self.grid().num_faces)
        bc_val[bound_faces] = self.bc_pressure
        return bc_val

    def source(self):
        val = np.zeros(self.grid().num_cells)
        wells_cell = self.grid().tags['wells_cell']
        wells_id = self.grid().tags['wells_id']
        if np.isfinite(wells_cell):
            val[wells_cell] = self.wells_flow_rate[wells_id]
        return val

#------------------------------------------------------------------------------#

class TransportData(pp.ParabolicDataAssigner):
    """
    Assign transport problem data to given grid.
    """

    def __init__(self, g, d, **kwargs):

        self.tol = kwargs.get('tol', 1e-6)
        self.domain = kwargs['domain']

        self.reference_aperture = kwargs['aperture']
        self.bc_temperature = kwargs['bc_temperature']
        self.wells_temperature = kwargs['wells_temperature']

        pp.ParabolicDataAssigner.__init__(self, g, d)

    def bc(self):
        bound_faces = self.grid().tags['domain_boundary_faces'].nonzero()[0]
        labels = np.array(['dir']*bound_faces.size)
        return pp.BoundaryCondition(self.grid(), bound_faces, labels)

    def bc_val(self, _):
        bound_faces = self.grid().tags['domain_boundary_faces'].nonzero()[0]
        bc_val = np.zeros(self.grid().num_faces)
        bc_val[bound_faces] = self.bc_temperature
        return bc_val

    def initial_condition(self):
        return self.bc_temperature*np.ones(self.grid().num_cells)

    def aperture(self):
        a = np.power(self.reference_aperture, 2-self.grid().dim)
        return np.ones(self.grid().num_cells) * a

    def source(self, _):
        val = np.zeros(self.grid().num_cells)
        wells_cell = self.grid().tags['wells_cell']
        wells_id = self.grid().tags['wells_id']
        if np.isfinite(wells_cell):
            val[wells_cell] = self.wells_temperature[wells_id]*\
                              self.grid().cell_volumes[wells_cell]
        return val

#------------------------------------------------------------------------------#

class TransportSolver(pp.ParabolicModel):
    """
    Make a ParabolicModel for the transport problem with specified parameters.
    """

    def __init__(self, gb, **data_problem):

        time_step = data_problem['end_time'] /\
                    float(data_problem['number_time_steps'])
        pp.ParabolicModel.__init__(self, gb, time_step=time_step,
                                          **data_problem)

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
        flow = pp.EllipticModel(gb, **data_problem)

        # transport problem
        assign_data(gb, data_problem, TransportData, 'transport_data')
        data_problem['file_name'] = 'temperature'
        transport = TransportSolver(gb, **data_problem)

        darcy_and_transport.DarcyAndTransport.__init__(self, flow, transport)

#------------------------------------------------------------------------------#
