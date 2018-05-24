import numpy as np
import matplotlib.pyplot as plt

import porepy as pp

from porepy.numerics import darcy_and_transport

#------------------------------------------------------------------------------#

def detect_wells(pb):

    pts, edges = pp.importer.lines_from_csv(pb['file'], tol=pb['tol'])

    # detect the fractures closest to the wells
    f, color = detect_fractures(pb['wells']['position'], pts, edges)

    # save the data
    pb['dfn'] = dict()
    pb['dfn']['pts'] = pts
    pb['dfn']['edges'] = edges

    pb['wells']['fracture'] = f
    pb['wells']['color'] = color

#------------------------------------------------------------------------------#

def plot(pb):

    color_f = pb['wells']['color']

    pb['domain'] = pp.cg.bounding_box(pb['dfn']['pts'])
    pp.plot_fractures(pb['domain'], pb['dfn']['pts'], pb['dfn']['edges'],
                      plot=False, fig_id=1, colortag=color_f)

    pp.plot_wells(pb['domain'], pb['wells']['position'],
                  plot=False, fig_id=1, colortag=pb['wells']['fracture'])

    plt.figure(1)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.show()

#------------------------------------------------------------------------------#

def create_grid(pb):

    # pack the data for the grid generation
    mesh_kwargs = {'tol': pb['tol']}
    mesh_kwargs['mesh_size_frac'] = pb['mesh_size']
    kwargs = {'assign_fracture_id': True}

    # import the fractures and create the grid
    file_name, domain = pb['file'], pb['domain']
    gb = pp.importer.dfm_2d_from_csv(file_name, mesh_kwargs, domain, **kwargs)
    gb.compute_geometry()
    gb.assign_node_ordering()

    return gb

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

    def __init__(self, g, d, **pb):

        self.pb = pb
        self.tol = self.pb['tol']
        self.is_fracture = g.dim < 2
        pp.EllipticDataAssigner.__init__(self, g, d)

    def aperture(self):
        if self.is_fracture:
            apert = self.pb['fracture']['aperture']
            return np.power(apert, 2-self.grid().dim)
        else:
            return 1.

    def permeability(self):
        if self.is_fracture:
            k = self.pb['fracture']['permeability']
        else:
            k = self.pb['rock'].PERMEABILITY

        mu = self.pb['water'].dynamic_viscosity()
        rho = self.pb['water'].density()
        kxx = k*rho*pp.GRAVITY_ACCELERATION/mu
        return pp.SecondOrderTensor(3, kxx*np.ones(self.grid().num_cells))

    def bc(self):
        bound_faces = self.grid().tags['domain_boundary_faces'].nonzero()[0]
        labels = np.array(['dir']*bound_faces.size)
        return pp.BoundaryCondition(self.grid(), bound_faces, labels)

    def bc_val(self):
        return np.zeros(self.grid().num_faces)

    def source(self):
        val = np.zeros(self.grid().num_cells)
        wells_cell = self.grid().tags['wells']['cell']
        wells_id = self.grid().tags['wells']['id']
        if np.isfinite(wells_cell):
            val[wells_cell] = self.pb['wells']['flow_rate'][wells_id]
        return val

#------------------------------------------------------------------------------#

class TransportData(pp.ParabolicDataAssigner):
    """
    Assign transport problem data to given grid.
    """

    def __init__(self, g, d, **pb):

        self.pb = pb
        self.tol = self.pb['tol']
        self.is_fracture = g.dim < 2
        pp.ParabolicDataAssigner.__init__(self, g, d)

    def bc(self):
        bound_faces = self.grid().tags['domain_boundary_faces'].nonzero()[0]
        labels = np.array(['dir']*bound_faces.size)
        return pp.BoundaryCondition(self.grid(), bound_faces, labels)

    def bc_val(self, t):
        bound_faces = self.grid().tags['domain_boundary_faces'].nonzero()[0]
        bc_val = np.zeros(self.grid().num_faces)
        bc_val[bound_faces] = self.pb['theta_ref']
        return bc_val

    def initial_condition(self):
        return self.pb['theta_ref']

    def source(self, t):
        val = np.zeros(self.grid().num_cells)
        wells_cell = self.grid().tags['wells']['cell']
        wells_id = self.grid().tags['wells']['id']
        if np.isfinite(wells_cell):
            val[wells_cell] = self.pb['wells']['temperature'][wells_id]*\
                              self.grid().cell_volumes[wells_cell]
        return val

    def porosity(self):
        if self.is_fracture:
            return self.pb['fracture']['porosity']
        else:
            return self.pb['rock'].POROSITY

    def diffusivity(self):
        lambda_w = self.pb['water'].thermal_conductivity()
        lambda_r = self.pb['rock'].thermal_conductivity()
        if self.is_fracture:
            phi = self.pb['fracture']['porosity']
        else:
            phi = self.pb['rock'].POROSITY
        lambda_e = np.power(lambda_w, phi)*np.power(lambda_r, 1.-phi)

        kxx = lambda_e*np.ones(self.grid().num_cells)
        return pp.SecondOrderTensor(3, kxx)

    def aperture(self):
        if self.is_fracture:
            apert = self.pb['fracture']['aperture']
            return np.power(apert, 2-self.grid().dim)
        else:
            return 1.

    def rock_specific_heat(self):
        return self.pb['rock'].specific_heat_capacity()

    def fluid_specific_heat(self):
        return self.pb['water'].specific_heat_capacity()

    def rock_density(self):
        return self.pb['rock'].DENSITY

    def fluid_density(self):
        return self.pb['water'].density()

#------------------------------------------------------------------------------#

class TransportSolver(pp.ParabolicModel):
    """
    Make a ParabolicModel for the transport problem with specified parameters.
    """

    def __init__(self, gb, **pb):

        time_step = pb['end_time']/float(pb['number_time_steps'])
        pp.ParabolicModel.__init__(self, gb, time_step=time_step, **pb)

    def space_disc(self):
        '''Space discretization. Returns the discretization terms that should be
        used in the model'''
        return self.advective_disc(), self.source_disc()

#------------------------------------------------------------------------------#

class Simulation(darcy_and_transport.DarcyAndTransport):
    """
    Combine the two problems for convenience.
    """

    def __init__(self, pb):

        pb['theta_ref'] = pb['temperature_at_depth']
        pb['water'] = pp.Water(pb['theta_ref'])
        pb['rock'] = pp.Granite(pb['theta_ref'])

        # create the computational grid
        gb = create_grid(pb)

        # Darcy problem
        self.assign_data(gb, pb, FlowData, 'problem')
        pb['file_name'] = 'hydraulic_head'
        flow = pp.EllipticModel(gb, **pb)

        # transport problem
        self.assign_data(gb, pb, TransportData, 'transport_data')
        pb['file_name'] = 'temperature'
        transport = TransportSolver(gb, **pb)

        darcy_and_transport.DarcyAndTransport.__init__(self, flow, transport)

    def assign_data(self, gb, pb, data_class, data_key):
        """
        Loop over grids to assign flow problem data.
        Darcy data_key: problem
        Transport data_key: transport_data
        """
        gb.add_node_props([data_key])

        wells = pb['wells']['position']
        for g, _ in gb:
            if g.dim == 1:
                well_id = np.isin(pb['wells']['fracture'], g.tags['fracture_id'])
                if np.any(well_id):
                    wells_id = np.where(well_id)[0]
                    wells_cell = g.closest_cell(wells[:, wells_id])
                else:
                    wells_cell, wells_id = [np.nan]*2
            else:
                wells_cell, wells_id = [np.nan]*2

            g.tags['wells'] = {'cell': wells_cell, 'id': wells_id}

        for g, d in gb:
            d[data_key] = data_class(g, d, **pb)

#------------------------------------------------------------------------------#
