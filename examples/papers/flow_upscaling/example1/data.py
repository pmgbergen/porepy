import numpy as np
import porepy as pp

# ------------------------------------------------------------------------------#


def flow(gb, data, tol):
    physics = data["physics"]

    mu = data["fluid"].dynamic_viscosity()

    for g, d in gb:
        param = pp.Parameters(g)
        d["is_tangential"] = True

        unity = np.ones(g.num_cells)
        zeros = np.zeros(g.num_cells)
        empty = np.empty(0)

        # set the permeability
        if g.dim == 2:
            kxx = data['km'] / mu * unity
            perm = pp.SecondOrderTensor(2, kxx=kxx, kyy=kxx, kzz=1)
        else: #g.dim == 1:
            kxx = data['kf'] / mu * unity
            perm = pp.SecondOrderTensor(1, kxx=kxx, kyy=1, kzz=1)
        param.set_tensor(physics, perm)

        # Assign apertures
        aperture = np.power(data['aperture'], 2-g.dim)
        param.set_aperture(aperture)

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size:
            (top, bottom, left, right), boundary = bc_flag(g, data["domain"], tol)

            labels = np.array(["neu"] * b_faces.size)
            labels[left + right] = "dir"
            param.set_bc(physics, pp.BoundaryCondition(g, b_faces, labels))

            bc_val = np.zeros(g.num_faces)
            bc_val[b_faces[right]] = data["bc_flow"]
            param.set_bc_val(physics, bc_val)
        else:
            param.set_bc(physics, pp.BoundaryCondition(g, empty, empty))

        d["param"] = param

    for e, d in gb.edges():
        g_l = gb.nodes_of_edge(e)[0]
        mg = d["mortar_grid"]
        check_P = mg.slave_to_mortar_avg()

        aperture = gb.node_props(g_l, "param").get_aperture()
        gamma = check_P * np.power(aperture, 1/(2.-g.dim))
        d["kn"] = data["kf"] / mu * np.ones(mg.num_cells) / gamma


# ------------------------------------------------------------------------------#

def advdiff(gb, data, tol):
    physics = data["physics"]

    cm = data["rock"].specific_heat_capacity()
    cw = data["fluid"].specific_heat_capacity()

    rhom = data["rock"].DENSITY
    rhow = data["fluid"].density()

    lm = data["rock"].thermal_conductivity()
    lw = data["fluid"].thermal_conductivity()

    for g, d in gb:
        param = d["param"]

        unity = np.ones(g.num_cells)
        zeros = np.zeros(g.num_cells)
        empty = np.empty(0)

        # Assign the porosity
        if g.dim == 2:
            phi = data["rock"].POROSITY
        else:
            phi = data["porosity_f"]

        ce = phi * rhow * cw + (1-phi) * rhom * cm
        param.set_porosity(ce)

        # Assign the diffusivity
        l = np.power(lw, phi)*np.power(lm, 1-phi)
        perm = pp.SecondOrderTensor(3, l * unity)
        param.set_tensor(physics, perm)

        # Assign apertures
        aperture = np.power(data['aperture'], 2-g.dim)
        param.set_aperture(aperture)

        # Flux
        flux_name = data["keys_flow"][2]
        d[flux_name] *= rhow * cw

        # Boundaries
        b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
        if b_faces.size:
            (top, bottom, left, right), boundary = bc_flag(g, data["domain"], tol)

            labels = np.array(["neu"] * b_faces.size)
            labels[left + right] = ["dir"]
            param.set_bc(physics, pp.BoundaryCondition(g, b_faces, labels))

            bc_val = np.zeros(g.num_faces)
            bc_val[b_faces[right]] = data["bc_advdiff"]
            param.set_bc_val(physics, bc_val)
        else:
            param.set_bc(physics, pp.BoundaryCondition(g, np.empty(0), np.empty(0)))

        d["param"] = param

        # Assign time step
        d["deltaT"] = data["deltaT"]

    lambda_flux = "lambda_flux" #+ data["flux"]
    # Assign coupling discharge and diffusivity
    for e, d in gb.edges():
        d["flux_field"] = d[lambda_flux] * rhow * cw

        g_l = gb.nodes_of_edge(e)[0]
        mg = d["mortar_grid"]
        check_P = mg.slave_to_mortar_avg()

        gamma = check_P * gb.node_props(g_l, "param").get_aperture()
        k = check_P * gb.node_props(g_l, "param").get_tensor(physics).perm[0, 0, :]
        d["kn"] = k * np.ones(mg.num_cells) / gamma


#    def bc_val(self, t):
#        # we are assuming that the solution variable is split and saved on the grid
#        # bucket every step. we avoid the first one by assuming t_0 = 0.
#        if t > 0:
#            sol = self.gb.node_props(self.grid(), self.field_name)
#        else:
#            sol = 80*np.ones(self.grid().num_cells)
#
#        bc_val = np.zeros(self.grid().num_faces)
#        b_faces = self.grid().tags["domain_boundary_faces"].nonzero()[0]
#        if b_faces.size > 0:
#            f_inflow = b_faces[self.inflow]
#            bc_val[f_inflow] = 20
#
#            # reconstruct the cells on the outflow boundary, and then take the
#            # solution at previous time step and impose it as boundary condition
#            f_outflow = b_faces[self.outflow]
#            bc_val[f_outflow] = 80
##            outflow = np.zeros(self.grid().num_faces, dtype=np.bool)
##            outflow[f_outflow] = True
##            c_outflow = np.where(self.grid().cell_faces.transpose()*outflow)[0]
##            bc_val[f_outflow] = sol[c_outflow]
##            print(sol[c_outflow])
#        return bc_val

# ------------------------------------------------------------------------------#

def bc_flag(g, domain, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    top = b_face_centers[1] > domain["ymax"] - tol
    bottom = b_face_centers[1] < domain["ymin"] + tol
    left = b_face_centers[0] < domain["xmin"] + tol
    right = b_face_centers[0] > domain["xmax"] - tol
    boundary = top + bottom + left + right

    return (top, bottom, left, right), boundary
