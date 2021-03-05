"AD implementation of Richards' equation. TPFA solver for manufactured solution on unit square."

import porepy as pp
import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sps

from porepy.numerics.fv.generaltpfaad import GeneralTpfaAd, UpwindAd, HarmAvgAd
from porepy.numerics.ad.functions import exp, sin, cos, tanh, heaviside, RegularizedHeaviside
from porepy.numerics.ad.equation_manager import Expression
from porepy.numerics.ad.operators import SecondOrderTensorAd
from porepy.numerics.ad.grid_operators import DirBC
from porepy.numerics.ad.local_forward_mode import initLocalAdArrays

# Van Genuchten hydraulic constitutive laws
sr_vg = 0.
a_vg = 0.2
n_vg = 2
m_vg = 1 - 1/n_vg

def sat(p):
    return sr_vg + (1 - sr_vg) * (1 + (a_vg * (p**2) ** 0.5 ) ** n_vg ) ** (-(1 - 1/n_vg))

def krw(s):
    return s ** 0.5 * (1 - (1 - s ** (1 / m_vg)) ** m_vg) ** 2


def mobility(p):
    return krw(sat(p))

# Choose either a parabolic or a trigonometric pressure profile and either linear or trigonometric increase in time:
exact_solution = "trigonometric"
#exact_solution = "parabolic"

if exact_solution == "parabolic":

    # Exact parabolic pressure profile
    def p_ex(x, y, t):
        return t * x * (1-x) * y * (1-y) - 0.1
    
    def dpdx_ex(x, y, t):
        return t * (1-2*x) * y * (1-y)
    
    def dpdy_ex(x, y, t):
        return t * x * (1-x) * (1-2 * y)

elif exact_solution == "trigonometric":
    
    # Exact trigonometric pressure profile
    pi = np.pi
    def p_ex(x, y, t):
        return sin(pi * t) * sin(pi * x) * cos(pi * y) - 2
    
    def dpdx_ex(x, y, t):
        return sin(pi * t) * pi * cos(pi * x) * cos(pi * y)
    
    def dpdy_ex(x, y, t):
        return sin(pi * t) * sin(pi * x) * (-pi) * sin(pi * y)

else:
    raise RuntimeError("No available exact solution chosen.")

# Exact hydraulics
def sat_ex(x, y, t):
    return sat(p_ex(x,y,t))

def flux_ex(x, y, t):
    return [- mobility(p_ex(x,y,t)) * dpdx_ex(x,y,t), - mobility(p_ex(x,y,t)) * dpdy_ex(x,y,t)]

# Setup grid, gridbucket and data
refinementlvl = 1

gb = pp.meshing.cart_grid([], nx = [4 * 2 ** refinementlvl, 4 * 2 ** refinementlvl], physdims = [1, 1])
g = gb.grids_of_dimension(2)[0]
d = gb.node_props(g)

# Time step size
final_time = 0.5
num_time_steps = 4 * 2 ** (2 * refinementlvl)
time = 0
dt = final_time / num_time_steps

# Define keywords
pressure_variable = "pressure"
flow_kw = "flow"

# Flow parameters
perm = SecondOrderTensorAd(np.ones(g.num_cells))

all_faces = g.tags['domain_boundary_faces'].nonzero()[0]
is_neu_faces = g.face_centers[1, all_faces] == 1
is_dir_faces = [not e for e in is_neu_faces]
neu_faces = all_faces[is_neu_faces]
dir_faces = all_faces[is_dir_faces]

bc_labels = np.array(['dir']*all_faces.size)
bc_labels[is_neu_faces] = 'neu'
bc_flow = pp.BoundaryCondition(g, all_faces, bc_labels)

bc_val_flow = np.zeros(g.num_faces)

flow_param = {"second_order_tensor": perm,
        "bc": bc_flow,
        "bc_values": bc_val_flow}
pp.initialize_default_data(g, d, flow_kw, flow_param)

# AD Boundary condition
bc_flow = pp.ad.BoundaryCondition(flow_kw, [g])
dir_bc_flow = DirBC(bc_flow, [g])

# Primary variables / DOFs
d[pp.PRIMARY_VARIABLES] = {
    pressure_variable : {"cells" : 1}
}

# Initialize current states
cc = g.cell_centers
state = p_ex(cc[0], cc[1], np.ones_like(cc[0]))
d[pp.STATE] = {}
d[pp.STATE][pressure_variable] = state
d[pp.STATE][pp.ITERATE] = {}
d[pp.STATE][pp.ITERATE][pressure_variable] = state.copy()

# Managers
dof_manager = pp.DofManager(gb)
manager = pp.ad.EquationManager(gb, dof_manager)

# Ad variables
p = manager.variables[g][pressure_variable]

# Discretization operators
tpfa_ad = GeneralTpfaAd(flow_kw)
tpfa_ad.discretize(g, d)

mass = pp.MassMatrix(flow_kw)
mass.discretize(g, d)
mass_ad = pp.ad.Discretization(g, discretization = pp.MassMatrix(flow_kw))
div = pp.ad.Divergence([g])

# AD functions
satAd = pp.ad.Function(sat, name='sat')
mobilityAd = pp.ad.Function(mobility, name='mobility')

# Transmissibility operators
harmAvg = HarmAvgAd(g, d, tpfa_ad)
hs = heaviside
# Could also use a regularized version of the Heaviside function (only used for determining the Jacobian!)
#def reg(x):
#    reg_parameter = 1e-3
#    return 0.5 * (tanh(x / reg_parameter) + 1)
#hs = RegularizedHeaviside(reg)
upwind = UpwindAd(g, tpfa_ad, hs)
weighting_type = "lazy upstream"
if weighting_type == "harmonic":
    # TODO can we in a general fashion allow for mobilityAd(p) * perm and perm * rhoAd(p) here? In the latter curently __rmul__ is ran for Local_Ad_array many many times. Is this related since rhoAd(p) is a row vector?
    mobilityAd = pp.ad.Function(mobility, name='mobility', local = True)
    face_transmissibility = harmAvg(mobilityAd(p) * perm)
elif weighting_type == "lazy harmonic":
    mobilityAd = pp.ad.Function(mobility, name='mobility', local = True)
    face_transmissibility = harmAvg(mobilityAd(p.previous_iteration()) * perm)
elif weighting_type == "upstream":
    face_transmissibility = upwind(mobilityAd(p), p, mobilityAd(dir_bc_flow), dir_bc_flow) * harmAvg(perm)
elif weighting_type == "lazy upstream":
    face_transmissibility = upwind(mobilityAd(p.previous_iteration()), p, mobilityAd(dir_bc_flow), dir_bc_flow) * harmAvg(perm)
else:
    raise RuntimeError("weighting scheme does not exist")

# Time loop
total_iteration_counter = 0
for n in range(1,num_time_steps+1):
    tol = 1e-8
    iteration_counter = 0
    residual_norm = 1
    rel_res = 1

    time += dt 

    # Dirichlet boundary condition
    fc = g.face_centers[:, dir_faces]
    p_bc = p_ex(fc[0], fc[1], time * np.ones_like(fc[0]))
    flow_param["bc_values"][dir_faces] = p_bc

    # Neumann boundary condition
    fc = g.face_centers[:, neu_faces]
    flux_bc = flux_ex(fc[0], fc[1], time * np.ones_like(fc[0]))
    fnormal = g.face_normals[:, neu_faces]
    flow_param["bc_values"][neu_faces] = sum([fnormal[i] * flux_bc[i] for i in range(0,2)])

    # Source term
    cc = g.cell_centers
    x_ad, y_ad, t_ad = initLocalAdArrays([cc[0,:], cc[1,:], time * np.ones_like(cc[0,:])])
    sat_ad = sat_ex(x_ad, y_ad, t_ad)
    flux_ad = flux_ex(x_ad, y_ad, t_ad)
    dsdt = sat_ad.jac[2]
    div_flux_ad = sum([flux_ad[i].jac[i] for i in range(0,2)])
    source = dsdt + div_flux_ad

    print("time step", n)

    while iteration_counter <= 30 and not (rel_res<1e-12 or residual_norm < 1e-12):

        # Lhs of the flow equation
        sat_linearization = "l-scheme"
        if sat_linearization=="newton":
            flow_momentum_eq = mass_ad.mass * satAd(p) + dt * div * tpfa_ad.flux(face_transmissibility, p, bc_flow)
        elif sat_linearization=="l-scheme":
            L = 0.075 # sort of optimized...
            flow_momentum_eq = (dt * div * tpfa_ad.flux(face_transmissibility, p, bc_flow)
                    + mass_ad.mass * (satAd(p.previous_iteration()))
                    + mass_ad.mass * (L * (p - p.previous_iteration())))
        # TODO the following does not work - TypeError...
        # mass_ad.mass * (satAd(p.previous_iteration()) + L * (p - p.previous_iteration()))
        # TODO the following does raise an error stemming from equation_manager.py
        #flow_momentum_eq = mass_ad.mass * satAd(p.previous_iteration()) + dt * div * tpfa_ad.flux(face_transmissibility, p, bc_flow)
        #flow_momentum_eq = dt * div * tpfa_ad.flux(face_transmissibility, p, bc_flow) + mass_ad.mass * satAd(p.previous_iteration())

        # Rhs of the flow equation
        flow_rhs = mass_ad.mass * satAd(p.previous_timestep()) + dt * mass_ad.mass * source

        # Flow equation
        flow_eq = pp.ad.Expression(flow_momentum_eq - flow_rhs, dof_manager=dof_manager, name="flow equation")
        manager.equations.clear()
        manager.equations.append(flow_eq)

        # Assembler problem
        A, b = manager.assemble_matrix_rhs()

        # Solve for pressure increment and update pressure
        pressure_increment = sps.linalg.spsolve(A, b)
        # Update approximation
        d[pp.STATE][pp.ITERATE][pressure_variable] += pressure_increment

        # Compute 'error' as norm of the residual
        residual_norm = np.linalg.norm(b, 2)
        if iteration_counter == 0:
            initial_residual_norm = residual_norm
        else:
            initial_residual_norm = max(residual_norm, initial_residual_norm)
        rel_res = residual_norm / initial_residual_norm
        print("iteration", iteration_counter, "abs res", residual_norm, "rel res", residual_norm / initial_residual_norm)

        # Prepare next iteration
        iteration_counter += 1
        total_iteration_counter += 1

    print()

    # Update next time step solution
    d[pp.STATE][pressure_variable] = d[pp.STATE][pp.ITERATE][pressure_variable].copy()

    # Export pressure field
    exporter = pp.Exporter(gb, "flow"+str(n))
    exporter.write_vtu("pressure")

    # Compute discretization error
    p_exact = p_ex(cc[0], cc[1], time * np.ones_like(cc[0]))
    error = p_exact - d[pp.STATE][pressure_variable]
    l2_error = (error @ (mass_ad.mass.parse(gb) @ error)) ** 0.5
    print("accuracy", l2_error)

print("Total iteration count:", total_iteration_counter)
