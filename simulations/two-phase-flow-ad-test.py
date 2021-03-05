"With the goal of a 2p2c simulation, implement incompressible 2p1c."

import time
import porepy as pp
import numpy as np
import scipy.sparse as sps
import scipy.optimize as spo

from porepy.numerics.fv.generaltpfaad import GeneralTpfaAd, UpwindAd, FluxBasedUpwindAd, HarmAvgAd
from porepy.numerics.ad.functions import exp, sin, cos, tanh, heaviside, RegularizedHeaviside
from porepy.numerics.ad.equation_manager import Expression
from porepy.numerics.ad.operators import SecondOrderTensorAd
from porepy.numerics.ad.grid_operators import DirBC
from porepy.numerics.ad.forward_mode import initAdArrays
from porepy.numerics.solvers.andersonacceleration import AndersonAcceleration

# Material parameters

# Rock - arbitrary, exact solution will integrate these
porosity = 1e-1
permeability_value = 1e-12

# Brine
rho_b = 1000 # kg m**-3
mu_b = 300.3 * 1e-6 # Pa s
kappa_b = 3 # 1 - arbitrary

def lambda_b(sat_b, sat_c):
    sat_b_scaled = sat_b / (sat_b + sat_c)
    return sat_b_scaled ** kappa_b / mu_b

# CO2
rho_c = 750 # kg m**-3
mu_c = 42.5 * 1e-6 # Pa s
kappa_c = 3 # 1 - arbitrary
krc_max = 0.4 # 1 - arbitrary

def lambda_c(sat_b, sat_c):
    sat_c_scaled = sat_c / (sat_b + sat_c)
    return krc_max * sat_c_scaled ** kappa_c / mu_c

# Define fractional flow function as function of merely sat_c; and its derivative.
def total_lambda(sat_c):
    return lambda_b(1-sat_c, sat_c) + lambda_c(1-sat_c, sat_c)

def frac_c(sat_c):
    return lambda_c(1-sat_c, sat_c) / total_lambda(sat_c)

def lambda_b_prime(sat_c):
    return -kappa_b * (1-sat_c) ** (kappa_b - 1) / mu_b

def lambda_c_prime(sat_c):
    return krc_max * kappa_c * sat_c ** (kappa_c - 1) / mu_c

def frac_c_prime(sat_c):
    return lambda_c_prime(sat_c) / total_lambda(sat_c) - lambda_c(1-sat_c, sat_c) / total_lambda(sat_c) ** 2 * (lambda_b_prime(sat_c) + lambda_c_prime(sat_c))

# Setup grid, gridbucket and data
dx = 1000
dy = 200
dz = 40
N = 50
gb = pp.meshing.cart_grid([], nx = [N, 1, 1], physdims =[dx, dy, dz])
g = gb.grids_of_dimension(3)[0]
d = gb.node_props(g)

# Time step size
one_day = 24 * 60 * 60 # in seconds
final_time = 100 * one_day
current_time = 0
dt = 0.1 * one_day
num_time_steps = 10 #int(final_time / dt)

# Define keywords
pressure_variable = "pressure"
brine_saturation_variable = "brine"
co2_saturation_variable = "co2"
flow_kw = "flow"

# Flow parameters
perm = SecondOrderTensorAd(permeability_value * np.ones(g.num_cells))

# Injection over whole cross section; rest impermeable
all_faces = g.tags['domain_boundary_faces'].nonzero()[0]
injection_faces  = all_faces[g.face_centers[0, all_faces] < 1e-6]
extraction_faces = all_faces[g.face_centers[0, all_faces] > dx - 1e-6]

bc_labels = np.array(['neu'] * all_faces.size)
bc_labels[injection_faces] = 'dir'
bc_flow = pp.BoundaryCondition(g, all_faces, bc_labels)

bc_val_flow = np.zeros(g.num_faces)
one_year = 365 * one_day
injection_area = dy * dz
p_ref = 5e6 # Pa
q_ext = 0.1 # ca. 1.427e6 / one_year / porosity # m**3/s
bc_val_flow[injection_faces]  =  p_ref
bc_val_flow[extraction_faces] = q_ext # outflow = inflow

flow_param = {
    "second_order_tensor": perm,
    "bc": bc_flow,
    "bc_values": bc_val_flow
}
pp.initialize_default_data(g, d, flow_kw, flow_param)

# AD Boundary condition
bc_flow = pp.ad.BoundaryCondition(flow_kw, [g])
dir_bc_flow = DirBC(bc_flow, [g])

# Primary variables / DOFs
d[pp.PRIMARY_VARIABLES] = {
    pressure_variable : {"cells" : 1},
    brine_saturation_variable : {"cells" : 1},
    co2_saturation_variable : {"cells" : 1}
}

# Initialize current states - constant reference pressure; 100% saturated with brine.
pressure_state = p_ref * np.ones(g.num_cells)
brine_state = np.ones_like(pressure_state)
co2_state = np.zeros_like(pressure_state)

d[pp.STATE] = {}
d[pp.STATE][pressure_variable] = pressure_state
d[pp.STATE][brine_saturation_variable] = brine_state
d[pp.STATE][co2_saturation_variable] = co2_state

d[pp.STATE][pp.ITERATE] = {}
d[pp.STATE][pp.ITERATE][pressure_variable] = pressure_state.copy()
d[pp.STATE][pp.ITERATE][brine_saturation_variable] = brine_state.copy()
d[pp.STATE][pp.ITERATE][co2_saturation_variable] = co2_state.copy()

# Managers
dof_manager = pp.DofManager(gb)
manager = pp.ad.EquationManager(gb, dof_manager)

# Ad variables
p  = manager.variables[g][pressure_variable]
sb = manager.variables[g][brine_saturation_variable]
sc = manager.variables[g][co2_saturation_variable]

# Discretization operators
tpfa_ad = GeneralTpfaAd(flow_kw)
tpfa_ad.discretize(g, d)

mass = pp.MassMatrix(flow_kw)
mass.discretize(g, d)
mass_ad = pp.ad.Discretization(g, discretization = pp.MassMatrix(flow_kw))
mass_mat = mass_ad.mass

div = pp.ad.Divergence([g])

one_vector = np.ones(g.num_cells)
zero_vector = np.zeros(g.num_cells)
z_component = g.cell_centers[2]

# Transmissibility operators
harmAvg = HarmAvgAd(g, d, tpfa_ad)
hs = heaviside
fluxBasedUpwind = FluxBasedUpwindAd(g, tpfa_ad, hs)

# AD functions
lam_b = pp.ad.Function(lambda_b, name='mobility brine')
lam_c = pp.ad.Function(lambda_c, name='mobility co2')

# Solver
total_iteration_counter = 0
abs_tol = 1e-8
rel_tol = 1e-8
max_iter = 2
res_vol_factor = 1.

# Define the coupled two-phase problem
source_b = np.zeros(g.num_cells)
source_c = np.zeros(g.num_cells)
lam_b_old = lam_b(sb.previous_timestep(), sc.previous_timestep())
lam_c_old = lam_c(sb.previous_timestep(), sc.previous_timestep())
total_flux = tpfa_ad.flux(harmAvg(perm) * harmAvg(lam_b_old + lam_c_old) , p, bc_flow)
lam_b_old_up = fluxBasedUpwind(lam_b(sb.previous_timestep(), sc.previous_timestep()), pp.ad.Array(lambda_b(np.zeros_like(bc_val_flow), np.ones_like(bc_val_flow))), total_flux)
lam_c_old_up = fluxBasedUpwind(lam_c(sb.previous_timestep(), sc.previous_timestep()), pp.ad.Array(lambda_c(np.zeros_like(bc_val_flow), np.ones_like(bc_val_flow))), total_flux)

v_res = sb.previous_timestep() + sc.previous_timestep() - pp.ad.Array(one_vector)
total_flow_source = source_c / rho_c + source_b / rho_b + porosity * res_vol_factor * v_res
pressure_eq = pp.ad.Expression(div * total_flux, dof_manager=dof_manager, name="pressure equation")
manager.equations.append(pressure_eq)

brine_transport_eq = pp.ad.Expression(
        rho_b * porosity * mass_mat * (sb - sb.previous_timestep())
        + dt * div * (rho_b * lam_b_old_up / (lam_b_old_up + lam_c_old_up) * total_flux),
        dof_manager=dof_manager, name="brine transport equation")
manager.equations.append(brine_transport_eq)

co2_transport_eq = pp.ad.Expression(
        rho_c * porosity * mass_mat * (sc - sc.previous_timestep())
        + dt * div * (rho_c * lam_c_old_up / (lam_b_old_up + lam_c_old_up) * total_flux),
        dof_manager=dof_manager, name="co2 transport equation")
manager.equations.append(co2_transport_eq)

start = time.time()

# Time loop
for n in range(1,num_time_steps+1):

    print("time step", n)
    current_time += dt

    #######################################
    # Exact solution
    #######################################
    # Compute shock saturation sc_shock
    def sc_shock_function(sat_c):
        return frac_c_prime(sat_c) - frac_c(sat_c) / sat_c
    sc_shock = spo.root_scalar(sc_shock_function, x0=0.2, x1=0.9).root
    # Compute shock velocity vc_shock
    vc_shock = q_ext / porosity / injection_area * frac_c_prime(sc_shock)
    # Compute p_exact
    p_exact = np.zeros(g.num_cells)
    for k in range(0, g.num_cells):
        delta_x = dx / N
        xk = k * delta_x
        xkp1 = (k + 1) * delta_x
        xkc = 0.5 * (xkp1 + xk)

        # Try first with shock front, otherwise set either saturation equal zero or solve implicit function.
        x_eval = vc_shock * current_time
        sc_eval = sc_shock
        if (x_eval > xkp1):
            x_eval = xkc
            def tmp_fct(sat_c):
                return x_eval - q_ext / porosity / injection_area * frac_c_prime(sat_c) * current_time
            if tmp_fct(1.1) * tmp_fct(0.7*sc_shock) < 0:
                sc_eval = spo.bisect(tmp_fct, a=1.1, b=0.7*sc_shock)
            else:
                sc_eval = spo.root_scalar(tmp_fct, x0=1.1, x1=0.7*sc_shock).root
        elif x_eval < xk:
            x_eval = xkc
            sc_eval = 0

        # Solve 1d pressure equation by hand.
        if k==0:
            p_exact[k] = p_ref - q_ext * (xkc - xk) / injection_area / total_lambda(sc_eval) / permeability_value 
        if k > 0:
            p_exact[k] = p_exact[k-1] - q_ext * (xkp1 - xk) / injection_area / total_lambda(sc_eval) / permeability_value

    #######################################
    # Pressure equation step
    #######################################
    iteration_counter = 0
    residual_norm = 1
    rel_res = 1

    # TODO all three problems read the same... this can be significantly simplified
    while iteration_counter <= max_iter and not (rel_res < rel_tol or residual_norm < abs_tol):

        # Assembler problem
        A, b = manager.assemble_matrix_rhs(equations=["pressure equation"], ad_var=[p])
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
        print("Pressure eq: iteration", iteration_counter, "abs res", residual_norm, "rel res", residual_norm / initial_residual_norm)

        # Prepare next iteration
        iteration_counter += 1
        total_iteration_counter += 1

    #######################################
    # Transport step - brine
    #######################################
    iteration_counter = 0
    residual_norm = 1
    rel_res = 1

    while iteration_counter <= max_iter and not (rel_res < rel_tol or residual_norm < abs_tol):

        # Assembler problem
        A, b = manager.assemble_matrix_rhs(equations=["brine transport equation"], ad_var=[sb])
        # Solve for pressure increment and update pressure
        sb_increment = sps.linalg.spsolve(A, b)
        # Update approximation
        d[pp.STATE][pp.ITERATE][brine_saturation_variable] += sb_increment

        # Compute 'error' as norm of the residual
        residual_norm = np.linalg.norm(b, 2)
        if iteration_counter == 0:
            initial_residual_norm = residual_norm
        else:
            initial_residual_norm = max(residual_norm, initial_residual_norm)
        if initial_residual_norm < 1e-8:
            initial_residual_norm = 1
        rel_res = residual_norm / initial_residual_norm
        print("Brine eq: iteration", iteration_counter, "abs res", residual_norm, "rel res", residual_norm / initial_residual_norm)

        # Prepare next iteration
        iteration_counter += 1
        total_iteration_counter += 1

    #######################################
    # Transport step - co2
    #######################################
    iteration_counter = 0
    residual_norm = 1
    rel_res = 1

    while iteration_counter <= max_iter and not (rel_res < rel_tol or residual_norm < abs_tol):

        # Assembler problem
        A, b = manager.assemble_matrix_rhs(equations=["co2 transport equation"], ad_var=[sc])

        # Solve for pressure increment and update pressure
        sc_increment = sps.linalg.spsolve(A, b)
        # Update approximation
        d[pp.STATE][pp.ITERATE][co2_saturation_variable] += sc_increment

        # Compute 'error' as norm of the residual
        residual_norm = np.linalg.norm(b, 2)
        if iteration_counter == 0:
            initial_residual_norm = residual_norm
        else:
            initial_residual_norm = max(residual_norm, initial_residual_norm)
        if initial_residual_norm < 1e-8:
            initial_residual_norm = 1
        rel_res = residual_norm / initial_residual_norm
        print("CO2 eq: iteration", iteration_counter, "abs res", residual_norm, "rel res", residual_norm / initial_residual_norm)

        # Prepare next iteration
        iteration_counter += 1
        total_iteration_counter += 1

    #######################################
    # Finish time step with some organization
    #######################################
    # Update next time step solution
    d[pp.STATE][pressure_variable]         = d[pp.STATE][pp.ITERATE][pressure_variable].copy()
    d[pp.STATE][brine_saturation_variable] = d[pp.STATE][pp.ITERATE][brine_saturation_variable].copy()
    d[pp.STATE][co2_saturation_variable]   = d[pp.STATE][pp.ITERATE][co2_saturation_variable].copy()

    # Export pressure field
    #exporter = pp.Exporter(gb, "pressure"+str(n))
    #exporter.write_vtu(pressure_variable)
    #exporter = pp.Exporter(gb, "brine"+str(n))
    #exporter.write_vtu(brine_saturation_variable)
    #exporter = pp.Exporter(gb, "co"+str(n))
    #exporter.write_vtu(co2_saturation_variable)

    print()

end = time.time()

#######################################
# Compare p and p_exact
#######################################
print("Exact pressure")
print(p_exact)
print("Computed pressure")
print(d[pp.STATE][pp.ITERATE][pressure_variable])

print("Elapsed time", end-start)
