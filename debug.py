import porepy as pp
import numpy as np
import scipy.sparse as sps

variable_val = np.ones(3)
# This is the Jacobian matrix of the returned expression.
jac = sps.csr_matrix(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

# This is the expression to be used in the tests. The numerical values of val will
# be np.array([6, 15, 24]), and its Jacobian matrix is jac.
expression_val = jac @ variable_val

g = pp.CartGrid([3, 1])
mdg = pp.MixedDimensionalGrid()
mdg.add_subdomains([g])

eq_system = pp.ad.EquationSystem(mdg)
eq_system.create_variables("foo", subdomains=[g])
var = eq_system.variables[0]
d = mdg.subdomain_data(g)

pp.set_solution_values(
    name="foo", values=variable_val, data=d, time_step_index=0
)
pp.set_solution_values(name="foo", values=variable_val, data=d, iterate_index=0)
mat = pp.ad.SparseArray(jac)

v1 = mat @ var
v2 = v1 + v1
v3 = v2 + v1

eq_system.operator_value(v3)
