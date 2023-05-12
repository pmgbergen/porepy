import numpy as np
import scipy.sparse as sps

A = sps.diags([5, 6])
A = np.array([5, 6])
jac = np.array([[1, 2], [3, 4]])
jac_sps = sps.csc_matrix(jac)
# print(f"A {A.todense()}")
# print(f"jac sparse: {jac_sps} sparse.todense() {jac_sps.todense()} dense {jac}")
# print(f"dense: np.array([A * J for J in jac] {np.array([A * J for J in jac])}")
# print(f"dense: np.array([J * A for J in jac] {np.array([J * A for J in jac])}")
# print(f"sparse: A * jac_sps {(A * jac_sps).todense()}")
# print(f"sparse: jac_sps * A {(jac_sps * A).todense()}")

A = np.array([5, 6])
jac = np.array([[1, 2], [3, 4]])
jac_sps = sps.csc_matrix(jac)

# This is just the function from `forward_mode.Ad_array` without the class. Instead, jac
# is passed directly.
def diagvec_mul_jac(a, jac):
    try:
        A = sps.diags(a)
    except TypeError:
        A = a
    if isinstance(jac, np.ndarray):
        return np.array([A * J for J in jac.T]).T
    else:
        return A * jac


print(f"jac dense: diagvec_mul_jac {(diagvec_mul_jac(A, jac))}")
print(f"jac sparse: diagvec_mul_jac {(diagvec_mul_jac(A, jac_sps)).todense()}")

A = sps.diags([5, 6])
jac = np.array([[1, 2], [3, 4]])
print(A * jac.T[0])
