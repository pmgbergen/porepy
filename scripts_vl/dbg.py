import numpy as np
import porepy as pp
import scipy.sparse as sps

A = pp.ad.Ad_array(np.arange(3), sps.identity(3, format='csr'))

b = np.array([False, True, False], dtype=bool)

c = A[b]

print("done")