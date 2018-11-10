import numpy as np
import scipy.sparse as sps
import porepy as pp

class SlightlyCompressibleModel(pp.ParabolicModel):
    """
    Inherits from ParabolicProblem
    This class solves equations of the type:
    phi *c_p dp/dt  - \nabla K \nabla p = q

    Init:
    - gb (Grid/GridBucket) Grid or grid bucket for the problem
    - physics (string) Physics key word. See Parameters class for valid physics

    functions:
    discharge(): computes the discharges and saves it in the grid bucket as 'pressure'
    Also see functions from ParabolicProblem

    Example:
    # We create a problem with standard data

    gb = meshing.cart_grid([], [10,10], physdims=[1,1])
    for g, d in gb:
        d['problem'] = SlightlyCompressibleData(g, d)
    problem = SlightlyCompressible(gb)
    problem.solve()
   """

    def __init__(self, gb, keyword="flow", physics="flow", **kwargs):
        self.is_GridBucket = isinstance(gb, pp.GridBucket)
        pp.ParabolicModel.__init__(self, gb, keyword, physics, **kwargs)

    def space_disc(self):
        return self.diffusive_disc(), self.source_disc()

    def time_disc(self):
        """
        Returns the time discretization.
        """

        class TimeDisc(object):
            def __init__(self, deltaT, keyword):
                self.keyword = keyword
                self.deltaT = deltaT

            def assemble_matrix_rhs(self, g, data):
                ndof = g.num_cells
                aperture = data["param"].get_aperture()
                coeff = g.cell_volumes * aperture / self.deltaT
                lhs = sps.dia_matrix((coeff, 0), shape=(ndof, ndof))
                rhs = np.zeros(ndof)

                return lhs * data["compressibility"], rhs * data["compressibility"]

        time_discretization = TimeDisc(self.time_step(), self.keyword)
        return (time_discretization, None)

    def discharge(self, d_name="discharge", p_name="pressure"):
        self.pressure(p_name)
        fvutils.compute_discharges(self.grid(), d_name=d_name, p_name=p_name)

    def pressure(self):
        if self.is_GridBucket:
            self.split()
        else:
            self._data[self.keyword] = self._solver.p


class SlightlyCompressibleDataAssigner(pp.ParabolicDataAssigner):
    """
    Inherits from ParabolicData
    Base class for assigning valid data for a slighly compressible problem.
    Init:
    - g    (Grid) Grid that data should correspond to
    - d    (dictionary) data dictionary that data will be assigned to
    - physics (string) Physics key word. See Parameters class for valid physics

    Functions:
        compressibility: (float) the compressibility of the fluid
        permeability: (tensor.SecondOrderTensor) The permeability tensor for the rock.
                      Setting the permeability is equivalent to setting
                      the ParabolicData.diffusivity() function.
    Example:
    # We set an inflow and outflow boundary condition by overloading the
    # bc_val term
    class ExampleData(SlightlyCompressibleData):
        def __init__(g, d):
            SlightlyCompressibleData.__init__(self, g, d)
        def bc_val(self):
            left = self.grid().nodes[0] < 1e-6
            right = self.grid().nodes[0] > 1 - 1e-6
            val = np.zeros(g.num_faces)
            val[left] = 1
            val[right] = -1
            return val
    gb = meshing.cart_grid([], [10,10], physdims=[1,1])
    for g, d in gb:
        d['problem'] = ExampleData(g, d)
    """

    def __init__(self, g, data, physics="flow"):
        pp.ParabolicDataAssigner.__init__(self, g, data, physics)

    def _set_data(self):
        pp.ParabolicDataAssigner._set_data(self)
        self.data()["compressibility"] = self.compressibility()

    def compressibility(self):
        return 1.0

    def permeability(self):
        kxx = np.ones(self.grid().num_cells)
        return pp.SecondOrderTensor(self.grid().dim, kxx)

    def diffusivity(self):
        return self.permeability()
