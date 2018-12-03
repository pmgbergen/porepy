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
    - keyword (string) See Parameters class for valid keyword

    functions:
    darcy_flux(): computes the darcy_flux and saves it in the grid bucket as 'pressure'
    Also see functions from ParabolicProblem

    Example:
    # We create a problem with standard data

    gb = meshing.cart_grid([], [10,10], physdims=[1,1])
    for g, d in gb:
        d['problem'] = SlightlyCompressibleData(g, d)
    problem = SlightlyCompressible(gb)
    problem.solve()
   """

    def __init__(self, gb, keyword="flow", **kwargs):
        self.is_GridBucket = isinstance(gb, pp.GridBucket)
        pp.ParabolicModel.__init__(self, gb, keyword, **kwargs)

    def space_disc(self):
        return self.diffusive_disc(), self.source_disc()

    def time_disc(self):
        """
        Returns the time discretization.
        """

        class TimeDisc(object):
            def __init__(self, time_step, keyword):
                self.keyword = keyword
                self.time_step = time_step

            def assemble_matrix_rhs(self, g, data):
                ndof = g.num_cells
                parameter_dictionary = data[pp.PARAMETERS][self.keyword]
                aperture = parameter_dictionary["aperture"]
                coeff = g.cell_volumes * aperture / self.time_step
                lhs = sps.dia_matrix((coeff, 0), shape=(ndof, ndof))
                rhs = np.zeros(ndof)

                return (
                    lhs * parameter_dictionary["compressibility"],
                    rhs * parameter_dictionary["compressibility"],
                )

        time_discretization = TimeDisc(self.time_step(), self.keyword)
        return (time_discretization, None)

    def darcy_flux(self, d_name="darcy_flux", p_name="pressure"):
        self.pressure(p_name)
        fvutils.compute_darcy_flux(self.grid(), d_name=d_name, p_name=p_name)

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
    - keyword (string) Keyword key word. See Parameters class for valid keyword

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

    def __init__(self, g, data, keyword="flow"):
        pp.ParabolicDataAssigner.__init__(self, g, data, keyword)

    def _set_data(self):
        pp.ParabolicDataAssigner._set_data(self)
        parameter_dictionary = self.data()[pp.PARAMETERS][self.keyword]
        parameter_dictionary["compressibility"] = self.compressibility()

    def compressibility(self):
        return 1.0

    def permeability(self):
        kxx = np.ones(self.grid().num_cells)
        return pp.SecondOrderTensor(self.grid().dim, kxx)

    def diffusivity(self):
        return self.permeability()
