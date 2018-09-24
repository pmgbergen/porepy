import numpy as np
import porepy as pp

class Water():

    def __init__(self, theta_ref=None):
        if theta_ref is None:
            self.theta_ref = 20*(pp.CELSIUS)
        else:
            self.theta_ref = theta_ref

    def thermal_expansion(self, delta_theta):
        return 0.0002115+\
               1.32*1e-6*delta_theta+\
               1.09*1e-8*np.power(delta_theta, 2)

    def density(self, theta=None): # theta in CELSIUS
        if theta is None:
            theta = self.theta_ref
        theta_0 = 10*(pp.CELSIUS)
        rho_0 = 999.8349*(pp.KILOGRAM/pp.METER**3)
        return rho_0/(1.+self.thermal_expansion(theta-theta_0))

    def thermal_conductivity(self, theta=None): # theta in CELSIUS
        if theta is None:
            theta = self.theta_ref
        return 0.56+\
               0.002*theta-\
               1.01*1e-5*np.power(theta, 2)+\
               6.71*1e-9*np.power(theta, 3)

    def specific_heat_capacity(self, theta=None): # theta in CELSIUS
        if theta is None:
            theta = self.theta_ref
        return (4245-1.841*theta)/self.density(theta)

    def dynamic_viscosity(self, theta=None): # theta in CELSIUS
        if theta is None:
            theta = self.theta_ref
        theta = pp.CELSIUS_to_KELVIN(theta)
        mu_0 = 2.414*1e-5*(pp.PASCAL*pp.SECOND)
        return mu_0*np.power(10, 247.8/(theta-140))

    def hydrostatic_pressure(self, depth, theta=None):
        rho = self.density(theta)
        return rho*depth*pp.GRAVITY_ACCELERATION+pp.ATMOSPHERIC_PRESSURE

