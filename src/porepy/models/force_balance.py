"""
Class types:
    Generic VectorBalanceEquation
    Specific ForceBalanceEquations defines subdomain and interface equations through the
        terms entering. Force balance between opposing fracture interfaces is imposed.
    TODO: Specific ConstitutiveEquations and
    TODO: specific SolutionStrategy

Notes:
    - The class ForceBalanceEquations is a mixin class, and should be inherited by a class
        that defines the variables and discretization.

    - Refactoring needed for constitutive equations. Modularisation and moving to the library.

"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Union

import constit_library
import numpy as np
import scipy.sparse as sps

import porepy as pp

logger = logging.getLogger(__name__)


class VectorBalanceEquation:
    """Generic class for scalar balance equations on the form

    d_t(accumulation) + div(flux) - source = 0

    All terms need to be specified in order to define an equation.
    """

    def balance_equation(
        self, subdomains: list[pp.Grid], accumulation, stress, source
    ) -> pp.ad.Operator:
        """Balance equation for a vector variable.

        Args:
            subdomains: List of subdomains where the balance equation is defined.
            accumulation: Operator for the accumulation term.
            stress: Operator for the stress term.
            source: Operator for the source term.

        Returns:
            Operator for the balance equation.
        """

        dt = self.time_increment_method
        div = pp.ad.Divergence(subdomains, nd=self.nd)
        return dt(accumulation) + div * stress - source

    def volume_integral(
        self,
        integrand: pp.ad.Operator,
        grids: Union[list[pp.Grid], list[pp.MortarGrid]],
    ) -> pp.ad.Operator:
        """Numerical volume integral over subdomain or interface cells.

        Includes cell volumes and specific volume.

        Args:
            integrand: Operator for the integrand.
            grids: List of subdomain or interface grids over which the integral is to be
                computed.

        Returns:
            Operator for the volume integral.

        """
        geometry = pp.ad.Geometry(grids, nd=self.nd)
        # First factor expands from scalar to vector.
        vol = geometry.scalar_to_nd_cell * (geometry.cell_volumes * self.specific_volume(grids))
        return vol * integrand


class ForceBalanceEquations(VectorBalanceEquation):
    """Class for force balance equations and fracture deformation equations.
    """
    def set_equations(self):
        """Set equations for the subdomains and interfaces.

        The following equations are set:
            - Force balance in the matrix.
            - Force balance between fracture interfaces.
            - Deformation constraints for fractures, split into normal and tangential part.
        See individual equation methods for details.
        """
        matrix_subdomains = self.mdg.subdomains(dim=self.nd)
        fracture_subdomains = self.mdg.subdomains(dim=self.nd - 1)
        interfaces = self.mdg.interfaces(dim=se)
        matrix_eq = self.matrix_force_balance_equation(matrix_subdomains)
        # We split the fracture deformation equations into two parts, for the normal and
        # tangential components for convenience.
        fracture_eq_normal = self.normal_fracture_deformation_equation(fracture_subdomains)
        fracture_eq_tangential = self.tangential_fracture_deformation_equation(fracture_subdomains)
        intf_eq = self.interface_force_balance_equation(interfaces)
        self.system_manager.set_equation(matrix_eq, (matrix_subdomains, "cells", self.nd))
        self.system_manager.set_equation(fracture_eq_normal, (fracture_subdomains, "cells", 1))
        self.system_manager.set_equation(fracture_eq_tangential, (fracture_subdomains, "cells", self.nd-1))
        self.system_manager.set_equation(intf_eq, (interfaces, "cells", self.nd))

    def matrix_force_balance_equation(self, subdomains: list[pp.Grid]):
        """Force balance equation in the matrix.

        Inertial term is not included.

        Args:
            subdomains: List of subdomains where the force balance is defined. Only known usage
                is for the matrix domain(s).

        Returns:
            Operator for the force balance equation in the matrix.

        """
        accumulation = 0
        stress = self.stress(subdomains)
        body_force = self.body_force(subdomains)
        return self.balance_equation(subdomains, accumulation, stress, body_force)

    def rock_mass(self, subdomains: list[pp.Grid]):
        """Rock mass.

        Porosity weighting (1-porosity) is included to comply with other porous media models
        even if not usually included for (non-coupled) force/momentum balance.

        Args:
            subdomains: List of subdomains where the rock is defined.

        Returns:
            Operator for the rock mass.
        """
        density = self.rock_density(subdomains) * (1 - self.porosity(subdomains))
        mass = self.volume_integral(density, subdomains)
        mass.set_name("rock_mass")
        return mass

    def body_force(self, subdomains: list[pp.Grid]):
        """Body force.
        FIXME: See FluidMassBalanceEquations.fluid_source.
        Args:
            subdomains: List of subdomains where the body force is defined.

        Returns:
            Operator for the body force.

        """
        num_cells = sum([sd.num_cells for sd in subdomains])
        vals = np.zeros(num_cells * self.nd)
        source = pp.ad.Array(vals, "body_force")
        return source

class ConstitutiveEquationsForceBalance:
    """Class for constitutive equations for force balance equations.

    """



class VariablesForceBalance:
    """
    Variables for mixed-dimensional force balance and fracture deformation:
        Displacement in matrix and on fracture-matrix interfaces.
        Fracture contact traction.

    .. note::
        Implementation postponed till Veljko's more convenient SystemManager is available.

    """

    def __init__(self, mdg: pp.MortarDiscretization, system_manager: pp.ad.SystemManager):
        self.mdg = mdg
        self.system_manager = system_manager
        self._var_names = ["displacement", "contact_traction"]
        self._nd = mdg.dim_max()

    def set_variables(self):
        """Set variables for the subdomains and interfaces.

        The following variables are set:
            - Displacement in the matrix.
            - Displacement on fracture-matrix interfaces.
            - Fracture contact traction.
        See individual variable methods for details.
        """
        matrix_subdomains = self.mdg.subdomains(dim=self.nd)
        fracture_subdomains = self.mdg.subdomains(dim=self.nd - 1)
        interfaces = self.mdg.interfaces(dim=self.nd - 1)
        matrix_var = self.matrix_displacement(matrix_subdomains)
        intf_var = self.interface_displacement(interfaces)
        contact_var = self.fracture_contact_traction(fracture_subdomains)
        self.system_manager.set_variable(matrix_var, (matrix_subdomains, "cells", self.nd))
        self.system_manager.set_variable(intf_var, (interfaces, "cells", self.nd))
        self.system_manager.set_variable(contact_var, (interfaces, "cells", self.nd))

    def matrix_displacement(self, subdomains: list[pp.Grid]):
        """Displacement in the matrix.

        Args:
            subdomains: List of subdomains where the displacement is defined. Only known usage
                is for the matrix domain(s).

        Returns:
            Operator for the displacement.

        """
        displacement = self._eq_manager.merge_variables([(sd, "displacement") for sd in subdomains])
        return displacement

    def interface_displacement(self, interfaces: list[pp.MortarGrid]):
        """Displacement on fracture-matrix interfaces.

        Args:
            interfaces: List of interfaces where the displacement is defined.

        Returns:
            Operator for the displacement.

        """
        displacement = self._eq_manager.merge_variables([(intf, "displacement") for intf in interfaces])
        return displacement

    def contact_traction(self, subdomains: list[pp.Grid]):
        """Fracture contact traction.

        Args:
            subdomains: List of subdomains where the contact traction is defined. Only known usage
                is for the fracture subdomains.

        Returns:
            Variable for fracture contact traction.

        """
        contact_traction = self._eq_manager.merge_variables([(sd, "traction") for sd in subdomains])

        return contact_traction
