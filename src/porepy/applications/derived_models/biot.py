"""
Model class for poromechanical equations under the Biot assumptions [1, 2, 3].

This class can be seen as a subset of the full poromechanical system of equations,
where certain simplifications are introduced so that the classical system of
equations following Biot's consolidation theory is recovered.

In particular, we set the fluid density as constant by requiring the fluid
compressibility :math:`c_f` to be zero,

.. math::

    \rho := \rho_0 \exp{c_f (p - p_0)} = \rho_0,

and define the poromechanics porosity as:

.. math::

    \phi(p, \mathbf{u}) :=
        \phi_0
        + S_\varepsilon (p - p_0)
        + \alpha \mathrm{div}(\mathbf{u}),

where :math:`\rho_0`, :math:`\phi_0`, and :math:`p_0` are the reference density,
porosity, and pressure, and :math:`S_\varepsilon`, :math:`\alpha`, :math:`p`,
and :math:`\mathbf{u}`, are the specific storage, Biot's coefficient, pressure, and
displacement, respectively.

Note, however, that the above simplifications do not reflect the actual _physical_
assumptions considered in Biot's theory. e.g., the fluid is actually slightly
compressible rather than incompressible. Thus, the above simplifications must be seen
as implementation shortcuts rather than proper physical assumptions.

For a domain without fractures and in the absence of gravity, the governing equations
solved by this class are given by:

    - Momentum conservation:
    .. math::

        div(\sigma) = \mathbf{F},

    - Generalized Hooke's law:
    .. math::

        \sigma = \mathcal{C} : \mathbf{\mathrm{grad}}(\mathbf{u}) - \alpha p \mathbf{I},

    - Mass conservation:
    .. math::

        \rho_0  \left( S_\varepsilon \frac{\partial p}{\partial t} + \alpha
        \frac{\partial}{\partial t} \mathrm{div}(\mathbf{u}) \right) + \mathrm{div}(
        \rho_0 \mathbf{q}) = f,

    - Darcy flux:
    .. math::

        \mathbf{q} = - \frac{\mathcal{K}}{\mu_f} \mathrm{\mathbf{grad}}(p),

where :math:`\sigma` is the poroelastic (total) stress, :math:`\mathbf{F}` is a
vector source (usually a body force), :math:`\mathcal{C}` is the stiffness matrix (
written in terms of Lamé parameters in the case of isotropic solids),
and :math:`\mathbf{I}` is the identity matrix. In the mass conservation equation,
:math:`\mathbf{q}` is the Darcy flux, and :math:`f` is an external source of fluid mass.
Finally, :math:`\mathcal{K}` and :math:`\mu_f` denote the intrinsic permeability and the
fluid dynamic viscosity.

References:

    - [1] Biot, M. A. (1941). General theory of three‐dimensional consolidation.
      Journal of applied physics, 12(2), 155-164.

    - [2] Lewis, R. W., & Schrefler, B. A. (1987). The finite element method in the
      deformation and consolidation of porous media.

    - [3] Coussy, O. (2004). Poromechanics. John Wiley & Sons. ISO 690.

"""


import porepy as pp
from porepy.models.poromechanics import (
    ConstitutiveLawsPoromechanics,
    Poromechanics,
    SolutionStrategyPoromechanics,
)


class SolutionStrategyBiot(SolutionStrategyPoromechanics):
    """Modified solution strategy for the Biot class"""

    def set_materials(self):
        """Set the material constants."""
        super().set_materials()
        # Check that fluid compressibility is zero, otherwise Biot class doesn't hold
        assert self.fluid.compressibility() == 0


class ConstitutiveLawsBiot(
    pp.constitutive_laws.SpecificStorage,
    pp.constitutive_laws.BiotPoroMechanicsPorosity,
    ConstitutiveLawsPoromechanics,
):
    ...


class BiotPoromechanics(  # type: ignore[misc]
    ConstitutiveLawsBiot,
    SolutionStrategyBiot,
    Poromechanics,
):
    ...
