"""
Model class for poromechanic equations under the Biot assumptions [1, 2, 3].

This class can be seen as a subset of the full poromechanical system of equations,
where certain simplifications are introduced so that the classical system of
equations following Biot's consolidation is recovered.

In particular, we set the fluid density as constant:

    rho = rho_0,

and define the poromechanics porosity as:

    phi(p, u) = phi_0 + S_epsilon * (p - p0) + alpha_biot * div(u),

where rho_0, phi_0, and p_0 are the reference density, porosity, and pressure,
and S_epsilon, alpha_biot, p, and u, are the specific storage, Biot's coefficient,
pressure and displacement, respectively.

For a domain without fractures and in the absence of gravity, the governing equations
solved by this class are given by:

    Momentum conservation:
        div(sigma) = F,

    Generalized Hooke's law:
        sigma = C : grad(u) - alpha_biot * p * I,

    Mass conservation:
        rho_0 * (S_epsilon p_t + alpha_biot * div(u)_t) + div(rho_0 * q) = f,

    Darcy flux:
        q = - (K/mu) * grad(p),

where sigma is the poroelastic (total) stress, F is a vector (usually a body) force,
C is the stiffness matrix (written in terms of Lamé parameters in the case of
isotropic solids), and I is the identity matrix. In the mass conservation equation,
x_t represents the time derivative of the quantity x, q is the Darcy flux, and f is
an external source of fluid mass. Finally, K and mu denote the intrinsic permeability
and the fluid dynamic viscosity.

Examples:



References:

    [1] Biot, M. A. (1941). General theory of three‐dimensional consolidation.
      Journal of applied physics, 12(2), 155-164.

    [2] Lewis, R. W., & Schrefler, B. A. (1987). The finite element method in the
      deformation and consolidation of porous media.

    [3] Coussy, O. (2004). Poromechanics. John Wiley & Sons. ISO 690

"""


import porepy as pp
from porepy.models.poromechanics import ConstitutiveLawsPoromechanics, Poromechanics


class ConstitutiveLawsBiot(
    pp.constitutive_laws.SpecificStorage,
    pp.constitutive_laws.BiotPoromechanicsPorosity,
    pp.constitutive_laws.ConstantFluidDensity,
    ConstitutiveLawsPoromechanics,
):
    ...


class BiotPoromechanics(  # type: ignore[misc]
    ConstitutiveLawsBiot,
    Poromechanics,
):
    ...
