import numpy as np
import porepy.composite as ppc

def dummy_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Dummy function for computing the values and derivatives of a thermodynamic
    property.

    Correlations are designed to depend on p, h, z_H2O and z_CO2,
    i.e. ``thermodynamic_properties`` will be a 4-tuple of ``(nc,)`` arrays
    """
    nc = len(thermodynamic_dependencies[0])
    vals = np.ones(nc)
    # row-wise storage of derivatives, (4, nc) array
    diffs = np.ones((len(thermodynamic_dependencies), nc))

    return vals, diffs

class DriesnerCorrelations(ppc.AbstractEoS):
    """Class implementing the calculation of thermodynamic properties.

    Note:
        By thermodynamic properties, this framework refers to the below
        indicated quantities, and **not** quantitie which are variables.

        Fractions (partial and saturations) and other intensive quantities like
        temperature need a separate treatment because they are always modelled as
        variables, whereas properties are always dependent expressions.

    """

    def compute_phase_state(
            self, phase_type: int, *thermodynamic_input: np.ndarray
    ) -> ppc.PhaseState:
        """Function will be called to compute the values for a phase.
        ``phase_type`` indicates the phsycal type (0 - liq, 1 - gas).
        ``thermodynamic_dependencies`` are as defined by the user.
        """

        p, h, z_NaCl = thermodynamic_input
        n = len(p)  # same for all input (number of cells)

        # specific volume of phase
        v, dv = dummy_func(p, h, z_NaCl)  # (n,), (3, n) array
        # NOTE specific molar density rho not required since always computed as
        # reciprocal of v by PhaseState class

        # specific enthalpy of phase
        h, dh = dummy_func(p, h, z_NaCl)  # (n,), (3, n) array
        # dynamic viscosity of phase
        mu, dmu = dummy_func(p, h, z_NaCl)  # (n,), (3, n) array
        # thermal conductivity of phase
        kappa, dkappa = dummy_func(p, h, z_NaCl)  # (n,), (3, n) array

        # Fugacity coefficients
        # not required for this formulation, since no equilibrium equations
        # just show-casing it here
        phis = np.empty((2, n))  # (2, n) array  (2 components)
        dphis = np.empty(
            (2, 3, n)
        )  # (2, 3, n)  array (2 components, 3 dependencies, n cells)

        return ppc.PhaseState(
            phasetype=phase_type,
            v=v,
            dv=dv,
            h=h,
            dh=dh,
            mu=mu,
            dmu=dmu,
            kappa=kappa,
            dkappa=dkappa,
            phis=phis,
            dphis=dphis,
        )

def gas_saturation_func(
    *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    p, h, z_NaCl = thermodynamic_dependencies
    # same for all input (number of cells)
    assert len(p) == len(h) == len(z_NaCl)
    n = len(p)

    nc = len(thermodynamic_dependencies[0])
    vals = np.zeros(nc)
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))

    return vals, diffs


def temperature_func(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_NaCl = thermodynamic_dependencies
    # same for all input (number of cells)
    assert len(p) == len(h) == len(z_NaCl)
    n = len(p)

    nc = len(thermodynamic_dependencies[0])

    factor  = 773.5 / 3.0e6
    vals = np.array(h) * factor
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.ones((len(thermodynamic_dependencies), nc))
    diffs[1, :] = 1.0*factor
    return vals, diffs

def H2O_liq_func(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_NaCl = thermodynamic_dependencies
    # same for all input (number of cells)
    assert len(p) == len(h) == len(z_NaCl)
    n = len(p)

    nc = len(thermodynamic_dependencies[0])
    vals = np.array(1-z_NaCl)
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = -1.0
    return vals, diffs

def NaCl_liq_func(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_NaCl = thermodynamic_dependencies
    # same for all input (number of cells)
    assert len(p) == len(h) == len(z_NaCl)
    n = len(p)

    nc = len(thermodynamic_dependencies[0])
    vals = np.array(z_NaCl)
    # row-wise storage of derivatives, (4, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = +1.0
    return vals, diffs

def H2O_gas_func(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_NaCl = thermodynamic_dependencies
    # same for all input (number of cells)
    assert len(p) == len(h) == len(z_NaCl)
    n = len(p)

    nc = len(thermodynamic_dependencies[0])
    vals = np.array(1-z_NaCl)
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = -1.0
    return vals, diffs

def NaCl_gas_func(
        *thermodynamic_dependencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p, h, z_NaCl = thermodynamic_dependencies
    # same for all input (number of cells)
    assert len(p) == len(h) == len(z_NaCl)
    n = len(p)

    nc = len(thermodynamic_dependencies[0])
    vals = np.array(z_NaCl)
    # row-wise storage of derivatives, (3, nc) array
    diffs = np.zeros((len(thermodynamic_dependencies), nc))
    diffs[2, :] = +1.0
    return vals, diffs

chi_functions_map = {'H2O_liq': H2O_liq_func,'NaCl_liq': NaCl_liq_func,'H2O_gas': H2O_gas_func,'NaCl_gas': NaCl_gas_func}

class LiquidLikeTracerCorrelations(ppc.AbstractEoS):
    """Class implementing the calculation of thermodynamic properties.

    Note:
        By thermodynamic properties, this framework refers to the below
        indicated quantities, and **not** quantitie which are variables.

        Fractions (partial and saturations) and other intensive quantities like
        temperature need a separate treatment because they are always modelled as
        variables, whereas properties are always dependent expressions.

    """

    def rho_func(self,
            *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        # vals = (55508.435061792) * np.ones(nc)
        vals = (1000.0) * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def v_func(self,
            *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        vals, diffs = self.rho_func(*thermodynamic_dependencies)
        vals = 1.0 / vals
        return vals, diffs

    def mu_func(self,
            *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = (0.001) * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def h(self,
            *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = (1000.0) * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def kappa(self,
            *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = (0.5) * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def compute_phase_state(
            self, phase_type: int, *thermodynamic_input: np.ndarray
    ) -> ppc.PhaseState:
        """Function will be called to compute the values for a phase.
        ``phase_type`` indicates the phsycal type (0 - liq, 1 - gas).
        ``thermodynamic_dependencies`` are as defined by the user.
        """

        p, h, z_NaCl = thermodynamic_input

        # same for all input (number of cells)
        assert len(p) == len(h) == len(z_NaCl)
        nc = len(p)

        # specific volume of phase
        v, dv = self.v_func(*thermodynamic_input)  # (n,), (3, n) array

        # specific enthalpy of phase
        h, dh = self.h(*thermodynamic_input)  # (n,), (3, n) array
        # dynamic viscosity of phase
        mu, dmu = self.mu_func(*thermodynamic_input)  # (n,), (3, n) array
        # thermal conductivity of phase
        kappa, dkappa = self.kappa(*thermodynamic_input)  # (n,), (3, n) array

        # Fugacity coefficients
        # not required for this formulation, since no equilibrium equations
        # just show-casing it here
        phis = np.empty((2, nc))  # (2, n) array  (2 components)
        dphis = np.empty(
            (2, 3, nc)
        )  # (2, 3, n)  array (2 components, 3 dependencies, n cells)

        return ppc.PhaseState(
            phasetype=phase_type,
            v=v,
            dv=dv,
            h=h,
            dh=dh,
            mu=mu,
            dmu=dmu,
            kappa=kappa,
            dkappa=dkappa,
            phis=phis,
            dphis=dphis,
        )

class GasLikeTracerCorrelations(ppc.AbstractEoS):
    """Class implementing the calculation of thermodynamic properties.

    Note:
        By thermodynamic properties, this framework refers to the below
        indicated quantities, and **not** quantitie which are variables.

        Fractions (partial and saturations) and other intensive quantities like
        temperature need a separate treatment because they are always modelled as
        variables, whereas properties are always dependent expressions.

    """

    def rho_func(self,
            *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = 1.0 * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def v_func(self,
            *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        vals, diffs = self.rho_func(*thermodynamic_dependencies)
        vals = 1.0 / vals
        return vals, diffs

    def mu_func(self,
            *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = (0.00001) * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def h(self,
            *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = (1000.0) * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def kappa(self,
            *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        nc = len(thermodynamic_dependencies[0])
        vals = (1.0e-2) * np.ones(nc)
        # row-wise storage of derivatives, (4, nc) array
        diffs = np.zeros((len(thermodynamic_dependencies), nc))
        return vals, diffs

    def compute_phase_state(
            self, phase_type: int, *thermodynamic_input: np.ndarray
    ) -> ppc.PhaseState:
        """Function will be called to compute the values for a phase.
        ``phase_type`` indicates the phsycal type (0 - liq, 1 - gas).
        ``thermodynamic_dependencies`` are as defined by the user.
        """

        p, h, z_NaCl = thermodynamic_input

        # same for all input (number of cells)
        assert len(p) == len(h) == len(z_NaCl)
        nc = len(p)

        # specific volume of phase
        v, dv = self.v_func(*thermodynamic_input)  # (n,), (3, n) array

        # specific enthalpy of phase
        h, dh = self.h(*thermodynamic_input)  # (n,), (3, n) array
        # dynamic viscosity of phase
        mu, dmu = self.mu_func(*thermodynamic_input)  # (n,), (3, n) array
        # thermal conductivity of phase
        kappa, dkappa = self.kappa(*thermodynamic_input)  # (n,), (3, n) array

        # Fugacity coefficients
        # not required for this formulation, since no equilibrium equations
        # just show-casing it here
        phis = np.empty((2, nc))  # (2, n) array  (2 components)
        dphis = np.empty(
            (2, 3, nc)
        )  # (2, 3, n)  array (2 components, 3 dependencies, n cells)

        return ppc.PhaseState(
            phasetype=phase_type,
            v=v,
            dv=dv,
            h=h,
            dh=dh,
            mu=mu,
            dmu=dmu,
            kappa=kappa,
            dkappa=dkappa,
            phis=phis,
            dphis=dphis,
        )

