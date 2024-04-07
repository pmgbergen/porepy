import numpy as np
import porepy as pp
from typing import Callable, Sequence
import porepy.composite as ppc
from porepy.models.compositional_flow import SecondaryEquationsMixin


class LiquidDriesnerCorrelations(ppc.AbstractEoS):

    @property
    def obl(self):
        return self._obl

    @obl.setter
    def obl(self, obl):
        self._obl = obl

    def kappa(
        self,
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

        h_scale = 1.0 / 1000.0
        p_scale = 1.0 / 100000.0
        if not hasattr(self, "_obl"):
            raise AttributeError(
                "The operator-based linearization (OBL) object is not present. Set up a unique OBL of the type DriesnerBrineOBL"
            )

        p, h, z_NaCl = thermodynamic_input
        par_points = np.array((z_NaCl, h * h_scale, p * p_scale)).T
        self.obl.sample_at(par_points)
        n = len(p)  # same for all input (number of cells)

        # specific volume of phase
        v = self.obl.sampled_could.point_data["nu_l"]
        dvdz = self.obl.sampled_could.point_data["grad_nu_l"][:, 0]
        dvdH = self.obl.sampled_could.point_data["grad_nu_l"][:, 1] * h_scale
        dvdp = self.obl.sampled_could.point_data["grad_nu_l"][:, 2] * p_scale
        dv = np.vstack((dvdp, dvdH, dvdz))

        # specific enthalpy of phase
        h = self.obl.sampled_could.point_data["H_l"]
        dhdz = self.obl.sampled_could.point_data["grad_H_l"][:, 0]
        dhdH = self.obl.sampled_could.point_data["grad_H_l"][:, 1] * h_scale
        dhdp = self.obl.sampled_could.point_data["grad_H_l"][:, 2] * p_scale
        dh = np.vstack((dhdp, dhdH, dhdz))

        # dynamic viscosity of phase
        mu = self.obl.sampled_could.point_data["mu_l"]
        dmudz = self.obl.sampled_could.point_data["grad_mu_l"][:, 0]
        dmudH = self.obl.sampled_could.point_data["grad_mu_l"][:, 1] * h_scale
        dmudp = self.obl.sampled_could.point_data["grad_mu_l"][:, 2] * p_scale
        dmu = np.vstack((dmudp, dmudH, dmudz))

        # thermal conductivity of phase
        kappa, dkappa = self.kappa(*thermodynamic_input)  # (n,), (3, n) array

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


class GasDriesnerCorrelations(ppc.AbstractEoS):

    @property
    def obl(self):
        return self._obl

    @obl.setter
    def obl(self, obl):
        self._obl = obl

    def kappa(
        self,
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

        h_scale = 1.0 / 1000.0
        p_scale = 1.0 / 100000.0
        if not hasattr(self, "_obl"):
            raise AttributeError(
                "The operator-based linearization (OBL) object is not present. Set up a unique OBL of the type DriesnerBrineOBL"
            )

        p, h, z_NaCl = thermodynamic_input
        par_points = np.array((z_NaCl, h * h_scale, p * p_scale)).T
        self.obl.sample_at(par_points)
        n = len(p)  # same for all input (number of cells)

        # specific volume of phase
        v = self.obl.sampled_could.point_data["nu_v"]
        dvdz = self.obl.sampled_could.point_data["grad_nu_v"][:, 0]
        dvdH = self.obl.sampled_could.point_data["grad_nu_v"][:, 1] * h_scale
        dvdp = self.obl.sampled_could.point_data["grad_nu_v"][:, 2] * p_scale
        dv = np.vstack((dvdp, dvdH, dvdz))

        # specific enthalpy of phase
        h = self.obl.sampled_could.point_data["H_v"]
        dhdz = self.obl.sampled_could.point_data["grad_H_v"][:, 0]
        dhdH = self.obl.sampled_could.point_data["grad_H_v"][:, 1] * h_scale
        dhdp = self.obl.sampled_could.point_data["grad_H_v"][:, 2] * p_scale
        dh = np.vstack((dhdp, dhdH, dhdz))

        # dynamic viscosity of phase
        mu = self.obl.sampled_could.point_data["mu_v"]
        dmudz = self.obl.sampled_could.point_data["grad_mu_v"][:, 0]
        dmudH = self.obl.sampled_could.point_data["grad_mu_v"][:, 1] * h_scale
        dmudp = self.obl.sampled_could.point_data["grad_mu_v"][:, 2] * p_scale
        dmu = np.vstack((dmudp, dmudH, dmudz))

        # thermal conductivity of phase
        kappa, dkappa = self.kappa(*thermodynamic_input)  # (n,), (3, n) array

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


class FluidMixture(ppc.FluidMixtureMixin):
    """Mixture mixin creating the brine mixture with two components."""

    enthalpy: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Provided by :class:`~porepy.models.compositional_flow.VariablesEnergyBalance`."""

    def get_components(self) -> Sequence[ppc.Component]:
        """Setting H20 as first component in Sequence makes it the reference component.
        z_H20 will be eliminated."""
        species = ppc.load_species(["H2O", "NaCl"])
        components = [ppc.Component.from_species(s) for s in species]
        return components

    def get_phase_configuration(
        self, components: Sequence[ppc.Component]
    ) -> Sequence[tuple[ppc.AbstractEoS, int, str]]:
        eos_L = LiquidDriesnerCorrelations(components)
        eos_G = GasDriesnerCorrelations(components)
        # assign common OBL object
        eos_L.obl = self.obl
        eos_G.obl = self.obl
        return [(eos_L, 0, "liq"), (eos_G, 1, "gas")]

    def dependencies_of_phase_properties(
        self, phase: ppc.Phase
    ) -> Sequence[Callable[[pp.GridLikeSequence], pp.ad.Operator]]:
        z_NaCl = [
            comp.fraction
            for comp in self.fluid_mixture.components
            if comp != self.fluid_mixture.reference_component
        ]
        return [self.pressure, self.enthalpy] + z_NaCl

    def set_components_in_phases(
        self, components: Sequence[ppc.Component], phases: Sequence[ppc.Phase]
    ) -> None:
        """By default, the unified assumption is applied: all components are present
        in all phases."""
        super().set_components_in_phases(components, phases)


class SecondaryEquations(SecondaryEquationsMixin):
    """Mixin to provide expressions for dangling variables.

    The CF framework has the following quantities always as independent variables:

    - independent phase saturations
    - partial fractions (independent since no equilibrium)
    - temperature (needs to be expressed through primary variables in this model, since
      no p-h equilibrium)

    """

    dependencies_of_phase_properties: Sequence[
        Callable[[pp.GridLikeSequence], pp.ad.Operator]
    ]
    """Defined in the Brine mixture mixin."""

    temperature: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Provided by :class:`~porepy.models.energy_balance.VariablesEnergyBalance`."""

    def gas_saturation_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        h_scale = 1.0 / 1000.0
        p_scale = 1.0 / 100000.0
        p, h, z_NaCl = thermodynamic_dependencies
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h * h_scale, p * p_scale)).T
        self.obl.sample_at(par_points)

        # Gas saturationn
        S_v = self.obl.sampled_could.point_data["S_v"]
        dS_vdz = self.obl.sampled_could.point_data["grad_S_v"][:, 0]
        dS_vdH = self.obl.sampled_could.point_data["grad_S_v"][:, 1] * h_scale
        dS_vdp = self.obl.sampled_could.point_data["grad_S_v"][:, 2] * p_scale
        dS_v = np.vstack((dS_vdp, dS_vdH, dS_vdz))
        return S_v, dS_v

    def temperature_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        h_scale = 1.0 / 1000.0
        p_scale = 1.0 / 100000.0
        p, h, z_NaCl = thermodynamic_dependencies
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h * h_scale, p * p_scale)).T
        self.obl.sample_at(par_points)

        # Gas saturationn
        T = self.obl.sampled_could.point_data["Temperature"]
        dTdz = self.obl.sampled_could.point_data["grad_Temperature"][:, 0]
        dTdH = self.obl.sampled_could.point_data["grad_Temperature"][:, 1] * h_scale
        dTdp = self.obl.sampled_could.point_data["grad_Temperature"][:, 2] * p_scale
        dT = np.vstack((dTdp, dTdH, dTdz))
        return T, dT

    def H2O_liq_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        h_scale = 1.0 / 1000.0
        p_scale = 1.0 / 100000.0
        p, h, z_NaCl = thermodynamic_dependencies
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h * h_scale, p * p_scale)).T
        self.obl.sample_at(par_points)

        # Gas saturationn
        X_w = 1.0 - self.obl.sampled_could.point_data["Xl"]
        dX_wdz = -self.obl.sampled_could.point_data["grad_Xl"][:, 0]
        dX_wdH = -self.obl.sampled_could.point_data["grad_Xl"][:, 1] * h_scale
        dX_wdp = -self.obl.sampled_could.point_data["grad_Xl"][:, 2] * p_scale
        dX_w = np.vstack((dX_wdp, dX_wdH, dX_wdz))
        return X_w, dX_w

    def NaCl_liq_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        h_scale = 1.0 / 1000.0
        p_scale = 1.0 / 100000.0
        p, h, z_NaCl = thermodynamic_dependencies
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h * h_scale, p * p_scale)).T
        self.obl.sample_at(par_points)

        # Gas saturationn
        X_s = self.obl.sampled_could.point_data["Xl"]
        dX_sdz = self.obl.sampled_could.point_data["grad_Xl"][:, 0]
        dX_sdH = self.obl.sampled_could.point_data["grad_Xl"][:, 1] * h_scale
        dX_sdp = self.obl.sampled_could.point_data["grad_Xl"][:, 2] * p_scale
        dX_s = np.vstack((dX_sdp, dX_sdH, dX_sdz))
        return X_s, dX_s

    def H2O_gas_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        h_scale = 1.0 / 1000.0
        p_scale = 1.0 / 100000.0
        p, h, z_NaCl = thermodynamic_dependencies
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h * h_scale, p * p_scale)).T
        self.obl.sample_at(par_points)

        # Gas saturationn
        X_w = 1.0 - self.obl.sampled_could.point_data["Xv"]
        dX_wdz = -self.obl.sampled_could.point_data["grad_Xv"][:, 0]
        dX_wdH = -self.obl.sampled_could.point_data["grad_Xv"][:, 1] * h_scale
        dX_wdp = -self.obl.sampled_could.point_data["grad_Xv"][:, 2] * p_scale
        dX_w = np.vstack((dX_wdp, dX_wdH, dX_wdz))
        return X_w, dX_w

    def NaCl_gas_func(
        self,
        *thermodynamic_dependencies: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        h_scale = 1.0 / 1000.0
        p_scale = 1.0 / 100000.0
        p, h, z_NaCl = thermodynamic_dependencies
        assert len(p) == len(h) == len(z_NaCl)
        par_points = np.array((z_NaCl, h * h_scale, p * p_scale)).T
        self.obl.sample_at(par_points)

        # Gas saturationn
        X_s = self.obl.sampled_could.point_data["Xv"]
        dX_sdz = self.obl.sampled_could.point_data["grad_Xv"][:, 0]
        dX_sdH = self.obl.sampled_could.point_data["grad_Xv"][:, 1] * h_scale
        dX_sdp = self.obl.sampled_could.point_data["grad_Xv"][:, 2] * p_scale
        dX_s = np.vstack((dX_sdp, dX_sdH, dX_sdz))
        return X_s, dX_s

    def set_equations(self) -> None:
        subdomains = self.mdg.subdomains()

        chi_functions_map = {
            "H2O_liq": self.H2O_liq_func,
            "NaCl_liq": self.NaCl_liq_func,
            "H2O_gas": self.H2O_gas_func,
            "NaCl_gas": self.NaCl_gas_func,
        }

        ### Providing constitutive law for gas saturation based on correlation
        rphase = self.fluid_mixture.reference_phase  # liquid phase
        # gas phase is independent
        independent_phases = [p for p in self.fluid_mixture.phases if p != rphase]

        for phase in independent_phases:
            self.eliminate_by_constitutive_law(
                phase.saturation,  # callable giving saturation on ``subdomains``
                self.dependencies_of_phase_properties(
                    phase
                ),  # callables giving primary variables on subdoains
                self.gas_saturation_func,  # numerical function implementing correlation
                subdomains,  # all subdomains on which to eliminate s_gas
                # dofs = {'cells': 1},  # default value
            )

        ### Providing constitutive laws for partial fractions based on correlations
        for phase in self.fluid_mixture.phases:
            for comp in phase:
                self.eliminate_by_constitutive_law(
                    phase.partial_fraction_of[comp],
                    self.dependencies_of_phase_properties(phase),
                    chi_functions_map[comp.name + "_" + phase.name],
                    subdomains,
                )

        ### Provide constitutive law for temperature
        self.eliminate_by_constitutive_law(
            self.temperature,
            self.dependencies_of_phase_properties(rphase),  # since same for all.
            self.temperature_func,
            subdomains,
        )
