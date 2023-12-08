"""This module contains a classes representing multiphase multicomponent mixtures using
the unified formulation for phase stability and split calculations.

The base class :class:`BasicMixture` is a starting point to derive mixture classes
including various physics.

Fractional variables in the compositional framework are set and initiated by the
basic class during set-up.

The intensive state of a mixture is characterized by

- pressure,
- temperature and
- feed fractions (see component :attr:`~porepy.composite.component.Component.fraction`).

While the feed fractions are initialized by this framework, the modeller must provide
values once a mixture is set up.

Mixtures can be set up using an external system
(:class:`~porepy.numerics.ad.equation_system.EquationSystem`) to couple the
thermodynamic equilibrium problem with other physics.

They can also work on a stand-alone basis, where the mixture uses PorePy's AD framework
internally.

The crucial set-up method is :meth:`BasicMixture.set_up`, introducing new degrees of
freedom into the system.

Mixtures are set up using various :class:`~porepy.composite.phase.Phase` and
:class:`~porepy.composite.component.Component` instances. They have consistent
thermodynamic properties which can be computed, accessed and combined with other models.

Important:
    As of now it is recommended to use at least 2 components and 2 phases.

    Due to the phase rule ``F=C-P+2``, the thermodynamic degree of freedom reduces to 1
    if ``C=1`` and ``P=2`` f.e., causing the unified formulation to lose its
    injectivity. This leads to a potentially singular Jacobian of the system.

Once a mixture is modelled, flash calculations can be performed using a
:class:`~porepy.composite.flash.Flash` instance.

"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generator, Literal, Optional, overload

import numpy as np
import scipy.sparse as sps

import porepy as pp
from porepy.numerics.ad.operator_functions import NumericType

from ._core import COMPOSITIONAL_VARIABLE_SYMBOLS
from .component import Component, Compound
from .composite_utils import normalize_fractions, safe_sum
from .phase import Phase, PhaseProperties

__all__ = ["ThermodynamicState", "BasicMixture", "NonReactiveMixture"]

FlashSystemDict = dict[
    Literal["equations", "primary-variables", "secondary-variables"], list[str]
]
"""A type alias for subsystem dictionaries which contain:

- 'equations': A list of names of equations belonging to this subsystem.
- 'primary_vars': A list of names of primary variables in this subsystem.
- 'secondary_vars': A list of names of secondary variables in this subsystem.

"""


def _process(
    q: pp.ad.AdArray, as_ad: bool, projection: Optional[sps.spmatrix] = None
) -> NumericType:
    """Auxiliary function to process Ad-arrays."""
    if as_ad:
        if projection is not None:
            q.jac = q.jac * projection
        return q
    else:
        return q.val


@dataclass
class ThermodynamicState:
    """Data class for storing the thermodynamic state of a mixture..

    The name of the attributes ``p, T, X`` is designed such that they can be used
    as keyword arguments for

    1. :meth:`~porepy.composite.phase.AbstractEoS.compute`,
    2. :meth:`~porepy.composite.phase.Phase.compute_properties` and
    3. :meth:`~porepy.composite.mixture.BasicMixture.compute_properties`,

    and should not be meddled with (especially capital ``X``).

    Important:
        Upon inheritance, always provide default values for :meth:`initialize` to work.

    """

    p: NumericType = 0.0
    """Pressure."""

    T: NumericType = 0.0
    """Temperature."""

    h: NumericType = 0.0
    """Specific molar enthalpy of the mixture."""

    v: NumericType = 0.0
    """Molar volume of the mixture."""

    rho: NumericType = 0.0
    """Molar density of the mixture.

    As of now, density is always considered a secondary expression and never an
    independent variable.

    """

    z: list[NumericType] = field(default_factory=lambda: [])
    """Feed fractions per component. The first fraction is always the feed fraction of
    the reference component."""

    y: list[NumericType] = field(default_factory=lambda: [])
    """Phase fractions per phase. The first fraction is always the phase fraction of the
    reference phase."""

    s: list[NumericType] = field(default_factory=lambda: [])
    """Volume fractions (saturations) per phase. The first fraction is always the phase
    saturation of the reference phase."""

    X: list[list[NumericType]] = field(default_factory=lambda: [[]])
    """Phase compositions per phase (outer list) per component in phase (inner list)."""

    def __str__(self) -> str:
        """Returns a string representation of the stored state values."""
        vals = self.values()
        nc = len(self.z)
        np = len(self.y)

        msg = f"Thermodynamic state with {nc} components and {np} phases:\n"
        msg += f"\nIntensive state:\n\tPressure: {vals.p}\n\tTemperature: {vals.T}"
        for i, z in enumerate(vals.z):
            msg += f"\n\tFeed fraction {i}: {z}"
        for j, y in enumerate(vals.y):
            msg += f"\n\tPhase fraction {j}: {y}"
        for j, s in enumerate(vals.s):
            msg += f"\n\tPhase saturation {j}: {s}"
        for j in range(np):
            msg += f"\n\tComposition phase {j}:"
            for i in range(nc):
                msg += f"\n\t\t Component {i}: {vals.X[j][i]}"
        msg += (
            f"\nExtensive state:\n\tSpec. Enthalpy: {vals.h}"
            + f"\n\tMol. Density: {vals.rho}\n\tMol. Volume: {vals.v}"
        )

        return msg

    def diff(self, other: ThermodynamicState) -> ThermodynamicState:
        """Returns a state containing the absolute difference between this instance
        and another state.

        The difference is calculated per state function and fraction and uses only
        values (no derivatives, if any is given as an AD-array).

        Parameters:
            other: The other thermodynamic state.

        Returns:
            A new data class instance containing absolute difference values.

        """
        sv = self.values()
        ov = other.values()

        p = np.abs(sv.p - ov.p)
        T = np.abs(sv.T - ov.T)
        h = np.abs(sv.h - ov.h)
        v = np.abs(sv.v - ov.v)
        rho = np.abs(sv.rho - ov.rho)
        z = [np.abs(sz - oz) for sz, oz in zip(sv.z, ov.z)]
        y = [np.abs(sy - oy) for sy, oy in zip(sv.y, ov.y)]
        s = [np.abs(ss - os) for ss, os in zip(sv.s, ov.s)]
        X = [[np.abs(sx - ox) for sx, ox in zip(Xs, Xo)] for Xs, Xo in zip(sv.X, ov.X)]

        return ThermodynamicState(p=p, T=T, h=h, v=v, rho=rho, z=z, y=y, s=s, X=X)

    def values(self) -> ThermodynamicState:
        """Returns a derivative-free state in case any state function is stored as
        an :class:`~porepy.numerics.ad.forward_mode.AdArray`."""

        p = self.p.val if isinstance(self.p, pp.ad.AdArray) else self.p
        T = self.T.val if isinstance(self.T, pp.ad.AdArray) else self.T
        h = self.h.val if isinstance(self.h, pp.ad.AdArray) else self.h
        v = self.v.val if isinstance(self.v, pp.ad.AdArray) else self.v
        rho = self.rho.val if isinstance(self.rho, pp.ad.AdArray) else self.rho
        z = [z.val if isinstance(z, pp.ad.AdArray) else z for z in self.z]
        y = [y.val if isinstance(y, pp.ad.AdArray) else y for y in self.y]
        s = [s.val if isinstance(s, pp.ad.AdArray) else s for s in self.s]
        X = [
            [x.val if isinstance(x, pp.ad.AdArray) else x for x in x_j]
            for x_j in self.X
        ]

        return ThermodynamicState(p=p, T=T, h=h, v=v, rho=rho, z=z, y=y, s=s, X=X)

    @classmethod
    def initialize(
        cls,
        num_comp: int = 1,
        num_phases: int = 1,
        num_vals: int = 1,
        as_ad: bool = False,
        is_independent: Optional[
            list[Literal["p", "T", "z_r", "z_i", "s_r", "s_i", "y_r"]]
        ] = None,
        values_from: Optional[ThermodynamicState] = None,
    ) -> ThermodynamicState:
        """Initializes a thermodynamic state with zero values, based on given
        configurations.

        If the AD format with derivatives is requested, the order of derivatives is
        as follows:

        1. (optional) pressure
        2. (optional) temperature
        3. (optional) feed fractions as ordered in :attr:`z`
        4. (optional) phase saturations as ordered in :attr:`vf`
        5. phase fractions as ordered in :attr:`y`
        6. phase compositions as ordered in :attr:`X`

           .. math::

               (x_{00},\\dots,x_{0, num_comp},\\dots, x_{num_phases, num_comp})

        Note:
            The default arguments are such that a derivative-free state for a p-T flash
            (fixed pressure, temperature, feed) with eliminated reference phase fraction
            is created.

        Parameters:
            num_comp: ``default=1``

                Number of components. Must be at least 1.
            num_phases: ``default=1``

                Number of phases. Must be at least 1.

                Important:
                    For the case of 1 phase, the phase fraction will never have a
                    derivative.

            num_vals: ``default=1``

                Number of values per state function. States can be vectorized using
                numpy.
            as_ad: ``default=False``

                If True, the values are initialized as
                :class:`~porepy.numerics.ad.forward_mode.AdArray` instances, with
                proper derivatives in csr format.

                If False, the values are initialized as numpy arrays with length
                ``num_vals``.
            is_independent: ``default=None``

                Some additional states can be marked as independent, meaning they are
                considered as variables and have unity as its derivative
                (hence increasing the whole Jacobian, if Ad arrays are requested)

                States which can be marked as independent include

                - ``'p'``: pressure
                - ``'T'``: temperature
                - ``'z_r'``: reference component (feed) fraction
                - ``'z_i'``: feed fractions of other components
                - ``'s_r'``: reference phase saturation
                - ``'s_i'``: saturations of other phases
                - ``'y_r'``: reference phase fraction

                Phase compositions :attr:`X` and other phase fractions are **always**
                considered independent.
            values_from: ``default=None``

                If another state structure is passed, copy the values.
                Assumes the other state structure has values **only**
                (no derivatives in form of AD-arrays).

        Raises:
            ValueError: If an unsupported state is requested in ``is_independent``.
            AssertionError: If ``num_comp,num_vals < 1`` or ``num_phases<2``.

        Returns:
            A state data structure with above configurations

        """

        assert num_phases >= 1, "Number of phases must be at least 1."
        assert num_comp >= 1, "Number of components must be at least 1."
        assert num_vals >= 1, "Number of values per state must be at least 1."

        indp = num_phases - 1  # number of independent phases

        if values_from:
            p = values_from.p
            T = values_from.T
            h = values_from.h
            v = values_from.v
            rho = values_from.rho
            z = [values_from.z[i] for i in range(num_comp)]
            y = [values_from.y[j] for j in range(num_phases)]
            s = [values_from.s[j] for j in range(num_phases)]
            X = [
                [values_from.X[j][i] for i in range(num_comp)]
                for j in range(num_phases)
            ]
        else:
            vec = np.zeros(num_vals)  # default zero values
            # default state
            p = vec.copy()
            T = vec.copy()
            h = vec.copy()
            v = vec.copy()
            rho = vec.copy()
            z = [vec.copy() for _ in range(num_comp)]
            y = [vec.copy() for _ in range(num_phases)]
            s = [vec.copy() for _ in range(num_phases)]
            X = [[vec.copy() for _ in range(num_comp)] for _ in range(num_phases)]

        # update state with derivatives if requested
        if as_ad:
            # identity derivative per independent state
            id_block = sps.identity(num_vals, dtype=float, format="lil")
            # determining the number of column blocks per independent state
            # defaults to phase compositions and independent phase fractions
            N_default = num_comp * num_phases + indp

            # The default global matrices are always created
            jac_glob_d = sps.lil_matrix((num_vals, N_default * num_vals))
            y_jacs: list[sps.lil_matrix] = list()
            X_jacs: list[list[sps.lil_matrix]] = list()
            # number of columns belonging to independent phases
            n_p = (indp) * num_vals
            # dependent phase composition
            X_jacs.append(list())
            for i in range(num_comp):
                jac_x_0i = jac_glob_d.copy()
                jac_x_0i[:, n_p + i * num_vals : n_p + (i + 1) * num_vals] = id_block
                X_jacs[-1].append(jac_x_0i)
            # update the column number based on reference phase composition vals
            n_p += num_comp * num_vals
            for j in range(indp):
                jac_y_j = jac_glob_d.copy()
                jac_y_j[:, j * num_vals : (j + 1) * num_vals] = id_block
                y_jacs.append(jac_y_j)
                X_jacs.append(list())
                for i in range(num_comp):
                    jac_x_ji = jac_glob_d.copy()
                    jac_x_ji[
                        :, n_p + i * num_vals : n_p + (i + 1) * num_vals
                    ] = id_block
                    X_jacs[-1].append(jac_x_ji)

            # reference phase fraction is dependent by unity
            if len(y_jacs) > 0:
                y = [pp.ad.AdArray(y[0], -1 * safe_sum(y_jacs))] + [
                    pp.ad.AdArray(y[j + 1], y_jacs[j]) for j in range(indp)
                ]
            X = [
                [pp.ad.AdArray(X[j][i], X_jacs[j][i]) for i in range(num_comp)]
                for j in range(num_phases)
            ]

            if is_independent:
                # Number of blocks with new independent vars
                N = N_default
                # make unique
                is_independent = list(set(is_independent))
                for i in is_independent:
                    # adding blocks per independent feed fraction if requested
                    if i == "z_i":
                        N += num_comp - 1
                    # Adding blocks per independent phase saturation
                    elif i == "s_i":
                        N += num_phases - 1
                    # adding other blocks per independent state
                    elif i in ["p", "T", "y_r", "z_r", "s_r"]:
                        N += 1
                    else:
                        raise ValueError(f"Independent state {i} not supported.")

                # number of column blocks to pre-append to existing states
                N_new = N - N_default
                pre_block = sps.lil_matrix((num_vals, num_vals * N_new))

                # update derivatives of independent phases
                for j in range(1, num_phases):
                    y[j].jac = sps.hstack([pre_block, y[j].jac])
                # update derivative of reference phase
                if indp > 0:
                    y[0].jac = sps.hstack([pre_block, y[0].jac])
                else:
                    y[0] = pp.ad.AdArray(y[0], pre_block.copy())

                # update derivatives of phase compositions
                for j in range(num_phases):
                    for i in range(num_comp):
                        X[j][i].jac = sps.hstack([pre_block, X[j][i].jac])

                # Global Jacobian for new, independent states
                jac_glob = sps.lil_matrix((num_vals, N * num_vals))
                # occupied column indices, counted from right to left
                occupied = (num_comp * num_phases + indp) * num_vals

                # update derivative of reference phase fraction if requested
                if "y_r" in is_independent and not indp:
                    jac_y_r = jac_glob.copy()
                    jac_y_r[:, -(occupied + num_vals) : -occupied] = id_block
                    y[0] = pp.ad.AdArray(y[0], jac_y_r)
                    occupied += num_vals  # update occupied

                # construct derivatives w.r.t to saturations of independent phases
                jac_s_0_dep = None
                if "s_i" in is_independent and indp:
                    jac_s_0_dep = jac_glob.copy()
                    for j in range(indp):
                        jac_s_i = jac_glob.copy()
                        jac_s_i[:, -(occupied + num_vals) : -occupied] = id_block
                        jac_s_0_dep = jac_s_0_dep - jac_s_i
                        s[indp - j] = pp.ad.AdArray(s[indp - j], jac_s_i)
                        occupied += num_vals  # update occupied
                if "s_r" in is_independent and not indp:
                    jac_s_0 = jac_glob.copy()
                    jac_s_0[:, -(occupied + num_vals) : -occupied] = id_block
                    s[0] = pp.ad.AdArray(s[0], jac_s_i)
                    occupied += num_vals  # update occupied
                # eliminate reference saturation by unity
                elif jac_s_0_dep is not None:
                    s[0] = pp.ad.AdArray(s[0], jac_s_0_dep)

                # construct derivatives w.r.t. feed fractions
                if "z_i" in is_independent:
                    for i in range(num_comp - 1):
                        jac_z_i = jac_glob.copy()
                        jac_z_i[:, -(occupied + num_vals) : -occupied] = id_block
                        z[num_comp - 1 - i] = pp.ad.AdArray(
                            z[num_comp - 1 - i], jac_z_i
                        )
                        occupied += num_vals

                # construct derivative w.r.t. reference feed fraction
                if "z_r" in is_independent:
                    jac_z_r = jac_glob.copy()
                    jac_z_r[:, -(occupied + num_vals) : -occupied] = id_block
                    z[0] = pp.ad.AdArray(z[0], jac_z_r)
                    occupied += num_vals

                # construct derivatives for states which are not given as list
                # in reverse order, right to left
                modified_quantities: list[NumericType] = list()
                for key, quantity in zip(["T", "p"], [T, p]):
                    # modify quantity if requested
                    if key in is_independent:
                        jac_q = jac_glob.copy()
                        jac_q[:, -(occupied + num_vals) : -occupied] = id_block
                        quantity = pp.ad.AdArray(quantity, jac_q)
                        occupied += num_vals
                    modified_quantities.append(quantity)
                T, p = modified_quantities

        return cls(p=p, T=T, h=h, v=v, rho=rho, z=z, y=y, s=s, X=X)


class BasicMixture:
    """Base mixture class managing present components and modelled phases.

    This is a layer-1 implementation of a mixture, containing the core functionality
    and objects representing the variables and basic thermodynamic properties.

    The equilibrium problem is set in the unified formulation (unified flash).
    It allows one gas-like phase and an arbitrary number of liquid-like phases.

    For performing the flash computations, see respective Flash classes.

    Important:
        - The first, non-gas-like phase is treated as the reference phase.
          Its molar fraction and saturation will not be part of the primary variables.
        - The first component is set as reference component.
          Its mass conservation will not be part of the equilibrium equations.
        - Choice of reference phase and component influence the choice of equations and
          variables, keep that in mind. It might have numeric implications.

    Notes:
        If the user wants to model a single-component mixture, a dummy component must be
        added as the first component (reference component for elimination),
        with a feed fraction small enough s.t. its effects on the
        thermodynamic properties are negligible.

        This approximates a single-component mixture, due to the flash system being
        inherently singular in this case. Numerical issues can appear if done so!

    Parameters:
        components: A list of components to be added to the mixture.
            This are the chemical species which can appear in multiple phases.
        phases: A list of phases to be modelled.

    Raises:
        AssertionError: If the model assumptions are violated.

            - 1 gas phase must be modelled.
            - At least 2 components must be present.
            - At least 2 phases must be modelled.

    """

    def __init__(
        self,
        components: list[Component],
        phases: list[Phase],
    ) -> None:
        # modelled phases and components
        self._components: list[Component] = []
        """A list of components passed at instantiation."""
        self._phases: list[Phase] = []
        """A list of phases passed at instantiation."""

        # a container holding objects already added, to avoid double adding
        doubles = []
        # Lists of gas-like and liquid-like phases
        gaslike_phases: list[Phase] = list()
        other_phases: list[Phase] = list()

        for comp in components:
            # add, Avoid double components
            if comp.name not in doubles:
                doubles.append(comp.name)
                self._components.append(comp)

        for phase in phases:
            # add, avoid double phases
            if phase.name not in doubles:
                doubles.append(phase.name)
                # add phase
                if phase.type == 1:
                    gaslike_phases.append(phase)
                else:
                    other_phases.append(phase)

        self._phases = other_phases + gaslike_phases
        # adding all components to every phase, according to unified procedure
        for phase in self.phases:
            phase.components = list(self.components)

        # checking model assumptions
        assert len(gaslike_phases) == 1, "Only 1 gas-like phase is permitted."
        assert len(self._components) > 1, "At least 2 components required."
        assert len(self._phases) > 1, "At least 2 phases required."

        ### PUBLIC

        self.system: pp.ad.EquationSystem
        """The AD-system set during :meth:`set_up`.

        This attribute is not available prior to that.

        """

        self.dofs: int
        """The number of DOFs per state function. This is calculated during
        :meth:`set_up`."""

        self.reference_phase_eliminated: bool
        """Flag if the reference phase variables have been eliminated in this mixture.

        See :meth:`set_up`.

        """

        self.molar_fraction_variables: list[str]
        """A list of names of molar fractional variables, which are unknowns in the
        equilibrium problem.

        These include

        - phase fractions
        - phase compositions

        This list is created in :meth:`set_up`.

        """

        self.saturation_variables: list[str]
        """A list of names of saturation variables, which are unknowns in the
        equilibrium problem.

        Note:
            Saturations are only relevant in equilibrium problems involving the volume
            of the mixtures.
            Otherwise they can be calculated a posterior.

        This list is created in :meth:`set_up`.

        """

        self.feed_fraction_variables: list[str]
        """A list of names of feed fraction variables per present components.

        Note:
            Feed fractions are constant in non-reactive mixtures, since there the number
            of moles of a species is assumed to be constant.

        This list is created in :meth:`set_up`.

        """

        self.solute_fraction_variables: dict[Compound, list[str]] = dict()
        """A dictionary containing per compound (key) names of solute fractions
        for each solute in that compound.

        Note:
            Solute fractions are assumed constant in non-reactive mixtures.
            They are not used anywhere in the flash by default.

        This map is created in :meth:`set_up`.

        """

        self.y_R: pp.ad.Operator
        """A representation of the :meth:`~porepy.composite.phase.Phase.fraction` of the
        :meth:`reference_phase` by unity, using the fractions of other present phases.

        This operator is created in :meth:`set_up`.

        """

        self.s_R: pp.ad.Operator
        """A representation of the :meth:`~porepy.composite.phase.Phase.saturation` of
        the :meth:`reference_phase` by unity, using the saturations of other present
        phases.

        This operator is created in :meth:`set_up`.

        """

        self.z_R: pp.ad.Operator
        """A representation of the
        :meth:`~porepy.composite.component.Component.fraction` of the
        :meth:`reference_component` by unity, using the overall fractions of other
        present components.

        This operator is created in :meth:`set_up`.

        """

        self.enthalpy: pp.ad.Operator
        """An operator representing the mixture enthalpy as a sum of
        :attr:`~porepy.composite.phase.Phase.enthalpy` weighed with
        :attr:`~porepy.composite.phase.Phase.fraction`.

        This operator is created in :meth:`set_up`.

        """

        self.density: pp.ad.Operator
        """An operator representing the mixture density as a sum of
        :attr:`~porepy.composite.phase.Phase.density` weighed with
        :attr:`~porepy.composite.phase.Phase.saturation`.

        This operator is created in :meth:`set_up`.

        """

        self.volume: pp.ad.Operator
        """An operator representing the mixture volume as a reciprocal of
        :attr:`density`.

        This operator is created in :meth:`set_up`.

        """

        self.phase_composition_unities: dict[Phase, pp.ad.Operator]
        """A dictionary containing per phase (key) its composition unity in
        operator form.

        .. math::

            1 - \\sum_i x_{ij}~,~\\forall j~,

        with :math:`x_{ij}` being the fraction component :math:`i` in phase :math:`j`.

        This operator is created in :meth:`set_up`.

        """

        self.phase_fraction_relation: dict[Phase, pp.ad.Operator]
        """A dictionary containing per phase (key) the relation between molar fraction
        :math:`y` and saturation :math:`s`, using the mixture and phase densities
        :math:`\\rho`.

        .. math::

            y_j \\rho - s_j \\rho_j~,~\\forall j~.

        This operator is created in :meth:`set_up`.

        """

    def __str__(self) -> str:
        """
        Returns:
            A string representation of the composition, with information about present
            components and phases.

        """
        out = f"Composition with {self.num_components} components:"
        for component in self.components:
            out += f"\n\t{component.name}"
        out += f"\nand {self.num_phases} phases:"
        for phase in self.phases:
            out += f"\n\t{phase.name}"
        return out

    @property
    def num_components(self) -> int:
        """Number of components in this mixture."""
        return len(self._components)

    @property
    def num_phases(self) -> int:
        """Number of phases phases in this mixture."""
        return len(self._phases)

    @property
    def num_equilibrium_equations(self) -> int:
        """Number of necessary equilibrium equations for this composition, based on the
        number of added components and phases:

        .. math::

            n_c (n_p - 1)

        Equilibrium equations are formulated with respect to the reference phase,
        hence ``n_p - 1``.

        """
        return self.num_components * (self.num_phases - 1)

    @property
    def components(self) -> Generator[Component, None, None]:
        """
        Note:
            The first component is always the reference component.

        Yields:
            Components added to the composition.

        """
        for C in self._components:
            yield C

    @property
    def phases(self) -> Generator[Phase, None, None]:
        """
        Note:
            The first phase is always the reference phase.

        Yields:
            Phases modelled by the composition class.

        """
        for P in self._phases:
            yield P

    @property
    def reference_phase(self) -> Phase:
        """Returns the reference phase.

        As of now, the first **non-gas-like** phase is declared as reference
        phase (based on :attr:`~porepy.composite.phase.Phase.gaslike`).

        The implications of a phase being the reference phase include:

        - Phase fraction and saturation can be dependent expression, not variables
          (Elimination by unity).

        """
        return self._phases[0]

    @property
    def reference_component(self) -> Component:
        """Returns the reference component.

        As of now, the first component is declared as reference component.

        The implications of a component being the reference component include:

        - The mass constraint can be eliminated, since it is linear dependent on the
          other mass constraints due to various unity constraints.

        """
        return self._components[0]

    def _instantiate_frac_var(
        self, ad_system: pp.ad.EquationSystem, name: str, subdomains: list[pp.Grid]
    ) -> pp.ad.MixedDimensionalVariable:
        """Auxiliary function to instantiate variables with 1 degree per cell.

        STATE and ITERATE are set to zero.

        """
        var = ad_system.create_variables(name=name, subdomains=subdomains)
        ad_system.set_variable_values(
            np.zeros(self.dofs), variables=[name], iterate_index=0, time_step_index=0
        )
        return var

    def set_up(
        self,
        *args,
        ad_system: Optional[pp.ad.EquationSystem] = None,
        subdomains: list[pp.Grid] = None,
        num_vals: int = 1,
        eliminate_ref_phase: bool = True,
        eliminate_ref_feed_fraction: bool = True,
        **kwargs,
    ) -> list[pp.Grid]:
        """Basic set-up of mixture.

        This creates the fractional variables in the phase-equilibrium problem.

        The following AD operators are created (with 0-values, if variable):

        - component overall fractions (feed fractions)
          :attr:`~porepy.composite.component.Component.fraction`.
        - phase fractions :attr:`~porepy.composite.phase.Phase.fractions`
          (variables, except for reference phase).
        - phase saturations :attr:`~porepy.composite.phase.Phase.saturation`
          (variables, except for reference phase).
        - phase compositions :attr:`~porepy.composite.phase.Phase.fraction_of`
          (variables).
        - normalized phase compositions
          :attr:`~porepy.composite.phase.Phase.normalized_fraction_of`
          (expressions dependent on phase compositions).

        Important:
            This set-up does **not** introduce equations into the AD framework.
            E.g., no call to
            :meth:`~porepy.numerics.ad.equation_system.EquationSystem.set_equation`
            is performed.
            This is up to child classes implementing specific mixture dynamics.

        Parameters:
            *args: Placeholders in case of inheritance.
            ad_system: ``default=None``

                If given, this class will use the AD system and the respective
                mixed-dimensional domain to represent all involved variables cell-wise
                in each subdomain.

                If not given (None), a single-cell domain and respective AD system are
                created.
            subdomains: ``default=None``

                If ``ad_system`` is not None, restrict this mixture to a given set of
                grids by defining this keyword argument.

                Otherwise, every subdomain found in
                :attr:`~porepy.numerics.ad.equation_system.EquationSystem.mdg`
                will be used as domain for this mixture.

                Important:
                    All components and phases are present in each subdomain-cell.
                    Their fractions are introduced as cell-wise, scalar unknowns.

            num_vals: ``default=1``

                Number of values per state function for the default AD system
                (and its grid).

                Use this to vectorize the flash procedure, such that multiple different
                thermodynamic states are passed in vector form and the flash system
                is assembled in a block-diagonal manner.

                Only used if ``ad_system=None`` and the default system is created.

                Warning:
                    In some problematic cases, the vectorization causes a purely
                    mathematical coupling between the formally independent flash cases.

                    This is due to the condition number of the flash system being
                    inherently high. Over-iterations necessary for problematic
                    vector-components can cause convergence issues for other,
                    already converged vector-components.

            eliminate_reference_phase: ``default=True``

                An optional flag to eliminate reference phase variables from the
                system, and hence reduce the system.

                The saturation and fraction can be eliminated by unity using other
                saturations and fractions.

                If True, the attributes
                :attr:`~porepy.composite.phase.Phase.fraction` and
                :attr:`~porepy.composite.phase.Phase.saturation` will **not** be
                variables, but expressions.
            eliminate_ref_feed_Fraction: ``default=True``

                An optional flag to eliminate the feed fraction of the reference
                component from the system as a variable, hence to reduce the system.

                It can be eliminated by unity using the other feed fractions.

                If True, the attribute
                :attr:`~porepy.composite.phase.component.fraction` of
                :meth:`reference_component` will **not** be a variable, but an
                expression.
            **kwargs: Placeholder in case of inheritance.

        Returns:
            A list of domains on which the mixture is defined.

            If the default domain is created, it returns the created instance.

            If a domain is defined using ``ad_system``, the mixture will be defined on
            all subdomains found in the system.
            If additionally ``subdomains`` is defined, the mixture will be defined on
            the intersection of domains found in ``subdomains`` and those found in
            ``ad_system.mdg``.

        """
        domains: list[pp.Grid]
        if ad_system is None:
            sg = pp.CartGrid([num_vals, 1], [1, 1])
            mdg = pp.MixedDimensionalGrid()
            mdg.add_subdomains(sg)
            mdg.compute_geometry()

            ad_system = pp.ad.EquationSystem(mdg)
            domains = mdg.subdomains()
        else:
            if subdomains is None:
                domains = ad_system.mdg.subdomains()
            else:
                domains = [sd for sd in subdomains if sd in ad_system.mdg.subdomains()]

        self.system = ad_system
        self.dofs = int(sum([sd.num_cells for sd in domains]))
        self.reference_phase_eliminated = bool(eliminate_ref_phase)

        ## Creating fractional variables.
        # First, create all component fractions and solute fraction for compounds
        variables: list[str] = []
        solute_variables: dict[Compound, list[str]] = dict()
        for comp in self.components:
            if comp != self.reference_component:
                name = (
                    f"{COMPOSITIONAL_VARIABLE_SYMBOLS['component_fraction']}"
                    + f"_{comp.name}"
                )
                comp.fraction = self._instantiate_frac_var(ad_system, name, domains)
                variables.append(name)

            if isinstance(comp, Compound):
                solute_variables.update({comp: []})
                for solute in comp.solutes:
                    name = (
                        f"{COMPOSITIONAL_VARIABLE_SYMBOLS['solute_fraction']}"
                        + f"_{solute.name}_{comp.name}"
                    )
                    comp.solute_fraction_of[solute] = self._instantiate_frac_var(
                        ad_system, name, domains
                    )
                    solute_variables[comp].append(name)

        z_R: pp.ad.Operator = self.evaluate_unity(
            [c.fraction for c in self.components if c != self.reference_component]
        )  # type: ignore
        z_R.set_name("ref-component-fraction-by-unity")
        if eliminate_ref_feed_fraction:
            self.reference_component.fraction = z_R
        else:
            name = (
                f"{COMPOSITIONAL_VARIABLE_SYMBOLS['component_fraction']}"
                + f"_{self.reference_component.name}"
            )
            self.reference_component.fraction = self._instantiate_frac_var(
                ad_system, name, domains
            )
            variables.append(name)
        self.feed_fraction_variables = variables
        self.solute_fraction_variables = solute_variables

        # Second, create all saturations.
        # Eliminate ref phase saturation by unity, if requested.
        variables: list[str] = []
        for phase in self.phases:
            if phase != self.reference_phase:
                name = (
                    f"{COMPOSITIONAL_VARIABLE_SYMBOLS['phase_saturation']}_{phase.name}"
                )
                phase.saturation = self._instantiate_frac_var(ad_system, name, domains)
                variables.append(name)

        s_R: pp.ad.Operator = self.evaluate_unity(
            [p.saturation for p in self.phases if p != self.reference_phase]
        )  # type: ignore
        s_R.set_name("ref-phase-saturation-by-unity")
        self.s_R = s_R
        if self.reference_phase_eliminated:
            self.reference_phase.saturation = s_R
        else:
            name = (
                f"{COMPOSITIONAL_VARIABLE_SYMBOLS['phase_saturation']}"
                + f"_{self.reference_phase.name}"
            )
            self.reference_phase.saturation = self._instantiate_frac_var(
                ad_system, name, domains
            )
            variables.append(name)
        self.saturation_variables = variables

        # Third, create all phase molar fractions
        # Eliminate ref phase fraction by unity if requested.
        variables: list[str] = []
        for phase in self.phases:
            if phase != self.reference_phase:
                name = (
                    f"{COMPOSITIONAL_VARIABLE_SYMBOLS['phase_fraction']}_{phase.name}"
                )
                phase.fraction = self._instantiate_frac_var(ad_system, name, domains)
                variables.append(name)

        y_R: pp.ad.Operator = self.evaluate_unity(
            [p.fraction for p in self.phases if p != self.reference_phase]
        )  # type: ignore
        y_R.set_name("ref-phase-fraction-by-unity")
        self.y_R = y_R
        if self.reference_phase_eliminated:
            self.reference_phase.fraction = y_R
        else:
            name = (
                f"{COMPOSITIONAL_VARIABLE_SYMBOLS['phase_fraction']}"
                + f"_{self.reference_phase.name}"
            )
            self.reference_phase.fraction = self._instantiate_frac_var(
                ad_system, name, domains
            )
            variables.append(name)

        # Fourth, create all phase compositions (extended fractions)
        for phase in self.phases:
            for comp in self.components:
                name = (
                    f"{COMPOSITIONAL_VARIABLE_SYMBOLS['phase_composition']}"
                    + f"_{comp.name}_{phase.name}"
                )
                phase.fraction_of.update(
                    {comp: self._instantiate_frac_var(ad_system, name, domains)}
                )
                variables.append(name)
        self.molar_fraction_variables = variables

        # Fifth, create operators representing normalized fractions
        for phase in self.phases:
            sum_j: pp.ad.Operator = safe_sum(
                list(phase.fraction_of.values())
            )  # type: ignore
            for comp in self.components:
                name = f"{phase.fraction_of[comp].name}_normalized"
                x_ij_n = phase.fraction_of[comp] / sum_j
                x_ij_n.set_name(name)
                phase.normalized_fraction_of.update({comp: x_ij_n})

        ## Creating mixture properties
        # First, mixture density and volume
        self.density: pp.ad.Operator = safe_sum(
            [phase.saturation * phase.density for phase in self.phases]
        )  # type: ignore
        self.density.set_name("mixture-density")
        self.volume: pp.ad.Operator = self.density ** (-1)
        self.volume.set_name("mixture-volume")

        # Second, mixture enthalpy
        self.enthalpy: pp.ad.Operator = safe_sum(
            [phase.fraction * phase.enthalpy for phase in self.phases]
        )  # type: ignore
        self.enthalpy.set_name("mixture-enthalpy")

        ## Creating other, common operators

        # phase composition unity
        self.phase_composition_unities: dict[Phase, pp.ad.Operator] = dict()
        for phase in self.phases:
            unity: pp.ad.Operator = self.evaluate_unity(
                [phase.fraction_of[comp] for comp in self.components]
            )  # type: ignore
            unity.set_name(f"phase-composition-unity-{phase.name}")
            self.phase_composition_unities.update({phase: unity})

        # phase fraction relations
        self.phase_fraction_relation: dict[Phase, pp.ad.Operator] = dict()
        for phase in self.phases:
            relation: pp.ad.Operator = (
                phase.fraction - phase.saturation * phase.density / self.density
            )
            relation.set_name(f"phase-fraction-relation-{phase.name}")
            self.phase_fraction_relation.update({phase: relation})

        return domains

    @overload
    def compute_properties(
        self,
        p: NumericType,
        T: NumericType,
        X: list[list[NumericType]],
        store: Literal[True] = True,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        # Typing overload for default return value: None, properties are stored
        ...

    @overload
    def compute_properties(
        self,
        p: NumericType,
        T: NumericType,
        X: list[list[NumericType]],
        store: Literal[False] = False,
        normalize: bool = True,
        **kwargs,
    ) -> list[PhaseProperties]:
        # Typing overload for default return value: Properties are returned
        ...

    def compute_properties(
        self,
        p: NumericType,
        T: NumericType,
        X: list[list[NumericType]],
        store: bool = True,
        normalize: bool = True,
        **kwargs,
    ) -> None | list[PhaseProperties]:
        """This is a wrapper to compute the properties of each phase.

        See :meth:`~porepy.composite.phase.Phase.compute_properties` for more.

        This method loops over modelled phases and calls respective phase-specific
        methods.

        Parameters:
            p: Pressure.
            T: Temperature.
            store: ``default=True``

                Flag to store or return the results
            normalize: ``default=True``

                Normalizes phase compositions if True, otherwise takes as is.
            X: ``len=n_p``

                A nested list containing for each phase a sub-list of (normalized)
                component fractions in that phase.
            **kwargs: Keyword arguments to be passed to the phase's method.

        """
        results: list[PhaseProperties] = list()
        for j, phase in enumerate(self.phases):
            if normalize:
                X_j = normalize_fractions(X[j])
            else:
                X_j = X[j]
            props = phase.compute_properties(p, T, X_j, store=store, **kwargs)
            results.append(props)
        if not store:
            return results

    def get_thermodynamic_state_from_vector(
        self,
        state: Optional[np.ndarray] = None,
        as_ad: bool = False,
        derivatives: Optional[list[str]] = None,
    ) -> ThermodynamicState:
        """Evaluates the state variables of the mixture as stored in the AD framework.

        Note:
            Thermodynamic state functions like density, volume and enthalpy are
            evaluated **as is**. If the right values are required, consider
            :meth:`compute_properties_from_state_vector` first.

        Parameters:
            state: ``default=None``

                A state vector from which the compositional state should be extracted.
                (see :meth:`~porepy.numerics.ad.operators.evaluate`).

                If ``None``, the values stored as ``ITERATE`` are obtained.
            as_ad: ``default=False``

                If ``True``, the variables are returned as
                :class:`~porepy.numerics.ad.forward_mode.AdArray` containing
                the global derivatives, as assigned by the AD framework.

                If ``False``, only the values in form of numpy arrays are returned.
            derivatives: ``default=None``

                A list of variables for which derivatives in the AD-array should be
                included, if ``as_ad`` is ``True``.

                If given, the Jacobian of the assembled AD-arrays will be sliced
                accordingly.

        Returns:
            A data structure containing the evaluated values for fractional variables
            (see :meth:`~porepy.numerics.ad.operators.Operator.evaluate`) without
            derivatives.

        """

        if derivatives:
            projection = self.system.projection_to(derivatives).transpose()
        else:
            projection = None

        z: list[NumericType] = [
            _process(comp.fraction.evaluate(self.system, state), as_ad, projection)
            for comp in self.components
        ]
        y: list[NumericType] = [
            _process(phase.fraction.evaluate(self.system, state), as_ad, projection)
            for phase in self.phases
        ]
        s: list[NumericType] = [
            _process(phase.saturation.evaluate(self.system, state), as_ad, projection)
            for phase in self.phases
        ]
        X: list[list[NumericType]] = [
            [
                _process(
                    phase.fraction_of[comp].evaluate(self.system, state),
                    as_ad,
                    projection,
                )
                for comp in self.components
            ]
            for phase in self.phases
        ]

        h = _process(self.enthalpy.evaluate(self.system, state), as_ad, projection)
        v = _process(self.volume.evaluate(self.system, state), as_ad, projection)
        rho = _process(self.density.evaluate(self.system, state), as_ad, projection)

        return ThermodynamicState(h=h, v=v, rho=rho, z=z, y=y, s=s, X=X)

    def get_fractional_state_from_vector(
        self,
        state: Optional[np.ndarray] = None,
        as_ad: bool = False,
        derivatives: Optional[list[str]] = None,
    ) -> ThermodynamicState:
        """Same as :meth:`get_thermodynamic_state`, with the difference that only
        fractions are evaluated (to save time)."""

        if derivatives:
            projection = self.system.projection_to(derivatives).transpose()
        else:
            projection = None

        z: list[NumericType] = [
            _process(comp.fraction.evaluate(self.system, state), as_ad, projection)
            for comp in self.components
        ]
        y: list[NumericType] = [
            _process(phase.fraction.evaluate(self.system, state), as_ad, projection)
            for phase in self.phases
        ]
        s: list[NumericType] = [
            _process(phase.saturation.evaluate(self.system, state), as_ad, projection)
            for phase in self.phases
        ]
        X: list[list[NumericType]] = [
            [
                _process(
                    phase.fraction_of[comp].evaluate(self.system, state),
                    as_ad,
                    projection,
                )
                for comp in self.components
            ]
            for phase in self.phases
        ]

        return ThermodynamicState(z=z, y=y, s=s, X=X)

    def compute_properties_from_vector(
        self,
        pressure: pp.ad.Operator,
        temperature: pp.ad.Operator,
        state: Optional[np.ndarray] = None,
        as_ad: bool = True,
        derivatives: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """This is a wrapper function for :meth:`compute_properties`,
        which extracts relevant input from a global AD state vector.

        The fractional input arguments are extracted using
        :meth:`get_compositional_state` and subsequently fed to
        :meth:`~FluidMixture.compute_properties`.

        The other intensive state variables, pressure and temperature, must be
        given by input. The user must ensure their domains of definition coincide with
        the domains of the fractional variables.

        As an additional step, it normalizes the phase compositions for consistency.

        For a detailed explanation of input arguments, see
        :meth:`get_compositional_state`.

        Parameters:
            pressure: The pressure variable to be found in :attr:`system`.
            temperature: The temperature variable to be found in :attr:`system`.
            as_ad: ``default=True``
            **kwargs: Keyword-arguments to be passed to
                :meth:`compute_properties` and subsequently to
                :meth:`~porepy.composite.phase.Phase.compute_properties`.

        """

        if derivatives:
            projection = self.system.projection_to(derivatives).transpose()
        else:
            projection = None

        frac_state = self.get_fractional_state_from_vector(state, as_ad, derivatives)

        p: NumericType = _process(
            pressure.evaluate(self.system, state=state), as_ad, projection
        )
        T: NumericType = _process(
            temperature.evaluate(self.system, state=state), as_ad, projection
        )

        self.compute_properties(p, T, frac_state.X, **kwargs)

    @staticmethod
    def evaluate_unity(x: list[Any]) -> Any:
        """Method to evaluate the deviation from unity for any family of quantities
        ``x``.

        A safe sum function is used, avoiding an allocation of zero as the first
        summand.

        Parameters:
            x: A family of quantities which ought to sum up to 1.

        Returns:
            ``1 - sum(x)``, with 1 as a Python integer.

        """
        return 1 - safe_sum(x)

    @staticmethod
    def evaluate_weighed_sum(q: list[Any], w: Optional[list[Any]] = None) -> Any:
        """Method to evaluate a weighed sum of any quantities.

        A safe sum function is used, avoiding an allocation of zero as the first
        summand.

        Parameters:
            q: A family of quantities which ought to be summed up.
            w: ``default=None, len(weights)==len(quantities)``

                Optional weights for each each element in ``quantities``.

        Returns:
            ``sum(q * w)``, with ``*`` as the dot product.

            If ``w`` is not given, performs only ``sum(q)``.

        """
        if w:
            return safe_sum([q_ * w_ for q_, w_ in zip(q, w)])
        else:
            return safe_sum(q)


class NonReactiveMixture(BasicMixture):
    """A class modelling non-reactive mixtures using the unified formulation.

    This is a layer-2 mixture implementing equilibrium equations on top of the
    standard expressions and variables in the base class.

    It contains some generic methods to evaluate or construct equations, compatible with
    PorePy's AD framework. These include:

    - Static mass constraints (fixed feed fractions)
    - Equilibrium equations formulated with respect to the reference phase.
    - Complementary conditions for phase fractions using the unified framework.
    - Equality constraints for thermodynamic quantities (e.g. enthalpy, volume).

    The equations above are introduced into the AD framework (:attr:`system`) on top of
    the base class' method :meth:`BasicMixture.set_up`.

    """

    def __init__(self, components: list[Component], phases: list[Phase]) -> None:
        super().__init__(components, phases)

        self.equations: list[str]
        """A list of names of equations, which are introduced into the :attr:`system`
        by this mixture class.

        This list is created during :meth:`set_up`, where the equations are set.

        The equations here are such, that they are closed for a mixture with
        ``n_c`` components and ``n_p`` phases, if the reference phase fraction was
        eliminated.

        If the reference phase fraction was eliminated (keyword argument
        ``eliminate_ref_phase`` during :meth:`set_up`), there are
        ``n_c * n_p + n_p - 1`` unknowns introduced into the system. Namely

        - ``n_c * n_p`` phase composition variables
          (see :attr:`~porepy.composite.phase.Phase.fraction_of`) and
        - ``n_p - 1`` phase molar fractions
          (see :attr:`~porepy.composite.phase.Phase.fraction`).

        The equations contained here represent

        - ``n_c - 1`` mass constraints per component (except reference component)
          (see :attr:`mass_constraints`),
        - ``n_c * (n_p - 1)`` isofugacity constraints
          (see :attr:`equilibrium_equations`) and
        - ``n_p`` complementary conditions (possible semi-smooth,
          if ``semismooth_complementarity=True`` during :meth:`set_up`)
          (see :attr:`complementary_conditions`).

        If the reference phase fraction was eliminated, the equations presented here
        form a closed system with the variables in :attr:`molar_fraction_variables`
        and are suitable for the p-T-flash in the unified setting.

        Important:
            If the reference phase fraction was **not** eliminated, there is one
            variable more than equations are introduced, namely the reference phase
            fraction. The system is not closed in this case.

        Note:
            No other equations (such as state constraints) are introduced by default.
            E.g., an isenthalpic flash requires an additional enthalpy constraint for
            an additional variable, the temperature.

        """

        self.mass_constraints: dict[Component, pp.ad.Operator]
        """A map containing mass constraints per components (key), except for the
        reference component.

        .. math::

            z_i - \\sum_j x_{ij} y_j~,~\\forall j~,

        using feed fractions :math:`z`, phase fraction :math:`y` and phase compositions
        :math:`x`.

        This operator is created in :meth:`set_up`.

        """

        self.equilibrium_equations: dict[Component, dict[Phase, pp.ad.Operator]]
        """A map containing equilibrium equations per component (key).

        Equilibrium equations are formulated between a phase (second key) and the
        reference phase.

        Per component, there are ``num_phases -1`` equilibrium equations.

        .. math::

            x_{ij} \\varphi_{ij} - x_{iR} \\varphi_{iR}~,~\\forall i~,~j \\neq R~.

        This dictionary is filled in :meth:`set_up`.

        """

        self.complementary_conditions: dict[Phase, pp.ad.Operator]
        """A map containing complementary conditions per phase (key) as per the unified
        setting.

        .. math::

            y_j (1 - \\sum_i x_{ij})~,
            \\min \\{y_j, (1 - \\sum_i x_{ij}) \\}~\\forall j~.

        Note that fraction of the reference phase :math:`y_R` is possibly represented
        through unity.

        Complementary conditions are either given as is, or in semi-smooth form
        (see this class' :meth:`set_up`).

        This dictionary is filled in :meth:`set_up`.

        """

    def set_up(
        self,
        *args,
        ad_system: Optional[pp.ad.EquationSystem] = None,
        subdomains: list[pp.Grid] = None,
        num_vals: int = 1,
        eliminate_ref_phase: bool = True,
        eliminate_ref_feed_fraction: bool = True,
        semismooth_complementarity: bool = True,
        **kwargs,
    ) -> list[pp.Grid]:
        """Performs on top of the base class set-up the creation of equations relevant
        for non-reactive mixtures.

        These include:

        - mass constraints (:attr:`mass_constraints`)
        - equilibrium equations (:attr:`equilibrium_equations`)
        - complementary conditions (:attr:`complementary_conditions`)

        The equations are introduced into :attr:`system`.
        Names of set equations are stored :attr:`equations`.

        Parameters:
            semismooth_complementarity: ``default=True``

                If True, the complementary conditions are set using a semi-smooth
                min-function (see :attr:`complementary_conditions`).

        """
        domains = super().set_up(
            *args,
            ad_system=ad_system,
            subdomains=subdomains,
            num_vals=num_vals,
            eliminate_ref_phase=eliminate_ref_phase,
            eliminate_ref_feed_fraction=eliminate_ref_feed_fraction,
            **kwargs,
        )

        # Setting up mass constraints
        mass_constraints: dict[Component, pp.ad.Operator] = dict()
        y_j = [phase.fraction for phase in self.phases]
        for comp in self.components:
            if comp != self.reference_component:
                constraint: pp.ad.Operator = self.evaluate_homogenous_constraint(
                    comp.fraction,
                    y_j,
                    [phase.fraction_of[comp] for phase in self.phases],
                )  # type: ignore
                constraint.set_name(f"mass-constraint-{comp.name}")
                mass_constraints.update({comp: constraint})
        self.mass_constraints = mass_constraints

        # Setting up equilibrium equations
        equilibrium: dict[Component, dict[Phase, pp.ad.Operator]] = dict()
        for comp in self.components:
            comp_equ: dict[Phase, pp.ad.Operator] = dict()
            for phase in self.phases:
                if phase != self.reference_phase:
                    equ = (
                        phase.fraction_of[comp] * phase.fugacity_of[comp]
                        - self.reference_phase.fraction_of[comp]
                        * self.reference_phase.fugacity_of[comp]
                    )
                    equ.set_name(
                        f"isofugacity-constraint-"
                        + f"{comp.name}-{phase.name}-{self.reference_phase.name}"
                    )
                    comp_equ.update({phase: equ})
            equilibrium.update({comp: comp_equ})
        self.equilibrium_equations = equilibrium

        # Setting up complementary conditions
        ss_min: pp.ad.Operator = pp.ad.SemiSmoothMin()
        cc_conditions: dict[Phase, pp.ad.Operator] = dict()
        for phase in self.phases:
            comp_unity = self.phase_composition_unities[phase]
            if semismooth_complementarity:
                equ = ss_min(phase.fraction, comp_unity)
                equ.set_name(f"semismooth-complementary-condition-{phase.name}")
            else:
                equ = phase.fraction * comp_unity
                equ.set_name(f"complementary-condition-{phase.name}")
            cc_conditions.update({phase: equ})
        self.complementary_conditions = cc_conditions

        ## Introducing equations into the system
        equations: list[str] = list()  # equations introduced by default

        # mass constraints
        for comp in self.components:
            if comp != self.reference_component:
                equ = self.mass_constraints[comp]
                self.system.set_equation(
                    equ,
                    grids=domains,
                    equations_per_grid_entity={"cells": 1},
                )
                equations.append(equ.name)

        # Equilibrium constraints
        for comp in self.components:
            for phase, equ in self.equilibrium_equations[comp].items():
                self.system.set_equation(
                    equ,
                    grids=domains,
                    equations_per_grid_entity={"cells": 1},
                )
                equations.append(equ.name)

        # Complementary conditions
        for _, equ in self.complementary_conditions.items():
            self.system.set_equation(
                equ,
                grids=domains,
                equations_per_grid_entity={"cells": 1},
            )
            equations.append(equ.name)

        self.equations = equations

        return domains

    @staticmethod
    def evaluate_fractional_complementarity(y: Any, x: list[Any]) -> Any:
        """Method to evaluate the complementarity of ``y`` and the unity of ``x``.

        A safe sum function is used, avoiding an allocation of zero as the first
        summand.

        Parameters:
            y: Any quantity.
            x: A family of quantities fulfilling unity.

        Returns:
            ``y * (1 - sum(x))`` with 1 as a Python integer.

        """
        return y * BasicMixture.evaluate_unity(x)

    @staticmethod
    def evaluate_homogenous_constraint(
        phi: Any, y_j: list[Any], phi_j: list[Any]
    ) -> Any:
        """Method to evaluate the equality between a quantity ``phi`` and its
        sub-quantities ``phi_j``.

        A safe sum function is used, avoiding an allocation of zero as the first
        summand.

        This method can be used with any first-order homogenous quantity, i.e.
        quantities which are a sum of phase-related quantities weighed with phase
        fractions.

        Examples include mass, enthalpy and any other energy of the thermodynamic model.

        Parameters:
            phi: Any homogenous quantity
            y_j: Fractions of how ``phi`` is split into sub-quantities
            phi_j: ``len(phi_j) == len(y_j)``

                Sub-quantities of ``phi``, if entity ``j`` where saturated (``y_j=1``).

        Returns:
            ``sum(y_j*phi_j) - phi``, with ``*`` being a scalar product.

        """
        return BasicMixture.evaluate_weighed_sum(phi_j, y_j) - phi
