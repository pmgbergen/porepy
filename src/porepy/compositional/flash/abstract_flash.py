"""Module containing an abstraction layer for the flash procedure."""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    NotRequired,
    Optional,
    Sequence,
    TypeAlias,
    TypedDict,
    cast,
    Callable,
)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from numpy.typing import NDArray

import porepy as pp

from ..utils import FlashSpec, normalize_rows

__all__ = [
    "IsobaricSpecifications",
    "IsochoricSpecifications",
    "StateSpecDict",
    "FlashSpec",
    "FlashResults",
    "AbstractFlash",
]

logger = logging.getLogger(__name__)


class IsobaricSpecifications(TypedDict):
    """Typed dictionary for isobaric equilibrium specifications.

    The pressure values are obligatory and one energy-related variable is required.

    """

    p: np.ndarray | pp.number
    """Pressure at equilibrium."""

    T: NotRequired[np.ndarray | pp.number]
    """Temperature at equilibrium."""

    h: NotRequired[np.ndarray | pp.number]
    """Specific fluid enthalpy at equilibrium."""


class IsochoricSpecifications(TypedDict):
    """Typed dictionary for isochoric equilibrium specifications.

    The specific volume values are obligatory and one energy-related variable is
    required.

    """

    v: np.ndarray | pp.number
    """Specific fluid volume at equilibrium."""

    T: NotRequired[np.ndarray | pp.number]
    """Temperature at equilibrium."""

    u: NotRequired[np.ndarray | pp.number]
    """Specific fluid internal energy at equilibrium."""

    h: NotRequired[np.ndarray | pp.number]
    """Specific fluid enthalpy at equilibrium."""


StateSpecDict: TypeAlias = IsobaricSpecifications | IsochoricSpecifications
"""Alias for typed dictionaries for state specifications for equilibrium calculations.

Supported are specifications with either pressure or specific volume defined.
One additional energy-related state value is required.

"""


@dataclass
class FlashResults(pp.compositional.FluidProperties):
    """Data class for storing flash results.

    Adds additional fields to store information about convergence and numerical
    performance.

    The information is stored in arrays for vectorized flash computations and indices
    correspond to the indices of the (vectorized) state specifications.

    """

    specification: FlashSpec = FlashSpec.none
    """"Equilibrium state specification."""

    dofs: int = 0
    """Degrees of freedom of the flash problem, which depends on the number of phases,
    components and equilibrium specifiations.
    
    Example:

    For a 2-component, 2-phase flash with specified pressure and specific enthalpy,
    the number of DOFs is 6:
    
    - vapor fraction (1),
    - partial fraction per component per phase (4).
    - temperature (1).

    """

    size: int = 0
    """Size of the flash input for vectorized calculations.

    This will always be at least 1 after calculations, since results are always stored
    in arrays for simplicity.

    """

    exitcode: NDArray[np.int_] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int_)
    )
    """Ëxit codes for individual flash procedures.
    
    Following exit codes are reserved and universal for all procedures:

    - 0: Flash converged.
    - 1: Flash reached maximal number of iterations.
    - 2: Flash diverged.

    """

    num_iter: NDArray[np.int_] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int_)
    )
    """Number of iterations performed in the flash algorithm."""

    @property
    def converged(self) -> NDArray[np.bool_]:
        """ "Flags indicating where the flash converged."""
        return self.exitcode == 0

    @property
    def max_iter_reached(self) -> NDArray[np.bool_]:
        """Flags indicating where the flash reached the maximal number of iterations."""
        return self.exitcode == 1

    @property
    def stationary(self) -> NDArray[np.bool_]:
        """Flags indicating where the flash problem became stationary."""
        return self.exitcode == 2

    @property
    def diverged(self) -> NDArray[np.bool_]:
        """Flags indicating where the flash diverged."""
        return self.exitcode == 3

    @property
    def failure(self) -> NDArray[np.bool_]:
        """Flags indicating where the flash failed for unknown reasons."""
        return (self.exitcode == 4) | (self.exitcode == 5)

    @property
    def exitcode_parsed(self) -> NDArray[np.str_]:
        """Human-readable exitcode translating to the code to its classification."""
        # NOTE object-dtype to avoid capping of strings.
        v = np.array(["converged"] * self.exitcode.size, dtype=object)
        v[self.max_iter_reached] = "max. iterations reached"
        v[self.stationary] = "stationary point"
        v[self.diverged] = "diverged"
        v[self.exitcode == 4] = "failure in evaluation"
        v[self.exitcode == 5] = "unspecified exception (multisolver)"
        v[self.exitcode > 5] = "unknown"
        return v.astype(str)

    def postprocess_fractions(self, eps: float = 1e-8) -> None:
        """Ensures fractions are strictly in the interval [0, 1] and fulfill the
        unity constraint.

        Overall fractions :attr:`z` are not modified.

        Parameters:
            eps: ``default=1e-8``

                Used to determine absence of a phase and whether partial fractions
                should be normalized.

        """

        self.y[self.y < 0] = 0.0
        self.y[self.y > 1] = 1.0
        self.y = normalize_rows(self.y.T).T
        self.sat[self.sat < 0] = 0.0
        self.sat[self.sat > 1] = 1.0
        self.sat = normalize_rows(self.sat.T).T

        for y, phase in zip(self.y, self.phases):
            idx = y > eps
            phase.x[phase.x < 0] = 0.0
            phase.x[phase.x > 1] = 1.0
            if np.any(idx):
                phase.x[:, idx] = normalize_rows(phase.x[:, idx].T).T

    def log_summary(self, logger: logging.Logger | None = None) -> None:
        """Logs a summary of flash results.

        Parameters:
            logger: If given, logs the summary as an information log.
                By default, it is printed to the console.

        """
        notc = (~self.converged).sum()
        sr = (1 - notc / self.size) * 100
        mi = self.max_iter_reached.sum()
        st = self.stationary.sum()
        div = self.diverged.sum()
        f = self.failure.sum()
        msg = f"Not converged: {notc} / {self.size} (success rate {(sr):.2f} %)\n"
        msg += f"Max iter / Stationary / Diverged / Failure: {mi} / {st} / {div} / {f}"
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)


class AbstractFlash(abc.ABC):
    """Abstract base class for flash algorithms defining the interface of flash objects.

    The definition of the interface is done mainly as an orientation for compatibility
    with the remainder of PorePy's framework (especially the compositional flow).

    """

    def __init__(
        self,
        fluid: pp.Fluid[pp.FluidComponent, pp.Phase],
        params: Optional[dict] = None,
    ) -> None:
        super().__init__()

        if fluid.num_phases < 2:
            raise pp.compositional.CompositionalModellingError(
                f"Flash calculations require at least 2 phases, got {fluid.num_phases}."
            )

        if params is None:
            params = {}

        self.params: dict = params
        """Flash parameters given at instantiation."""

        self.params["num_phases"] = fluid.num_phases
        self.params["num_components"] = fluid.num_components
        self.params["components_per_phase"] = tuple(
            [phase.num_components for phase in fluid.phases]
        )
        self.params["gas_phase_index"] = fluid.gas_phase_index

        self.solver_params: dict[str, float] = {
            "atol_res": 1e-7,
            "max_iterations": 150.0,
            "num_phases": float(fluid.num_phases),
            "num_components": float(fluid.num_components),
        }
        """A dictionary containing solver parameters.

        Base class provides all general solver parameters, except the flash dimension
        which must be provided inside a call to :meth:`flash`.

        Note:
            Expects values which are convertible to floats. Numba is extensively used
            and it supports only dictionaries with a single type for key-value pairs.

        """

        if "solver_params" in self.params:
            solver_params = self.params.get("solver_params")
            assert isinstance(solver_params, dict)
            self.solver_params.update(solver_params)

        self._phasesplit_code_shift = 1000
        """Used for encoding phasesplit plot."""

    def compile(self, *args: FlashSpec) -> None:
        """Compilation interface for flash methods per flash specification.

        The base method is empty, not abstract.

        Parameters:
            *args: Specify subset of flash types which should be compiled to safe time.

        """

    def parse_flash_arguments(
        self,
        specification: StateSpecDict,
        z: Optional[Sequence[np.ndarray | pp.number]] = None,
        /,
        *,
        initial_state: Optional[pp.compositional.FluidProperties] = None,
    ) -> FlashResults:
        """Helper method to parse the input and construct a preliminary container for
        flash results with uniform input (casting into arrays of same size).

        The parameters are described in :meth:`flash`.

        Primary aim of this method is to determin the flash specifications, the DOFs and
        the size of vectorized input, as well as a uniform casting of flash arguments
        into numpy arrays of the same size.

        The returned data structure can be used by :meth:`flash` to fill it up with
        results and return it to the caller.

        Raises:
            ValueError: If compositions violate assumptions.
            NotImplementedError: If unsupported flash specifications were passed.
            AssertionError: If any family of fractions contained in the initial state
                violates assumptions.
            ValueError: If broadcasting of arguments and/or initial state to uniform
                input failed.
            TypeError:
                If state specifications are not of the expected type.

        Returns:
            A preliminary flash result data structure. If ``initial_state`` is defined,
            the data are copied into place. Note however that only data associated with
            degrees of freedom is copied.

        """

        ncomp = self.params["num_components"]
        nphase = self.params["num_phases"]

        # Parsing compositions.
        err_z = None
        if z is None:
            if ncomp == 1:
                z = [1.0]
            else:
                err_z = f"Expecting {ncomp} feed fractions, none given."
        else:
            if len(z) != ncomp:
                err_z = f"Expecting {ncomp} feed fractions, {len(z)} given."

        z = cast(Sequence[np.ndarray | pp.number], z)

        for i, z_ in enumerate(z):
            if np.any(z_ < 0) or np.any(z_ > 1):
                err_z = f"Violation of bound [0, 1] for feed fraction {i + 1}."

        z_sum = pp.compositional.safe_sum(z)
        if not np.all(z_sum == 1.0):
            err_z = "Feed fractions violate unity."

        if err_z:
            raise ValueError(err_z)

        # Declaring output.
        spec: FlashSpec = FlashSpec.none
        # Base dofs include independent phase fractions and partial fractions.
        dofs: int = nphase - 1 + nphase * ncomp
        size: int = 1
        state1: np.ndarray | pp.number = 0.0
        state2: np.ndarray | pp.number = 0.0
        raise_spec_error = False
        isochoric_spec = False
        isobaric_spec = False
        isothermal_spec = False

        if "p" in specification:
            state1 = specification["p"]  # type:ignore[typeddict-item]
            isobaric_spec = True
            if "T" in specification:
                state2 = specification["T"]
                spec = FlashSpec.pT
                isothermal_spec = True
            elif "h" in specification:
                state2 = specification["h"]
                spec = FlashSpec.ph
                # Temperature is an additional unknown.
                dofs += 1
            else:
                raise_spec_error = True
        elif "v" in specification:
            state1 = specification["v"]
            isochoric_spec = True
            if "T" in specification:
                state2 = specification["T"]
                spec = FlashSpec.vT
                isothermal_spec = True
                # No additional unknowns as it is equivalent to pT
            elif "u" in specification:
                state2 = specification["u"]
                spec = FlashSpec.vu
                # Pressure and temperature are additional unknowns.
                dofs += 2
            elif "h" in specification:
                state2 = specification["h"]
                spec = FlashSpec.vh
                # Pressure and temperature are additional unknowns.
                dofs += 2
            else:
                raise_spec_error = True
        else:
            raise_spec_error = True

        # Sanity check
        assert isobaric_spec != isochoric_spec and not (
            isobaric_spec and isochoric_spec
        ), "Must be either isobaric or isochoric."

        if raise_spec_error:
            raise NotImplementedError(
                f"Unsupported flash specifications {list(specification.keys())}."
            )

        # Isochoric specifications also have saturations as independent variables.
        # NOTE: for vT this might change, as pressure is not necessarily a DOF.
        if isochoric_spec:
            dofs += nphase - 1

        # Simple way to determine system size and check if input is broadcastable.
        try:
            t = z_sum + state1 + state2
        except Exception as err:
            raise ValueError(
                "Failed to broadcast flash arguments into uniform shape"
            ) from err

        if isinstance(t, np.ndarray):
            size = t.shape[0]
        elif not isinstance(t, (int, float)):
            raise TypeError(
                "Could not unify types of input arguments: "
                + f"z_sum={type(z_sum)} "
                + f"{spec.name} = {type(state1)}, {type(state2)}"
            )

        # Output structure
        results = FlashResults(specification=spec, size=size, dofs=dofs)
        # Uniformization of state values.
        s1 = np.zeros(size)
        s2 = np.zeros(size)
        s1[:] = state1
        s2[:] = state2

        Z = list()
        for z_ in z:
            _z = np.zeros(size)
            _z[:] = z_
            Z.append(_z)
        results.z = np.array(Z)

        if isobaric_spec:
            results.p = s1
        elif isochoric_spec:
            results.rho = 1.0 / s1
        else:
            # For reminding future developers.
            assert False, "Missing parsing of flash input"

        setattr(results, spec.name[1], s2)

        # Uniformization of initial values if provided.
        if isinstance(initial_state, pp.compositional.FluidProperties):
            # Check initial values.
            n = len(initial_state.y)
            assert n == nphase, f"Expecting {nphase} phase fractions, {n} provided."
            assert np.allclose(initial_state.y.sum(axis=0), 1.0), (
                "Initial phase fractions violate strong unity constraint."
            )

            for j in range(nphase):
                assert np.all(initial_state.phases[j].x.sum(axis=0) <= 1.0 + 1e-7), (
                    f"Component fractions in phase {j} violate weak unity constraint."
                )
                n = len(initial_state.phases[j].x)
                n_j = self.params["components_per_phase"][j]
                assert n == n_j, (
                    f"Expexting {n_j} partial fractions in phase {j}, {n} provided."
                )

            if isochoric_spec:
                n = len(initial_state.sat)
                assert n == nphase, (
                    f"Expecting {nphase} phase saturations, {n} provided."
                )
                assert np.allclose(initial_state.sat.sum(axis=0), 1.0), (
                    "Initial phase saturations violate strong unity constraint."
                )

            # Broadcast initial values.
            try:
                # Molar fractions.
                Y = list()
                phases = []
                for j in range(nphase):
                    y = np.zeros(size)
                    y[:] = initial_state.y[j]
                    Y.append(y)
                    # Fractions of components in phase.
                    X = list()
                    for i in range(self.params["components_per_phase"][j]):
                        x = np.zeros(size)
                        x[:] = initial_state.phases[j].x[i]
                        X.append(x)
                    phases.append(
                        pp.compositional.PhaseProperties(
                            x=np.array(X), state=initial_state.phases[j].state
                        )
                    )
                results.y = np.array(Y)
                results.phases = phases

                if isochoric_spec:
                    S = list()
                    for j in range(nphase):
                        s = np.zeros(size)
                        s[:] = initial_state.sat[j]
                        S.append(s)
                    results.sat = np.array(S)
                    p = np.zeros(size)
                    p[:] = initial_state.p
                    results.p = p
                if not isothermal_spec:
                    T = np.zeros(size)
                    T[:] = initial_state.T
                    results.T = T
            except Exception as err:
                raise ValueError(
                    "Failed to uniformize initial state for:\n"
                    + f"y: {initial_state.y}\n"
                    + f"s: {initial_state.sat}\n"
                    + f"x per phase: {[phase.x for phase in initial_state.phases]}"
                ) from err

        return results

    @abc.abstractmethod
    def flash(
        self,
        specification: StateSpecDict,
        z: Optional[Sequence[np.ndarray | pp.number]] = None,
        /,
        *,
        initial_state: Optional[pp.compositional.FluidProperties] = None,
        params: Optional[dict] = None,
        **kwargs,
    ) -> FlashResults:
        """Abstract method for performing a flash procedure.

        The equilibrium state must be defined in terms of compositions ``z`` and two
        state functions declared in ``specification``.
        One state must relate to pressure or volume. The other to energy.

        Parameters:
            specifications: Equilibrium specifications in terms of state functions.
            z: ``default=None``

                Overall fractions of mass per component.

                It is only optional for pure fluids (``z`` is implicitly assumed to be
                1). For fluid mixtures with multiple components it must be of length
                ``num_components``.
            initial_state: ``default=None``

                If not given, an initial guess must be computed by the flash class.

                If given, it must have at least values for phase fractions and
                compositions.

                It must have additionally values for temperature, for a state definition
                where temperature is not known at equilibrium.

                It must have additionally values for pressure and saturations, for state
                definitions where pressure is not known at equilibrium.
            params: ``default=None``

                Optional dictionary containing anything else required for custom flash
                classes.

        Returns:
            A data structure containing the flash results.

        """
        ...

    def plot_phase_diagram(
        self,
        specification: FlashSpec,
        spec1range: np.ndarray | pp.number,
        spec2range: np.ndarray | pp.number,
        zrange: Optional[Sequence[np.ndarray | pp.number]] = None,
        /,
        *,
        field: str | list[str] = "phasesplit",
        zindex: int = 0,
        transpose: bool = False,
        plotkwargs: Optional[dict] = None,
        **kwargs,
    ) -> tuple[Figure, FlashResults]:
        """ "Plot a 2D phase diagram for specified ranges.

        The type of flash performed is indicated with ``specification``.

        The three given ranges are used as follows:

        Two must be given as arrays, one must be given in scalar format.

        The first non-scalar range is used as the vertical axis.
        The second non-scalar range is used as the horizontal axis.
        If ```zrange`` is an array, the fluid must be a mixture (multiple componets) and
        which ``z`` value`` to plot on the vertical axis can be defined using
        ``zindex``.

        The values to be plotted are indicated using a string.
        In general, the strings are expected to denote fields of the
        :class:`FlashResults` returned by :meth:`flash`.

        For phase-related quantities, use ``f'*_{int}'`` to specify which phase.
        For quantities related to components in phases, use two integers seperated by
        an underscore, where the first integer denotes the phase index and the
        second the component index. For derivatives of phase properties an additional
        integer is expected. In any case the last integer denotes the derivative index.
        For admissible field names, see
        :class:`~porepy.compositional.states.PhaseProperties`

        Examples:

            - ``'h_1'`` will plot the specific enthalpy of the first phase.
            - ``'x_1_2'`` will plot the partial fraction of the second component in the
              first phase.
            - ``'phis_2_4'`` will plot the fugacity coefficient of the fourth component
              in the second phase.
            - ``'dmu_2_1'`` will plot the derivative of the viscosity of the second
              phase found in the first row of ``dmu`` of respective phase properties.
            - ``'dphis_1_2_3' will return the derivative of the fugacity coefficient of
              the second component in the first phase with respect to its third
              dependency, i.e., the third row ``dphis[...,..., 3]``.

        The default value for ``field``, ``'phasesplit'`` will plot the number of phases
        and their physical catecorization. I.e., liquid, vapor-liquid, vapor, etc.

        Supported keyword arguments are:

        - ``'xtransform'``: A callable transforming the values on the horizontal axis
          for the plot.
        - ``'ytransform'``: A callable transforming the values on the vertical axis for
          the plot.
        - ``'vtransform'``: A callable transforming the values to be plotted. Must take
          an array (values) and a string (field name).
        - ``'initial_state'``: See :meth:`flash`.
        - ``'params'``: See :meth:`flash`.
        - ``'flash_kwargs'``: See :meth:`flash`.
        - ``'eps'``: Used to define an absent phase numerically (i.e. values below this
          will plot a phase as absent.)
        - ``'show_not_converged'``: If ``True``, non-converged flash points will be
          plotted with a cross. Default is ``False``.
        - ``'max_cols'``: If given, sets the number of columns for multi-field plots.
          Defaults to 3.
        - ``'figsize'``: A 2-tuple of floats containing width and height of the figure.
        - ``'phasecontour'``: Boolean. If true, draws contour lines around different
          phase regions.

        Parameters:
            specification: The flash to be calculated.
            spec1range: Range or value of the first fixed state function
            spec2range: Range or value of the second fixed state function
            zrange: Ranges or values for the compositions. For pure fluids, this is
                always assumed to be just the value 1.
            field: One or multiple fields to be plotted.
            zindex: ``default=0``

                If ``zrange`` is a sequence of compositions, this defines which one
                should be used for the plot.
            transpose: ``default=False``

                Transposes the plot.
            plotkwargs: Keyword arguments for :obj:`pyplot.pcolormesh`
            **kwargs: Keyword arguments for this function.

        Returns:
            The handle to the created figure and the results of the flash performed to
            obtain the figure.

        """
        # Parsing compositions.
        ncomp = self.params["num_components"]
        err_z = None
        if zrange is None:
            if ncomp == 1:
                zrange = [1.0]
            else:
                err_z = f"Expecting {ncomp} feed fractions, none given."
        else:
            if len(zrange) != ncomp:
                err_z = f"Expecting {ncomp} feed fractions, {len(zrange)} given."

        zrange = cast(Sequence, zrange)

        if err_z:
            raise ValueError(err_z)

        # Parsing input to obtain axes, look for non-vectorized quantity.
        isrange = [False, False, False]
        if isinstance(spec1range, np.ndarray):
            isrange[0] = True
        if isinstance(spec2range, np.ndarray):
            isrange[1] = True
        if np.any([isinstance(z, np.ndarray) for z in zrange]):
            if not isinstance(zrange[zindex], np.ndarray):
                raise ValueError(f"Compositions at index {zindex} must be an array.")
            isrange[2] = True

        s1: np.ndarray | pp.number
        s2: np.ndarray | pp.number
        z: Sequence[np.ndarray | pp.number]

        match isrange:
            case [True, True, False]:
                yvals = spec1range
                xvals = spec2range
                xm, ym = np.meshgrid(xvals, yvals)
                s1 = ym.flatten()
                s2 = xm.flatten()
                z = zrange
                xlabel = specification.name[1]
                ylabel = specification.name[0]
            case [True, False, True]:
                yvals = spec1range
                xvals = zrange[zindex]
                xm, ym = np.meshgrid(xvals, yvals)

                s1 = ym.flatten()
                s2 = spec2range
                assert isinstance(s2, (int, float))
                z = [
                    np.meshgrid(_, yvals)[0].flatten()
                    if isinstance(_, np.ndarray)
                    else _
                    for _ in zrange
                ]
                xlabel = f"z_{zindex}"
                ylabel = specification.name[0]
            case [False, True, True]:
                yvals = spec2range
                xvals = zrange[zindex]
                xm, ym = np.meshgrid(xvals, yvals)

                s1 = spec1range
                assert isinstance(s1, (int, float))
                s2 = ym.flatten()
                z = [
                    np.meshgrid(_, yvals)[0].flatten()
                    if isinstance(_, np.ndarray)
                    else _
                    for _ in zrange
                ]
                xlabel = f"z_{zindex}"
                ylabel = specification.name[1]
            case _:
                raise ValueError("Exactly two of the 3 ranges must be arrays.")

        shape: tuple[int, int] = xm.shape
        assert ym.shape == shape, "Failed to reshape axis values."
        spec = {specification.name[0]: s1, specification.name[1]: s2}
        eps = kwargs.get("eps", 1e-7)

        # Compute results for plot.
        results = self.flash(
            # Just for typing, parse_flash_arguments will resolve issues.
            cast(IsobaricSpecifications, spec),
            z,
            initial_state=kwargs.get("initial_state", None),
            params=kwargs.get("params", None),
            **kwargs.get("flash_kwargs", {}),
        )
        results.postprocess_fractions(eps)

        xtransform: Callable[[np.ndarray], np.ndarray] = kwargs.get(
            "xtransform", lambda x: x
        )
        ytransform: Callable[[np.ndarray], np.ndarray] = kwargs.get(
            "ytransform", lambda x: x
        )
        vtransform: Callable[[np.ndarray, str], np.ndarray] = kwargs.get(
            "vtransform", lambda x, s: x
        )

        not_conv = ~results.converged
        s1f = spec[specification.name[0]][not_conv]
        s2f = spec[specification.name[1]][not_conv]

        # Shape axis values.
        xlabel = specification.name[1]
        ylabel = specification.name[0]
        if transpose:
            xm, ym = [ym.transpose(), xm.transpose()]
            xlabel, ylabel = [ylabel, xlabel]
            s1f, s2f = [s2f, s1f]
        xm = xtransform(xm.flatten()).reshape(shape)
        ym = ytransform(ym.flatten()).reshape(shape)
        s2f = xtransform(s2f)
        s1f = ytransform(s1f)

        # Parse field and format values to be plotted.
        vals: list[np.ndarray] = []
        split_v: np.ndarray | None = None
        if isinstance(field, str):
            fields = [field]
        else:
            fields = field
        for f in fields:
            v = self._parse_field(results, f, eps)
            v = vtransform(v, f).reshape(shape)
            if transpose:
                v = v.transpose()
            vals.append(v)
            if f == "phasesplit":
                split_v = v

        # Parse plotting options and create figure.
        nf = len(fields)
        ni = 1
        if nf == 1:
            nrows = 1
            ncols = 1
        else:
            max_cols = int(kwargs.get("max_cols", 3))
            assert max_cols > 0, "Require at least 1 column."
            if nf <= max_cols:
                ncols = nf
                nrows = 1
            else:
                ncols = max_cols
                nrows = int(np.floor(nf / ncols))
                if nf % ncols > 0:
                    nrows += 1
            assert nrows >= 1, "Require at least 1 row."

        plot_phase_contour = kwargs.get("phasecontour", False)
        if split_v is None and plot_phase_contour:
            split_v = self._parse_field(results, "phasesplit", eps).reshape(shape)
            assert isinstance(split_v, np.ndarray)
            if transpose:
                split_v = split_v.transpose()

        base_fh = 5.0  # Base height for 1 plot.
        base_fw = 5.5  # Base width for 1 plot.
        fig = plt.figure(
            figsize=kwargs.get("figsize", (ncols * base_fw, nrows * base_fh))
        )

        for v, f in zip(vals, fields):
            default_plotkwargs: dict[str, Any] = {
                "shading": "nearest",
                "label": f,
                "cmap": "viridis",
            }
            default_plotkwargs.update(
                plotkwargs if isinstance(plotkwargs, dict) else {}
            )
            if f == "phasesplit":
                cmap, norm, cbticks, cblabels = self._get_split_cmap(v)
                default_plotkwargs["cmap"] = cmap
                default_plotkwargs["norm"] = norm

            ax = fig.add_subplot(nrows, ncols, ni)
            ni += 1
            ax.set_box_aspect(1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f)

            img = ax.pcolormesh(
                xm,
                ym,
                v,
                **default_plotkwargs,
            )
            cax = ax.inset_axes((1.01, 0.2, 0.05, 0.6))
            cb_rr = fig.colorbar(img, ax=ax, cax=cax, orientation="vertical")
            if f == "phasesplit":
                cb_rr.set_ticks(cbticks)
                cb_rr.set_ticklabels(cblabels)

            if plot_phase_contour:
                assert isinstance(split_v, np.ndarray), "Failed to parse split values."
                lvls = np.unique(split_v)
                ax.contour(xm, ym, split_v, lvls, colors="black", linewidths=1.5)

            if np.any(not_conv) and kwargs.get("show_not_converged", False):
                img_nc = ax.scatter(
                    s2f,
                    s1f,
                    s=5,
                    c="firebrick",
                    marker="x",
                    label="No conv.",
                )
                ax.legend(handles=[img_nc], loc="lower center")

        fig.tight_layout()
        return fig, results

    def _parse_field(self, results: FlashResults, field: str, eps: float) -> np.ndarray:
        """ "Helper method to parse the field to be plotted in the phase diagram."""
        err_msg: str | None = None
        vals: np.ndarray | None = None
        nphase: int = self.params["num_phases"]
        gasidx: int = self.params["gas_phase_index"]

        # Phase split is determined based on the phase fractions.
        if field == "phasesplit":
            y = results.y
            y[y < eps] = 0.0
            y[y > 1] = 1.0
            y = pp.compositional.normalize_rows(y.T).T
            gid = np.zeros(nphase, dtype=bool)
            if gasidx is not None:
                gid[gasidx] = True
            yL = y[~gid]
            yG = y[gid]
            if not np.any(gid):
                yG = np.zeros(y.shape[0])
            else:
                # Reshape
                yG = yG[0]

            # Encoding values:
            # 0 - gas only
            # i in [1, 999], gas with i liquids
            # i in [1001, 2000], 1001 1 liquid, 1002 2 liquid ...
            vals = cast(np.ndarray, (yL > 0.0).sum(axis=0).astype(int))
            # Where no gas, shift by coding factor.
            vals[yG <= 0.0] += self._phasesplit_code_shift
            # Sanity check that pure gas is indicated by zero.
            assert np.all(vals[yG >= 1.0] == 0), "Phasesplit encoding failed."
            # Sanity check that where gas is also present, values are below code factor.
            assert np.all(vals[yG > 0] < self._phasesplit_code_shift), (
                "Non-unique encoding."
            )
            vals = vals.astype(int)
        # Other values are extraced from the results directly.
        else:
            field_spec = field.split("_")
            n = len(field_spec)

            if "ext" in field_spec or "normalized" in field_spec:
                base = f"{field_spec[0]}_{field_spec[1]}"
                n -= 2
                field_spec = field_spec[2:]
            else:
                base = field_spec[0]
                n -= 1
                field_spec = field_spec[1:]

            if n > 3:
                err_msg = f"Expecting at most 3 indices in field name, got {n}"
            # No index at all can only mean fluid property.
            elif n == 0:
                if hasattr(results, base):
                    vals = getattr(results, base)
                else:
                    err_msg = f"Fluid property {base} not defined."
            else:
                phase_index = int(field_spec[0])
                cd_index = None
                d_index = None
                if n >= 2:
                    cd_index = int(field_spec[1])
                if n >= 3:
                    d_index = int(field_spec[2])

                if phase_index > nphase or phase_index < 1:
                    err_msg = f"Phase index in {field} out of range."
                else:
                    # If n == 1, it can be saturations or fractions.
                    if n == 1 and hasattr(results, base):
                        vals = getattr(results, base)[phase_index - 1]
                    else:
                        phase = results.phases[phase_index - 1]
                        if hasattr(phase, base):
                            vals = getattr(phase, base)
                        else:
                            err_msg = f"Property {base} not defined."

                if cd_index is not None and vals is not None:
                    if vals.ndim != n:
                        err_msg = (
                            f"Dimension of phase property {base} must be exactly {n}."
                        )
                    if cd_index > vals.shape[0] or cd_index < 1:
                        err_msg = (
                            f"Index {cd_index} for phase property {base} out of range."
                        )
                    else:
                        if vals.ndim > 2:
                            vals = vals[cd_index - 1, :, :]
                            if d_index:
                                if d_index < 1 or d_index > vals.shape[0]:
                                    err_msg = (
                                        f"Index {cd_index} for phase property "
                                        f"{base} out of range."
                                    )
                                else:
                                    vals = vals[d_index - 1]
                        else:
                            vals = vals[cd_index - 1]

        if err_msg:
            raise ValueError(err_msg)

        assert isinstance(vals, np.ndarray), f"Failed to parse field {field}."
        assert vals.ndim == 1, f"Parsed field of unexpected dimension {vals.ndim}."
        return vals

    def _get_split_cmap(
        self, vals: np.ndarray
    ) -> tuple[ListedColormap, BoundaryNorm, list[float], list[str]]:
        """ "Helper method to construct a suitable colormap based on phase split."""

        vals = np.unique(vals).astype(int)
        colors: list[Any]
        if np.any(vals == 0):
            colors = ["wheat"]
            ticks = [0.0]
            labels = ["V"]
            bounds = [-0.5, 0.5]
        else:
            colors = []
            ticks = []
            labels = []
            bounds = [0.5]

        # Colormap for (V)L+ discrete spectra.
        vlcmap = LinearSegmentedColormap.from_list(
            "VL+", [(0.0, "honeydew"), (1.0, "seagreen")]
        )
        lcmap = LinearSegmentedColormap.from_list(
            "L+", [(0.0, "lightsteelblue"), (1.0, "darkblue")]
        )
        shift = self._phasesplit_code_shift
        # Map VL splits (1 to max L in VL+ region) to 0-1 and get colors.
        vlvals = vals[(vals > 0) & (vals < shift)]
        # Map L splits (1 to max L in L+ region) to 0-1 and get colors
        lvals = vals[vals > shift]

        if np.any(vlvals):
            m = vlvals.max()
            ticks.append(1.0)
            if m == 1:
                labels.append("VL")
                colors.append(vlcmap(0.0))
                bounds.append(1.5)
            else:
                labels.append("VL1")
                f = lambda x: -1 / (1 - m) * x + 1 / (1 - m)
                for i in range(1, m + 1):
                    colors.append(vlcmap(f(i)))
                    bounds.append(i + 0.5)
                ticks.append(m)
                labels.append(f"VL{int(m)}")

        if np.any(lvals):
            m = lvals.max()
            ticks.append(shift + 1)
            if m == shift + 1:
                labels.append("L")
                colors.append(lcmap(0.0))
                bounds.append(m + 1.5)
            else:
                labels.append("L1")
                f = lambda x: -1 / (1 - m) * x + 1 / (1 - m)
                for i in range(1 + shift, m + 1):
                    colors.append(lcmap(f(i)))
                    bounds.append(i + 0.5)
                ticks.append(m)
                labels.append(f"L{int(m - shift)}")

        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds, cmap.N, clip=True)
        return cmap, norm, ticks, labels
