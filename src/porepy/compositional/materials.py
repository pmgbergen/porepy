"""Storage classes for solid constants and fluid components.

Material contants are values representing either constant physical properties (e.g.
critical pressure) or parameters for constitutive laws (e.g. constant compressibility
for exponential-type density law). A material is instantiated with a
:class:`~porepy.models.units.Units` object, which defines the units of the physical
properties for a simulation setup. While the constants must be given in base SI units
when instantiating a material class, its constants are converted and stored in the
target units, to be used subsequently.

For converting values on the fly, see :meth:`~porepy.models.units.Units.convert_units`.

"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, dataclass, field, is_dataclass
from typing import Any, Callable, ClassVar, TypeVar, cast

import porepy as pp

from .base import Component

__all__ = [
    "Constants",
    "FluidComponent",
    "SolidConstants",
    "NumericalConstants",
    "ReferenceVariableValues",
    "load_fluid_constants",
]

number = pp.number


# 1. By using keyword_only arguments for the construction of materials, the user is
# forced to instantiate the constants with the right names of constants, otherwise
# errors are raised (no separate checks required).
# 2. By providing default values to the fields, not every constant is required. The user
# is expected to be aware which physics are used in the model.
# 3. By instructing dataclass to not override the __eq__ provided by object, the classes
# remain hashable (can be used in dicts), see for instance
# https://stackoverflow.com/a/52390734.
# We do not need to compare constants by equality of field values (what eq=True would
# do), since the class is intended for instances like solids and fluids, which *are*
# distinct by design.
# Note however, it inherits the default eq from object, which is a comparison of memory
# addresses.
@dataclass(kw_only=True, eq=False)
class Constants:
    """Material property container and conversion class.

    The base class identifies a material using a given :attr:`name`. To define material
    properties of some kind, derive a dataclass with this class as its base (as
    keywords-only and ``eq==False`` data class).

    Constants are declared as fields (float or int) with default values.

    Derived classes must have a class attribute :attr:`SI_units` containing information
    about the the physical unit of each declared constant.

    If the user wants the material to be presented in other than SI units, a ``units=``
    kw-argument can be passed to declare the target, non-SI units (e.g. MPa instead of
    Pa).

    The base class provides a check that constants defined as dataclass fields
    are not assignable, once the instance is created. This is motivated by the fact that
    material parameters are not supposed to change (or converted to another unit) once
    given in a simulation.

    Note:
        When inheriting from this class and making another data class, use
        ``@dataclass(kw_only=True, eq=False)`` to be consistent with the base class
        and to make the child hashable (important when storing it in dictionaries).

        Having an overload of ``__eq__`` by comparing fields (as ``dataclass`` does)
        makes little sense for this class, as it is designed to hold different constants
        per instance. We loose only the hashability with the redundant ``__eq__``.

    Important:
        When instantiating a ``Constants`` data class, the constants must all be
        strictly given in SI units. The conversion happens post-initialization for the
        subsequent simulation.

    Important:
        Every derived class must have a class attribute :attr:`SI_units`, **annotated as
        ClassVar**. This is to inform the base class about the used SI units.

        For examples, see :class:`FluidComponent` or :class:`SolidConstants`. For
        instructions on how to write composed units, see
        :meth:`~porepy.models.units.Units.convert_units`.

    """

    # NOTE: Annotating it as a ClassVar leads to the dataclasses decorator ignoring this
    # in its machinery. The annotation must not be forgotten in derived classes.
    SI_units: ClassVar[dict[str, str]] = {}
    """A dictionary containing the SI unit of every material constant defined by a
    derived class.

    E.g., ``{'pressure': 'Pa'}``

    Every derived data class must have one implemented and annotated as a ``ClassVar``.

    Note:
        The ``ClassVar`` hints to ``dataclass`` that this field should not be part of
        the dataclass mechanism, but a single attribute accessed by all instances.

    """

    name: str = ""
    """Name of material given at instantiation."""

    units: pp.Units = field(default_factory=lambda: pp.Units())
    """Units of the species constants.

    This defines the physical units used in the simulation for this species. Constants
    with a physical dimension passed during instantiation will be converted to the
    units defined here, while constants with dimension ``['-'], [''], ['1']`` will be
    considered dimensionless and left untouched.

    """

    # NOTE init=False makes this a data class field, which is not in the constructor
    # signature. Its value is computed in the post-initialization procedure.
    constants_in_SI: dict[str, number] = field(init=False, default_factory=dict)
    """A dictionary containing the original constants in SI units, passed at the
    instantiation of this class.

    Dictionary keys are given by the field/parameter name as string.

    Note:
        This object is not given during instantiation of a material class, but populated
        in the post-initialization procedure.

    """

    _initialized: bool = field(init=False, default=False)
    """Flag marking the end of the initialization and post-initialization procedure.

    Set to be ``True``, once :meth:`__post_init__` is done. Used to disallow the
    assignment of material parameters after construction.

    Once True, constants cannot be set anymore.

    """

    def __post_init__(self, *args, **kwargs) -> None:
        """Post-initialization procedure to convert the given species parameters from
        SI units to the units given by :attr:`units`.

        The original constants in SI units are stored in :attr:`constants_in_SI`.

        """

        # Get the constants using the data class field, excluding utilities
        constants = asdict(self)
        constants.pop("name")
        constants.pop("units")
        constants.pop("constants_in_SI")
        constants.pop("_initialized")

        # Safety check if the user forgot to annotate the SI_units as a class variable.
        # If yes, dataclass would declare it as a data field.
        if "SI_units" in constants:
            raise AttributeError(
                f"The attribute 'SI_units' in class {type(self)} must be annotated as"
                + " SI_units: typing.ClassVar[dict[str,str]]."
            )

        # By logic, what remains are constants as numbers storing original constants.
        self.constants_in_SI.update(cast(dict[str, number], dict(constants)))

        # Transform the constants to SI and store them in this instance.
        # Bypass the freezing of attributes https://stackoverflow.com/questions/
        # 53756788/how-to-set-the-value-of-dataclass-field-in-post-init-when-frozen-true
        for k, v in self.constants_in_SI.items():
            # Convert the value to SI if it has a physical unit.
            if k not in type(self).SI_units:
                raise AttributeError(
                    f"Material constant {k} requires a declaration of its physical unit"
                    + f" in SI in {type(self)}.SI_units."
                )
            si_unit = self.SI_units[k]
            v_in_custom_units = self.units.convert_units(v, si_unit)
            object.__setattr__(self, k, v_in_custom_units)

        # Flag the initialization procedure as done.
        object.__setattr__(self, "_initialized", True)

    def __init_subclass__(cls) -> None:
        """If a data-class inherits this class, the base class post-initialization
        procedure must be inherited as well for the conversion from SI units (at
        instantiation) to simulation :attr:`units`.

        Also, the blockage of rewriting the material parameters must be inherited as
        well.

        Both is achieved by assigning the base class ``__post_init__`` and
        ``__setattr__`` of this base class to any child class which is a data class.

        """

        if is_dataclass(cls):
            if hasattr(cls, "__post_init__"):
                if cls.__post_init__ != Constants.__post_init__:
                    raise AttributeError(
                        f"Data classes inheriting form {Constants} must not"
                        + " have __post_init__ defined, but inherit it."
                    )
            else:
                object.__setattr__(cls, "__post_init__", Constants.__post_init__)
            if hasattr(cls, "__setattr__"):
                if cls.__setattr__ != Constants.__setattr__:
                    raise AttributeError(
                        f"Data classes inheriting form {Constants} must not"
                        + " have __setattr__ defined, but inherit it."
                    )
            else:
                object.__setattr__(cls, "__setattr__", Constants.__setattr__)

    def __setattr__(self, name: str, value: Any) -> None:
        """Custom attribute setter imitating dataclasses with frozen attributes, but
        only disallowing the setting of attributes which correspond to a field defined
        as part of the material constant framework.

        Parameters:
            name: The name of an attribute.
            value: The new value to be set.

        Raises:
            FrozenInstanceError: If the user attempts to set any of the following
                attributes:

                - :attr:`name`
                - :attr:`units`
                - :attr:`constants_in_SI`
                - and any field of material parameters indirectly defined in
                  :attr:`SI_units`

        """

        frozen = ["name", "units", "constants_in_SI", "_initialized"] + list(
            type(self).SI_units.keys()
        )
        if name in frozen and self._initialized:
            raise FrozenInstanceError(
                f"Cannot assign to field {name}. Names, units and material parameters"
                + " are frozen once set."
            )
        else:
            super().__setattr__(name, value)

    def to_units(self: _Constants, units: pp.Units) -> _Constants:
        """Utility to quickly convert constants to new units.

        Parameters:
            units: A new unit system. Note that ``units`` must cover all SI units
                declared in :attr:`SI_units` of the constants (child) class.

        Returns:
            A new instance of of the data class using this method, with parameters
            converted according to ``units``.

        """
        return type(self)(name=self.name, units=units, **self.constants_in_SI)


_Constants = TypeVar("_Constants", bound=Constants)
"""Type variable for Constants-like objects. Mainly used to type the units conversion
of :class:`Constants` correctly."""


# See comment above (in the definition of Constants) for the reasoning behind the
# eq=False.
@dataclass(kw_only=True, eq=False)
class FluidComponent(Constants, Component):
    """Material data class for fluid components.

    It declares parameters relevant for fluid-like components, which are expected of a
    fluid by the remaining framework, especially flow and transport equations.

    This class is intended for 1 fluid component only. Fluid mixtures with multiple
    components require multiple sets of constants to define individual fluid components.

    This class is used as the default representation of fluid components inside a
    mixture.

    """

    SI_units: ClassVar[dict[str, str]] = dict(
        {
            "density": "kg * m^-3",
            "molar_mass": "kg * mol^-1",
            "critical_pressure": "Pa",
            "critical_temperature": "K",
            "critical_specific_volume": "m^3 * kg^-1",
            "acentric_factor": "-",
            "compressibility": "Pa^-1",
            "specific_heat_capacity": "J * kg^-1 * K^-1",
            "thermal_expansion": "K^-1",
            "viscosity": "Pa * s",
            "thermal_conductivity": "W * m^-1 * K^-1",
            "normal_thermal_conductivity": "W * m^-1 * K^-1",
        }
    )

    acentric_factor: number = 0.0

    compressibility: number = 0.0

    critical_pressure: number = 1.0

    critical_specific_volume: number = 1.0

    critical_temperature: number = 1.0

    density: number = 1.0

    molar_mass: number = 1.0

    normal_thermal_conductivity: number = 1.0

    thermal_conductivity: number = 1.0

    thermal_expansion: number = 0.0

    specific_heat_capacity: number = 1.0

    viscosity: number = 1.0


# Strictly speaking, it is not necessary to have eq=False for SolidConstants. The issues
# with hashability noted for Constants (see above), do not apply here, since the class
# could have been made frozen (this is in contrast to the FluidComponent, which
# inherits from the non-constant Component class). However, for consistency, we keep
# eq=False; although this means objects of type SolidConstants cannot be compared by
# fields, this is also not an expected use case.
@dataclass(kw_only=True, eq=False)
class SolidConstants(Constants):
    """Material data class for solid species present in the porous medium.

    It declares solid species parameters, which are expected of a solid by the remaining
    framework, especially poro- & fracture-mechanics.

    This class is meant for 1 solid species only. Different domains in the mD-setting
    require each their own material constants instance in the case of heterogeneity.

    """

    SI_units: ClassVar[dict[str, str]] = dict(
        {
            "density": "kg * m^-3",
            "biot_coefficient": "-",
            "dilation_angle": "rad",
            "fracture_gap": "m",
            "fracture_normal_stiffness": "Pa * m^-1",
            "fracture_tangential_stiffness": "Pa * m^-1",
            "friction_coefficient": "-",
            "lame_lambda": "Pa",
            "maximum_elastic_fracture_opening": "m",
            "normal_permeability": "m^2",
            "permeability": "m^2",
            "porosity": "-",
            "residual_aperture": "m",
            "shear_modulus": "Pa",
            "skin_factor": "-",
            "specific_heat_capacity": "J * kg^-1 * K^-1",
            "specific_storage": "Pa^-1",
            "thermal_conductivity": "W * m^-1 * K^-1",
            "thermal_expansion": "K^-1",
            "well_radius": "m",
        }
    )

    biot_coefficient: number = 1.0

    density: number = 1.0

    dilation_angle: number = 0.0

    fracture_gap: number = 0.0

    fracture_normal_stiffness: number = 1.0
    """Intended use is in Barton-Bandis-type models for elastic fracture deformation."""

    fracture_tangential_stiffness: number = -1.0
    """
    Note:
        The current default value is -1.0, with the convention that negative
        values correspond to a fracture that does not deform elastically in the
        tangential direction.
    """

    friction_coefficient: number = 1.0

    lame_lambda: number = 1.0
    """Lame's first parameter"""

    maximum_elastic_fracture_opening: number = 0.0
    """Intended use is in Barton-Bandis-type models for elastic fracture deformation."""

    normal_permeability: number = 1.0

    permeability: number = 1.0

    porosity: number = 0.1

    residual_aperture: number = 0.1

    shear_modulus: number = 1.0

    skin_factor: number = 0.0

    specific_heat_capacity: number = 1.0

    specific_storage: number = 1.0

    thermal_conductivity: number = 1.0

    thermal_expansion: number = 0.0

    well_radius: number = 0.1


@dataclass(kw_only=True, eq=False)
class FractureDamageSolidConstants(SolidConstants):
    """Solid parameters for fracture damage models."""

    # NOTE this makes a deep copy of the solid constants dict.
    SI_units: ClassVar[dict[str, str]] = dict(**SolidConstants.SI_units)
    SI_units.update(
        {
            "initial_dilation_damage": "-",
            "initial_friction_damage": "-",
            "dilation_damage_decay": "-",
            "friction_damage_decay": "-",
        }
    )
    initial_friction_damage: float = 1.0
    friction_damage_decay: float = 0.0
    initial_dilation_damage: float = 1.0
    dilation_damage_decay: float = 0.0


@dataclass(kw_only=True, eq=False)
class NumericalConstants(Constants):
    """Data class containing numerical method parameters,
    including characteristic sizes.

    """

    SI_units: ClassVar[dict[str, str]] = dict(
        {
            "characteristic_displacement": "m",
            "characteristic_contact_traction": "Pa",
            "open_state_tolerance": "-",
            "contact_mechanics_scaling": "-",
        }
    )

    characteristic_contact_traction: number = 1.0
    """Characteristic traction used for scaling of contact mechanics."""

    characteristic_displacement: number = 1.0
    """Characteristic displacement used for scaling of contact mechanics."""

    contact_mechanics_scaling: number = 1e-1
    """Safety scaling factor, making fractures softer than the matrix."""

    open_state_tolerance: number = 1e-10
    """Tolerance parameter for the tangential characteristic contact mechanics."""


@dataclass(kw_only=True)
class ReferenceVariableValues(Constants):
    """A data class storing reference values for a model.

    Intended use is for defining for example reference pressure and temperature,
    where the perturbation from the reference value :math:`(p - p_{ref})` is used in
    constitutive laws for example.

    The aim is to have a single set of (scalar) reference values, which is to be used
    consistently in the whole model, with the correct units.

    """

    SI_units: ClassVar[dict[str, str]] = dict(
        {
            "pressure": "Pa",
            "temperature": "K",
        }
    )

    pressure: number = 0.0

    temperature: number = 0.0


def load_fluid_constants(names: list[str], package: str) -> list[FluidComponent]:
    """Creates a fluid species, if identifiable by ``name`` in ``package``.

    Utility function to extract parameters for a fluid, like critical values.

    Important:
        The ``name`` is passed directly to the package. There is no guarantee if the
        returned values are correct, of if the third-party package will work without
        throwing errors.

    Parameters:
        names: A list of names or chemical formulae to look up the chemical species.
        package: Name of one of the supported packages containing chemical databases.
            Currently supported:

            - ``'chemicals'``: :mod:`chemicals`

    Raises:
        NotImplementedError: If an unsupported package is passed as argument.
        ModuleNotFoundError: If the ``package`` could not be imported.

    Returns:
        If the look-up was successful, extracts the relevant data and returns respective
        data structure.

    """

    species: list[FluidComponent] = []

    cas: str

    cas_loader: Callable
    mw_loader: Callable
    pc_loader: Callable
    Tc_loader: Callable
    vc_loader: Callable
    omega_loader: Callable

    if package == "chemicals":
        # Will raise import error if not found.
        import chemicals  # type: ignore

        cas_loader = chemicals.CAS_from_any
        mw_loader = lambda x: chemicals.MW(x) * 1e-3  # molar mass in kg / mol
        pc_loader = chemicals.Pc  # critical pressure in Pa
        Tc_loader = chemicals.Tc  # critical temperature in K
        vc_loader = chemicals.Vc  # critical volume in m^3 / mol
        omega_loader = chemicals.acentric.omega  # acentric factor

    else:
        raise NotImplementedError(f"Unsupported package `{package}`.")

    for name in names:
        cas = str(cas_loader(name))

        # critical volume is molar, need conversion
        mm = float(mw_loader(cas))
        v_crit = float(vc_loader(cas)) / mm

        species.append(
            FluidComponent(
                name=name,
                molar_mass=mm,
                critical_pressure=float(pc_loader(cas)),
                critical_temperature=float(Tc_loader(cas)),
                critical_specific_volume=v_crit,
                acentric_factor=float(omega_loader(cas)),
            )
        )

    return species
