"""Storage classes for material constants.

Materials are storage classes for values of physical properties. They are typically used
when composing constitutive laws. A material is instantiated with a Units object, which
defines the units of the physical properties. The material can then be used to convert
from units set by the user (standard SI units) to those specified by the Units object,
which are the ones used internally in the simulation. The conversion hopefully reduces
problems with scaling/rounding errors and condition numbers.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import ClassVar, Optional, Union, cast, overload

import numpy as np

import porepy as pp

__all__ = [
    "MaterialConstants",
    "FluidConstants",
    "SolidConstants",
]

number = pp.number


class _HashableDict(dict):
    """See https://stackoverflow.com/questions/1151658/python-hashable-dicts.

    We require hashable dictionaries for the below material constant classes which
    contain various constants and unit declarations, all in simple formats and per se
    hashable.

    """

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


# The dataclass is frozen, since material parameters are not supposed to change, as well
# as the units they are given in.
# By using this in combination with keyword_only arguments for the construction of
# materials, the user is forced to instantiate the constants with the right names of
# constants, otherwise errors are raised (no separate checks required).
# By providing default values to the fields, not every constant is required. The user
# is expected to be aware which physics are used in the model.
@dataclass(frozen=True, kw_only=True)
class MaterialConstants:
    """Material property container and conversion class.

    The base clase identifies a material using a given :attr:`name`.
    To define material properties of some kind, derive a dataclass with this class
    as its base (as frozen and keywords-only data class).

    Material constants are declared as fields (float or int) with default values.

    Derived classes must have a class attribute :attr:`SI_units` containing information
    about the the physical unit of each declared constant.

    If the user wants the material to be presented in other than SI units, a ``units=``
    kw-argument can be passed to declare the target, non-SI units (f.e. MPa instead of
    Pa). The base class functionality will take conversion

    Important:
        When instantiating a material constants data class, the constants must all
        be strictly given in SI units. The conversion happens post-initialization for
        the subsequent simulation.

    Important:
        Every derived class must have a class attribute :attr:`SI_units`, **annotated as
        ClassVar**. This is to inform the base class about the used SI units.

        For Examples, see :class:`FluidConstants`.
        For instructions on how to write composed units, see :meth:`convert_units`

    This class is intended for 1 species only. Different domains or fluids in the
    mD-setting require each their own material constants instance in the case of
    heterogenity.

    """

    SI_units: ClassVar[_HashableDict[str, str]] = _HashableDict()
    """A dictionary containing the SI unit of every material constant defined by
    a derived class.

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
    units defined here.

    """

    constants_in_SI: _HashableDict[str, number] = field(
        init=False, default_factory=_HashableDict
    )
    """A dictionary containing the original constants in SI units, passed at the
    instantiation of this class.

    Dictionary keys are given by the field/parameter name as string.

    Note:
        This object is not given during instantiation of a material class, but populated
        in the post-initialization procedure.

    """

    def __post_init__(self, *args, **kwargs) -> None:
        """Post-initialization procedure to convert the given species parameters from
        SI units to the units given by :attr:`units`.

        The original constants in SI units are stored in :attr:`constants_in_SI`.

        """

        # Get the material constants using the data class field, excluding utilities
        constants = asdict(self)
        constants.pop("name")
        constants.pop("units")
        constants.pop("constants_in_SI")

        # safety check if the user forgot to annotate the SI_units as a class variabel
        # If yes, dataclass would declare it as a data field
        if "SI_units" in constants:
            raise AttributeError(
                f"The attribute 'SI_units' in class {type(self)} must be annotated as"
                + " SI_units: typing.ClassVar[dict[str,str]]."
            )

        # by logic, what remains are constants as numbers
        # storing original constants
        self.constants_in_SI.update(
            cast(_HashableDict[str, number], _HashableDict(constants))
        )

        # transform the constants to SI and store them in this instance
        # Bypass the freezing of attributes https://stackoverflow.com/questions/
        # 53756788/how-to-set-the-value-of-dataclass-field-in-post-init-when-frozen-true
        for k, v in self.constants_in_SI.items():
            # convert the value to SI if it has a physical unit
            if k not in type(self).SI_units:
                raise AttributeError(
                    f"Material constant {k} requires a declaration of its physical unit"
                    + f" in SI in {type(self)}.SI_units."
                )
            si_unit = self.SI_units[k]
            v_in_custom_units = self.convert_units(v, si_unit)
            object.__setattr__(self, k, v_in_custom_units)

    def __init_subclass__(cls) -> None:
        """If a data-class inherits this class, the base class post-initialization
        procedure must be inherited as well for the conversion from SI units
        (at instantiation) to simulation :attr:`units`."""

        if is_dataclass(cls):
            if hasattr(cls, "__post_init__"):
                if cls.__post_init__ != MaterialConstants.__post_init__:
                    raise AttributeError(
                        f"Data classes inheriting form {MaterialConstants} must not"
                        + " have __post_init__ defined, but inherit it."
                    )
            else:
                object.__setattr__(
                    cls, "__post_init__", MaterialConstants.__post_init__
                )

    @overload
    def convert_units(
        self, value: number, units: str, to_si: Optional[bool] = False
    ) -> number: ...

    @overload
    def convert_units(
        self,
        value: np.ndarray,
        units: str,
        to_si: Optional[bool] = False,
    ) -> np.ndarray: ...

    def convert_units(
        self,
        value: Union[number, np.ndarray],
        units: str,
        to_si: Optional[bool] = False,
    ) -> Union[number, np.ndarray]:
        """Convert value between SI and user specified units.

        The method divides the value by the units as defined by the user. As an example,
        if the user has defined the unit for pressure to be 1 MPa, then a value of  1e8
        will be converted to 1e8 / 1e6 = 1e2. Conversely, if to_si is True, the value
        will be converted to SI units, i.e. a value of 1e-2 results in 1e-2 * 1e6 = 1e4.

        Parameters:
            value: Value to be converted.
            units: Units of ``value`` defined as a string in
                the form of ``unit1 * unit2 * unit3^-1``, e.g., ``"Pa*m^3/kg"``.

                Valid units are the attributes and properties of the :attr:`units`
                defined at instantiation.
                Validoperators are * and ^, including negative powers (e.g. m^-2). A
                dimensionless value can be specified by setting units to "", "1" or "-".
            to_si: ``default=False``

                If True, the value is converted from given ``units`` to SI units.
                If False, the value is assumed to be in SI units, andconverted to the
                :attr:`units` specified by the user during instantiation.

        Returns:
            Value in the user specified units to be used in the simulation.

        """
        # Make a copy of the value to avoid modifying the original.
        # This is not strictly necessary for scalars, but is done in case the method is
        # used for arrays.
        if isinstance(value, np.ndarray):
            value = value.copy()
        # Trim any spaces
        units = units.replace(" ", "")
        if units in ["", "1", "-"]:
            return value
        # Traverse string specifying units, and convert to SI units
        # The string is traversed by first splitting at *.
        # If the substring contains a ^, the substring is split again, and the first
        # element is raised to the power of the second.
        for sub_unit in units.split("*"):
            if "^" in sub_unit:
                sub_unit, power = sub_unit.split("^")
                factor = getattr(self.units, sub_unit) ** float(power)
            else:
                factor = getattr(self.units, sub_unit)
            if to_si:
                value *= factor
            else:
                value /= factor
        return value


@dataclass(frozen=True, kw_only=True)
class FluidConstants(MaterialConstants):
    """Material data class for fluid species.

    It declares fluid species parameters, which are expected of a fluid by the remaining
    framework, especially flow & transport equations.

    """

    SI_units: ClassVar[_HashableDict[str, str]] = _HashableDict(
        {
            "density": "kg * m^-3",
            "molar_mass": "kg * mol^-1",
            "critical_pressure": "Pa",
            "critical_temperature": "K",
            "critical_specific_volume": "m^3 * kg^-1",
            "pressure": "Pa",
            "temperature": "K",
            "compressibility": "Pa^-1",
            "specific_heat_capacity": "J * kg^-1 * K^-1",
            "thermal_expansion": "K^-1",
            "viscosity": "Pa * s",
            "thermal_conductivity": "W * m^-1 * K^-1",
            "normal_thermal_conductivity": "W * m^-1 * K^-1",
        }
    )

    molar_mass: number = 1

    critical_pressure: number = 1

    critical_temperature: number = 1

    critical_specific_volume: number = 1

    density: number = 1

    pressure: number = 0

    temperature: number = 0

    compressibility: number = 0

    specific_heat_capacity: number = 1

    thermal_expansion: number = 0

    viscosity: number = 1

    thermal_conductivity: number = 1

    normal_thermal_conductivity: number = 1


@dataclass(frozen=True, kw_only=True)
class SolidConstants(MaterialConstants):
    """Material data class for solid species present in the porous medium.

    It declares solid species parameters, which are expected of a solid by the remaining
    framework, especially poro- & fracture-mechanics.

    """

    SI_units: ClassVar[_HashableDict[str, str]] = _HashableDict(
        {
            "density": "kg * m^-3",
            "biot_coefficient": "-",
            "characteristic_displacement": "m",
            "characteristic_contact_traction": "Pa",
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
            "temperature": "K",
            "thermal_conductivity": "W * m^-1 * K^-1",
            "thermal_expansion": "K^-1",
            "well_radius": "m",
            "open_state_tolerance": "-",
            "contact_mechanics_scaling": "-",
        }
    )

    density: number = 1

    biot_coefficient: number = 1

    characteristic_displacement: number = 1

    characteristic_contact_traction: number = 1

    dilation_angle: number = 0

    fracture_gap: number = 0

    fracture_normal_stiffness: number = 1
    """Intended use is in Barton-Bandis-type models for elastic fracture deformation."""

    fracture_tangential_stiffness: number = -1.0
    """
    Note:
        The current default value is -1.0, with the convention that negative
        values correspond to a fracture that does not deform elastically in the
        tangential direction.
    """

    friction_coefficient: number = 1

    lame_lambda: number = 1
    """Lame's first parameter"""

    maximum_elastic_fracture_opening: number = 0
    """Intended use is in Barton-Bandis-type models for elastic fracture deformation."""

    normal_permeability: number = 1

    permeability: number = 1

    porosity: number = 0.1

    residual_aperture: number = 0.1

    shear_modulus: number = 1

    skin_factor: number = 0

    specific_heat_capacity: number = 1

    specific_storage: number = 1

    temperature: number = 0

    thermal_conductivity: number = 1

    thermal_expansion: number = 0

    well_radius: number = 0.1

    open_state_tolerance: number = 1e-5
    """Numerical method parameter.

    Tolerance parameter for the tangential characteristic contact mechanics.

    FIXME: Revisit the tolerance.

    """

    contact_mechanics_scaling: number = 1e-1
    """Numerical method parameter

    Safety scaling factor, making fractures softer than the matrix

    """
