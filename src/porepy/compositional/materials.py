"""Storage classes for material constants.

Material contants are values representing either constant physical properties (e.g.
critical pressurre) or parameters for constitutive laws (e.g. constant compressibility
for exponential-type density law). A material is instantiated with a
:class:`~porepy.models.units.Units` object, which defines the units of the physical
properties for a simulation setup. While the constants must be given in base SI units
when instantiating a material class, its constants are converted and stored in the
target units, to be used subsequently.

For converting values on the fly, see :meth:`~porepy.models.units.Units.convert_units`.

"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Callable, ClassVar, Generic, TypeVar, cast

import porepy as pp

__all__ = [
    "MaterialConstants",
    "FluidConstants",
    "SolidConstants",
    "ReferenceValues",
    "load_fluid_constants",
]

number = pp.number

_K = TypeVar("_K")
_V = TypeVar("_V")


class _HashableDict(Generic[_K, _V], OrderedDict):
    """See https://stackoverflow.com/questions/1151658/python-hashable-dicts.

    We require hashable dictionaries for the below material constant classes which
    contain various constants and unit declarations in dicts, all in simple formats and
    per se hashable.

    The need for hashable material constants arises when fluid constants are used as a
    base class for components and tracers, which are dynamically created and themselves
    stored in standard dictionaries at various points in ``porepy.compositional``.

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

    The base class identifies a material using a given :attr:`name`. To define material
    properties of some kind, derive a dataclass with this class as its base (as frozen
    and keywords-only data class).

    Material constants are declared as fields (float or int) with default values.

    Derived classes must have a class attribute :attr:`SI_units` containing information
    about the the physical unit of each declared constant.

    If the user wants the material to be presented in other than SI units, a ``units=``
    kw-argument can be passed to declare the target, non-SI units (e.g. MPa instead of
    Pa).

    Important:
        When instantiating a material constants data class, the constants must all
        be strictly given in SI units. The conversion happens post-initialization for
        the subsequent simulation.

    Important:
        Every derived class must have a class attribute :attr:`SI_units`, **annotated as
        ClassVar**. This is to inform the base class about the used SI units.

        For examples, see :class:`FluidConstants` or :class:`SolidConstants`. For
        instructions on how to write composed units, see
        :meth:`~porepy.models.units.Units.convert_units`.

    """

    # NOTE Annotating it as a ClassVar leads to the dataclasss decorator ignoring this
    # in its machinery. The annotation must not be forgotten in derived classes.
    SI_units: ClassVar[_HashableDict[str, str]] = _HashableDict()
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

        # Safety check if the user forgot to annotate the SI_units as a class variable.
        # If yes, dataclass would declare it as a data field.
        if "SI_units" in constants:
            raise AttributeError(
                f"The attribute 'SI_units' in class {type(self)} must be annotated as"
                + " SI_units: typing.ClassVar[dict[str,str]]."
            )

        # By logic, what remains are constants as numbers storing original constants.
        self.constants_in_SI.update(
            cast(_HashableDict[str, number], _HashableDict(constants))
        )

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

    def __init_subclass__(cls) -> None:
        """If a data-class inherits this class, the base class post-initialization
        procedure must be inherited as well for the conversion from SI units (at
        instantiation) to simulation :attr:`units`."""

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

    def to_units(self: _MaterialConstants, units: pp.Units) -> _MaterialConstants:
        """Utility to quickly convert material constants to new units.

        Parameters:
            units: A new unit system. Note that ``units`` must cover all SI units
                declared in :attr:`SI_units` of the material constants (child) class.

        Returns:
            A new instance of of the data class using this method, with parameters
            converted according to ``units``.

        """
        kwargs = dict(self.constants_in_SI)
        kwargs["name"] = self.name
        kwargs["units"] = units
        return type(self)(**kwargs)


_MaterialConstants = TypeVar("_MaterialConstants", bound=MaterialConstants)


@dataclass(frozen=True, kw_only=True)
class FluidConstants(MaterialConstants):
    """Material data class for fluid species.

    It declares fluid species parameters, which are expected of a fluid by the remaining
    framework, especially flow and transport equations.

    This class is intended for 1 fluid species only. Fluid mixtures with multiple
    components require multiple sets of constants to define individual fluid components.

    """

    SI_units: ClassVar[_HashableDict[str, str]] = _HashableDict(
        {
            "density": "kg * m^-3",
            "molar_mass": "kg * mol^-1",
            "critical_pressure": "Pa",
            "critical_temperature": "K",
            "critical_specific_volume": "m^3 * kg^-1",
            "acentric_factor": "-",
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

    # NOTE due to legacy code (Only 1-phase 1-component fluid), the fluid constants have a
    # pressure and temperature property, which is not a material property. This will likely be
    # refactored in the future, but is left here for now. See GH issue 1244 and respective
    # comments.
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


@dataclass(frozen=True, kw_only=True)
class SolidConstants(MaterialConstants):
    """Material data class for solid species present in the porous medium.

    It declares solid species parameters, which are expected of a solid by the remaining
    framework, especially poro- & fracture-mechanics.

    This class is meant for 1 solid species only. Different domains in the mD-setting
    require each their own material constants instance in the case of heterogenity.

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

    biot_coefficient: number = 1.0

    characteristic_contact_traction: number = 1.0

    characteristic_displacement: number = 1.0

    contact_mechanics_scaling: number = 1e-1
    """Numerical method parameter

    Safety scaling factor, making fractures softer than the matrix

    """

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

    open_state_tolerance: number = 1e-5
    """Numerical method parameter.

    Tolerance parameter for the tangential characteristic contact mechanics.

    FIXME: Revisit the tolerance.

    """

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


@dataclass(frozen=True, kw_only=True)
class ReferenceValues(MaterialConstants):
    """A data class storing reference values for a model.

    Intended use is for defining for example reference pressure and temperature,
    where the perturbation from the reference value :math:`(p - p_{ref})` is used in
    constitutive laws for example.

    The aim is to have a single set of (scalar) reference values, which is to be used
    consistently in the whole model, with the correct units.

    """

    SI_units: ClassVar[_HashableDict[str, str]] = _HashableDict(
        {
            "pressure": "Pa",
            "temperature": "K",
        }
    )

    pressure: number = 0.0

    temperature: number = 0.0


def load_fluid_constants(names: list[str], package: str) -> list[FluidConstants]:
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

    species: list[FluidConstants] = []

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
            FluidConstants(
                name=name,
                molar_mass=mm,
                critical_pressure=float(pc_loader(cas)),
                critical_temperature=float(Tc_loader(cas)),
                critical_specific_volume=v_crit,
                acentric_factor=float(omega_loader(cas)),
            )
        )

    return species
