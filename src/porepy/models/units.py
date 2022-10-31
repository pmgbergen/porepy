import warnings

import porepy as pp
import porepy.number as number


class Units:
    """Units for material properties.

    This is a sketch of a class for scaling material properties. The idea is that the
    material properties should be stored in SI units, but that the user may want to
    specify them in other units. These are defined in init.
    Example:
        Running a simulation in km, days and  MPa is achieved by setting
        my_material = Units(m=1e3, s=86400, Pa=1e6)

    Base units are attributes of the class, and can be accessed as e.g. my_material.length
    Derived units are properties computed from the base units, and can be accessed as e.g.
    my_material.Pa. This ensures consistency between the base and derived units while allowing
    reference to derived units in usage of the class.

    TODO: Consider whether this needs to be incorporated in TimeStepManager.

    """

    m: number = 1 * pp.METER
    """Length unit, defaults to 1 m."""
    s: number = 1 * pp.SECOND
    """Time unit, defaults to 1 s."""
    kg: number = 1 * pp.KILOGRAM
    """Mass unit, defaults to 1 kg."""
    K: number = 1 * pp.KELVIN
    """Temperature unit, defaults to 1 K."""
    mol: number = 1
    """Mole unit, defaults to 1 mol."""

    def __init__(
        self,
        m: number = 1,
        s: number = 1,
        kg: number = 1,
        K: number = 1,
        mol: number = 1,
    ):
        """Initialize the units.

        Parameters:
            kwargs (dict): Dictionary of units. The keys are the name of the unit, and the
                values are the scaling factor. For example, if the user wants to specify
                length in kilometers, time in hours and substance amount in millimolar, the
                dictionary should be
                    dict(m=1e3, s=3600, mol=1e-3)
                or, equivalently,
                    dict(m=pp.KILO * pp.METER, s=pp.HOUR, mol=pp.MILLI * pp.MOLE)
        """
        # Check that all units are numbers and assign them as attributes
        for unit in ["m", "s", "kg", "K", "mol"]:
            val = locals()[unit]
            if not isinstance(val, number):
                raise ValueError(
                    f"All units must be numbers. Parameter {unit} is {type(val)}"
                )
            if val <= 0:
                warnings.warn(
                    "Expected positive value for " + unit + ", got " + str(val)
                )
            setattr(self, unit, val)

    @property
    def Pa(self):
        """Pressure (or stress) unit, derived from kg, m and s."""
        return self.kg / (self.m * self.s**2)

    @property
    def J(self):
        """Energy unit, derived from m, kg and s."""
        return self.kg * self.m**2 / self.s**2

    @property
    def N(self):
        """Force unit, derived from m, kg and s."""
        return self.kg * self.m / self.s**2

    @property
    def W(self):
        """Power unit, derived from m, kg and s."""
        return self.kg * self.m**2 / self.s**3
