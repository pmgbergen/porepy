import numpy as np

import porepy as pp

number = pp.number


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
    rad: number = 1
    """Angle unit, defaults to 1 rad."""

    def __init__(
        self,
        **kwargs: number,
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
        # Sanity check on input values
        for key, value in kwargs.items():
            if not isinstance(value, number):
                raise ValueError("Input values must be of type number.")
            if key not in ["m", "s", "kg", "K", "mol", "rad"]:
                # If for some reason the user wants to change the base units, this can be
                # done by assigning to the attributes directly or overwriting the __init__.
                # Since we do not recommend this, we do not allow it here.
                raise ValueError("Input keys must be valid base units.")

        # Set known base units
        self.m: number = kwargs.get("m", 1)
        """Length unit in meters."""
        self.s: number = kwargs.get("s", 1)
        """Time unit in seconds."""
        self.kg: number = kwargs.get("kg", 1)
        """Mass unit in kilograms."""
        self.K: number = kwargs.get("K", 1)
        """Temperature unit in Kelvin."""
        self.mol: number = kwargs.get("mol", 1)
        """Mole unit in moles."""
        self.rad: number = kwargs.get("rad", 1)
        """Angle unit in radians."""

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

    @property
    def Celsius(self):
        """Temperature unit, derived from K."""
        return self.K - 273.15

    @property
    def degree(self):
        """Angle unit, derived from rad."""
        return self.rad * 180 / np.PI
