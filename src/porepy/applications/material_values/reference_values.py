"""This file contains reference values for fluid and solid material constants
implemented in :mod:`~porepy.application.material_values`.

They can be used by adding ``params['reference_values'] = porepy.ReferenceValues(...)``
to a model's parameters at instantiation.

"""

extended_reference_values_for_testing = {
    "pressure": 101325.0,  # [Pa]
    "temperature": 293.15,  # [K]
}
