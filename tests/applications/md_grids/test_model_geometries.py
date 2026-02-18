"""
Testing the functionality related to model geometries. There functions are covered:
- SubsurfaceCuboidDomain
- TwoWells3d
- TwoEllipticFractures3d

"""

import numpy as np

import porepy as pp
from porepy.applications.md_grids.model_geometries import (
    SubsurfaceCuboidDomain,
    TwoEllipticFractures3d,
    TwoWells3d,
)


class SubsurfaceDomainModel(SubsurfaceCuboidDomain):
    def __init__(self, params):
        self.params = params
        self.units = params.get("units", pp.Units)


def test_subsurface_set_domain():
    """
    Check whether the domain in x, y, and z directions are correctly created from
    the given domain size.

    """
    params = {
        "domain_sizes": np.array([10.0, 20.0, 30.0]),
        "units": pp.Units(m=1.0),
    }
    model = SubsurfaceDomainModel(params)
    model.set_domain()
    box = model._domain.bounding_box

    assert box["xmin"] == 0.0
    assert box["xmax"] == 10.0
    assert box["ymin"] == 0.0
    assert box["ymax"] == 20.0
    assert box["zmin"] == -30.0
    assert box["zmax"] == 0.0


class TwoWells3dModel(TwoWells3d):
    def __init__(self, params):
        self.params = params
        self.units = params.get("units", pp.Units)


def test_created_well_network():
    """
    Check whether the expected well network for the model are correctly created. This
    test specifically checks that two wells are exactly created and assigned with the
    corresponding well names.

    """
    params = {
        "domain_sizes": np.array([10.0, 20.0, 30.0]),
        "units": pp.Units(m=1.0),
    }
    model = TwoWells3dModel(params)
    model.set_domain()
    model.set_well_network()
    wells = model.well_network.wells
    names = [well.tags["well_name"] for well in wells]

    assert len(wells) == 2
    assert names == ["injection_well", "production_well"]


class TwoEllipticFractures3dModel(TwoEllipticFractures3d):
    def __init__(self, params):
        self.params = params
        self.units = params.get("units", pp.Units)


def test_created_elliptic_fracs():
    """
    Check whether elliptic fractures are correctly created and parameterized.
    This test specifically check that two elliptic fractures are exactly created
    with expected major axis length.

    """
    params = {
        "domain_sizes": np.array([10.0, 20.0, 30.0]),
        "units": pp.Units(m=1.0),
    }
    model = TwoEllipticFractures3dModel(params)
    model.set_domain()
    model.set_fractures()
    fractures = model._fractures
    expected_major_axis = 0.2 * 10.0

    assert len(fractures) == 2
    assert np.allclose(model.fracture_major_axes, expected_major_axis)
