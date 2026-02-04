import numpy as np
import porepy as pp

import pytest

from porepy.applications.md_grids.model_geometries import (
    SubsurfaceCuboidDomain,
    TwoEllipticFractures3d,
    TwoWells3d,
)

class subsurface_domain_model(
    SubsurfaceCuboidDomain
):
    def __init__(self, params):
        self.params = params
        self.units = params.get("units", pp.Units)

def test_subsurface_set_domain():
    params = {
        "domain_sizes": np.array([10.0, 20.0, 30.0]),
        "units": pp.Units(m=1.0),
    }
    model = subsurface_domain_model(params)
    model.set_domain()
    box = model._domain.bounding_box

    assert box["xmin"] == 0.0
    assert box["xmax"] == 10.0
    assert box["ymin"] == 0.0
    assert box["ymax"] == 20.0
    assert box["zmin"] == -30.0
    assert box["zmax"] == 0.0


class TwoWells3d_model(
    TwoWells3d
):
    def __init__(self, params):
        self.params = params
        self.units = params.get("units", pp.Units)

def test_created_well_network():
    params = {
        "domain_sizes": np.array([10.0, 20.0, 30.0]),
        "units": pp.Units(m=1.0),
    }
    model = TwoWells3d_model(params)
    model.set_domain()
    model.set_well_network()
    wells = model.well_network.wells
    names = [well.tags["well_name"] for well in wells]

    assert len(wells) == 2
    assert names == ["injection_well", "production_well"]

class TwoEllipticFractures3d_model(
    TwoEllipticFractures3d
):
    def __init__(self, params):
        self.params = params
        self.units = params.get("units", pp.Units)

def test_created_well_network():
    params = {
        "domain_sizes": np.array([10.0, 20.0, 30.0]),
        "units": pp.Units(m=1.0),
    }
    model = TwoEllipticFractures3d_model(params)
    model.set_domain()
    model.set_fractures()
    fractures = model._fractures
    expected_major_axis = 0.2 * 10.0

    assert len(fractures) == 2
    assert np.allclose(model.fracture_major_axes, expected_major_axis)

