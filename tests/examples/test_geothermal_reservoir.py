"""
Test functionalities in the example case of the geomthermal reservoir.

"""

import numpy as np
import porepy as pp

import pytest
from porepy.examples.geothermal_reservoir import (
    GeothermalReservoirWellBCs,
)

@pytest.fixture
def geothermal_model():

    model = GeothermalReservoirWellBCs()
    model.prepare_simulation()

    return model

@pytest.fixture
def well_subdomains(geothermal_model): 
    model = geothermal_model
    wells = [
        sd for sd in model.mdg.subdomains()
        if "parent_well_index" in sd.tags
    ]
    assert len(wells) > 0
    return wells

def test_NeumannWellBCs_in_FirstTimeInterval(
    geothermal_model, well_subdomains
): 
    """
    Test that well grids have Neumann BCs during the first time interval.
    """
    model = geothermal_model
    model.time_manager.time = model.time_manager.schedule[0]

    for sd in well_subdomains:
        bc = model.bc_type_darcy_flux(sd)
        assert not np.any(bc.is_dir)

