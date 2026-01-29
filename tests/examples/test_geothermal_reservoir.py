"""
Test functionalities in the example case of the geomthermal reservoir.

"""

import numpy as np
import porepy as pp

import pytest
from porepy.examples.geothermal_reservoir import (
    GeothermalReservoirWellBCs,
    NeumannWellBCsFirstTimeInterval, 
)
import porepy.applications.md_grids.model_geometries
from porepy.applications.test_utils import models, well_models

class geothermal_model(
    well_models.OneVerticalWell,
    porepy.applications.md_grids.model_geometries.OrthogonalFractures3d,
    NeumannWellBCsFirstTimeInterval,
    pp.Poromechanics,
):
    pass

model = geothermal_model()
model.prepare_simulation()

def well_subdomains(): 
    wells = [sd for sd in model.mdg.subdomains() if model.is_well_grid(sd)]
 
    assert len(wells) > 0
    return wells

def test_NeumannWellBCs_in_FirstTimeInterval(): 
    """
    Test that well grids have Neumann BCs during the first time interval.
    """
    model.time_manager.time = model.time_manager.schedule[0]
    for sd in well_subdomains():
        bc = model.bc_type_darcy_flux(sd)
        assert not np.any(bc.is_dir)


