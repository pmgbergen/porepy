"""
Test functionalities in the example case of the geomthermal reservoir.

"""

import numpy as np
import porepy as pp

import pytest
from porepy.examples.geothermal_reservoir import (
    GeothermalReservoirWellBCs,
    NeumannWellBCsFirstTimeInterval,
    WellBoundaryConditions,
    BoundaryConditionsMechanicsNeumann, 
)
import porepy.applications.md_grids.model_geometries
from porepy.applications.test_utils import well_models

class geothermal_model_neu(
    well_models.OneVerticalWell,
    porepy.applications.md_grids.model_geometries.OrthogonalFractures3d,
    NeumannWellBCsFirstTimeInterval,
    pp.Poromechanics,
):
    pass

@pytest.fixture
def neuBC_model(): 
    model = geothermal_model_neu()
    model.prepare_simulation()
    return model

@pytest.fixture
def well_subdomains(neuBC_model): 
    model = neuBC_model
    wells = [sd for sd in model.mdg.subdomains() if model.is_well_grid(sd)]
 
    assert len(wells) > 0
    return wells

def test_NeumannWellBCs_in_FirstTimeInterval(neuBC_model, well_subdomains): 
    """
    Test that well grids have Neumann BCs during the first time interval.
    """
    model = neuBC_model
    model.time_manager.time = model.time_manager.schedule[0]
    for sd in well_subdomains:
        bc = model.bc_type_darcy_flux(sd)
        assert not np.any(bc.is_dir)

class OneVerticalInjectionWell(well_models.OneVerticalWell): 
    def set_well_network(self) -> None:
        super().set_well_network()
        self.well_network.wells[0].tags["well_name"] = "injection_well"

class geomhermal_model_well(
    OneVerticalInjectionWell,
    porepy.applications.md_grids.model_geometries.OrthogonalFractures3d,
    WellBoundaryConditions,
    pp.Thermoporomechanics,
):
    @property
    def well_names(self):
        return["injection_well"]

@pytest.fixture
def well_bc_model(): 
    params = {
        "injection_well_pressures": [1e6, 1e6],
        "injection_well_temperatures": [300.00, 300.00],
    }
    model = geomhermal_model_well(params)
    model.prepare_simulation()
    return model

def test_well_bcs_pressure(well_bc_model):
    """
    Test the boundary conditions of one well for pressure.
    """
    model = well_bc_model
    model.time_manager.time = model.time_manager.schedule[0]
    wells = [sd for sd in model.mdg.subdomains() if model.is_well_grid(sd)]
    assert len(wells) == 1

    expected_value = model.units.convert_units(1e6, "Pa")

    bg = model.mdg.subdomain_to_boundary_grid(wells[0])
    values = model.bc_values_pressure(bg)
        
    assert np.any(np.isclose(values, expected_value))\
    
def test_well_bcs_temperature(well_bc_model):
    """
    Test the boundary conditions of one well for temperature.
    """
    model = well_bc_model
    model.time_manager.time = model.time_manager.schedule[0]
    wells = [sd for sd in model.mdg.subdomains() if model.is_well_grid(sd)]
    assert len(wells) == 1

    expected_value = model.units.convert_units(300.00, "K")

    bg = model.mdg.subdomain_to_boundary_grid(wells[0])
    values = model.bc_values_temperature(bg)
        
    assert np.any(np.isclose(values, expected_value))


class geothermal_model_mechanics(
    well_models.OneVerticalWell,
    porepy.applications.md_grids.model_geometries.OrthogonalFractures3d,
    BoundaryConditionsMechanicsNeumann,
    pp.Poromechanics,
):
    pass

@pytest.fixture
def mechcanics_bc_model(): 
    model = geothermal_model_mechanics()
    model.prepare_simulation()
    return model

def test_mechanics_bcs_neumann(mechcanics_bc_model):
    """
    Test the boundary conditions of mechanics.
    """
    model = mechcanics_bc_model
    matrix_grids = [sd for sd in model.mdg.subdomains() if sd.dim == model.nd]
    assert len(matrix_grids) == 1
    
    sd = matrix_grids[0]
    bc = model.bc_type_mechanics(sd)

    faces = model.faces_to_fix(sd)

    expected_dir = [
        np.array([False, True, True]),  
        np.array([False, True, True]),  
        np.array([True, False, True]),  
    ]

    assert len(faces) == 3
    assert np.any(bc.is_dir)
    assert not np.all(bc.is_dir)

    for i, value in zip(faces, expected_dir):
        assert np.array_equal(bc.is_dir[:, i], value)
        assert np.array_equal(bc.is_neu[:, i], ~value)





