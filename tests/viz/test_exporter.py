"""Tests of the export functionalities of Exporter, FractureNetwork2d, and
FractureNetwork3d.

The tests focus on the write-to-file capabilities of the Exporter. It is tested for
various sorts of relevant meshes in 1d, 2d, and 3d, including single domain as well as
mixed-dimensional domains. In addition, the export capability of  2d and 3d fracture
networks is tested. All tests have a similar character and are based on a simple
comparison with reference vtu files. It should be noted that failure of any test
indicates that something in the export filter, or in the vtk python bindings has
changed. If the change is external to PorePy, this does not necessarily mean that
something is wrong.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import porepy as pp
from porepy.applications.test_utils.vtk import (
    PathLike,
    compare_pvd_files,
    compare_vtu_files,
)
from porepy.fracs.utils import pts_edges_to_linefractures
from tests.models.test_poromechanics import NonzeroFractureGapPoromechanics
from porepy.applications.test_utils.models import Thermoporomechanics

# Globally store location of reference files
FOLDER_REFERENCE = (
    os.path.dirname(os.path.realpath(__file__)) + "/" + "test_vtk_reference"
)


@dataclass
class ExporterTestSetup:
    """Class to define where to store vtu files, and test the export functionality of
    the Exporter, FractureNetwork2d, and FractureNetwork3d.

    """

    folder = "./test_vtk"
    file_name = "grid"
    folder_reference = FOLDER_REFERENCE


@dataclass
class SingleSubdomain:
    """Data type for a single subdomain."""

    grid: pp.Grid
    ref_vtu_file: PathLike


@pytest.fixture
def setup() -> ExporterTestSetup:
    """Method to deliver a setup to all tests, and remove any temporary directory."""

    # Setup
    setup = ExporterTestSetup()
    yield setup

    # Teardown: remove temporary directory for vtu files.
    full_path = Path.cwd() / Path.resolve(Path(setup.folder)).name
    shutil.rmtree(full_path)


@pytest.fixture(params=range(7))
def subdomain(request: pytest.FixtureRequest) -> SingleSubdomain:
    """Helper for parametrization of test_single_subdomains.

    Define collection of single subdomains incl. 1d, 2d, 3d grids, of simplicial,
    Cartesian and polytopal element type.

    """

    # Construct 2d polytopal grid
    sd_polytop_2d = pp.StructuredTriangleGrid([2] * 2, [1] * 2)
    sd_polytop_2d.compute_geometry()
    pp.coarsening.generate_coarse_grid(sd_polytop_2d, [0, 1, 3, 3, 1, 1, 2, 2])

    # Construct 3d polytopal grid
    sd_polytop_3d = pp.CartGrid([3, 2, 3], [1] * 3)
    sd_polytop_3d.compute_geometry()
    pp.coarsening.generate_coarse_grid(
        sd_polytop_3d, [0, 0, 1, 0, 1, 1, 0, 2, 2, 3, 2, 2, 4, 4, 4, 4, 4, 4]
    )

    # Define the collection of subdomains
    subdomains: list[SingleSubdomain] = [
        # 1d grid
        SingleSubdomain(
            pp.CartGrid(3, 1),
            f"{FOLDER_REFERENCE}/single_subdomain_1d.vtu",
        ),
        # 2d simplex grid
        SingleSubdomain(
            pp.StructuredTriangleGrid([3] * 2, [1] * 2),
            f"{FOLDER_REFERENCE}/single_subdomain_2d_simplex_grid.vtu",
        ),
        # 2d Cartesian grid
        SingleSubdomain(
            pp.CartGrid([4] * 2, [1] * 2),
            f"{FOLDER_REFERENCE}/single_subdomain_2d_cart_grid.vtu",
        ),
        # 2d polytopal grid
        SingleSubdomain(
            sd_polytop_2d,
            f"{FOLDER_REFERENCE}/single_subdomain_2d_polytop_grid.vtu",
        ),
        # 3d simplex grid
        SingleSubdomain(
            pp.StructuredTetrahedralGrid([3] * 3, [1] * 3),
            f"{FOLDER_REFERENCE}/single_subdomain_3d_simplex_grid.vtu",
        ),
        # 3d Cartesian grid
        SingleSubdomain(
            pp.CartGrid([4] * 3, [1] * 3),
            f"{FOLDER_REFERENCE}/single_subdomain_3d_cart_grid.vtu",
        ),
        # 3d polytopal grid
        SingleSubdomain(
            sd_polytop_3d,
            f"{FOLDER_REFERENCE}/single_subdomain_3d_polytop_grid.vtu",
        ),
    ]
    subdomain = subdomains[request.param]
    subdomain.grid.compute_geometry()
    return subdomain


def test_single_subdomains(setup: ExporterTestSetup, subdomain: SingleSubdomain):
    """Test of the Exporter for single subdomains of different dimensionality and
    different grid type. Exporting of scalar and vectorial data is tested.
    """
    # Define grid
    sd = subdomain.grid
    # Define data
    dummy_scalar = np.ones(sd.num_cells) * sd.dim
    dummy_scalar_pt = np.ones(sd.num_nodes) * sd.dim
    dummy_vector = np.ones((3, sd.num_cells)) * sd.dim
    dummy_vector_pt = np.ones((3, sd.num_nodes)) * sd.dim
    # Export data
    save = pp.Exporter(
        sd,
        setup.file_name,
        setup.folder,
        export_constants_separately=False,
    )
    save.write_vtu(
        [("dummy_scalar", dummy_scalar), ("dummy_vector", dummy_vector)],
        data_pt=[
            ("dummy_scalar_pt", dummy_scalar_pt),
            ("dummy_vector_pt", dummy_vector_pt),
        ],
    )
    # Check that exported vtu file and reference file are the same
    assert compare_vtu_files(
        f"{setup.folder}/{setup.file_name}_{sd.dim}.vtu",
        f"{subdomain.ref_vtu_file}",
    )


def test_import_state_from_vtu_single_subdomains(
    setup: ExporterTestSetup, subdomain: SingleSubdomain
):
    # Test of the import routine of the Exporter for single subdomains. Consistent
    # with test_single_subdomains.

    # Define grid
    sd = subdomain.grid

    # Define exporter
    save = pp.Exporter(
        sd,
        setup.file_name,
        setup.folder,
        export_constants_separately=False,
    )

    # Define keys (here corresponding to all data stored in the vtu file to pass the
    # test).
    keys = ["dummy_scalar", "dummy_vector"]
    keys_pt = ["dummy_scalar_pt", "dummy_vector_pt"]

    # Import data
    save.import_state_from_vtu(
        vtu_files=f"{subdomain.ref_vtu_file}",
        keys=keys,
        keys_pt=keys_pt,
        automatic=False,
        dims=sd.dim,
    )

    # Perform comparison on vtu level (seems the easiest as it only involves a
    # comparison of dictionaries). This requires test_single_subdomains to pass all
    # tests.
    save.write_vtu(keys, data_pt=keys_pt)

    # Check that exported vtu file and reference file are the same
    assert compare_vtu_files(
        f"{setup.folder}/{setup.file_name}_{sd.dim}.vtu",
        f"{subdomain.ref_vtu_file}",
    )


def test_mdg(setup: ExporterTestSetup):
    """Test Exporter for 2d mixed-dimensional grids for a two-fracture domain.

    Exporting of scalar and vectorial data, separately defined on both subdomains and
    interfaces.

    """

    # Define grid
    mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "cartesian",
        meshing_args={"cell_size": 0.25},
        fracture_indices=[0, 1],
        fracture_endpoints=[np.array([0.25, 0.75]), np.array([0, 1])],
    )

    # Define data
    for sd, sd_data in mdg.subdomains(return_data=True):
        pp.set_solution_values(
            name="dummy_scalar",
            values=np.ones(sd.num_cells) * sd.dim,
            data=sd_data,
            time_step_index=0,
        )
        pp.set_solution_values(
            name="dummy_scalar_pt",
            values=np.ones(sd.num_nodes) * sd.dim,
            data=sd_data,
            time_step_index=0,
        )

        pp.set_solution_values(
            name="dummy_vector",
            values=np.ones((3, sd.num_cells)) * sd.dim,
            data=sd_data,
            time_step_index=0,
        )
        pp.set_solution_values(
            name="dummy_vector_pt",
            values=np.ones((3, sd.num_nodes)) * sd.dim,
            data=sd_data,
            time_step_index=0,
        )

    for intf, intf_data in mdg.interfaces(return_data=True):
        pp.set_solution_values(
            name="dummy_scalar",
            values=np.zeros(intf.num_cells),
            data=intf_data,
            time_step_index=0,
        )
        pp.set_solution_values(
            name="dummy_scalar_pt",
            values=np.zeros(intf.num_nodes),
            data=intf_data,
            time_step_index=0,
        )

        pp.set_solution_values(
            name="dummy_vector",
            values=np.ones((3, intf.num_cells)) * sd.dim,
            data=intf_data,
            time_step_index=0,
        )
        pp.set_solution_values(
            name="dummy_vector_pt",
            values=np.ones((3, intf.num_nodes)) * sd.dim,
            data=intf_data,
            time_step_index=0,
        )

        pp.set_solution_values(
            name="unique_dummy_scalar",
            values=np.zeros(intf.num_cells),
            data=intf_data,
            time_step_index=0,
        )
        pp.set_solution_values(
            name="unique_dummy_scalar_pt",
            values=np.zeros(intf.num_nodes),
            data=intf_data,
            time_step_index=0,
        )

    # Export data
    save = pp.Exporter(
        mdg,
        setup.file_name,
        setup.folder,
        export_constants_separately=False,
    )
    save.write_vtu(
        ["dummy_scalar", "dummy_vector", "unique_dummy_scalar"],
        data_pt=["dummy_scalar_pt", "dummy_vector_pt", "unique_dummy_scalar_pt"],
    )
    # Check that exported vtu files and reference files are the same.
    for appendix in ["1", "2", "mortar_1"]:
        assert compare_vtu_files(
            f"{setup.folder}/{setup.file_name}_{appendix}.vtu",
            f"{setup.folder_reference}/mdg_grid_{appendix}.vtu",
        )


@pytest.mark.parametrize("case", range(2))
def test_import_from_pvd_mdg(setup: ExporterTestSetup, case: int):
    """Test import-from-pvd functionality of the Exporter for 2d mixed-dimensional grids
    for a two-fracture domain.

    Here, purely reading functionality is tested, given a fixed set of input pvd files.

    Two cases are considered, testing two functionalities: importing from a mdg pvd
    file and a (conventional) pvd file, originating from pp.Exporter._export_mdg_pvd()
    and pp.Exporter.write_pvd(). These correspond to case equal 1 and 0, respectively.

    Exporting of scalar and vectorial data, separately defined on both subdomains and
    interfaces.

    """

    # Define grid
    mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "cartesian",
        meshing_args={"cell_size": 0.25},
        fracture_indices=[0, 1],
        fracture_endpoints=[np.array([0.25, 0.75]), np.array([0, 1])],
    )

    # Define exporter
    save = pp.Exporter(
        mdg,
        setup.file_name,
        setup.folder,
        export_constants_separately=False,
    )

    # Assume the following has been run for a previous simulation
    # save.write_vtu(
    #     ["dummy_scalar", "dummy_vector", "unique_dummy_scalar"],
    #     timestep=1
    # )
    # Yet, then the simulation crashed, now it is restarted from pvd file, picking up
    # the latest available timestep.
    pvd_file = Path(f"{setup.folder_reference}/restart/previous_grid.pvd")
    if case == 0:
        # Test restart from conventional pvd file.
        time_index = save.import_from_pvd(
            pvd_file=pvd_file,
            keys=["dummy_scalar", "dummy_vector", "unique_dummy_scalar"],
        )

    elif case == 1:
        # Test restart from
        time_index = save.import_from_pvd(
            pvd_file=Path(f"{setup.folder_reference}/restart/grid_000001.pvd"),
            is_mdg_pvd=True,
            keys=["dummy_scalar", "dummy_vector", "unique_dummy_scalar"],
        )

    # The above setup has been created such that the imported data corresponds to some
    # first time step. In case 0, this is encoded in the content of previous_grid.pvd.
    # In case 1, this is encoded in the appendix of grid_000001.pvd. Test whether the
    # time index has been identified correctly.
    assert time_index == 1

    # To trick the test, copy the current pvd file to the temporary folder before
    # continuing writing it through appending the next time step.
    Path(f"{setup.folder}").mkdir(parents=True, exist_ok=True)
    shutil.copy(pvd_file, Path(f"{setup.folder}/{setup.file_name}.pvd"))

    # Imitate the initialization of a simulation, i.e., export the initial condition.
    save.write_vtu(["dummy_scalar", "dummy_vector", "unique_dummy_scalar"], time_step=1)
    save.write_pvd(append=True)

    # Now, export both the vtu and the pvd (continuing using the previous one).
    # NOTE: Typically, the data would be modified by running the simulation
    # for one more timestep. This is irrelevant for testing the restarting capabilities.
    save.write_vtu(["dummy_scalar", "dummy_vector", "unique_dummy_scalar"], time_step=2)
    save.write_pvd(append=True)

    # Check that newly exported vtu files and reference files are the same.
    for appendix in ["1", "2", "mortar_1"]:
        assert compare_vtu_files(
            f"{setup.folder}/{setup.file_name}_{appendix}_000002.vtu",
            f"{setup.folder_reference}/restart/grid_{appendix}_000002.vtu",
        )

    # Check that the newly exported pvd files and reference file are the same.
    for appendix in ["_000002", ""]:
        assert compare_pvd_files(
            f"{setup.folder}/{setup.file_name}{appendix}.pvd",
            f"{setup.folder_reference}/restart/grid{appendix}.pvd",
        )


@pytest.mark.parametrize("addendum", ["", "nontrivial_data_"])
def test_import_state_from_vtu_mdg(setup: ExporterTestSetup, addendum: str):
    # Test of the import routine of the Exporter for 2d mixed-dimensional grids.
    # Consistent with test_mdg.
    # NOTE: In case the reference files for this test needs to be updated, the
    # references for the two addenda ("" and "nontrivial_data_") can be identical,
    # though they of course need to be separate files.
    # Define grid
    mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "cartesian",
        meshing_args={"cell_size": 0.25},
        fracture_indices=[0, 1],
        fracture_endpoints=[np.array([0.25, 0.75]), np.array([0, 1])],
    )
    # Define exporter
    save = pp.Exporter(
        mdg,
        setup.file_name,
        setup.folder,
        export_constants_separately=False,
    )
    # Define keys (here corresponding to all data stored in the vtu file to pass the
    # test).
    keys = ["dummy_scalar", "dummy_vector", "unique_dummy_scalar"]
    keys_pt = ["dummy_scalar_pt", "dummy_vector_pt", "unique_dummy_scalar_pt"]
    # Import data
    save.import_state_from_vtu(
        vtu_files=[
            Path(f"{setup.folder_reference}/mdg_{addendum}grid_2.vtu"),
            Path(f"{setup.folder_reference}/mdg_{addendum}grid_1.vtu"),
            Path(f"{setup.folder_reference}/mdg_{addendum}grid_mortar_1.vtu"),
        ],
        keys=keys,
        keys_pt=keys_pt,
    )

    # Perform comparison on vtu level (seems the easiest as it only involves a
    # comparison of dictionaries). This requires test_mdg to pass all tests.
    save.write_vtu(keys, data_pt=keys_pt)
    # Check that exported vtu files and reference files are the same.
    for appendix in ["1", "2", "mortar_1"]:
        assert compare_vtu_files(
            f"{setup.folder}/{setup.file_name}_{appendix}.vtu",
            f"{setup.folder_reference}/mdg_{addendum}grid_{appendix}.vtu",
        )


def test_mdg_data_selection(setup: ExporterTestSetup):
    """Test Exporter for 2d mixed-dimensional grids for a two-fracture domain.

    Exporting of scalar and vectorial data, separately defined on both subdomains and
    interfaces. Furthermore, the different possibilities of how to export data are
    tested: addressing selected data associated to all subdomains and interfaces, single
    ones, or defining external data (here simply cell centers).

    """

    # Define grid
    mdg, _ = pp.mdg_library.square_with_orthogonal_fractures(
        "cartesian",
        meshing_args={"cell_size": 0.25},
        fracture_indices=[0, 1],
        fracture_endpoints=[np.array([0.25, 0.75]), np.array([0, 1])],
    )

    # Define data
    for sd, sd_data in mdg.subdomains(return_data=True):
        pp.set_solution_values(
            name="dummy_scalar",
            values=np.ones(sd.num_cells) * sd.dim,
            data=sd_data,
            time_step_index=0,
        )
        pp.set_solution_values(
            name="dummy_scalar_pt",
            values=np.ones(sd.num_nodes) * sd.dim,
            data=sd_data,
            time_step_index=0,
        )

        pp.set_solution_values(
            name="dummy_vector",
            values=np.ones((3, sd.num_cells)) * sd.dim,
            data=sd_data,
            time_step_index=0,
        )
        pp.set_solution_values(
            name="dummy_vector_pt",
            values=np.ones((3, sd.num_nodes)) * sd.dim,
            data=sd_data,
            time_step_index=0,
        )

    for intf, intf_data in mdg.interfaces(return_data=True):
        pp.set_solution_values(
            name="dummy_scalar",
            values=np.zeros(intf.num_cells),
            data=intf_data,
            time_step_index=0,
        )
        pp.set_solution_values(
            name="dummy_scalar_pt",
            values=np.zeros(intf.num_nodes),
            data=intf_data,
            time_step_index=0,
        )

        pp.set_solution_values(
            name="unique_dummy_scalar",
            values=np.zeros(intf.num_cells),
            data=intf_data,
            time_step_index=0,
        )
        pp.set_solution_values(
            name="unique_dummy_scalar_pt",
            values=np.zeros(intf.num_nodes),
            data=intf_data,
            time_step_index=0,
        )

    # Fetch separate subdomains
    subdomains_1d = mdg.subdomains(dim=1)
    subdomains_2d = mdg.subdomains(dim=2)
    sd_2d = subdomains_2d[0]

    # Export data
    save = pp.Exporter(
        mdg,
        setup.file_name,
        setup.folder,
        export_constants_separately=False,
    )
    save.write_vtu(
        [
            (subdomains_1d, "dummy_scalar"),
            "dummy_vector",
            "unique_dummy_scalar",
            (sd_2d, "cc", sd_2d.cell_centers),
        ],
        data_pt=[
            (subdomains_1d, "dummy_scalar_pt"),
            "dummy_vector_pt",
            "unique_dummy_scalar_pt",
            (sd_2d, "nodes", sd_2d.nodes),
        ],
    )

    # Check that exported vtu files and reference files are the same.
    for appendix in ["1", "2", "mortar_1"]:
        assert compare_vtu_files(
            f"{setup.folder}/{setup.file_name}_{appendix}.vtu",
            f"{setup.folder_reference}/mdg_data_selection_grid_{appendix}.vtu",
        )


def test_constant_data(setup: ExporterTestSetup):
    """Test Exporter functionality to distinguish between constant and non-constant
    data during exporting.

    """

    # Define grid
    g = pp.StructuredTriangleGrid([3] * 2, [1] * 2)
    g.compute_geometry()

    # Define data
    dummy_scalar = np.ones(g.num_cells) * g.dim
    dummy_scalar_pt = np.ones(g.num_nodes) * g.dim

    dummy_vector = np.ones((3, g.num_cells)) * g.dim
    dummy_vector_pt = np.ones((3, g.num_nodes)) * g.dim

    # Export data
    save = pp.Exporter(
        g,
        setup.file_name,
        setup.folder,
    )
    # Add additional constant data (cell centers)
    save.add_constant_data([(g, "cc", g.cell_centers)], data_pt=[("x", g.nodes[0, :])])
    save.write_vtu(
        [("dummy_scalar", dummy_scalar), ("dummy_vector", dummy_vector)],
        data_pt=[
            ("dummy_scalar_pt", dummy_scalar_pt),
            ("dummy_vector_pt", dummy_vector_pt),
        ],
    )

    # Check that exported vtu files and reference files are the same
    for appendix in ["2", "constant_2"]:
        assert compare_vtu_files(
            f"{setup.folder}/{setup.file_name}_{appendix}.vtu",
            f"{setup.folder_reference}/constant_data_test_grid_{appendix}.vtu",
        )


def test_fracture_network_2d(setup: ExporterTestSetup):
    """Test of the export functionality of FractureNetwork2d."""

    # Define network
    p = np.array([[0, 2, 1, 2, 1], [0, 0, 0, 1, 2]])
    e = np.array([[0, 2, 3], [1, 3, 4]])
    fractures = pts_edges_to_linefractures(p, e)
    domain = pp.Domain({"xmin": -2, "xmax": 3, "ymin": -2, "ymax": 3})
    network_2d = pp.create_fracture_network(fractures, domain)

    # Define data
    dummy_scalar = np.ones(network_2d.num_frac())
    dummy_vector = np.ones((3, network_2d.num_frac()))
    data = {"dummy_scalar": dummy_scalar, "dummy_vector": dummy_vector}

    # Make directory if not existent
    if not os.path.exists(setup.folder):
        os.makedirs(setup.folder)

    # Export data
    network_2d.to_file(
        setup.folder + "/" + setup.file_name + ".vtu",
        data=data,
    )

    # Check that exported vtu file and reference file are the same.
    assert compare_vtu_files(
        f"{setup.folder}/{setup.file_name}.vtu",
        f"{setup.folder_reference}/fractures_2d.vtu",
    )


def test_fracture_network_3d(setup: ExporterTestSetup):
    """Test of the export functionality of FractureNetwork3d."""

    # Define network
    f_1 = pp.PlaneFracture(np.array([[0, 1, 2, 0], [0, 0, 1, 1], [0, 0, 1, 1]]))
    f_2 = pp.PlaneFracture(
        np.array([[0.5, 0.5, 0.5, 0.5], [-1, 2, 2, -1], [-1, -1, 2, 2]])
    )
    bbox = {"xmin": -2, "xmax": 3, "ymin": -2, "ymax": 3, "zmin": -3, "zmax": 3}
    network_3d = pp.create_fracture_network([f_1, f_2], domain=pp.Domain(bbox))

    # Define data
    num_frac = len(network_3d.fractures)
    dummy_scalar = [[1] for _ in range(num_frac)]
    dummy_vector = [[np.ones(3)] for _ in range(num_frac)]
    data = {"dummy_scalar": dummy_scalar, "dummy_vector": dummy_vector}

    # Make directory if not existent.
    if not os.path.exists(setup.folder):
        os.makedirs(setup.folder)

    # Export data
    network_3d.to_file(
        setup.folder + "/" + setup.file_name + ".vtu",
        data=data,
    )

    # Check that exported vtu file and reference file are the same.
    assert compare_vtu_files(
        f"{setup.folder}/{setup.file_name}.vtu",
        f"{setup.folder_reference}/fractures_3d.vtu",
    )


class TailoredThermoporomechanics(
    NonzeroFractureGapPoromechanics,
    pp.model_boundary_conditions.TimeDependentMechanicalBCsDirNorthSouth,
    pp.model_boundary_conditions.BoundaryConditionsEnergyDirNorthSouth,
    pp.model_boundary_conditions.BoundaryConditionsMassDirNorthSouth,
    Thermoporomechanics,
):
    def grid_type(self):
        return "cartesian"


def test_rescaled_export(setup: ExporterTestSetup):
    """The test exports the scaled and unscaled versions of the same simulation and
    checks whether the output is the same. The output of the scaled model should be
    rescaled back to the SI units.

    """

    def run_simulation_save_results(units: pp.Units, file_name: str):
        nontrivial_solid = {
            "biot_coefficient": 0.47,  # [-]
            "density": 2.6,  # [kg * m^-3]
            "friction_coefficient": 0.6,  # [-]
            "lame_lambda": 7.02,  # [Pa]
            "permeability": 0.5,  # [m^2]
            "porosity": 1.3e-1,  # [-]
            "shear_modulus": 1.4,  # [Pa]
            "specific_heat_capacity": 7.2,  # [J * kg^-1 * K^-1]
            "specific_storage": 4e-1,  # [Pa^-1]
            "thermal_conductivity": 3.1,  # [W * m^-1 * K^-1]
            "thermal_expansion": 9.6e-2,  # [K^-1]
            "temperature": 2,  # [K]
        }
        nontrivial_fluid = {
            "compressibility": 1e-1,  # [Pa^-1], isentropic compressibility
            "density": 9.2,  # [kg m^-3]
            "specific_heat_capacity": 4.2,  # [J kg^-1 K^-1], isochoric specific heat
            "thermal_conductivity": 0.5,  # [kg m^-3]
            "thermal_expansion": 2.068e-1,  # [K^-1]
            "viscosity": 1.002e-1,  # [Pa s], absolute viscosity
            "pressure": 1,  # [Pa]
            "temperature": 2,  # [K]
        }

        model_params = {
            "suppress_export": False,
            "units": units,
            "file_name": file_name,
            "folder_name": setup.folder,
            "material_constants": {
                "solid": pp.SolidConstants(nontrivial_solid),
                "fluid": pp.FluidConstants(nontrivial_fluid),
            },
        }
        model = TailoredThermoporomechanics(params=model_params)
        pp.run_time_dependent_model(model)

    units_scaled = pp.Units(m=3.14, kg=42.0, K=3.79)
    units_unscaled = pp.Units()
    scaled_prefix = "scaled"
    unscaled_prefix = "unscaled"

    run_simulation_save_results(units=units_scaled, file_name=scaled_prefix)
    run_simulation_save_results(units=units_unscaled, file_name=unscaled_prefix)

    folder_path = Path(setup.folder)
    num_vtk_tested = 0
    for file_name_scaled in os.listdir(setup.folder):
        if not file_name_scaled.startswith(scaled_prefix):
            continue
        file_name_unscaled = f"{unscaled_prefix}{file_name_scaled[6:]}"
        if file_name_scaled.endswith("vtu"):
            assert compare_vtu_files(
                test_file=folder_path / file_name_scaled,
                reference_file=folder_path / file_name_unscaled,
            ), f"Files don't match: {file_name_scaled} and {file_name_unscaled}."
            num_vtk_tested += 1

    assert num_vtk_tested > 0, "No VTU files were tested."
