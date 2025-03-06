"""Provide a 5-spot model based on the geometry of the 11th SPE Comparative Solution
Project (SPE11), case A:

[J. M. Nordbotten, M. A. Ferno, B. Flemisch, A. R. Kovscek, and K.-A. Lie, “The 11th
Society of Petroleum Engineers Comparative Solution Project: Problem Definition,” SPE
Journal, vol. 29, no. 05, pp. 2507–2524, May 2024, doi: 10.2118/218015-PA.]

https://sccs.stanford.edu/sites/g/files/sbiybj17761/files/media/file/spe_csp11_description.pdf


Additionally, the module provides util functions to download and prepare geometric data
for the model.

"""

import logging
import pathlib
from typing import Any, Callable

import gmsh
import numpy as np
import requests

import porepy as pp
from porepy.fracs.fracture_importer import dfm_from_gmsh
from porepy.viz.exporter import DataInput

logger = logging.getLogger(__name__)

DATA_DIR: pathlib.Path = pathlib.Path(__file__).parent / "spe11_data"
ZIP_FILENAME: str = "por_perm_case2a.zip"
URL: str = (
    "https://raw.githubusercontent.com/Simulation-Benchmarks/11thSPE-CSP/refs/heads/main/geometries/spe11a.geo"
)

# region MODEL_PARAMETERS
WIDTH: float = 2.8  # [m]
HEIGHT: float = 1.2  # [m]


PERMEABILITY: dict[str, float] = {
    "facies 1": 4e-11,
    "facies 2": 5e-10,
    "facies 3": 1e-9,
    "facies 4": 2e-9,
    "facies 5": 4e-9,
    "facies 6": 1e-8,
    "facies 7": 1e-30,  # Epsilon to avoid division by zero when calculating face transmissibilities.
}
POROSITY: dict[str, float] = {
    "facies 1": 0.44,
    "facies 2": 0.43,
    "facies 3": 0.44,
    "facies 4": 0.45,
    "facies 5": 0.43,
    "facies 6": 0.46,
    "facies 7": 1e-20,  # Epsilon to avoid ill-defined problem.
}

# endregion


def download_spe11_data(data_dir: pathlib.Path) -> None:
    """Download the SPE11 geometric information and store them locally."""
    # Ensure the destination directory exists.
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download the ZIP file.
    logger.info(f"Downloading dataset from {URL}")
    response = requests.get(URL)
    response.raise_for_status()

    with (data_dir / "spe11a.geo").open("wb") as f:
        f.write(response.content)
    logger.info("Download completed.")


def read_refinement_factor(geo_file: pathlib.Path) -> float:
    """Read the refinement factor in the SPE11 geometric information."""
    with geo_file.open("r") as f:
        lines: list[str] = f.readlines()
    return float(lines[3][36:-3])


def write_refinement_factor(
    geo_file: pathlib.Path, refinement_factor: float = 1.0
) -> None:
    """Adjust the refinement factor in the SPE11 geometric information."""
    with geo_file.open("r+") as f:
        lines: list[str] = f.readlines()
        lines[3] = f"DefineConstant[ refinement_factor = {refinement_factor} ];\n"
        # Replace the current content of the file.
        f.seek(0)
        f.write("".join(lines))
        f.truncate()


def fix_face_normals(
    gmsh_file: pathlib.Path, mesh_normal: np.ndarray = np.array([0, 0, 1])
) -> pathlib.Path:
    """Fix the SPE11 mesh s.t. all face normals point in the same direction.

    Writes the fixed mesh to a `.msh` file with the same name as the input file.

    """
    if gmsh_file.suffix == ".msh":
        out_file: pathlib.Path = gmsh_file
        gmsh.open(str(gmsh_file))
    else:
        out_file = gmsh_file.with_suffix(".msh")

        # Generate the mesh.
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 3)
        gmsh.merge(str(gmsh_file))
        gmsh.model.mesh.generate(dim=2)
        gmsh.model.mesh.createGeometry()

    # Get all entities
    entities: list[tuple[int, int]] = gmsh.model.getEntities(2)
    points: np.ndarray = gmsh.model.mesh.getNodes(-1, -1)[1].reshape(-1, 3)  # type: ignore

    # Function to calculate the normal of a triangle
    def calculate_normal(triangle: np.ndarray) -> np.ndarray:
        nonlocal points
        p1, p2, p3 = points[triangle]
        normal: np.ndarray = np.cross(p2 - p1, p3 - p1)
        return normal / np.linalg.norm(normal)

    # Fix normals
    num_false_normals: int = 0
    num_cells: int = 0

    for entity in entities:
        element_types, element_tags, node_tags = gmsh.model.mesh.get_elements(
            entity[0], entity[1]
        )
        for elem_type, elem_tags, nodes in zip(element_types, element_tags, node_tags):
            if elem_type == 2:
                # Convert to zero-based index
                triangles: np.ndarray = nodes.reshape(-1, 3) - 1  # type: ignore
                # Loop through all simplices and swap order of vertices if normal points
                # in the wrong direction.
                for i, triangle in enumerate(triangles):
                    normal: np.ndarray = calculate_normal(triangle)
                    num_cells += 1
                    if np.dot(normal, mesh_normal) < 0:
                        triangles[i] = [triangle[1], triangle[0], triangle[2]]
                        num_false_normals += 1
                gmsh.model.mesh.remove_elements(entity[0], entity[1])
                gmsh.model.mesh.add_elements(
                    entity[0],
                    entity[1],
                    [elem_type],
                    [elem_tags],
                    [triangles.flatten() + 1],
                )

    logger.info(f"Fixed {num_false_normals} of {num_cells} normals.")

    gmsh.write(str(out_file))
    gmsh.finalize()
    return out_file


def load_spe11_data(
    data_dir: pathlib.Path, refinement_factor: float = 1.0
) -> pp.MixedDimensionalGrid:
    """Load the SPE11 data into a :class:`~numpy.ndarray`.

    Parameters:
        data_dir: The directory containing the .geo file.

    Returns:
        tuple: A tuple containing:

    """
    # Ensure the destination directory exists.
    data_dir.mkdir(parents=True, exist_ok=True)

    # Assure that the `.geo` file is available. In lieu of a goto statement, run the
    # following loop at maximum twice.
    i: int = 0
    geo_file: pathlib.Path | None = None
    while True:
        for filename in data_dir.iterdir():
            if filename.suffix == ".geo":
                geo_file = data_dir / filename
        if geo_file is None:
            if i >= 1:
                raise FileNotFoundError(
                    "Could not locate the .geo file. Perhaps, the download failed"
                )
            logger.info(".geo file not found. Downloading again ...")
            download_spe11_data(data_dir)
        else:
            break
        i += 1

    if read_refinement_factor(geo_file) != refinement_factor:
        logger.info(
            "Refinement factor in the .geo file is wrong. Adjusting and"
            + " recomputing mesh ..."
        )
        write_refinement_factor(geo_file, refinement_factor)

    gmsh_file: pathlib.Path = fix_face_normals(geo_file)

    logger.info("Loading mesh.")
    mdg = dfm_from_gmsh(str(gmsh_file), dim=2)
    return mdg


class EquationsSPE11(pp.PorePyModel):
    """Mixin class to provide the SPE11 model inspired equations and data.

    Takes care of:
    Updates the two-phase flow equations to include:
    - the SPE11 porosity field.
    - the SPE11 permeability field.
    - A volumetric source term for the water phase in the center cell.
    - Production wells in the corner cells.

    """

    def permeability(self, subdomains: list[pp.Grid]) -> dict[str, np.ndarray]:
        """Solid permeability. Units are set by :attr:`self.solid`."""
        g: pp.Grid = subdomains[0]
        permeability: np.ndarray = np.zeros(g.num_cells)
        for facies, perm in PERMEABILITY.items():
            permeability[g.tags[facies + "_simplices"]] = perm
        return {"kxx": permeability}

    def porosity(self, subdomains: list[pp.Grid]) -> np.ndarray:
        """Solid porosity."""
        g: pp.Grid = subdomains[0]
        porosity: np.ndarray = np.zeros(g.num_cells)
        for facies, por in POROSITY.items():
            porosity[g.tags[facies + "_simplices"]] = por
        return porosity


class ModelGeometrySPE11(pp.PorePyModel):

    def set_domain(self) -> None:
        """Set domain of the problem."""
        box: dict[str, pp.number] = {"xmax": WIDTH, "ymax": HEIGHT}
        self._domain = pp.Domain(box)

    def set_fractures(self) -> None:
        pass

    def set_geometry(self) -> None:
        self.set_domain()
        self.mdg = load_spe11_data(
            DATA_DIR,
            self.params["meshing_arguments"].get("spe11_refinement_factor", 10.0),
        )
        self.nd: int = self.mdg.dim_max()
        g: pp.Grid = self.mdg.subdomains()[0]

        # Check that the domain size is correct.
        height: float = np.max(g.nodes[1, :]) - np.min(g.nodes[1, :])
        width: float = np.max(g.nodes[0, :]) - np.min(g.nodes[0, :])
        assert np.isclose(height, HEIGHT)
        assert np.isclose(width, WIDTH)


class SolutionStrategySPE11(pp.PorePyModel):
    """Mixin class to provide the SPE11 model data."""

    porosity: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Porosity of the rock. Normally provided by a mixin instance of
    :class:`~porepy.models.constitutive_laws.ConstantPorosity` or a subclass thereof.

    """

    # For the next two methods, ignore type errors due do unknown methods/attributes and
    # the wrong type for self.permeability.
    def prepare_simulation(self) -> None:
        self.set_materials()  # type: ignore
        self.set_geometry()
        super().prepare_simulation()  # type: ignore
        self.add_constant_spe11_data()

    def add_constant_spe11_data(self) -> None:
        """Save the SPE11 data to the exporter."""
        data: list[DataInput] = []
        g: pp.Grid = self.mdg.subdomains()[0]
        for dim, perm in self.permeability([self.g]).items():  # type: ignore
            data.append((g, "permeability_" + dim, perm))
        data.append((g, "porosity", self.porosity([g])))
        self.exporter.add_constant_data(data)

        # For convenience, add the porosity and permeability to the iteration exporter
        # if it exists.
        if hasattr(self, "iteration_exporter"):
            self.iteration_exporter.add_constant_data(data)  # type: ignore


# The various protocols define different types for
# ``nonlinear_solver_statistics`` and cause a MyPy error. This is not a problem in
# practice, but ``nonlinear_solver_statistics`` needs to be called with care. We ignore
# the error.
class SPE11(EquationsSPE11, ModelGeometrySPE11, SolutionStrategySPE11, pp.SinglePhaseFlow): ...  # type: ignore


params: dict[str, Any] = {"meshing_arguments": {"spe11_refinement_factor": 10.0}}
model = SPE11(params)
pp.run_stationary_model(model, params)
