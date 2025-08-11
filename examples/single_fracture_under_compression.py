"""Setup for sliding fracture with analytical solution.

To activate the traction stabilization, uncomment the relevant lines in the Model class.
Note, oscillations also occur without friction.

"""

import logging
from typing import Callable

import numpy as np
import porepy as pp
from functools import partial


# Set logging level
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Geometry:
    params: dict
    units: pp.Units

    def meshing_arguments(self) -> dict:
        mesh_args = {}
        mesh_args["cell_size"] = self.units.convert_units(10, "m")
        mesh_args["cell_size_fracture"] = self.units.convert_units(0.1, "m")
        return mesh_args

    def grid_type(self) -> str:
        return "simplex"

    def set_fractures(self) -> None:
        """Set the fractures in the domain."""
        angle = self.params.get("fracture_angle", 20) * np.pi / 180
        length = 10
        points = []
        x = self.units.convert_units(-0.5 * length * np.cos(angle), "m")
        y = self.units.convert_units(-0.5 * length * np.sin(angle), "m")
        x_end = self.units.convert_units(0.5 * length * np.cos(angle), "m")
        y_end = self.units.convert_units(0.5 * length * np.sin(angle), "m")
        points.append(np.array([[x, y], [x_end, y_end]]))
        self._fractures = [pp.LineFracture(pts.T) for pts in points]

    def set_domain(self) -> None:
        """Set the cube domain."""
        bounding_box = {
            "xmin": self.units.convert_units(-1000, "m"),
            "xmax": self.units.convert_units(1000, "m"),
            "ymin": self.units.convert_units(-1000, "m"),
            "ymax": self.units.convert_units(1000, "m"),
        }
        self._domain = pp.Domain(bounding_box)


youngs_modulus = 70e3 * pp.GIGA
poisson_ratio = 0.2

solid_parameters: dict[str, float] = {
    "shear_modulus": youngs_modulus / (2 * (1 + poisson_ratio)),
    "lame_lambda": youngs_modulus
    * poisson_ratio
    / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio)),
    "friction_coefficient": np.tan(30 * np.pi / 180),
}

numerical_parameters: dict[str, float] = {
    "open_state_tolerance": 1e-10,
    "characteristic_contact_traction": 200.0 * pp.MEGA,
}


class BackgroundStress:
    """Mechanical boundary conditions."""

    units: pp.Units

    nd: int

    domain_boundary_sides: Callable[[pp.Grid | pp.BoundaryGrid], pp.domain.DomainSides]

    time_manager: pp.TimeManager

    onset: bool

    def background_stress(self, grid: pp.Grid) -> np.ndarray:
        s = np.zeros((self.nd, self.nd, grid.num_cells))
        s[0, 0] = self.units.convert_units(-200 * pp.MEGA, "Pa")
        return s

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        bc = pp.BoundaryConditionVectorial(sd, boundary_faces, "dir")
        bc.internal_to_dirichlet(sd)

        # Use Neumann virtually everywhere
        domain_sides = self.domain_boundary_sides(sd)
        bc.is_dir[:, domain_sides.all_bf] = False
        bc.is_neu[:, domain_sides.all_bf] = True

        # Fix x-coordinate in the mid cell of the north and south faces
        if np.any(domain_sides.north):
            north_center = np.where(domain_sides.north)[0][
                np.argmin(np.abs(sd.face_centers[0, domain_sides.north] - 0))
            ]
            bc.is_dir[0, north_center] = True
            bc.is_neu[0, north_center] = False

        if np.any(domain_sides.south):
            south_center = np.where(domain_sides.south)[0][
                np.argmin(np.abs(sd.face_centers[0, domain_sides.south] - 0))
            ]
            bc.is_dir[0, south_center] = True
            bc.is_neu[0, south_center] = False

        # Fix y-coordinate in the mid cell of the east and west faces
        if np.any(domain_sides.east):
            east_center = np.where(domain_sides.east)[0][
                np.argmin(np.abs(sd.face_centers[1, domain_sides.east] - 0))
            ]
            bc.is_dir[1, east_center] = True
            bc.is_neu[1, east_center] = False

        if np.any(domain_sides.west):
            west_center = np.where(domain_sides.west)[0][
                np.argmin(np.abs(sd.face_centers[1, domain_sides.west] - 0))
            ]
            bc.is_dir[1, west_center] = True
            bc.is_neu[1, west_center] = False

        return bc

    @property
    def onset(self) -> bool:
        return self.time_manager.time > self.time_manager.schedule[0] + 1e-5

    def bc_values_stress(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Background stress applied to boundary."""
        vals = np.zeros((self.nd, boundary_grid.num_cells))

        # Lithostatic stress on the domain boundaries.
        if boundary_grid.dim == self.nd - 1 and self.onset:
            background_stress_tensor = self.background_stress(boundary_grid)
            domain_sides = self.domain_boundary_sides(boundary_grid)

            # Apply stress from left and right in neumann faces
            if np.any(domain_sides.east):
                east_center = np.where(domain_sides.east)[0][
                    np.argmin(
                        np.abs(boundary_grid.cell_centers[1, domain_sides.east] - 0)
                    )
                ]
                east_others = np.array(
                    list(set(np.where(domain_sides.east)[0].tolist()) - {east_center})
                )
                vals[0, east_others] = (
                    background_stress_tensor[0, 0, east_others]
                    * boundary_grid.cell_volumes[east_others]
                )

            if np.any(domain_sides.west):
                west_center = np.where(domain_sides.west)[0][
                    np.argmin(
                        np.abs(boundary_grid.cell_centers[1, domain_sides.west] - 0)
                    )
                ]
                west_others = list(
                    set(np.where(domain_sides.west)[0].tolist()) - {west_center}
                )
                vals[0, west_others] = (
                    -background_stress_tensor[0, 0, west_others]
                    * boundary_grid.cell_volumes[west_others]
                )

        return vals.ravel("F")


class CustomExport:
    def compute_fracture_states(self):
        # Active sets
        states = []
        subdomains = self.mdg.subdomains(dim=self.nd - 1)

        # Compute the friction bound and the yield criterion
        f_norm = pp.ad.Function(partial(pp.ad.l2_norm, self.nd - 1), "norm_function")
        mu = self.friction_coefficient(subdomains)
        if np.isclose(mu.value(self.equation_system), 0):
            b = self.normal_component(subdomains) @ self.contact_traction(subdomains)
        else:
            b = self.friction_bound(subdomains)
        t_t = self.tangential_component(subdomains) @ self.contact_traction(subdomains)
        yield_criterion = self.friction_bound(subdomains) - f_norm(t_t)
        b_eval = b.value(self.equation_system)
        yield_criterion_eval = yield_criterion.value(self.equation_system)
        tol = self.numerical.open_state_tolerance

        # Determine the state of each fracture cell
        conversion = {
            "stick": 0,
            "slip": 1,
            "open": 2,
            "unknown": -1,
        }
        for b_val, yc_val in zip(b_eval, yield_criterion_eval):
            # print("Checking values", b_val, yc_val, tol)
            if b_val >= -tol:
                states.append(conversion["open"])
            elif yc_val > tol:
                states.append(conversion["stick"])
            elif yc_val <= tol:
                states.append(conversion["slip"])
            else:
                states.append(conversion["unknown"])

        # Split combined states vector into subdomain-corresponding vectors
        split_states = []
        num_cells = []
        for sd in subdomains:
            prev_num_cells = int(sum(num_cells))
            split_states.append(
                np.array(states[prev_num_cells : prev_num_cells + sd.num_cells])
            )
            num_cells.append(sd.num_cells)
        return split_states

    def data_to_export(self):
        """Add data to regular data export."""

        # Fetch current data
        data = super().data_to_export()

        # Exclude contact_traction from data and scale it.
        data = [d for d in data if d[1] != "contact_traction"]

        # Add data to the fracture
        for i, sd in enumerate(self.mdg.subdomains(dim=self.nd - 1)):
            scaled_contact_traction = self.characteristic_contact_traction(
                [sd]
            ) * self.contact_traction([sd])
            scaled_contact_traction_n = (
                self.normal_component([sd]) @ scaled_contact_traction
            )
            scaled_contact_traction_t = (
                self.tangential_component([sd]) @ scaled_contact_traction
            )
            data.append(
                (
                    sd,
                    "contact_traction",
                    self.units.convert_units(1, "Pa^-1")
                    * scaled_contact_traction.value(self.equation_system),
                )
            )
            data.append(
                (
                    sd,
                    "contact_traction_n",
                    self.units.convert_units(1, "Pa^-1")
                    * scaled_contact_traction_n.value(self.equation_system),
                )
            )
            data.append(
                (
                    sd,
                    "contact_traction_t",
                    self.units.convert_units(1, "Pa^-1")
                    * scaled_contact_traction_t.value(self.equation_system),
                )
            )

        # Add contact states
        states = self.compute_fracture_states()
        for i, sd in enumerate(self.mdg.subdomains(dim=self.nd - 1)):
            data.append((sd, "states", states[i]))

        return data


class Model(
    # pp.models.solution_strategy.TractionStabilization,
    Geometry,
    BackgroundStress,
    CustomExport,
    pp.constitutive_laws.CharacteristicDisplacementFromTraction,
    pp.momentum_balance.MomentumBalance,
): ...


# Model parameters
model_params = {
    # Time
    "time_manager": pp.TimeManager(
        schedule=[0, pp.DAY],
        dt_init=pp.DAY,
        constant_dt=True,
    ),
    # Material
    "material_constants": {
        "solid": pp.SolidConstants(**solid_parameters),
        "numerical": pp.NumericalConstants(**numerical_parameters),
    },
    "units": (pp.Units(kg=70e3 * pp.GIGA, m=1e0, s=1, rad=1)),
    "export_constants_separately": False,
    "traction_stabilization_scaling": 1e1,
}

# Solver parameters
solver_params = {
    "nl_convergence_tol": 1e-6,
    "nl_convergence_tol_res": 1e-6,
}

if __name__ == "__main__":
    # Run the model
    model = Model(model_params)
    pp.run_time_dependent_model(model, solver_params)
