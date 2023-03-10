"""
Module containing a function to generate mixed-dimensional grids. It encapsulates
different lower-level md-grid generation.
"""
from __future__ import annotations

import inspect
from typing import Dict, Literal, Optional, Union

import numpy as np

import porepy as pp
import porepy.grids.standard_grids.utils as utils
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d

FractureNetwork = Union[FractureNetwork2d, FractureNetwork3d]


def _validate_args_types(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    mesh_arguments: dict,
    domain: Optional[pp.Domain] = None,
    fracture_network: Optional[FractureNetwork] = None,
    well_network: Optional[pp.WellNetwork3d] = None,
    **kwargs,
):

    if not isinstance(grid_type, str):
        raise TypeError("grid_type must be str, not %r" % type(grid_type))

    if not isinstance(mesh_arguments, dict):
        raise TypeError(
            "mesh_arguments must be dict[str], not %r" % type(mesh_arguments)
        )

    valid_domain = isinstance(domain, pp.Domain) or domain is None
    if not valid_domain:
        raise TypeError("domain must be pp.Domain, not %r" % type(domain))

    valid_frac_network = (
        isinstance(fracture_network, FractureNetwork2d)
        or isinstance(fracture_network, FractureNetwork3d)
        or fracture_network is None
    )
    if not valid_frac_network:
        raise TypeError(
            "fracture_network must be none, FractureNetwork2d, or FractureNetwork3d, not %r"
            % type(fracture_network)
        )

    valid_well_network = (
        isinstance(well_network, pp.WellNetwork3d) or well_network is None
    )
    if not valid_well_network:
        raise TypeError(
            "valid_well_network must be none, or WellNetwork3d, not %r"
            % type(fracture_network)
        )


def _validate_mesh_arguments(mesh_arguments):

    # Common requirement among mesh types
    mesh_size = mesh_arguments.get("mesh_size")
    if not isinstance(mesh_size, float):
        raise TypeError("mesh_size must be float, not %r" % type(mesh_size))
    if not mesh_size > 0:
        raise ValueError("mesh_size must be strcitly positive %r" % mesh_size)


def _infer_dimension_from_domain_or_networks(
    domain: Optional[pp.Domain] = None,
    fracture_network: Optional[FractureNetwork] = None,
    well_network: Optional[pp.WellNetwork3d] = None,
) -> int:

    networks = [fracture_network, well_network]
    valued_objects = [network.domain for network in networks if network is not None]
    if domain is None and not any(valued_objects):
        raise ValueError(
            "Not able to infer the dimension. A pp.domain instance must be provided."
        )

    dimension: int = 0
    if domain is not None:
        if fracture_network is not None:
            if not domain.polytope == fracture_network.domain.polytope:
                raise ValueError("domain and fracture_network.domain differ.")
        if well_network is not None:
            if not domain.polytope == well_network.domain.polytope:
                raise ValueError("domain and well_network.domain differ.")
        dimension = domain.dim
    elif fracture_network is not None and domain is None:
        if well_network is not None:
            if not fracture_network.domain.polytope == well_network.domain.polytope:
                raise ValueError(
                    "fracture_network.domain and well_network.domain differ."
                )
        dimension = fracture_network.domain.dim
    elif well_network is not None and domain is None:
        dimension = well_network.domain.dim

    valid_dimension = dimension == 2 or dimension == 3
    if not valid_dimension:
        raise ValueError("Inferred dimension must be 2 or 3, not %r" % dimension)

    return dimension


def _validate_args(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    mesh_arguments: dict,
    domain: Optional[pp.Domain] = None,
    fracture_network: Optional[FractureNetwork] = None,
    well_network: Optional[pp.WellNetwork3d] = None,
    **kwargs,
):

    _validate_args_types(
        grid_type, mesh_arguments, domain, fracture_network, well_network
    )

    _validate_mesh_arguments(mesh_arguments)


def _setup_extra_simplex_2d_args(**extra_args):
    # fetch signature
    defaults = dict(inspect.signature(FractureNetwork2d.mesh).parameters.items())

    # filter arguments processed in an alternative manner
    defaults.pop("self")
    defaults.pop("mesh_args")
    defaults.pop("kwargs")

    # transfer defaults
    extra_args_list = [
        extra_args.get(item[0], item[1].default) for item in defaults.items()
    ]
    return extra_args_list


def _setup_extra_simplex_3d_args(**extra_args):
    # fetch signature
    defaults = dict(inspect.signature(FractureNetwork3d.mesh).parameters.items())

    # filter arguments processed in an alternative manner
    defaults.pop("self")
    defaults.pop("mesh_args")
    defaults.pop("kwargs")

    # transfer defaults
    extra_args_list = [
        extra_args.get(item[0], item[1].default) for item in defaults.items()
    ]
    return extra_args_list


def create_mdg(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    mesh_arguments: dict,
    domain: Optional[pp.Domain] = None,
    fracture_network: Optional[FractureNetwork] = None,
    well_network: Optional[pp.WellNetwork3d] = None,
    **kwargs,
) -> pp.MixedDimensionalGrid:
    """Creates a mixed dimensional grid.

    Examples:

        .. code:: python3

            import porepy as pp
            import numpy as np

            # Set a domain
            domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})

            # Generate and export a simple mdg
            mdg = pp.create_mdg("simplex",domain)
            pp.Exporter(mdg, "mdg").write_vtu()

            # Generate and export a mdg with a fracture network
            frac_1 = pp.LineFracture(np.array([[0.0, 2.0], [0.0, 0.0]]))
            frac_2 = pp.LineFracture(np.array([[1.0, 1.0], [0.0, 1.0]]))
            fractures = [frac_1, frac_2]
            fracture_network = pp.create_fracture_network(fractures, domain)

            mdg = pp.create_mdg("simplex",fracture_network)
            pp.Exporter(mdg, "mdg_with_fractures").write_vtu()

    Raises:
        - ValueError: If ``domain = None`` and ``fractures = None``and
            ``wells = None``.
        - TypeError: ...
        - Warning: If  ... .

    Parameters:
        grid_type: Type of the resulting grid
            instance of Literal["simplex", "cartesian", "tensor_grid"].
        mesh_arguments: ... .
        domain: Domain specfication. An instance of
            :class:`~porepy.geometry.domain.Domain`.
        fracture_network: ... .
        well_network: ... .
        **kwargs: ... .

    Returns:
        Mixed dimensional grid object.

    """

    _validate_args(
        grid_type, mesh_arguments, domain, fracture_network, well_network, **kwargs
    )

    dimension = _infer_dimension_from_domain_or_networks(
        domain, fracture_network, well_network
    )

    # lower level mdg generation
    if dimension == 2:
        if grid_type == "simplex":
            assert type(mesh_arguments) == type(kwargs)

            h_size = mesh_arguments["mesh_size"]
            lower_level_arguments = {}
            lower_level_arguments["mesh_size_frac"] = h_size
            lower_level_arguments.update(kwargs)
            utils.set_mesh_sizes(lower_level_arguments)
            extra_args = _setup_extra_simplex_2d_args(**kwargs)
            mdg = fracture_network.mesh(lower_level_arguments, *extra_args, **kwargs)
            mdg.compute_geometry()

        elif grid_type == "cartesian":

            # mesh_size
            xmin = fracture_network.domain.bounding_box["xmin"]
            xmax = fracture_network.domain.bounding_box["xmax"]
            ymin = fracture_network.domain.bounding_box["ymin"]
            ymax = fracture_network.domain.bounding_box["ymax"]

            h_size = mesh_arguments["mesh_size"]
            n_x = round((xmax - xmin) / h_size)
            n_y = round((ymax - ymin) / h_size)

            fractures = [f.pts for f in fracture_network.fractures]
            mdg = pp.meshing.cart_grid(
                fracs=fractures,
                physdims=[ymax, xmax],
                nx=np.array([n_x, n_y]),
                **kwargs,
            )

        elif grid_type == "tensor_grid":

            # mesh_size
            xmin = fracture_network.domain.bounding_box["xmin"]
            xmax = fracture_network.domain.bounding_box["xmax"]
            ymin = fracture_network.domain.bounding_box["ymin"]
            ymax = fracture_network.domain.bounding_box["ymax"]

            h_size = mesh_arguments["mesh_size"]
            n_x = round((xmax - xmin) / h_size) + 1
            n_y = round((ymax - ymin) / h_size) + 1
            x_space = np.linspace(xmin, xmax, num=n_x)
            y_space = np.linspace(ymin, ymax, num=n_y)

            fractures = [f.pts for f in fracture_network.fractures]
            mdg = pp.meshing.tensor_grid(
                fracs=fractures, x=x_space, y=y_space, **kwargs
            )

    elif dimension == 3:
        if grid_type == "simplex":

            h_size = mesh_arguments["mesh_size"]
            lower_level_arguments = {}
            lower_level_arguments["mesh_size_frac"] = h_size
            lower_level_arguments.update(kwargs)
            utils.set_mesh_sizes(lower_level_arguments)
            extra_args = _setup_extra_simplex_3d_args(**kwargs)
            mdg = fracture_network.mesh(lower_level_arguments, *extra_args, **kwargs)
            mdg.compute_geometry()

        elif grid_type == "cartesian":

            # mesh_size
            xmin = fracture_network.domain.bounding_box["xmin"]
            xmax = fracture_network.domain.bounding_box["xmax"]
            ymin = fracture_network.domain.bounding_box["ymin"]
            ymax = fracture_network.domain.bounding_box["ymax"]
            zmin = fracture_network.domain.bounding_box["zmin"]
            zmax = fracture_network.domain.bounding_box["zmax"]

            h_size = mesh_arguments["mesh_size"]
            n_x = round((xmax - xmin) / h_size)
            n_y = round((ymax - ymin) / h_size)
            n_z = round((zmax - zmin) / h_size)

            fractures = [f.pts for f in fracture_network.fractures]
            mdg = pp.meshing.cart_grid(
                fracs=fractures,
                physdims=[ymax, xmax, zmax],
                nx=np.array([n_x, n_y, n_z]),
                **kwargs,
            )

        elif grid_type == "tensor_grid":

            # mesh_size
            xmin = fracture_network.domain.bounding_box["xmin"]
            xmax = fracture_network.domain.bounding_box["xmax"]
            ymin = fracture_network.domain.bounding_box["ymin"]
            ymax = fracture_network.domain.bounding_box["ymax"]
            zmin = fracture_network.domain.bounding_box["zmin"]
            zmax = fracture_network.domain.bounding_box["zmax"]

            h_size = mesh_arguments["mesh_size"]
            n_x = round((xmax - xmin) / h_size) + 1
            n_y = round((ymax - ymin) / h_size) + 1
            n_z = round((zmax - zmin) / h_size) + 1

            x_space = np.linspace(xmin, xmax, num=n_x)
            y_space = np.linspace(ymin, ymax, num=n_y)
            z_space = np.linspace(zmin, zmax, num=n_y)

            fractures = [f.pts for f in fracture_network.fractures]
            mdg = pp.meshing.tensor_grid(
                fracs=fractures, x=x_space, y=y_space, z=z_space, **kwargs
            )

    return mdg
