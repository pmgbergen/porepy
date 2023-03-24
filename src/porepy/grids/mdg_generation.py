"""
Module containing a function to generate mixed-dimensional grids (mdg). It encapsulates
different lower-level mdg generation.
"""
from __future__ import annotations

import inspect
from typing import Literal, Union

import numpy as np

import porepy as pp
import porepy.grids.standard_grids.utils as utils
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d

FractureNetwork = Union[FractureNetwork2d, FractureNetwork3d]


def _validate_args_types(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    mesh_arguments: dict,
    fracture_network: FractureNetwork,
    **kwargs,
):

    if not isinstance(grid_type, str):
        raise TypeError("grid_type must be str, not %r" % type(grid_type))

    if not isinstance(mesh_arguments, dict):
        raise TypeError(
            "mesh_arguments must be dict[str], not %r" % type(mesh_arguments)
        )

    valid_fracture_network = isinstance(
        fracture_network, FractureNetwork2d
    ) or isinstance(fracture_network, FractureNetwork3d)
    if not valid_fracture_network:
        raise TypeError(
            "fracture_network must be FractureNetwork2d or FractureNetwork3d, not %r"
            % type(fracture_network)
        )


def _validate_grid_type(grid_type):
    valid_type = grid_type in ["simplex", "cartesian", "tensor_grid"]
    if not valid_type:
        raise ValueError(
            "grid_type must be in ['simplex', 'cartesian', 'tensor_grid'] not %r"
            % grid_type
        )


def _validate_mesh_arguments(mesh_arguments):

    # Common requirement among mesh types
    mesh_size = mesh_arguments.get("mesh_size")
    if not isinstance(mesh_size, float):
        raise TypeError("mesh_size must be float, not %r" % type(mesh_size))
    if not mesh_size > 0:
        raise ValueError("mesh_size must be strictly positive %r" % mesh_size)


def _validate_domain_instance(fracture_network: FractureNetwork) -> pp.Domain:

    omega: pp.Domain = fracture_network.domain
    valid_dimension = omega.dim == 2 or omega.dim == 3
    if not valid_dimension:
        raise ValueError("Inferred dimension must be 2 or 3, not %r" % omega.dim)

    return omega

def _infer_dimension_from_network(fracture_network: FractureNetwork) -> int:

    if isinstance(fracture_network, FractureNetwork2d):
        return 2
    elif isinstance(fracture_network, FractureNetwork3d):
        return 3


def _validate_args(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    mesh_arguments: dict,
    fracture_network: FractureNetwork,
    **kwargs,
):

    _validate_args_types(grid_type, mesh_arguments, fracture_network)
    _validate_grid_type(grid_type)
    _validate_mesh_arguments(mesh_arguments)


def _collect_extra_simplex_args(mesh_arguments, kwargs, mesh_function):
    # fetch signature
    defaults = dict(inspect.signature(mesh_function).parameters.items())

    # filter arguments processed in an alternative manner
    defaults.pop("self")
    defaults.pop("mesh_args")
    defaults.pop("kwargs")

    # transfer defaults
    extra_args_list = [
        kwargs.get(item[0], item[1].default) for item in defaults.items()
    ]

    # remove duplicate keys
    [kwargs.pop(item[0]) for item in defaults.items() if item[0] in kwargs]

    h_size = mesh_arguments["mesh_size"]
    lower_level_args = {}
    lower_level_args["mesh_size_frac"] = h_size

    if "mesh_size_bound" in kwargs:
        lower_level_args["mesh_size_bound"] = kwargs["mesh_size_bound"]
        kwargs.pop("mesh_size_bound")

    if "mesh_size_min" in kwargs:
        lower_level_args["mesh_size_min"] = kwargs["mesh_size_min"]
        kwargs.pop("mesh_size_min")

    # In case mesh_size_bound and mesh_size_min were not informed set them
    utils.set_mesh_sizes(lower_level_args)

    return (lower_level_args, extra_args_list, kwargs)


def _preprocess_cartesian_args(omega, mesh_arguments, kwargs):

    nx_cells = None
    phys_dims = None

    xmin = omega.bounding_box["xmin"]
    xmax = omega.bounding_box["xmax"]
    ymin = omega.bounding_box["ymin"]
    ymax = omega.bounding_box["ymax"]

    h_size = mesh_arguments["mesh_size"]
    n_x = round((xmax - xmin) / h_size)
    n_y = round((ymax - ymin) / h_size)

    nx_cells = [n_x, n_y]
    phys_dims = [xmax, ymax]

    if omega.dim == 3:
        zmin = omega.bounding_box["zmin"]
        zmax = omega.bounding_box["zmax"]
        n_z = round((zmax - zmin) / h_size)
        nx_cells = [n_x, n_y, n_z]
        phys_dims = [xmax, ymax, zmax]

    # Avoid duplicity in values for keyword argument
    nx_arg = kwargs.get("nx", None)
    if isinstance(nx_arg, nx_cells.__class__) and len(nx_arg) == omega.dim:
        # overwrite information if user provide a valid nx
        nx_cells = nx_arg
        kwargs.pop("nx")

    physdims_arg = kwargs.get("physdims", None)
    if isinstance(physdims_arg, phys_dims.__class__) and len(phys_dims) == omega.dim:
        # overwrite information if user provide a valid physdims
        phys_dims = physdims_arg
        kwargs.pop("physdims")

    # remove duplicate keys
    [kwargs.pop(item[0]) for item in mesh_arguments.items() if item[0] in kwargs]

    return (nx_cells, phys_dims, kwargs)


def _preprocess_tensor_grid_args(omega, mesh_arguments, kwargs):

    x_space = None
    y_space = None
    z_space = None

    xmin = omega.bounding_box["xmin"]
    xmax = omega.bounding_box["xmax"]
    ymin = omega.bounding_box["ymin"]
    ymax = omega.bounding_box["ymax"]

    h_size = mesh_arguments["mesh_size"]
    n_x = round((xmax - xmin) / h_size) + 1
    n_y = round((ymax - ymin) / h_size) + 1

    x_space = np.linspace(xmin, xmax, num=n_x)
    y_space = np.linspace(ymin, ymax, num=n_y)

    # Avoid duplicity in kwargs
    x_arg = kwargs.get("x", None)
    if isinstance(x_arg, x_space.__class__):
        x_space = x_arg
        kwargs.pop("x")
    y_arg = kwargs.get("y", None)
    if isinstance(y_arg, y_space.__class__):
        y_space = y_arg
        kwargs.pop("y")

    if omega.dim == 3:
        zmin = omega.bounding_box["zmin"]
        zmax = omega.bounding_box["zmax"]
        n_z = round((zmax - zmin) / h_size) + 1
        z_space = np.linspace(zmin, zmax, num=n_z)

        z_arg = kwargs.get("z", None)
        if isinstance(z_arg, z_space.__class__):
            z_space = z_arg
            kwargs.pop("z")

    # remove duplicate keys
    [kwargs.pop(item[0]) for item in mesh_arguments.items() if item[0] in kwargs]

    return (x_space, y_space, z_space, kwargs)


def create_mdg(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    mesh_arguments: dict,
    fracture_network: FractureNetwork,
    **kwargs,
) -> pp.MixedDimensionalGrid:
    """Creates a mixed dimensional grid.

    Examples:

        .. code:: python3

            import porepy as pp
            import numpy as np

            # Set a domain
            domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})

            # Generate a fracture network
            frac_1 = pp.LineFracture(np.array([[0.0, 2.0], [0.0, 0.0]]))
            frac_2 = pp.LineFracture(np.array([[1.0, 1.0], [0.0, 1.0]]))
            fractures = [frac_1, frac_2]
            fracture_network = pp.create_fracture_network(fractures, domain)

            # Generate a mixed-dimensional grid (MDG)
            mesh_args = {"mesh_size": 0.1}
            mdg = pp.create_mdg("simplex",mesh_args,fracture_network)


    Raises:
        - ValueError: If ``fracture_network.domain.dim`` is not 2 or 3.
            If
        - TypeError: Mandatory arguments types inconsistent (see type hints in signature).

    Parameters:
        grid_type: Type of grid. Use ``simplex`` for unstructured triangular and tetrahedral grids, ``cartesian`` for structured, uniform Cartesian grids, and ``tensor_grid`` for structured, non-uniform Cartesian grids.
            instance of Literal["simplex", "cartesian", "tensor_grid"].
        mesh_arguments: ... .
        fracture_network: fracture network specification. Instance of
            :class:`~porepy.fracs.fracture_network_2d.FractureNetwork2d` or
            :class:`~porepy.fracs.fracture_network_3d.FractureNetwork3d`.
        **kwargs: Extra arguments associated with each grid_type.

    Returns:
        Mixed dimensional grid object.

    """

    _validate_args(grid_type, mesh_arguments, fracture_network, **kwargs)

    dim = _infer_dimension_from_network(fracture_network)

    # Unstructure cases
    if grid_type == "simplex":
        if dim == 2:
            # preprocess user's arguments provided in kwargs
            (lower_level_args, extra_args, kwargs) = _collect_extra_simplex_args(
                mesh_arguments, kwargs, FractureNetwork2d.mesh
            )
            # perform the actual meshing
            mdg = fracture_network.mesh(lower_level_args, *extra_args, **kwargs)

        elif dim == 3:
            # preprocess user's arguments provided in kwargs
            (lower_level_args, extra_args, kwargs) = _collect_extra_simplex_args(
                mesh_arguments, kwargs, FractureNetwork3d.mesh
            )
            # perform the actual meshing
            mdg = fracture_network.mesh(lower_level_args, *extra_args, **kwargs)

    domain = _validate_domain_instance(fracture_network)
    # Structure cases
    if grid_type == "cartesian":
        (nx_cells, phys_dims, kwargs) = _preprocess_cartesian_args(
            domain, mesh_arguments, kwargs
        )
        fractures = [f.pts for f in fracture_network.fractures]
        mdg = pp.meshing.cart_grid(
            fracs=fractures, nx=nx_cells, physdims=phys_dims, **kwargs
        )

    if grid_type == "tensor_grid":
        fractures = [f.pts for f in fracture_network.fractures]
        (xs, ys, zs, kwargs) = _preprocess_tensor_grid_args(
            domain, mesh_arguments, kwargs
        )
        mdg = pp.meshing.tensor_grid(fracs=fractures, x=xs, y=ys, z=zs, **kwargs)

    return mdg
