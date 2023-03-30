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
    cell_arguments: dict,
    fracture_network: FractureNetwork,
    **kwargs,
):

    if not isinstance(grid_type, str):
        raise TypeError("grid_type must be str, not %r" % type(grid_type))

    if not isinstance(cell_arguments, dict):
        raise TypeError(
            "cell_arguments must be dict[str], not %r" % type(cell_arguments)
        )

    valid_fracture_network = isinstance(
        fracture_network, FractureNetwork2d
    ) or isinstance(fracture_network, FractureNetwork3d)
    if not valid_fracture_network:
        raise TypeError(
            "fracture_network must be FractureNetwork2d or FractureNetwork3d, not %r"
            % type(fracture_network)
        )


def _validate_grid_type_value(grid_type):
    valid_type = grid_type in ["simplex", "cartesian", "tensor_grid"]
    if not valid_type:
        raise ValueError(
            "grid_type must be in ['simplex', 'cartesian', 'tensor_grid'] not %r"
            % grid_type
        )


def _validate_simplex_meshing_args_values(meshing_args):

    # Get expected arguments
    cell_size = meshing_args.get("cell_size", None)
    cell_size_min = meshing_args.get("cell_size_min", None)
    cell_size_bound = meshing_args.get("cell_size_bound", None)
    cell_size_frac = meshing_args.get("cell_size_frac", None)
    constraints = meshing_args.get("constraints", None)

    # validate cell_size
    cell_size_q = cell_size is not None
    if cell_size_q:
        if not isinstance(cell_size, float):
            raise TypeError("cell_size must be float, not %r" % type(cell_size))
        if not cell_size > 0:
            raise ValueError("cell_size must be strictly positive %r" % cell_size)

    # validate cell_size_min
    cell_size_min_q = cell_size_min is not None
    if cell_size_min_q:
        if not isinstance(cell_size_min, float):
            raise TypeError("cell_size_min must be float, not %r" % type(cell_size_min))
        if not cell_size_min > 0:
            raise ValueError("cell_size_min must be strictly positive %r" % cell_size_min)

    # validate cell_size_bound
    cell_size_bound_q = cell_size_bound is not None
    if cell_size_bound_q:
        if not isinstance(cell_size_bound, float):
            raise TypeError("cell_size_bound must be float, not %r" % type(cell_size_bound))
        if not cell_size_bound > 0:
            raise ValueError("cell_size_bound must be strictly positive %r" % cell_size_bound)

    # validate cell_size_frac
    cell_size_frac_q = cell_size_frac is not None
    if cell_size_frac_q:
        if not isinstance(cell_size_frac, float):
            raise TypeError("cell_size_frac must be float, not %r" % type(cell_size_frac))
        if not cell_size_frac > 0:
            raise ValueError("cell_size_frac must be strictly positive %r" % cell_size_frac)

    if not cell_size_q and not cell_size_min_q:
        raise ValueError("cell_size or cell_size_min must be provided.")

    if not cell_size_q and not cell_size_bound_q:
        raise ValueError("cell_size or cell_size_bound must be provided.")

    if not cell_size_q and not cell_size_frac_q:
        raise ValueError("cell_size or cell_size_frac must be provided.")

    constraints_q = constraints is not None
    if constraints_q:
        if not isinstance(constraints, np.ndarray):
            raise TypeError("constraints must be np.array, not %r" % type(constraints))


def _validate_cartesian_meshing_args_values(dimension, meshing_args):

    # Common requirement among mesh types
    cell_size = meshing_args.get("cell_size", None)
    cell_size_x = meshing_args.get("cell_size_x", None)
    cell_size_y = meshing_args.get("cell_size_y", None)
    cell_size_z = meshing_args.get("cell_size_z", None)

    # validate cell_size
    cell_size_q = cell_size is not None
    if cell_size_q:
        if not isinstance(cell_size, float):
            raise TypeError("cell_size must be float, not %r" % type(cell_size))
        if not cell_size > 0:
            raise ValueError("cell_size must be strictly positive %r" % cell_size)

    # validate cell_size_x
    cell_size_x_q = cell_size_x is not None
    if cell_size_x_q:
        if not isinstance(cell_size_x, float):
            raise TypeError("cell_size_x must be float, not %r" % type(cell_size_x))
        if not cell_size_x > 0:
            raise ValueError("cell_size_x must be strictly positive %r" % cell_size_x)

    # validate cell_size_y
    cell_size_y_q = cell_size_y is not None
    if cell_size_y_q:
        if not isinstance(cell_size_y, float):
            raise TypeError("cell_size_y must be float, not %r" % type(cell_size_y))
        if not cell_size_y > 0:
            raise ValueError("cell_size_y must be strictly positive %r" % cell_size_y)

    # validate cell_size_z
    cell_size_z_q = cell_size_z is not None
    if cell_size_z_q:
        if not isinstance(cell_size_z, float):
            raise TypeError("cell_size_z must be float, not %r" % type(cell_size_z))
        if not cell_size_z > 0:
            raise ValueError("cell_size_z must be strictly positive %r" % cell_size_z)

    if not cell_size_q and not cell_size_x_q:
        raise ValueError("cell_size or cell_size_x must be provided.")
    if not cell_size_q and not cell_size_y_q:
        raise ValueError("cell_size or cell_size_y must be provided.")

    if dimension == 3:
        if not cell_size_q and not cell_size_z_q:
            raise ValueError("cell_size or cell_size_z must be provided.")

def _validate_tensor_grid_meshing_args_values(domain, meshing_args):

    # Common requirement among mesh types
    cell_size = meshing_args.get("cell_size", None)
    x_pts = meshing_args.get("x_pts", None)
    y_pts = meshing_args.get("y_pts", None)
    z_pts = meshing_args.get("z_pts", None)

    # validate cell_size
    cell_size_q = cell_size is not None
    if cell_size_q:
        if not isinstance(cell_size, float):
            raise TypeError("cell_size must be float, not %r" % type(cell_size))
        if not cell_size > 0:
            raise ValueError("cell_size must be strictly positive %r" % cell_size)

    # validate cell_size_x
    x_pts_q = x_pts is not None
    if x_pts_q:
        if not isinstance(x_pts, np.ndarray):
            raise TypeError("x_pts must be np.ndarray, not %r" % type(x_pts))
        x_range = np.array([np.min(x_pts), np.max(x_pts)])
        box_x_range = np.array([domain.bounding_box["xmin"], domain.bounding_box["xmax"]])
        valid_range = np.isclose(x_range, box_x_range)
        if not np.any(valid_range):
            raise ValueError("x_pts must contain boundary points %r" % x_pts)

    # validate cell_size_y
    y_pts_q = y_pts is not None
    if y_pts_q:
        if not isinstance(y_pts, np.ndarray):
            raise TypeError("y_pts must be np.ndarray, not %r" % type(y_pts))
        y_range = np.array([np.min(y_pts), np.max(y_pts)])
        box_y_range = np.array([domain.bounding_box["ymin"], domain.bounding_box["ymax"]])
        valid_range = np.isclose(y_range, box_y_range)
        if not np.any(valid_range):
            raise ValueError("y_pts must contain boundary points %r" % y_pts)

    # validate cell_size_z
    z_pts_q = z_pts is not None
    if z_pts_q:
        if not isinstance(z_pts, np.ndarray):
            raise TypeError("z_pts must be np.ndarray, not %r" % type(z_pts))
        z_range = np.array([np.min(z_pts), np.max(z_pts)])
        box_z_range = np.array([domain.bounding_box["zmin"], domain.bounding_box["zmax"]])
        valid_range = np.isclose(z_range, box_z_range)
        if not np.any(valid_range):
            raise ValueError("z_pts must contain boundary points %r" % z_pts)


def _retrieve_domain_instance(
    fracture_network: FractureNetwork,
) -> Union[pp.Domain, None]:

    # Needed to avoid mypy incompatible types in assignment
    if fracture_network.domain is not None:
        omega: pp.Domain = fracture_network.domain
        valid_dimension = omega.dim == 2 or omega.dim == 3
        if not valid_dimension:
            raise ValueError("Inferred dimension must be 2 or 3, not %r" % omega.dim)
        return omega
    else:
        return None


def _infer_dimension_from_network(fracture_network: FractureNetwork) -> int:

    if isinstance(fracture_network, FractureNetwork2d):
        return 2
    elif isinstance(fracture_network, FractureNetwork3d):
        return 3


def _validate_args(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    meshing_args: dict,
    fracture_network: FractureNetwork,
    **kwargs,
):

    _validate_args_types(grid_type, meshing_args, fracture_network)
    _validate_grid_type_value(grid_type)

    # DFN case is only supported for unstructured simplex meshes
    if fracture_network.domain is None and grid_type != "simplex":
        raise ValueError(
            "fracture_network without a domain is only supported for unstructured simplex"
            " meshes, not for %r" % grid_type
        )

    if grid_type == "simplex":
        _validate_simplex_meshing_args_values(meshing_args)
    elif grid_type == "cartesian":
        dimension = _infer_dimension_from_network(fracture_network)
        _validate_cartesian_meshing_args_values(dimension, meshing_args)
    elif grid_type == "tensor_grid":
        domain = fracture_network.domain
        _validate_tensor_grid_meshing_args_values(domain, meshing_args)




def _preprocess_simplex_args(meshing_args, kwargs, cell_function):

    # fetch signature
    defaults = dict(inspect.signature(cell_function).parameters.items())

    # filter arguments processed in an alternative manner
    defaults.pop("self")
    defaults.pop("mesh_args")
    defaults.pop("kwargs")

    # If provided update constrains
    constraints = meshing_args.get("constraints", None)
    if constraints is not None:
        kwargs.update([("constraints", constraints)])

    # Collects provided and default values
    extra_args_list = [
        kwargs.get(item[0], item[1].default) for item in defaults.items()
    ]



    # Removes duplicate keys in kwargs
    [kwargs.pop(item[0]) for item in defaults.items() if item[0] in kwargs]

    cell_size = meshing_args.get("cell_size", None)
    cell_size_min = meshing_args.get("cell_size_min", None)
    cell_size_bound = meshing_args.get("cell_size_bound", None)
    cell_size_frac = meshing_args.get("cell_size_frac", None)

    # translating quantities to lower level signature
    lower_level_args = {}
    # First set all cell sizes to cell_size
    if cell_size is not None:
        lower_level_args["mesh_size_min"] = cell_size
        lower_level_args["mesh_size_bound"] = cell_size
        lower_level_args["mesh_size_frac"] = cell_size

    # If provided overwrites with cell_size_min
    if cell_size_min is not None:
        lower_level_args["mesh_size_min"] = cell_size_min

    # If provided overwrites with cell_size_bound
    if cell_size_bound is not None:
        lower_level_args["mesh_size_bound"] = cell_size_bound

    # If provided overwrites with cell_size_frac
    if cell_size_frac is not None:
        lower_level_args["mesh_size_frac"] = cell_size_frac

    return (lower_level_args, extra_args_list, kwargs)


def _preprocess_cartesian_args(domain, meshing_args, kwargs):

    xmin = domain.bounding_box["xmin"]
    xmax = domain.bounding_box["xmax"]
    ymin = domain.bounding_box["ymin"]
    ymax = domain.bounding_box["ymax"]

    phys_dims = [xmax, ymax]
    if domain.dim == 3:
        zmax = domain.bounding_box["zmax"]
        phys_dims = [xmax, ymax, zmax]

    cell_size = meshing_args.get("cell_size", None)
    cell_size_x = meshing_args.get("cell_size_x", None)
    cell_size_y = meshing_args.get("cell_size_y", None)
    cell_size_z = meshing_args.get("cell_size_z", None)

    # If provided compute all the information from cell_size
    nx_cells = [None for i in range(domain.dim)]
    if cell_size is not None:
        n_x = round((xmax - xmin) / cell_size)
        n_y = round((ymax - ymin) / cell_size)
        nx_cells = [n_x, n_y]

        if domain.dim == 3:
            zmin = domain.bounding_box["zmin"]
            zmax = domain.bounding_box["zmax"]
            n_z = round((zmax - zmin) / cell_size)
            nx_cells = [n_x, n_y, n_z]

    # if provided overwrite information in x-direction
    if cell_size_x is not None:
        n_x = round((xmax - xmin) / cell_size_x)
        nx_cells[0] = n_x

    # if provided overwrite information in y-direction
    if cell_size_y is not None:
        n_y = round((ymax - ymin) / cell_size_y)
        nx_cells[1] = n_y

    # if provided overwrite information in z-direction
    if cell_size_z is not None and domain.dim == 3:
        zmin = domain.bounding_box["zmin"]
        zmax = domain.bounding_box["zmax"]
        n_z = round((zmax - zmin) / cell_size_z)
        nx_cells[2] = n_z

    # Remove duplicate keys
    [kwargs.pop(item[0]) for item in meshing_args.items() if item[0] in kwargs]

    return (nx_cells, phys_dims, kwargs)


def _preprocess_tensor_grid_args(domain, meshing_args, kwargs):

    x_pts = None
    y_pts = None
    z_pts = None

    xmin = domain.bounding_box["xmin"]
    xmax = domain.bounding_box["xmax"]
    ymin = domain.bounding_box["ymin"]
    ymax = domain.bounding_box["ymax"]

    cell_size = meshing_args.get("cell_size", None)
    user_x_pts = meshing_args.get("x_pts", None)
    user_y_pts = meshing_args.get("y_pts", None)
    user_z_pts = meshing_args.get("z_pts", None)

    # If provided compute all the information from cell_size
    if cell_size is not None:
        n_x = round((xmax - xmin) / cell_size) + 1
        n_y = round((ymax - ymin) / cell_size) + 1

        x_pts = np.linspace(xmin, xmax, num=n_x)
        y_pts = np.linspace(ymin, ymax, num=n_y)

        if domain.dim == 3:
            zmin = domain.bounding_box["zmin"]
            zmax = domain.bounding_box["zmax"]
            n_z = round((zmax - zmin) / cell_size) + 1
            z_pts = np.linspace(zmin, zmax, num=n_z)

    # if provided overwrite information in x-direction
    if user_x_pts is not None:
        x_pts = user_x_pts

    # if provided overwrite information in y-direction
    if user_y_pts is not None:
        y_pts = user_y_pts

    # if provided overwrite information in z-direction
    if user_z_pts is not None and domain.dim == 3:
        z_pts = user_z_pts

    # remove duplicate keys
    [kwargs.pop(item[0]) for item in meshing_args.items() if item[0] in kwargs]

    return (x_pts, y_pts, z_pts, kwargs)


def create_mdg(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    meshing_args: dict,
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
            meshing_args = {"cell_size": 0.1}
            mdg = pp.create_mdg("simplex",meshing_args,fracture_network)


    Raises:
        - ValueError: If ``fracture_network.domain.dim`` is not 2 or 3.
            If
        - TypeError: Mandatory arguments types inconsistent (see type hints in signature).

    See Also:
        - :method:`~porepy.fracs.fracture_network_2d.FractureNetwork2d.mesh`
        - :method:`~porepy.fracs.fracture_network_2d.FractureNetwork3d.mesh`
        - :method:`~porepy.fracs.meshing.cart_grid`
        - :method:`~porepy.fracs.meshing.tensor_grid`

    Parameters:
        grid_type: Type of grid. Use ``simplex`` for unstructured triangular and
            tetrahedral grids, ``cartesian`` for structured, uniform Cartesian grids,
            and ``tensor_grid`` for structured, non-uniform Cartesian grids.
            Instance of Literal["simplex", "cartesian", "tensor_grid"].
        meshing_args: A ``dict`` with meshing keys depending on each grid_type:
            if grid_type == "simplex"
                cell_size: ``float``: Overall minimum cell size. It is required, if one of
                    [cell_size_min, cell_size_frac, cell_size_bound] is not provided.
                cell_size_min: ``float``: minimum cell size. If not provided, cell_size
                    will be used for completeness.
                cell_size_frac: ``float``: size at the fracture. If not provided,
                    cell_size will be used for completeness.
                cell_size_bound: ``float``: boundary cell size. If not provided, cell_size
                    will be used for completeness.
                constraints: ``np.ndarray``: Index list of the fractures that should be
                    treated as constraints in meshing, but not added as separate fracture
                    grids (no splitting of nodes etc.). Useful to define subregions of
                    the domain (and assign e.g., sources, material properties, etc.)
            if grid_type == "cartesian"
                cell_size: ``float``: size in any direction. It is required, if one of
                    [cell_size_x, cell_size_y, cell_size_z] is not provided.
                cell_size_x: ``float``: size in x-direction. If cell_size_x is provided,
                    it overwrites cell_size in the x-direction.
                cell_size_y: ``float``: size in y-direction. If cell_size_y is provided,
                    it overwrites cell_size in the y-direction.
                cell_size_z: ``float``: size in z-direction. If cell_size_z is provided,
                    it overwrites cell_size in the z-direction.
            if grid_type == "tensor_grid"
                cell_size: ``float``: size in any direction. It is required, if one of
                    [x_pts, y_pts, z_pts] is not provided.
                x_pts: ``np.array``: points in x-direction including boundary points.
                    If x_pts is provided, it overwrites the information computed from
                    cell_size in the x-direction.
                y_pts: ``np.array``: points in y-direction including boundary points.
                    If y_pts is provided, it overwrites the information computed from
                    cell_size in the y-direction.
                z_pts: ``np.array``: points in z-direction including boundary points.
                    If z_pts is provided, it overwrites the information computed from
                    cell_size in the z-direction.
        fracture_network: fracture network specification. Instance of
            :class:`~porepy.fracs.fracture_network_2d.FractureNetwork2d` or
            :class:`~porepy.fracs.fracture_network_3d.FractureNetwork3d`.
        **kwargs: A ``dict`` with extra meshing keys associated with each grid_type:
            if grid_type == "simplex" see signature for the `mesh` function in:
                :class:`~porepy.fracs.fracture_network_2d.FractureNetwork2d`
                :class:`~porepy.fracs.fracture_network_2d.FractureNetwork3d`
            if grid_type == "simplex" or "tensor_grid":
                offset: ``float``: Defaults to 0. Parameter that quantifies a perturbation
                    to nodes around the faces that are split.
                    NOTE: this is only for visualization purposes.

    Returns:
        Mixed dimensional grid object.

    """

    _validate_args(grid_type, meshing_args, fracture_network, **kwargs)

    dim = _infer_dimension_from_network(fracture_network)

    # Unstructured cases
    if grid_type == "simplex":
        if dim == 2:
            # preprocess user's arguments provided in kwargs
            (lower_level_args, extra_args, kwargs) = _preprocess_simplex_args(
                meshing_args, kwargs, FractureNetwork2d.mesh
            )
            # perform the actual meshing
            mdg = fracture_network.mesh(lower_level_args, *extra_args, **kwargs)

        elif dim == 3:
            # preprocess user's arguments provided in kwargs
            (lower_level_args, extra_args, kwargs) = _preprocess_simplex_args(
                meshing_args, kwargs, FractureNetwork3d.mesh
            )
            # perform the actual meshing
            mdg = fracture_network.mesh(lower_level_args, *extra_args, **kwargs)

    domain = _retrieve_domain_instance(fracture_network)
    if domain is not None:

        # Structured cases
        if grid_type == "cartesian":
            (nx_cells, phys_dims, kwargs) = _preprocess_cartesian_args(
                domain, meshing_args, kwargs
            )
            fractures = [f.pts for f in fracture_network.fractures]
            mdg = pp.meshing.cart_grid(
                fracs=fractures, nx=nx_cells, physdims=phys_dims, **kwargs
            )

        if grid_type == "tensor_grid":
            fractures = [f.pts for f in fracture_network.fractures]
            (xs, ys, zs, kwargs) = _preprocess_tensor_grid_args(
                domain, meshing_args, kwargs
            )
            mdg = pp.meshing.tensor_grid(fracs=fractures, x=xs, y=ys, z=zs, **kwargs)

    return mdg
