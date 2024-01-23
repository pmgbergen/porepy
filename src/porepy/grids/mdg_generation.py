"""
Module containing a function to generate mixed-dimensional grids (mdg). It encapsulates
different lower-level mdg generation.
"""
from __future__ import annotations

import inspect
import warnings
from typing import Callable, Literal, Optional, Union, get_args

import numpy as np

import porepy as pp
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d

FractureNetwork = Union[FractureNetwork2d, FractureNetwork3d]


def _validate_args_types(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    meshing_args: dict,
    fracture_network: FractureNetwork,
):
    """Validates argument types.

    Parameters:
        grid_type: Type of grid.
        meshing_args: A dictionary with meshing keywords depending on each ``grid_type``.
        fracture_network: fracture network specification.

    Raises:
        - TypeError: Raises type error messages if:
            - ``grid_type`` is not ``str``.
            - ``meshing_args`` is not ``dict``.
            - fracture_network is not an instance of
            :class:`~porepy.fracs.fracture_network_2d.FractureNetwork2d` or
            :class:`~porepy.fracs.fracture_network_3d.FractureNetwork3d`.

    """

    if not isinstance(grid_type, str):
        raise TypeError("grid_type must be str, not %r" % type(grid_type))

    if not isinstance(meshing_args, dict):
        raise TypeError("meshing_args must be dict, not %r" % type(meshing_args))

    valid_fracture_network: bool = isinstance(
        fracture_network, get_args(FractureNetwork)
    )
    if not valid_fracture_network:
        raise TypeError(
            "fracture_network must be FractureNetwork2d or FractureNetwork3d, not %r"
            % type(fracture_network)
        )


def _validate_grid_type_value(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"]
):
    """Validates grid_type value.

    Parameters:
        grid_type: Type of grid.

    Raises:
        - ValueError: Raises value error messages if:
            - grid_type is not a member of ['simplex', 'cartesian', 'tensor_grid']

    """
    valid_type: bool = grid_type in ["simplex", "cartesian", "tensor_grid"]
    if not valid_type:
        raise ValueError(
            "grid_type must be in ['simplex', 'cartesian', 'tensor_grid'] not %r"
            % grid_type
        )


def _validate_simplex_meshing_args_values(meshing_args: dict):
    """Validates items in meshing_args for simplex mdg.

    Parameters:
        meshing_args: A dictionary with meshing keys depending on each grid_type.

    Raises:
        - TypeError: Raises type error messages if:
            - cell_size is not ``float``
            - cell_size_min is not ``float``
            - cell_size_boundary is not ``float``
            - cell_size_fracture is not ``float``
        - ValueError: Raises value error messages if:
            - cell_size is not strictly positive
            - cell_size_min is not strictly positive
            - cell_size_boundary is not strictly positive
            - cell_size_fracture is not strictly positive
            - cell_size or cell_size_min are not provided
            - cell_size or cell_size_boundary are not provided
            - cell_size or cell_size_fracture are not provided

    """
    # Get expected arguments
    cell_size: Optional[float] = meshing_args.get("cell_size", None)
    cell_size_min: Optional[float] = meshing_args.get("cell_size_min", cell_size)
    cell_size_boundary: Optional[float] = meshing_args.get(
        "cell_size_boundary", cell_size
    )
    cell_size_fracture: Optional[float] = meshing_args.get(
        "cell_size_fracture", cell_size
    )

    # validate cell_size
    cell_size_q: bool = cell_size is not None
    if cell_size_q:
        if not isinstance(cell_size, float):
            raise TypeError("cell_size must be float, not %r" % type(cell_size))
        if not cell_size > 0:
            raise ValueError("cell_size must be strictly positive %r" % cell_size)

    # validate cell_size_min
    cell_size_min_q: bool = cell_size_min is not None
    if cell_size_min_q:
        if not isinstance(cell_size_min, float):
            raise TypeError("cell_size_min must be float, not %r" % type(cell_size_min))
        if not cell_size_min > 0:
            raise ValueError(
                "cell_size_min must be strictly positive %r" % cell_size_min
            )
    else:
        raise ValueError("cell_size or cell_size_min must be provided.")

    # validate cell_size_boundary
    cell_size_boundary_q: bool = cell_size_boundary is not None
    if cell_size_boundary_q:
        if not isinstance(cell_size_boundary, float):
            raise TypeError(
                "cell_size_boundary must be float, not %r" % type(cell_size_boundary)
            )
        if not cell_size_boundary > 0:
            raise ValueError(
                "cell_size_boundary must be strictly positive %r" % cell_size_boundary
            )
    else:
        raise ValueError("cell_size or cell_size_boundary must be provided.")

    # validate cell_size_fracture
    cell_size_fracture_q: bool = cell_size_fracture is not None
    if cell_size_fracture_q:
        if not isinstance(cell_size_fracture, float):
            raise TypeError(
                "cell_size_fracture must be float, not %r" % type(cell_size_fracture)
            )
        if not cell_size_fracture > 0:
            raise ValueError(
                "cell_size_fracture must be strictly positive %r" % cell_size_fracture
            )
    else:
        raise ValueError("cell_size or cell_size_fracture must be provided.")


def _validate_cartesian_meshing_args_values(dimension: int, meshing_args: dict):
    """Validates items in meshing_args for cartesian mdg.

    Parameters:
        dimension: The dimension of the domain.
        meshing_args: A dictionary with meshing keys depending on each grid_type.

    Raises:
        - TypeError: Raises type error messages if:
            - cell_size is not ``float``
            - cell_size_x is not ``float``
            - cell_size_y is not ``float``
            - cell_size_z is not ``float``
        - ValueError: Raises value error messages if:
            - cell_size is not strictly positive
            - cell_size_x is not strictly positive
            - cell_size_y is not strictly positive
            - cell_size_z is not strictly positive
            - cell_size or cell_size_x are not provided
            - cell_size or cell_size_y are not provided
            - cell_size or cell_size_z are not provided if dimension == 3

    """

    # Common requirement among mesh types
    cell_size: Optional[float] = meshing_args.get("cell_size", None)
    cell_size_x: Optional[float] = meshing_args.get("cell_size_x", cell_size)
    cell_size_y: Optional[float] = meshing_args.get("cell_size_y", cell_size)
    cell_size_z: Optional[float] = meshing_args.get("cell_size_z", cell_size)

    # validate cell_size
    cell_size_q: bool = cell_size is not None
    if cell_size_q:
        if not isinstance(cell_size, float):
            raise TypeError("cell_size must be float, not %r" % type(cell_size))
        if not cell_size > 0:
            raise ValueError("cell_size must be strictly positive %r" % cell_size)

    # validate cell_size_x
    cell_size_x_q: bool = cell_size_x is not None
    if cell_size_x_q:
        if not isinstance(cell_size_x, float):
            raise TypeError("cell_size_x must be float, not %r" % type(cell_size_x))
        if not cell_size_x > 0:
            raise ValueError("cell_size_x must be strictly positive %r" % cell_size_x)
    else:
        raise ValueError("cell_size or cell_size_x must be provided.")

    # validate cell_size_y
    cell_size_y_q: bool = cell_size_y is not None
    if cell_size_y_q:
        if not isinstance(cell_size_y, float):
            raise TypeError("cell_size_y must be float, not %r" % type(cell_size_y))
        if not cell_size_y > 0:
            raise ValueError("cell_size_y must be strictly positive %r" % cell_size_y)
    else:
        raise ValueError("cell_size or cell_size_y must be provided.")

    # validate cell_size_z
    cell_size_z_q: bool = cell_size_z is not None
    if cell_size_z_q:
        if not isinstance(cell_size_z, float):
            raise TypeError("cell_size_z must be float, not %r" % type(cell_size_z))
        if not cell_size_z > 0:
            raise ValueError("cell_size_z must be strictly positive %r" % cell_size_z)
    else:
        if dimension == 3:
            raise ValueError("cell_size or cell_size_z must be provided.")


def _validate_tensor_grid_meshing_args_values(domain: pp.Domain, meshing_args: dict):
    """Validates items in meshing_args for tensor_grid mdg.

    Parameters:
        domain: An instance of :class:`~porepy.geometry.domain.Domain` representing
           the domain.
        meshing_args: A dictionary with meshing keys depending on each grid_type.

    Raises:
        - TypeError: Raises type error messages if:
            - cell_size is not ``float``
            - x_pts is not ``np.ndarray``
            - y_pts is not ``np.ndarray``
            - z_pts is not ``np.ndarray``
        - ValueError: Raises value error messages if:
            - cell_size is not strictly positive
            - The points np.min(x_pts) and np.max(x_pts) are not on the boundary.
            - The points np.min(y_pts) and np.max(y_pts) are not on the boundary.
            - The points np.min(z_pts) and np.max(z_pts) are not on the boundary.
            - cell_size or x_pts are not provided
            - cell_size or y_pts are not provided
            - cell_size or z_pts are not provided if dimension == 3

    """

    # Common requirement among mesh types
    cell_size: Optional[float] = meshing_args.get("cell_size", None)
    x_pts: Optional[np.ndarray] = meshing_args.get("x_pts", None)
    y_pts: Optional[np.ndarray] = meshing_args.get("y_pts", None)
    z_pts: Optional[np.ndarray] = meshing_args.get("z_pts", None)

    # validate cell_size
    cell_size_q: bool = cell_size is not None
    if cell_size_q:
        if not isinstance(cell_size, float):
            raise TypeError("cell_size must be float, not %r" % type(cell_size))
        if not cell_size > 0:
            raise ValueError("cell_size must be strictly positive %r" % cell_size)

    # validate cell_size_x
    x_pts_q: bool = x_pts is not None
    if x_pts_q:
        if not isinstance(x_pts, np.ndarray):
            raise TypeError("x_pts must be np.ndarray, not %r" % type(x_pts))
        x_range = np.array([np.min(x_pts), np.max(x_pts)])
        box_x_range = np.array(
            [domain.bounding_box["xmin"], domain.bounding_box["xmax"]]
        )
        valid_range = np.isclose(x_range, box_x_range)
        if not np.all(valid_range):
            raise ValueError(
                "The points np.min(x_pts), np.max(x_pts) must be on the boundary."
            )

    # validate cell_size_y
    y_pts_q: bool = y_pts is not None
    if y_pts_q:
        if not isinstance(y_pts, np.ndarray):
            raise TypeError("y_pts must be np.ndarray, not %r" % type(y_pts))
        y_range = np.array([np.min(y_pts), np.max(y_pts)])
        box_y_range = np.array(
            [domain.bounding_box["ymin"], domain.bounding_box["ymax"]]
        )
        valid_range = np.isclose(y_range, box_y_range)
        if not np.all(valid_range):
            raise ValueError(
                "The points np.min(y_pts), np.max(y_pts) must be on the boundary."
            )

    # validate cell_size_z
    z_pts_q: bool = z_pts is not None and domain.dim == 3
    if z_pts_q:
        if not isinstance(z_pts, np.ndarray):
            raise TypeError("z_pts must be np.ndarray, not %r" % type(z_pts))
        z_range = np.array([np.min(z_pts), np.max(z_pts)])
        box_z_range = np.array(
            [domain.bounding_box["zmin"], domain.bounding_box["zmax"]]
        )
        valid_range = np.isclose(z_range, box_z_range)
        if not np.all(valid_range):
            raise ValueError(
                "The points np.min(z_pts), np.max(z_pts) must be on the boundary."
            )

    if not cell_size_q and not x_pts_q:
        raise ValueError("cell_size or x_pts must be provided.")

    if not cell_size_q and not y_pts_q:
        raise ValueError("cell_size or y_pts must be provided.")

    if domain.dim == 3:
        if not cell_size_q and not z_pts_q:
            raise ValueError("cell_size or z_pts must be provided.")


def _retrieve_domain_instance(
    fracture_network: FractureNetwork,
) -> Optional[pp.Domain]:
    """Retrieve from fracture_network.

    Parameters:
        fracture_network: fracture network specification.

    Raises:
        - ValueError: Raises value error messages if:
            - Inferred dimension is not 2 or 3

    Returns:
         domain: An instance of :class:`~porepy.geometry.domain.Domain` representing
         the domain or an instance ``None``.

    """

    # Needed to avoid mypy incompatible types in assignment
    if fracture_network.domain is not None:
        domain: pp.Domain = fracture_network.domain
        valid_dimension = domain.dim == 2 or domain.dim == 3
        if not valid_dimension:
            raise ValueError("Inferred dimension must be 2 or 3, not %r" % domain.dim)
        return domain
    else:
        return None


def _infer_dimension_from_network(fracture_network: FractureNetwork) -> int:
    """Infer dimension from fracture_network type.

    Parameters:
        fracture_network: fracture network specification.

    Returns:
        Dimension equal to 2 if the fracture network is FractureNetwork2d and 3 if the
        fracture network is FractureNetwork3d.

    """

    if isinstance(fracture_network, FractureNetwork2d):
        return 2
    elif isinstance(fracture_network, FractureNetwork3d):
        return 3


def _validate_args(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    meshing_args: dict,
    fracture_network: FractureNetwork,
):
    """Validates grid_type, meshing_args and fracture_network types and values.

    Parameters:
        grid_type: Type of grid.
        meshing_args: A ``dict`` with meshing keys depending on each grid_type.
        fracture_network: fracture network specification.

    Raises:
        - ValueError: Raises value error messages if:
            - fracture_network without a domain is provided and grid_type != "simplex"

    """

    _validate_args_types(grid_type, meshing_args, fracture_network)
    _validate_grid_type_value(grid_type)

    # DFN case is only supported for unstructured simplex meshes
    if fracture_network.domain is None and grid_type != "simplex":
        raise ValueError(
            "fracture_network without a domain is only supported for unstructured simplex"
            " meshes, not for %r" % grid_type
        )
    elif grid_type != "simplex":
        dim: int = _infer_dimension_from_network(fracture_network)
        domain: Union[pp.Domain, None] = _retrieve_domain_instance(fracture_network)
        if dim == 2:
            assert isinstance(fracture_network, FractureNetwork2d)
            _, fractures_deleted = fracture_network.impose_external_boundary(domain)
        elif dim == 3:
            assert isinstance(fracture_network, FractureNetwork3d)
            _, fractures_deleted = fracture_network.impose_external_boundary(domain)
        # Take note of deleted fractures
        if (sz := fractures_deleted.size) > 0:
            # It seems most likely that this is an undesired effect (for a
            # Cartesian geomtetry it should be possible to make sure the
            # fractures are within the domain), but we cannot rule out that the
            # user on purpose use the domain to get rid of some fractures.
            # Giving a warning seems like a fair compromise between raising an
            # error and doing nothing.
            warnings.warn(f"Found {sz} fractures outside the domain boundary")

    if grid_type == "simplex":
        _validate_simplex_meshing_args_values(meshing_args)
    elif grid_type == "cartesian":
        dimension = _infer_dimension_from_network(fracture_network)
        _validate_cartesian_meshing_args_values(dimension, meshing_args)
    elif grid_type == "tensor_grid":
        if fracture_network.domain is not None:
            _validate_tensor_grid_meshing_args_values(
                fracture_network.domain, meshing_args
            )


def _preprocess_simplex_args(
    meshing_args: dict, kwargs: dict, mesh_function: Callable
) -> tuple:
    """Preprocess arguments for simplex mdg.

    See Also:
        - :meth:`~porepy.fracs.fracture_network_2d.FractureNetwork2d.mesh`
        - :meth:`~porepy.fracs.fracture_network_3d.FractureNetwork3d.mesh`

    Parameters:
        meshing_args: A dictionary with meshing keys depending on each grid_type.
        kwargs: A dictionary with extra keys depending on each grid_type.
        mesh_function: A callable mesh function of a fracture network specification.

    Returns:
        lower_level_args: An instance of mesh_args which contains arguments passed
        on to mesh size control.
        extra_args_list: A list with values of the following arguments:
            - for :class:`~porepy.fracs.fracture_network_2d.FractureNetwork2d`, It
            contains: tol, do_snap, constraints, file_name, dfn, tags_to_transfer,
                remove_small_fractures, write_geo, finalize_gmsh, and clear_gmsh.
            - for :class:`~porepy.fracs.fracture_network_2d.FractureNetwork3d`, It
            contains: dfn, file_name, constraints, write_geo, tags_to_transfer,
                finalize_gmsh, and clear_gmsh.
        kwargs: It could contain the item offset: ``float``: Defaults to 0. Parameter that
            quantifies a perturbation to nodes around the faces that are split. This is
            only for visualization purposes.

    """

    # The functions for meshing of 2d and 3d fracture networks have different
    # signatures, thus we need to allow for different arguments. First fetch the
    # signature of the function - this call gives dictionary of argument names and
    # default values (None if no default is provided). The values of this dictionary
    # are of type inspect.Parameter.
    defaults: dict[str, inspect.Parameter] = dict(
        inspect.signature(mesh_function).parameters.items()
    )
    # filter arguments processed in an alternative manner
    defaults.pop("self")
    defaults.pop("mesh_args")
    defaults.pop("kwargs")

    # transfer defaults
    # The values of the dictionary are of type inspect.Parameter, which has a
    # default attribute that we access.
    extra_args_list: list = [
        kwargs.get(key, val.default) for (key, val) in defaults.items()
    ]

    # remove duplicate keys
    [kwargs.pop(key) for key in defaults if key in kwargs]

    # translating quantities to lower level signature
    lower_level_args: dict = {}
    cell_size: Optional[float] = meshing_args.get("cell_size", None)
    lower_level_args["mesh_size_min"] = meshing_args.get("cell_size_min", cell_size)
    lower_level_args["mesh_size_bound"] = meshing_args.get(
        "cell_size_boundary", cell_size
    )
    lower_level_args["mesh_size_frac"] = meshing_args.get(
        "cell_size_fracture", cell_size
    )

    return (lower_level_args, extra_args_list, kwargs)


def _preprocess_cartesian_args(
    domain: pp.Domain, meshing_args: dict, kwargs: dict
) -> tuple:
    """Preprocess arguments for cartesian mdg.

    See Also:
        - :meth:`~porepy.fracs.meshing.cart_grid`

    Parameters:
        domain: An instance of :class:`~porepy.geometry.domain.Domain` representing
            the domain.
        meshing_args: A dictionary with meshing keys depending on each grid_type.
        kwargs: A dictionary with extra keys depending on each grid_type.

    Returns:
        nx_cells: Number of cells in each direction.
        phys_dims: Physical dimensions in each direction. It is inferred from domain.
        kwargs: It could contain the item offset: ``float``: Defaults to 0.

    """

    xmin: float = domain.bounding_box["xmin"]
    xmax: float = domain.bounding_box["xmax"]
    ymin: float = domain.bounding_box["ymin"]
    ymax: float = domain.bounding_box["ymax"]

    phys_dims: list[float] = [xmax, ymax]
    if domain.dim == 3:
        phys_dims = [xmax, ymax, domain.bounding_box["zmax"]]

    cell_size: Optional[float] = meshing_args.get("cell_size", None)
    cell_size_x: Optional[float] = meshing_args.get("cell_size_x", cell_size)
    cell_size_y: Optional[float] = meshing_args.get("cell_size_y", cell_size)
    cell_size_z: Optional[float] = meshing_args.get("cell_size_z", cell_size)

    # If provided compute all the information from cell_size
    nx_cells: list[int] = [-1 for i in range(domain.dim)]
    n_x: int = -1
    n_y: int = -1
    n_z: int = -1

    # if provided overwrite information in x-direction
    if cell_size_x is not None:
        n_x = round((xmax - xmin) / cell_size_x)
        if np.abs((xmax - xmin)) - cell_size_x < 0.0:
            warnings.warn(
                "In the x-direction, cell_size_x is greater than the domain. The domain "
                "size is used instead."
            )
            n_x = int(np.max([round((xmax - xmin) / cell_size_x), 1]))
        nx_cells[0] = n_x

    # if provided overwrite information in y-direction
    if cell_size_y is not None:
        n_y = round((ymax - ymin) / cell_size_y)
        if np.abs((ymax - ymin)) - cell_size_y < 0.0:
            warnings.warn(
                "In the y-direction, cell_size_y is greater than the domain. The domain "
                "size is used instead."
            )
            n_y = int(np.max([round((ymax - ymin) / cell_size_y), 1]))
        nx_cells[1] = n_y

    # if provided overwrite information in z-direction
    if cell_size_z is not None and domain.dim == 3:
        zmin: float = domain.bounding_box["zmin"]
        zmax: float = domain.bounding_box["zmax"]

        n_z = round((zmax - zmin) / cell_size_z)
        if np.abs((zmax - zmin)) - cell_size_z < 0.0:
            warnings.warn(
                "In the z-direction, cell_size_z is greater than the domain. The domain "
                "size is used instead."
            )
            n_z = int(np.max([round((zmax - zmin) / cell_size_z), 1]))
        nx_cells[2] = n_z

    # Remove duplicate keys
    [kwargs.pop(item[0]) for item in meshing_args.items() if item[0] in kwargs]

    return (nx_cells, phys_dims, kwargs)


def _preprocess_tensor_grid_args(
    domain: pp.Domain, meshing_args: dict, kwargs: dict
) -> tuple:
    """Preprocess arguments for tensor_grid mdg.

    See Also:
        - :meth:`~porepy.fracs.meshing.cart_grid`

    Parameters:
        domain: An instance of :class:`~porepy.geometry.domain.Domain` representing
            the domain.
        meshing_args: A dictionary with meshing keys depending on each grid_type.
        kwargs: A dictionary with extra keys depending on each grid_type.

    Returns:
        x_pts: An ``np.ndarray`` with points in x-direction.
        y_pts: An ``np.ndarray`` with points in y-direction.
        z_pts: An ``np.ndarray`` with points in z-direction. It is ``None`` for 2D.
        kwargs: It could contain the item offset: ``float``: Defaults to 0.

    """

    x_pts: Optional[np.ndarray] = None
    y_pts: Optional[np.ndarray] = None
    z_pts: Optional[np.ndarray] = None

    xmin: float = domain.bounding_box["xmin"]
    xmax: float = domain.bounding_box["xmax"]
    ymin: float = domain.bounding_box["ymin"]
    ymax: float = domain.bounding_box["ymax"]

    cell_size: Optional[float] = meshing_args.get("cell_size", None)
    n_x: int = -1
    n_y: int = -1
    n_z: int = -1

    # If provided compute all the information from cell_size
    if cell_size is not None:
        n_x = round((xmax - xmin) / cell_size) + 1
        if np.abs((xmax - xmin)) - cell_size < 0.0:
            warnings.warn(
                "In the x-direction, cell_size is greater than the domain. The domain "
                "size is used instead."
            )
            n_x = int(np.max([round((xmax - xmin) / cell_size), 1])) + 1

        n_y = round((ymax - ymin) / cell_size) + 1
        if np.abs((ymax - ymin)) - cell_size < 0.0:
            warnings.warn(
                "In the y-direction, cell_size is greater than the domain. The domain "
                "size is used instead."
            )
            n_y = int(np.max([round((ymax - ymin) / cell_size), 1])) + 1

        x_pts = np.linspace(xmin, xmax, num=n_x)
        y_pts = np.linspace(ymin, ymax, num=n_y)

        if domain.dim == 3:
            zmin: float = domain.bounding_box["zmin"]
            zmax: float = domain.bounding_box["zmax"]

            n_z = round((zmax - zmin) / cell_size) + 1
            if np.abs((zmax - zmin)) - cell_size < 0.0:
                warnings.warn(
                    "In the z-direction, cell_size is greater than the domain. The domain "
                    "size is used instead."
                )
                n_z = int(np.max([round((zmax - zmin) / cell_size), 1])) + 1
            z_pts = np.linspace(zmin, zmax, num=n_z)

    x_pts = meshing_args.get("x_pts", x_pts)
    y_pts = meshing_args.get("y_pts", y_pts)

    if domain.dim == 3:
        z_pts = meshing_args.get("z_pts", z_pts)

    # remove duplicate keys
    [kwargs.pop(item[0]) for item in meshing_args.items() if item[0] in kwargs]

    return (x_pts, y_pts, z_pts, kwargs)


def create_mdg(
    grid_type: Literal["simplex", "cartesian", "tensor_grid"],
    meshing_args: dict,
    fracture_network: FractureNetwork,
    **kwargs,
) -> pp.MixedDimensionalGrid:
    """Creates a mixed-dimensional grid.

    Note:

    In terms of meshing arguments, we provide two ways of creating md-grids.

        (1) By providing `meshing_args` with the key ``cell_size``, where ``cell_size``
        represents a target cell size and works for all types of grids.

        (2) By providing meshing_args with tailored keys. The keywords will vary depending
        on the type of grid and are meant to provide more flexibility in the meshing
        process, see e.g., the **Parameters** section below. Note that if one of the keys
        is absent, cell_size will be used instead.

        Examples:

            .. code:: python3

                import porepy as pp
                import numpy as np

                # Set a domain
                domain = pp.Domain({"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1})

                # Generate a fracture network
                frac_1 = pp.LineFracture(np.array([[0.2, 0.8], [0.5, 0.5]]))
                frac_2 = pp.LineFracture(np.array([[0.5, 0.5], [0.1, 0.9]]))
                fractures = [frac_1, frac_2]
                fracture_network = pp.create_fracture_network(fractures, domain)

                # Generate a mixed-dimensional grid (MDG)
                meshing_args = {"cell_size": 0.1}
                mdg = pp.create_mdg("simplex", meshing_args, fracture_network)

        See Also:
            - :meth:`~porepy.fracs.fracture_network_2d.FractureNetwork2d.mesh`
            - :meth:`~porepy.fracs.fracture_network_2d.FractureNetwork3d.mesh`
            - :meth:`~porepy.fracs.meshing.cart_grid`
            - :meth:`~porepy.fracs.meshing.tensor_grid`

        Parameters:
            grid_type: Type of grid. Use ``simplex`` for unstructured triangular and
                tetrahedral grids, ``cartesian`` for structured, uniform Cartesian grids,
                and ``tensor_grid`` for structured, non-uniform Cartesian grids.
            meshing_args: A dictionary with meshing keys depending on each grid_type:
                if grid_type == "simplex"
                    cell_size: ``float``: Overall target cell size. It is required, if one
                        of [cell_size_min, cell_size_fracture, cell_size_boundary] is not
                        provided.
                    cell_size_min: ``float``: minimum cell size. If not provided,
                        cell_size will be used for completeness.
                    cell_size_fracture: ``float``: target mesh size close to the fracture.
                        If not provided, cell_size will be used for completeness.
                    cell_size_boundary: ``float``: target mesh size close to the external
                        boundaries (can be seen as a far-field value). If not provided,
                        cell_size will be used for completeness.
                if grid_type == "cartesian"
                    cell_size: ``float``: side length of the grid elements (squares in 2d
                        and cubes 3d). It is required, if one of [cell_size_x, cell_size_y
                        , cell_size_z] is not provided.
                    cell_size_x: ``float``: size in x-direction. If cell_size_x is
                        provided, it overwrites cell_size in the x-direction.
                    cell_size_y: ``float``: size in y-direction. If cell_size_y is
                        provided, it overwrites cell_size in the y-direction.
                    cell_size_z: ``float``: size in z-direction. If cell_size_z is
                        provided, it overwrites cell_size in the z-direction.
                if grid_type == "tensor_grid"
                    cell_size: ``float``: size in all directions. It is required, if one
                        of [x_pts, y_pts, z_pts] is not provided.
                    x_pts: ``np.ndarray``: points in x-direction. The points np.min(x_pts)
                        , np.max(x_pts) must be on the boundary. If x_pts is provided,
                        it overwrites the information computed from cell_size in the
                        x-direction.
                    y_pts: ``np.ndarray``: points in y-direction. The points np.min(y_pts)
                        , np.max(y_pts) must be on the boundary. If y_pts is provided,
                        it overwrites the information computed from cell_size in the
                        y-direction.
                    z_pts: ``np.ndarray``: points in z-direction. The points np.min(z_pts)
                        , np.max(z_pts) must be on the boundary. If z_pts is provided,
                        it overwrites the information computed from cell_size in the
                        z-direction.
            fracture_network: fracture network specification.
            **kwargs: A dictionary with extra meshing keys associated with each grid_type:
                if grid_type == "simplex" see signature for the `mesh` function in:
                    constraints: ``np.ndarray``: Index list of the fractures that should
                        be treated as constraints in meshing, but not added as separate
                        fracture grids (no splitting of nodes etc.). Useful to define
                        subregions of the domain (and assign e.g., sources, material
                        properties, etc.).
                    dfn: ``bool``: Defaults to False. Directive for generating a DFN mesh.
                        Providing True activates the directive.
                if grid_type == "simplex" or "tensor_grid":
                    offset: ``float``: Defaults to 0. Parameter that quantifies a
                        perturbation to nodes around the faces that are split.
                        NOTE: this is only for visualization purposes.

        Raises:
            - TypeError: If invalid arguments types are provided. See validator functions:
                - :meth:`~_validate_args`
                - :meth:`~_validate_args_types`
            - ValueError: If invalid arguments values are provided. See validator
                functions:
                - :meth:`~_validate_args`
                - :meth:`~_validate_grid_type_value`
                - :meth:`~_validate_simplex_meshing_args_values`
                - :meth:`~_validate_cartesian_meshing_args_values`
                - :meth:`~_validate_tensor_grid_meshing_args_values`

        Returns:
            Mixed-dimensional grid object.

    """

    _validate_args(grid_type, meshing_args, fracture_network)

    mdg: pp.MixedDimensionalGrid

    # Unstructured cases
    dim: int = _infer_dimension_from_network(fracture_network)
    if grid_type == "simplex":
        # Elegant solutions can be implemented to made this part more compact.
        # However while running mypy on this file, large type hints for
        # FractureNetwork2d.mesh and FractureNetwork3d.mesh are need.
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

    # Structured cases
    domain: Union[pp.Domain, None] = _retrieve_domain_instance(fracture_network)
    if domain is not None:
        fractures = [f.pts for f in fracture_network.fractures]
        if dim == 3:
            # In 3d the bounding polygons for the fractures are added to the set of
            # fractures in the network. Since we will feed only the fractures, not
            # the fracture network, into the structured mesh generator, we need to
            # filter out those fractures that are tagged as being part of the
            # boundary.
            fractures = [
                f.pts
                for (fi, f) in enumerate(fracture_network.fractures)
                if not fracture_network.tags["boundary"][fi]
            ]

        if grid_type == "cartesian":
            (nx_cells, phys_dims, kwargs) = _preprocess_cartesian_args(
                domain, meshing_args, kwargs
            )
            mdg = pp.meshing.cart_grid(
                fracs=fractures, nx=nx_cells, physdims=phys_dims, **kwargs
            )

        if grid_type == "tensor_grid":
            (xs, ys, zs, kwargs) = _preprocess_tensor_grid_args(
                domain, meshing_args, kwargs
            )
            mdg = pp.meshing.tensor_grid(fracs=fractures, x=xs, y=ys, z=zs, **kwargs)

    return mdg
