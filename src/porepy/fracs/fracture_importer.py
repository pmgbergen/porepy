from __future__ import annotations

import csv
import warnings
from contextlib import contextmanager
from logging import getLogger
from pathlib import Path
from typing import Optional, Union

import gmsh
import numpy as np
from numpy.typing import ArrayLike

import porepy as pp
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.fracs.utils import pts_edges_to_linefractures

logger = getLogger(__name__)


def network_from_csv(
    file_name: Path, has_domain: bool = True, tol: float = 1e-4, **kwargs
) -> FractureNetwork2d | FractureNetwork3d:
    """Create the fracture network from a CSV file.

    The file is assumed to have the following structure:
    - If has_domain is True, the first line describes the domain as a cuboid with
        ``X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX`` for 3D or ``X_MIN, Y_MIN,
        X_MAX, Y_MAX`` for 2D.
    - In 2D, the remaining lines describe the fractures as a list of points (one line
        per fracture) ``START_X, START_Y, END_X, END_Y``.
    - In 3D, Polygonal fractures are described as a list of points (one line per
        fracture) ``P0_X, P0_Y, P0_Z, ..., PN_X, PN_Y, PN_Z``.
        Elliptic fractures are described as ``CENTER_X, CENTER_Y, CENTER_Z,
        MAJOR_AXIS, MINOR_AXIS, MAJOR_AXIS_ANGLE, STRIKE_ANGLE, DIP_ANGLE``.
    Lines starting with ``#`` will be ignored.

    Parameters:
        file_name: Path to the CSV file.
        has_domain: Whether the first line in the CSV file specifies the domain.
            Defaults to True.
        tol: Geometric tolerance used in the computations. Defaults to 1e-4.
        **kwargs: Keyword arguments passed to
            :meth:`~porepy.fracs.fracture_network_2d.FractureNetwork2d` or
            :meth:`~porepy.fracs.fracture_network_3d.FractureNetwork3d`.

    Raises:
        ValueError: If the CSV file contains no data.
        ValueError: If lines in the CSV file have an invalid number of entries.

    Returns:
        The loaded fracture network.

    """
    # Marker for whether the file contains any non-comment content.
    has_nontrivial_content = False
    # Marker for whether the domain line has been read.
    domain_read = False
    # The dimension of the network. Set to None, but inferred from the first non-comment
    # line.
    nd = None

    fractures: list[pp.LineFracture] | list[pp.PlaneFracture | pp.EllipticFracture] = []

    with open(file_name, "r") as csv_file:
        while True:
            line = csv_file.readline()
            if not line:
                # End of file.
                break
            if line.startswith("#") or line.strip() == "":
                # Skip comments and empty lines.
                continue

            # There is data to be read, the file is not trivial.
            has_nontrivial_content = True
            data = np.array([line.strip().split(",")], dtype=float).ravel("F")
            if nd is None:
                if data.size == 4:
                    # Both 2d box domains and fractures have four entries.
                    nd = 2
                else:
                    # This can be a 3d domain (six entries) or a 3d point-based
                    # fracture (elliptic or point-based), depending on the context.
                    if (
                        data.size == 6  # Domain in 3d
                        or data.size == 8  # Elliptic fracture in 3d
                        or (
                            data.size >= 9 and data.size % 3 == 0
                        )  # Point-based fracture in 3d
                    ):
                        nd = 3
                    else:
                        raise ValueError(
                            "Could not infer dimension from data, data size: "
                            + f"{data.size}."
                        )

            if has_domain and not domain_read:
                # Read the domain line.
                domain_points = data.ravel()
                domain_read = True
                if nd == 2 and domain_points.size != 4:
                    raise ValueError(
                        "Domain line should have four entries in 2d, but has "
                        + f"{domain_points.size}."
                    )
                elif nd == 3 and domain_points.size != 6:
                    raise ValueError(
                        "Domain line should have six entries in 3d, but has "
                        + f"{domain_points.size}."
                    )
                continue

            # This is a fracture. Process the data according to the spatial dimension.
            if nd == 2:
                if data.size != 4:
                    raise ValueError(
                        "Fracture line should have four entries in 2d, but has "
                        + f"{data.size}."
                    )
                # Mypy does not understand that fractures will only contain
                # LineFractures in this branch (nd does not change after being set).
                fractures.append(
                    pp.LineFracture(data.reshape((2, -1), order="F"))  # type: ignore
                )
            elif nd == 3:
                # In 3d, the number of entries must be used to distinguish between
                # elliptic and polygonal fractures.

                if data.size == 8:
                    # This will be interpreted as an elliptic fracture. The number of
                    # points should be represented as an integer.
                    frac = pp.EllipticFracture(
                        data[:3],  # center
                        data[3],  # major axis
                        data[4],  # minor axis
                        data[5],  # major axis angle
                        data[6],  # strike angle
                        data[7],  # dip angle
                    )
                    fractures.append(frac)  # type: ignore
                else:
                    if data.size < 9 or not data.size % 3 == 0:
                        raise ValueError(
                            "Fracture line should at least 9 and a multiple of 3"
                            f" entries in 3d, but has {data.size}."
                        )
                    fractures.append(
                        pp.PlaneFracture(  # type: ignore
                            data.reshape((3, -1), order="F")
                        )
                    )
            else:
                # This should not happen.
                raise ValueError("Could not infer dimension from data.")

    if not has_nontrivial_content:
        raise ValueError("The CSV file contains no data.")

    if has_domain:
        if nd == 2:
            domain = {
                "xmin": domain_points[0],
                "xmax": domain_points[2],
                "ymin": domain_points[1],
                "ymax": domain_points[3],
            }
        else:  # nd == 3
            domain = {
                "xmin": domain_points[0],
                "xmax": domain_points[3],
                "ymin": domain_points[1],
                "ymax": domain_points[4],
                "zmin": domain_points[2],
                "zmax": domain_points[5],
            }

    return pp.create_fracture_network(
        fractures, pp.Domain(domain) if has_domain else None, tol=tol
    )


def dfm_from_gmsh(file_name: Path, dim: int, **kwargs) -> pp.MixedDimensionalGrid:
    """Generate a mixed-dimensional grid from a gmsh file.

    If the provided extension of the input file for gmsh is ``.geo`` (not ``.msh``),
    gmsh will be called to generate the mesh before the mixed-dimensional grid is
    constructed.

    Parameters:
        file_name:
            Name of gmsh *in* and *out* file. Should have extension ``.geo`` or
            ``.msh``. In the former case, gmsh will be called upon to generate the
            mesh before the mixed-dimensional mesh is constructed.
        dim:
            Dimension of the problem. Should be 2 or 3.
        **kwargs:
            Optional keyword arguments.

            See :meth:`~porepy.fracs.fracture_network_2d.FractureNetwork2d.mesh` for
            ``dim=2``,
            and :meth:`~porepy.fracs.fracture_network_3d.FractureNetwork3d.mesh` for
            ``dim=3``.

    Returns:
        Mixed-dimensional grid as contained in the gmsh file.
        The physical names are stored in pp.Grid.tags of the subdomains.
    """

    # Run gmsh to create .msh file.
    if file_name.suffix == ".msh":
        out_file = file_name
    else:
        if file_name.suffix == ".geo":
            file_name = file_name.with_suffix("")
        in_file = file_name.with_suffix(".geo")
        out_file = file_name.with_suffix(".msh")

        # Initialize gmsh.
        gmsh.initialize()
        # Reduce verbosity.
        gmsh.option.setNumber("General.Verbosity", 3)
        # Read the specified file.
        gmsh.merge(str(in_file))

        # Generate mesh and write.
        gmsh.model.mesh.generate(dim=dim)
        gmsh.write(str(out_file))

        # Wipe Gmsh's memory
        gmsh.finalize()

    if dim == 2:
        subdomains = pp.fracs.simplex.triangle_grid_from_gmsh(out_file, **kwargs)
    elif dim == 3:
        subdomains = pp.fracs.simplex.tetrahedral_grid_from_gmsh(
            file_name=out_file, **kwargs
        )
    else:
        raise ValueError(f"Unknown dimension, dim: {dim}")
    return pp.meshing.subdomains_to_mdg(subdomains, **kwargs)
