from __future__ import annotations

import csv
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import gmsh
import numpy as np
from numpy.typing import ArrayLike

import porepy as pp
from porepy.fracs.fracture_network_2d import FractureNetwork2d
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.fracs.utils import pts_edges_to_linefractures


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
        MAJOR_AXIS, MINOR_AXIS, MAJOR_AXIS_ANGLE, STRIKE_ANGLE, DIP_ANGLE, NUM_POINTS``.
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

    fractures: list[pp.LineFracture] | list[pp.PlaneFracture] = []

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
                    # This can be an elliptic fracture, a 3d domain or a 3d point-based
                    # fracture, depending on the context.
                    nd = 3

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

            # This is a fracture.
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
            else:  # nd == 3
                if data.size == 8:
                    # This will be interpreted as an elliptic fracture. The number of
                    # points should be represented as an integer.
                    frac = pp.create_elliptic_fracture(
                        data[:3], data[3], data[4], data[5], data[6], int(data[7])
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


def dfm_3d_from_fab(
    file_name: Path,
    tol: float = 1e-4,
    domain: Optional[pp.Domain] = None,
    return_domain: bool = False,
    **mesh_kwargs,
) -> Union[pp.MixedDimensionalGrid, tuple[pp.MixedDimensionalGrid, pp.Domain]]:
    """Create the mdg from a set of 3d fractures stored in a fab file and domain.

    Parameters:
        file_name: Name of the file.
        tol: ``default=1e-4``

            Tolerance for the methods.
        domain: ``default=None``

            Domain specification. If not given, the bounding box is considered.
        return_domain:  ``default=False``

            Whether to return the domain.
        mesh_kwargs: ``kwargs`` for the gridding, see e.g.,
            :meth:`~porepy.fracs.simplex.tetrahedral_grid_from_gmsh`.

    Returns:
        The resulting mixed-dimensional grid, and if :attr:`return_domain` is ``True``,
        also the domain.

    """
    msg = "This functionality is deprecated and will be removed in a future version"
    warnings.warn(msg, DeprecationWarning)

    network = network_3d_from_fab(file_name, return_all=False, tol=tol)
    assert isinstance(network, FractureNetwork3d)

    # Compute the domain if not given
    if domain is None:
        domain = pp.Domain(network.bounding_box())

    network.domain = domain
    mdg = network.mesh(mesh_kwargs)

    if return_domain:
        return mdg, domain
    else:
        return mdg


def network_3d_from_fab(
    f_name: Path, return_all: bool = False, tol: Optional[float] = None
) -> Union[FractureNetwork3d, tuple[FractureNetwork3d, list[np.ndarray], np.ndarray]]:
    r"""Create 3D fracture network from a ``.fab`` file, as specified by FracMan.

    The filter is based on the ``.fab``-files available at the time of writing and
    may not cover all options available.

    Note:
        The function also reads various other information of unknown usefulness,
        see implementation for details. This information is currently not returned.

    Parameters:
        f_name: Path to the ``.fab`` file.
        return_all: ``default=False``

            Whether to return additional information (see the Returns section).
        tol: ``default=None``

            Tolerance passed on instantiation of the returned
            :class:`~porepy.fracs.fracture_network_3d.FractureNetwork3d`.

    Returns:
        3D fracture network, and if ``return_all=True`` also

        - A list of numpy arrays of ``shape=(nd, num_points)``, where each
          item of the list contains the fractures cut by the domain boundary,
          represented by their ``num_points`` vertexes.
        - A numpy array, where for each element in the list of numpy arrays from
          above, a :math:`\pm 1` is associated, establishing which boundary the
          fracture is on.

    """
    msg = "This functionality is deprecated and will be removed in a future version"
    warnings.warn(msg, DeprecationWarning)

    def read_keyword(line):
        # Read a single keyword, on the form  key = val
        words = line.split("=")
        key = words[0].strip()
        val = words[1].strip()
        return key, val

    def read_section(f, section_name):
        # Read a section of the file, surrounded by a BEGIN / END wrapping
        d = {}
        for line in f:
            if line.strip() == "END " + section_name.upper().strip():
                return d
            k, v = read_keyword(line)
            d[k] = v

    def read_fractures(f, is_tess=False):
        # Read the fracture
        fracs = []
        fracture_ids = []
        trans = []
        nd = 3
        for line in f:
            if not is_tess and line.strip() == "END FRACTURE":
                return fracs, np.asarray(fracture_ids), np.asarray(trans)
            elif is_tess and line.strip() == "END TESSFRACTURE":
                return fracs, np.asarray(fracture_ids), np.asarray(trans)
            if is_tess:
                ids, num_vert = line.split()
            else:
                ids, num_vert, t = line.split()[:3]

                trans.append(float(t))

            ids = int(ids)
            num_vert = int(num_vert)
            vert = np.zeros((num_vert, nd))
            for i in range(num_vert):
                data = f.readline().split()
                vert[i] = np.asarray(data[1:])

            # Transpose to nd x n_pt format
            vert = vert.T

            # Read line containing normal vector, but disregard result
            data = f.readline().split()
            if is_tess:
                trans.append(int(data[1]))
            fracs.append(vert)
            fracture_ids.append(ids)

    with open(f_name, "r") as f:
        for line in f:
            if line.strip() == "BEGIN FORMAT":
                # Read the format section, but disregard the information for
                # now
                _ = read_section(f, "FORMAT")
            elif line.strip() == "BEGIN PROPERTIES":
                # Read in properties section, but disregard information
                _ = read_section(f, "PROPERTIES")
            elif line.strip() == "BEGIN SETS":
                # Read set section, but disregard information.
                _ = read_section(f, "SETS")
            elif line.strip() == "BEGIN FRACTURE":
                # Read fractures
                fracs, _, _ = read_fractures(f, is_tess=False)
            elif line.strip() == "BEGIN TESSFRACTURE":
                # Read tess_fractures
                tess_fracs, _, tess_sgn = read_fractures(f, is_tess=True)
            elif line.strip() == "BEGIN ROCKBLOCK":
                # Not considered block
                pass
            elif line.strip()[:5] == "BEGIN":
                # Check for keywords not yet implemented.
                raise ValueError("Unknown section type " + line)

    fractures = [pp.PlaneFracture(f) for f in fracs]
    if tol is not None:
        network = pp.create_fracture_network(fractures, tol=tol)
    else:
        network = pp.create_fracture_network(fractures)
    assert isinstance(network, FractureNetwork3d)

    if return_all:
        return network, tess_fracs, tess_sgn
    else:
        return network
