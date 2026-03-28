"""This module defines the abstract base class for fracture networks. Concrete
implementations are found in derived classes in the same subpackage.

Additionally, the module contains helper classes for mesh size control point insertion
and identification of Gmsh point indices.
"""

from __future__ import annotations

import copy
import heapq
import logging
import multiprocessing
import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union, cast

import gmsh
import numpy as np
from matplotlib import pyplot as plt

import porepy as pp

logger = logging.getLogger(__name__)


class FractureNetwork(ABC):
    """Abstract base class for fracture networks."""

    def __init__(
        self,
        nd: Literal[2, 3],
        domain: Optional[pp.Domain] = None,
        tol: float = 1e-8,
    ) -> None:
        self.nd = nd
        """Number of spatial dimensions (2 or 3)."""

        self.domain: Optional[pp.Domain] = domain
        """Domain specification for the fracture network."""

        self._tol = tol
        """Tolerance for geometric computations."""

        self.fractures: (
            list[pp.LineFracture] | list[pp.PlaneFracture | pp.EllipticFracture]
        ) = []
        """List of fractures in the network. Will be populated in derived classes."""

        self._extra_meshing_args: dict[str, float | int] = {
            # See the usage in code for definition of this parameter.
            "extend_mesh_size_from_boundary": 0,
            # Use Frontal-Delaunay (5) for 2d, HXT (10) for 3d as default meshing
            # algorithms.
            "meshing_algorithm": 10 if self.nd == 3 else 5,
            # Use a single processor by default to ensure deterministic meshing.
            "num_processors": 1,
            # Set a moderate verbosity level for Gmsh output. See the Gmsh documentation
            # for details on the available levels.
            "gmsh_verbosity_level": 3,
            # Whether to plot mesh quality metrics after meshing. This is set to False
            # by default.
            "plot_mesh_quality_metrics": False,
        }
        """Extra meshing arguments for fracture network meshing.

        This dictionary is meant to store extra meshing arguments that can be used for
        customizing the meshing process. While some such arguments get their defaults
        defined in the module 'mdg_generation', this dictionary defines what is
        considered more exotic arguments that are not commonly used, and therefore not
        exposed at a higher level.

        The default arguments can be overridden by providing keyword arguments to the
        mesh method; the override happens in the method _prepare_mesh_inputs.

        More information on the individual arguments may be found in the code where they
        are used.

        """

    def num_frac(self) -> int:
        """Return the number of fractures in the network."""
        return len(self.fractures)

    @abstractmethod
    def domain_to_gmsh(self) -> int:
        """Define the domain in gmsh."""
        pass

    def fractures_to_gmsh(self) -> list[int]:
        """Export the fractures to Gmsh using the OpenCASCADE kernel.

        Returns:
            A list of gmsh tags corresponding to the fractures in the network.

        """
        fracture_tags = [fracture.fracture_to_gmsh() for fracture in self.fractures]

        return fracture_tags

    @abstractmethod
    def mesh(
        self,
        mesh_args: dict[str, float],
        file_name: Optional[Path] = None,
        constraints: Optional[np.ndarray] = None,
        dfn: bool = False,
        **kwargs,
    ) -> pp.MixedDimensionalGrid:
        """Generate a mixed-dimensional grid by meshing the fracture network.

        Parameters:
            mesh_args: Dictionary with mesh size parameters. See
                :class:`~porepy.fracs.fracture_network.MeshSizeComputer` for details.
            file_name: Path to the output Gmsh .msh file. If ``None``, the default name
                ``gmsh_frac_file.msh`` is used.
            constraints: Numpy array with indices of fractures to be treated as
                constraints during meshing. The indices refer to the ordering of
                fractures in the fracture network. If ``None``, no constraints are
                applied.
            dfn: If ``True``, a discrete fracture network (DFN) style meshing is
                performed, where only the fractures are meshed (no volume mesh is
                created).
            **kwargs: Additional keyword arguments passed to Gmsh.

        Returns:
            A :class:`~porepy.meshing.mixed_dimensional_grid.MixedDimensionalGrid`
            representing the meshed fracture network.

        """
        pass

    def mesh_quality_metrics(self) -> None:
        """Visualize and log elementwise mesh quality metrics using gmsh.

        The evaluated metrics include:
            - minDetJac / maxDetJac : Minimum and maximum determinant of the Jacobian.
            - minSJ                 : Minimum scaled Jacobian (element regularity).
            - minSICN / minSIGE     : Inverse condition numbers measuring element
                                      skewness.
            - gamma                 : Shape quality factor (close to 1 → good element).
            - innerRadius / outerRadius : Ratio of inscribed to circumscribed radii.
            - minIsotropy           : Degree of isotropy (1 → perfectly isotropic).
            - angleShape            : Angular distortion indicator.
            - minEdge / maxEdge     : Minimum and maximum edge lengths.
            - volume                : Element area or volume measure.

        """
        # Compute mesh quality metrics using gmsh.
        all_element_tags = gmsh.model.mesh.getElements(2)[1][0]
        quality_types = [
            "minDetJac",
            "maxDetJac",
            "minSJ",
            "minSICN",
            "minSIGE",
            "gamma",
            "innerRadius",
            "outerRadius",
            "minIsotropy",
            "angleShape",
            "minEdge",
            "maxEdge",
            "volume",
        ]
        results = {}
        for qtype in quality_types:
            try:
                qvalues = gmsh.model.mesh.getElementQualities(all_element_tags, qtype)
                if len(qvalues) > 0:
                    results[qtype] = qvalues
            except Exception as e:
                print(f"Skipping {qtype}: {e}")

        # Plot histogram of mesh quality metrics.
        n = len(results)
        cols = 5
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = axes.flatten()

        for ax, (qtype, qvalues) in zip(axes, results.items()):
            ax.hist(
                qvalues,
                bins=30,
                color="#4C72B0",
                edgecolor="white",
                alpha=0.8,
            )

            ax.set_title(qtype, fontsize=11, fontweight="bold")
            ax.set_xlabel("Value", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.4)

        # Hide unused subplots
        for ax in axes[len(results) :]:
            ax.axis("off")

        fig.suptitle("Mesh Quality Distributions", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

        # Log mesh quality metrics.
        for qtype, qvalues in results.items():
            if len(qvalues) > 0:
                logger.info(
                    f"{qtype:15s}: min = {qvalues.min():.4e}, max = {qvalues.max():.4e}"
                    + f"avg = {qvalues.mean():.4e}, std = {qvalues.std():.4e}"
                )
            else:
                logger.info(f"{qtype:15s}: (no values returned)")

    def _prepare_mesh_inputs(
        self,
        file_name: Optional[Path] = None,
        constraints: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """Prepare inputs for the meshing process.

        Parameters:
            file_name: Optional path to the Gmsh mesh file to be created.
            constraints: Optional array of fracture indices to be constrained during
                meshing.
            **kwargs: Additional keyword arguments, including:
                - num_processors: Number of processors to use during meshing. If
                  ``None``, the default is to use all available processors minus two.
        Returns:
            A tuple containing:
            - file_name: The prepared file name for the Gmsh mesh file.
            - constraints: The prepared array of fracture indices to be constrained.
        """
        if file_name is None:
            file_name = Path("gmsh_frac_file.msh")
        file_name = file_name.with_suffix(".msh")

        if constraints is None:
            constraints = np.array([], dtype=int)
        else:
            constraints = np.atleast_1d(constraints)
            constraints.sort()

        # Update extra meshing arguments based on provided kwargs.
        for key in self._extra_meshing_args:
            if key in kwargs:
                self._extra_meshing_args[key] = kwargs[key]

        # Set the number of processors for Gmsh.
        gmsh.option.setNumber(
            "General.NumThreads", self._extra_meshing_args["num_processors"]
        )
        gmsh.option.setNumber(
            "General.Verbosity", self._extra_meshing_args["gmsh_verbosity_level"]
        )
        # Set Gmsh to use the same geometric tolerance as the fracture network. This
        # should maximize the chances that the PorePy and Gmsh algorithms are consistent
        # in this manner (though, from bitter experience regarding the complexity of
        # this problem, EK will not vouch for the implementation actually being
        # consistent in all relevant cases).
        gmsh.option.setNumber("Geometry.Tolerance", self._tol)

        # See the Gmsh documentation for an overview of the available algorithms.
        meshing_algorithm = self._extra_meshing_args.get("meshing_algorithm", 10)
        if self.nd == 3:
            # Implementation note: It is important to use Mesh.Algorithm3D, not
            # Mesh.Algorithm.
            gmsh.option.setNumber("Mesh.Algorithm3D", meshing_algorithm)
        else:
            gmsh.option.setNumber("Mesh.Algorithm", meshing_algorithm)

        return file_name, constraints

    def _fragment_fractures(self, fracture_tags: list[int], domain_tag: int):
        """Helper method to do fracture fragmentation.

        Parameters:
            fracture_tags: List of gmsh indices representing fractures.
            domain_tag: Gmsh tag of the domain.

        Returns:
            Intersection information, see the Gmsh manual or usage elsewhere in this
            code for more information.

        """
        dim_fracture_tags = [(self.nd - 1, tag) for tag in fracture_tags]

        if domain_tag >= 0:
            _, isect_mapping = gmsh.model.occ.fragment(
                dim_fracture_tags,
                [(self.nd, domain_tag)],
                removeObject=True,
                removeTool=True,
            )
        else:
            # Special handling of DFN-style meshing.
            # No intersections possible with only one fracture and no domain.
            if len(fracture_tags) == 1:
                # Gmsh did not seem to like fragmenting a single object without a
                # secondary object. Hence, we handle this case separately.
                isect_mapping = [[(self.nd - 1, fracture_tags[0])]]
            else:
                _, isect_mapping = gmsh.model.occ.fragment(
                    dim_fracture_tags, [], removeObject=True, removeTool=True
                )

        gmsh.model.occ.synchronize()
        return isect_mapping

    def _entity_on_domain_boundary(self, target_dim: int, ind: list[int]) -> bool:
        """Helper function to determine if an entity lies on the domain boundary.

        The intended use is to determine if a line or set of points lie on the boundary
        of the domain, in which case it should not be considered an intersection line or
        point between fractures.

        The implementation could have been generalized in various ways, but is kept as
        it is, since it concerns very specific use cases that are covered by the current
        implementation.

        Known possible issue: If the domain is fully split by a fracture, the original
        boundary sides of the domain may have been split into multiple parts (this will
        manifest as the variable boundary_surfaces containing more surfaces than the
        original domain boundary, for instance more than six surfaces for a box domain).
        In this case, a line that extends over multiple of these 'sub-sides', but is
        really part of one side in the original boundary definition, may be
        misidentified as an intersection line. In EK's understanding, this should not
        happen, since that line will also have been split into multiple parts during
        fragmentation, and each part will be on one of the sub-sides. Still, Gmsh works
        in mysterious ways, so it was considered wise to take note of this possible
        issue.

        Parameters:
            dim: Dimension of the entity to check (0 for points, 1 for line).
            ind: List of Gmsh tags identifying the entity to check.

        Returns:
            bool: ``True`` if the entity lies on a single part (a single member of
                the boundary_surfaces, see below), ``False`` otherwise.

        """
        assert target_dim <= 1, "The implementation is not intended for surfaces."
        # Get the domain boundary surfaces, accounting for the domain possibly
        # having been split into multiple parts.
        domain_entities = gmsh.model.get_entities(self.nd)
        boundary_entities = gmsh.model.get_boundary(
            [(self.nd, tag) for _, tag in domain_entities]
        )
        # Get hold of the boundary points of the entity to check.
        if target_dim == 0:
            boundary_points = [(target_dim, i) for i in ind]
        else:
            assert len(ind) == 1, "Only single entity indices are supported."
            boundary_points = gmsh.model.get_boundary([(target_dim, ind[0])])

        # For each boundary surface of the domain, compute the distance between the
        # entity and all boundary points to check if they are all zero.
        # Note that the other way around (checking if each entity point is on any of the
        # boundary entities) risks false positives for a line extending across the
        # domain between two boundaries.
        for ent in boundary_entities:
            dist = [gmsh.model.occ.get_distance(*bp, *ent)[0] for bp in boundary_points]
            # It is not 100% clear what an empty list of boundary points (i.e.,
            # len(dist) == 0) implies - the case arose while working with disc
            # fractures. However, it seems safest that the lack of boundary points does
            # not automatically lead to the fracture being classified as being on the
            # boundary, hence we rule out this case.
            if len(dist) > 0 and np.all(np.array(dist) < self._tol):
                return True
        # Having come this far, the entity is not on the domain boundary.
        return False

    def _insert_mesh_size_control_points(
        self, mesh_size_computer: MeshSizeComputer
    ) -> dict[int, list[tuple[np.ndarray, float]]]:
        """Insert control points for mesh size specification on fractures and
        boundaries.

        The method identifies points on fracture surfaces and domain boundaries where
        mesh size control points should be inserted. Later in the meshing process Gmsh
        mesh size fields will be assigned based on the distances from these points to
        surrounding objects. For a detailed description of the approach, see the
        documentation of the MeshSizeComputer class.

        NOTE: The mesh size control points are identified in terms of their coordinates,
        not their Gmsh tags. This is a pragmatic choice to avoid keeping track of point
        tags during the various operations in Gmsh, which seems futile to EK.

        Parameters:
            mesh_size_computer: Instance of MeshSizeComputer providing the mesh size
                parameters.

        Returns:
            dict: A dictionary mapping Gmsh entity tags to lists of tuples, each
                  containing the coordinates of an inserted mesh size control point
                  and its distance to the nearest other fracture or boundary.

        """

        ### Get hold of entities representing fractures and boundaries.
        domain_entities = gmsh.model.get_entities(self.nd)
        boundaries = gmsh.model.get_boundary(
            [(self.nd, tag) for _, tag in domain_entities]
        )
        fractures = [
            f for f in gmsh.model.get_entities(self.nd - 1) if f not in boundaries
        ]
        boundary_tags = set(tag for _, tag in boundaries)
        fracture_tags = set(tag for _, tag in fractures)
        entities = set(tag for _, tag in gmsh.model.get_entities(self.nd - 1))

        # Note to self: keeping track of gmsh tags of points is futile. Instead, we need
        # to identify points by their coordinates, and do a tolerance-based search.
        mesh_size_points: dict[int, list[tuple[np.ndarray, float]]] = {}
        for f in fracture_tags | boundary_tags:
            mesh_size_points[f] = []

        # To avoid inserting the same point multiple times on the same line, and to
        # prune doubly defined points from the gmsh specification, we keep track of
        # which points have already been inserted where.

        # Coordinates of the inserted mesh size control points.
        inserted_points: list[np.ndarray] = []
        # Coordinates of the mesh size control points already inserted. Used to avoid
        # duplicates.
        inserted_on_entity: list[int] = []

        # Take note of the boundary points of all entities, to avoid inserting points
        # there (doing so may confuse Gmsh).
        for ent in entities:
            bp = gmsh.model.get_boundary([(self.nd - 1, ent)], recursive=True)
            for b in bp:
                if b[0] != 0:
                    continue
                coord = gmsh.model.occ.get_bounding_box(*b)[:3]
                inserted_points.append(np.array(coord))
                inserted_on_entity.append(ent)

        # Create helper object responsible for computing the points to be inserted.
        inserter = MeshSizeControlPointInserter(self.nd, mesh_size_computer)

        def point_already_present(pt: np.ndarray, ind: int) -> tuple[bool, bool]:
            """Check if a point is already present among the inserted points.

            Parameters:
                pt: Coordinates of the point to be checked.
                end: Gmsh tag of the entity where the point is to be inserted.

            Returns:
                A tuple of three elements:
                - A boolean indicating whether the point is already present within
                  tolerance.
                - A boolean indicating whether the point is already present on the
                  specified entity.

            """
            if len(inserted_points) == 0:
                return False, False
            dists = np.linalg.norm(
                np.array(inserted_points) - np.array(pt).reshape((1, 3)), axis=1
            )
            i = np.argmin(dists)
            return dists[i] < self._tol, inserted_on_entity[i] == ind

        def insert_point(
            frac: int, points: list[tuple[int, np.ndarray, float]]
        ) -> None:
            """Insert mesh size control points on a fracture or boundary.

            Parameters:
                frac: Gmsh tag of the fracture or boundary where points are to be
                    inserted.
                points: List of tuples, each containing:
                    - Gmsh tag of the point to be inserted.
                    - Coordinates of the point to be inserted.
                    - Distance from the point to the nearest other fracture or boundary.

            The method inserts the specified mesh size control points into the
            dictionary mesh_size_points, and also keeps track of the inserted points to
            avoid duplicates.

            """
            for pi, pt, dist in points:
                point_present, on_entity = point_already_present(pt, frac)
                if point_present and on_entity:
                    # The point is already present, thus there will be a mesh size field
                    # for it on this entity. Remove the newly created point.
                    gmsh.model.occ.remove([(0, pi)])
                    continue
                # The mesh size control point is to be kept.
                mesh_size_points[frac].append((np.array(pt), dist))
                # Keep track of the inserted point, so that we avoid duplicates.
                inserted_points.append(np.array(pt))
                inserted_on_entity.append(frac)

        # Loop over all pairs of entities, compute distances and insert points as
        # needed.
        for f_0, f_1 in combinations(entities, 2):
            if f_0 in boundary_tags and f_1 in boundary_tags:
                # No refinement between two boundary lines.
                continue

            distances = gmsh.model.occ.getDistance(self.nd - 1, f_0, self.nd - 1, f_1)

            if distances[0] > mesh_size_computer.refinement_threshold():
                continue

            # Compute the points to be inserted on both fractures. Insert them.
            points_0, points_1 = inserter.compute_points(
                f_0,
                f_1,
                distances[1:4],
                distances[4:7],
                distances[0],
                f_0 in fracture_tags,
                f_1 in fracture_tags,
            )
            insert_point(f_0, points_0)
            insert_point(f_1, points_1)
            gmsh.model.occ.synchronize()

        return mesh_size_points

    def _assign_distance_based_mesh_size_field(
        self,
        entity: int,
        points: np.ndarray,
        dist: np.ndarray,
        mesh_size_computer: MeshSizeComputer,
        gmsh_point_finder: GmshPointIdentifier,
        is_boundary: bool,
        on_lower_dim: bool,
        surface_lines: Optional[list[int]] = None,
    ) -> list:
        """Assign mesh size field based on distances from points to fractures.

        The mesh size field is either restricted to the entity itself (if
        ``on_lower_dim=True``), or set in the surrounding domain (if
        ``on_lower_dim=False``).

        Parameters:
            entity: Gmsh tag identifying the entity where the mesh size field is to
                be applied. The entity is of dimension self.nd - 1.
            points: Array containing the coordinates of the N points on the entity where
                distances have been computed.
            dist: Array containing the distances from the points to other entities.
            mesh_size_computer: Instance of MeshSizeComputer providing the mesh size
                parameters.
            gmsh_point_finder: Instance of GmshPointIdentifier for mapping points to
                Gmsh indices.
            is_boundary: ``True`` if the entity is on the domain boundary, ``False``
                otherwise.
            on_lower_dim: ``True`` if the mesh size is to be restricted to the entity
                itself (codimension 1), ``False`` if the mesh size is to be set in the
                surrounding domain.
            surface_lines: (3D only) List of Gmsh tags identifying lines on the surface,
                e.g., intersection lines with other fractures. The mesh size field will
                also be applied on these lines.

        Returns:
            list: List of Gmsh size fields.

        """

        gmsh_fields = []
        # Get all domain entities to be used in the restriction step.
        domain_entities = gmsh.model.get_entities(self.nd)

        if self.nd == 3:
            entity_str = "SurfacesList"
            domain_str = "VolumesList"
        else:
            entity_str = "CurvesList"
            domain_str = "SurfacesList"

        # Loop over all points where distances have been computed, assign a mesh size
        # field based on the distance at that point, unless the distance is larger than
        # the refinement threshold.
        #
        # Implementation note: The points must be handled one by one, since each point
        # have a different distance to other fractures, leading to different mesh size
        # specifications.
        for i, d in enumerate(dist):
            if d > mesh_size_computer.refinement_threshold():
                # No refinement needed at this point.
                continue

            # Set distance field for the given point, then a threshold field to set the
            # mesh size based on the distance, and finally a restriction to either the
            # entity itself (on_lower_dim=True) or the surrounding domain
            # (on_lower_dim=False).
            pi = gmsh_point_finder.index(points[:, i])
            field = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(field, "PointsList", [pi])

            threshold = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(threshold, "InField", field)
            gmsh.model.mesh.field.setNumber(
                threshold, "DistMin", mesh_size_computer.dist_min(d)
            )
            gmsh.model.mesh.field.setNumber(
                threshold, "SizeMin", mesh_size_computer.size_min(d)
            )
            if on_lower_dim:
                gmsh.model.mesh.field.setNumber(
                    threshold,
                    "DistMax",
                    mesh_size_computer.dist_farfield(is_boundary, on_lower_dim=True),
                )
                gmsh.model.mesh.field.setNumber(
                    threshold,
                    "SizeMax",
                    mesh_size_computer.h_end(is_boundary),
                )
            else:
                gmsh.model.mesh.field.setNumber(
                    threshold,
                    "DistMax",
                    mesh_size_computer.dist_farfield(is_boundary, on_lower_dim=False),
                )
                gmsh.model.mesh.field.setNumber(
                    threshold, "SizeMax", mesh_size_computer.h_background()
                )

            # Implementation note: The order is important here - the restriction must
            # refer to the threshold field, not the other way around.
            restriction = gmsh.model.mesh.field.add("Restrict")
            gmsh.model.mesh.field.setNumber(restriction, "InField", threshold)
            if on_lower_dim:
                if self.nd == 3 and surface_lines is not None:
                    # Intersection lines on the surface should also have the same mesh
                    # size field applied.
                    gmsh.model.mesh.field.setNumbers(
                        restriction, "CurvesList", surface_lines
                    )
                gmsh.model.mesh.field.setNumbers(restriction, entity_str, [entity])
            else:
                gmsh.model.mesh.field.setNumbers(
                    restriction,
                    domain_str,
                    [entity[1] for entity in domain_entities],
                )
            gmsh_fields.append(restriction)

        return gmsh_fields

    def _set_uniform_mesh_field(
        self,
        entities: list[int],
        mesh_size_computer: MeshSizeComputer,
        boundary_tags: set[int],
        on_lower_dim: bool,
    ) -> list:
        """Set uniform mesh size fields on the given entities.

        The mesh is either restricted to the entities themselves (if
        ``on_lower_dim=True``), or in the surrounding domain (if
        ``on_lower_dim=False``). In the former case, the mesh size is constant on the
        entities, while in the latter case, the mesh size transitions from a fine mesh
        size close to the entities to the background mesh size.

        Parameters:
            entities: Set of Gmsh tags identifying the entities where the mesh size
                field should be applied. The entities are of dimension self.nd - 1.
            mesh_size_computer: Instance of MeshSizeComputer providing the mesh size
                parameters.
            boundary_tags: Set of Gmsh tags identifying the boundary entities of the
                domain.
            on_lower_dim: ``True`` if the mesh size is to be restricted to the entities
                themselves (codimension 1), ``False`` if the mesh size is to be set in
                the surrounding domain.

        Returns:
            list: Updated list of Gmsh size fields including the newly created ones.

        """
        gmsh_fields = []

        if self.nd == 3:
            entity_str = "SurfacesList"
            domain_str = "VolumesList"
        else:
            entity_str = "CurvesList"
            domain_str = "SurfacesList"
        if on_lower_dim:
            # This will set a uniform mesh size on the entities themselves. Set the
            # mesh size on boundary and interior entities separately.
            for is_boundary in [True, False]:
                uniform_field = gmsh.model.mesh.field.add("Constant")
                loc_entities = [
                    ent for ent in entities if (ent in boundary_tags) == is_boundary
                ]
                # Assign the entities to the field.
                gmsh.model.mesh.field.setNumbers(
                    uniform_field, entity_str, loc_entities
                )
                Vin = mesh_size_computer.h_end(is_boundary)
                gmsh.model.mesh.field.setNumber(uniform_field, "VIn", Vin)
                # Set the mesh size outside the entities to the background size. Since
                # we will take the minimum over all mesh size fields later, this will in
                # practice not affect the mesh size outside the entities, but it seems
                # Gmsh requires that a value is set.
                gmsh.model.mesh.field.setNumber(
                    uniform_field, "VOut", mesh_size_computer.h_background()
                )
                restriction = gmsh.model.mesh.field.add("Restrict")
                gmsh.model.mesh.field.setNumber(restriction, "InField", uniform_field)
                gmsh.model.mesh.field.setNumbers(restriction, entity_str, loc_entities)
                gmsh_fields.append(restriction)
        else:
            # This will set a mesh size field in the surrounding domain, transitioning
            # from a fine mesh size close to the entities to the background mesh size.

            # Get all domain entities to be used in the restriction step.
            domain_entities = gmsh.model.get_entities(self.nd)
            for ent in entities:
                # The below code sets up three nested fields, with the following logic:
                # 1. A Distance field computing the distance from the entity.
                # 2. A Threshold field setting the mesh size according to the distance
                #    from the entity. The interpretation of the parameters is documented
                #    in the MeshSizeComputer class.
                # 3. A Restrict field restricting the Threshold field to the domain.
                #
                # EK is fairly confident that the fields must be composed in this way.
                field = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(field, entity_str, [ent])
                threshold = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(threshold, "InField", field)
                gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0)
                gmsh.model.mesh.field.setNumber(
                    threshold,
                    "SizeMin",
                    mesh_size_computer.h_end(ent in boundary_tags),
                )
                gmsh.model.mesh.field.setNumber(
                    threshold,
                    "DistMax",
                    mesh_size_computer.dist_farfield(
                        ent in boundary_tags, on_lower_dim=False
                    ),
                )
                gmsh.model.mesh.field.setNumber(
                    threshold,
                    "SizeMax",
                    mesh_size_computer.h_end(True),
                )
                restriction = gmsh.model.mesh.field.add("Restrict")
                gmsh.model.mesh.field.setNumber(restriction, "InField", threshold)
                gmsh.model.mesh.field.setNumbers(
                    restriction,
                    domain_str,
                    [entity[1] for entity in domain_entities],
                )
                gmsh_fields.append(restriction)

        return gmsh_fields

    def _set_background_mesh_field(self, gmsh_fields: list[int]) -> None:
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", gmsh_fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        # The background mesh incorporates all mesh size specifications. We turn off
        # most other mesh size options in Gmsh to avoid conflicts, as is recommended in
        # the Gmsh documentation.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        # In cases where the background mesh size is large compared to the fracture
        # size, but where refinement is triggered by fractures close to the boundary, we
        # may end up in situations where the mesh size on the boundary is much smaller
        # than in the interior. To avoid this, the user is given the option to extend
        # the mesh size from the boundary into the domain. This is off by default, in
        # line with the recommendations in the Gmsh documentation.
        gmsh.option.setNumber(
            "Mesh.MeshSizeExtendFromBoundary",
            int(self._extra_meshing_args["extend_mesh_size_from_boundary"]),
        )

    def _uniquify_mesh_size_dictionary(
        self, mesh_size_points: dict[int, list[tuple[np.ndarray, float]]]
    ) -> None:
        """Helper function to uniquify mesh size control points.

        This will remove duplicate points from the mesh size control point dictionary,
        added during the insertion process (presumably for different fractures or
        boundaries that are close to each other). The minimum mesh size among the
        duplicates is kept.

        Parameters:
            mesh_size_points: Dictionary mapping Gmsh entity tags to lists of tuples,
                each containing the coordinates of an inserted mesh size control point
                and its distance to the nearest other fracture or boundary.

        The dictionary is modified in place.
        """
        if len(mesh_size_points) > 0:
            all_pts = []
            mesh_sizes = []
            entity_item_comb = []
            # Loop over all entities and their points, collect all points and mesh
            # sizes.
            for entity, info in mesh_size_points.items():
                for i, d in enumerate(info):
                    all_pts.append(d[0])
                    mesh_sizes.append(d[1])
                    entity_item_comb.append((entity, i))
            if len(all_pts) > 0:
                # Uniquify points, then map back to the entities. The mesh size is set
                # to the minimum among duplicates.
                mesh_size_array = np.array(mesh_sizes)
                _, ind_map, inv_map = pp.array_operations.uniquify_point_set(
                    np.array(all_pts).T, tol=self._tol
                )
                min_size = np.empty(ind_map.size, dtype=float)

                # Loop over unique points, find minimum mesh size among duplicates.
                for i in range(ind_map.size):
                    inds = inv_map == i
                    min_size[i] = np.min(mesh_size_array[inds])

                # Map back to entities.
                for line_ind, pt_ind in enumerate(inv_map):
                    entity = entity_item_comb[line_ind][0]
                    item = entity_item_comb[line_ind][1]
                    mesh_size_points[entity][item] = (
                        mesh_size_points[entity][item][0],
                        min_size[pt_ind],
                    )

    def _subfracture_to_fracture_mapping(
        self,
        isect_mapping: list[list[tuple[int, int]]],
        gmsh_to_porepy_fracture_ind_map: dict[int, int],
    ) -> dict[int, list[int]]:
        """Map information from a subfracture (after intersections have been split) back
        to the full fractures.

        Parameters:
            isect: Intersection information as obtained by Gmsh's fragment method.
            gmsh_to_porepy_fracture_ind_map: Mapping from the Gmsh (1-offset)
                bookkeeping of fractures to PorePy's 0-offset system.

        Returns:
            Dictionary mapping from the PorePy bookkeeping to the Gmsh one. The mapping
            is in general one to many, since in Gmsh, the fractures may be have been
            split into multiple objects due to intersections.

        """
        fracture_to_surface: dict[int, list[int]] = {}
        # Count the number of fracture objects that survived both the fragmentation and
        # the distance-based domain trimming.
        num_real_frac = len(set(gmsh_to_porepy_fracture_ind_map.values()))
        for fracture_group in isect_mapping[:num_real_frac]:
            # A fracture_group here was formed after intersection removal. It may
            # contain either a full fracture, or be one of several subfractures forming
            # a fracture.
            all_fracs = []
            if (
                fracture_group
                and fracture_group[0][1] not in gmsh_to_porepy_fracture_ind_map
            ):
                # Skip fractures on the boundary.
                continue

            for fracture in fracture_group:
                if fracture[0] == self.nd - 1:  # Only nd - 1 objects.
                    all_fracs.append(fracture[1])
            if all_fracs:
                frac_ind = gmsh_to_porepy_fracture_ind_map[all_fracs[0]]
                fracs = fracture_to_surface.get(frac_ind, [])
                fracs.extend(all_fracs)
                fracture_to_surface[frac_ind] = fracs

        return fracture_to_surface


ij = namedtuple("ij", ["i", "j"])
Point = namedtuple("Point", ["x", "y", "z"])


class Direction(Enum):
    WEST = "west"
    EAST = "east"
    SOUTH = "south"
    NORTH = "north"


class MeshSizeControlPointInserter:
    """Helper class to insert points on fracture surfaces for mesh size control.

    This class is used to manage the insertion of points on fracture surfaces in a
    fracture network. The points are used to control the mesh size during the meshing
    process, ensuring that the mesh is refined in areas of interest, such as near
    fracture intersections.

    For each fracture, in a pair of close fractures, a Cartesian grid of mesh size
    control points is constructed on the fracture surface, centered at the closest point
    to the other fracture. In 2d (1d fractures), this grid extends along the fracture
    line, while in 3d (2d fractures), the grid extends in two orthogonal directions on
    the fracture surface. The spacing of the grid points is determined by the desired
    mesh size at the closest point, and the mesh size control points are inserted with
    sufficient density to ensure that the mesh size transitions smoothly from fine to
    coarse away in regions where the fractures are not proximate. In particular, for two
    parallel fractures, the mesh size control points are inserted such that the mesh
    size is uniform along the overlap region between the fractures.

    """

    def __init__(self, nd: int, mesh_size_computer: MeshSizeComputer) -> None:
        self._nd = nd
        self._mesh_size_computer = mesh_size_computer

    def compute_points(
        self,
        f_0: int,
        f_1: int,
        cp_0: list[float],
        cp_1: list[float],
        distance: float,
        f_0_is_fracture: bool,
        f_1_is_fracture: bool,
    ) -> tuple[list, list]:
        """Compute points to be inserted on the surfaces of two fractures.

        Given two fractures and their respective closest points (to the other fracture),
        this method computes the points that need to be inserted on the surfaces of both
        fractures to ensure proper mesh size control.

        See the class documentation for details on the approach.

        Parameters:
            f_0: Gmsh index of the first fracture object.
            f_1: Gmsh index of the second fracture object.
            cp_0: Coordinates of the closest point on the first fracture.
            cp_1: Coordinates of the closest point on the second fracture.
            distance: Minimum distance between the two fractures.
            f_0_is_fracture: Boolean indicating if the first fracture is a fracture
                (as opposed to a boundary).
            f_1_is_fracture: Boolean indicating if the second fracture is a fracture
                (as opposed to a boundary).

        Returns:
            A tuple containing two lists:
                - List of points to be inserted on the first fracture.
                - List of points to be inserted on the second fracture.

        """
        points_0 = self._control_points(f_0, f_1, cp_0, cp_1, distance, f_0_is_fracture)
        points_1 = self._control_points(f_1, f_0, cp_1, cp_0, distance, f_1_is_fracture)
        return points_0, points_1

    def _control_points(
        self,
        f_main: int,
        f_other: int,
        cp_0: list[float],
        cp_1: list[float],
        init_distance: float,
        f_main_is_fracture: bool,
    ):
        """Compute control points to be inserted on a fracture surface.

        Parameters:
            f_main: Gmsh index of the fracture where points are to be inserted.
            f_other: Gmsh index of the other fracture.
            cp_0: Coordinates of the closest point on the main fracture.
            cp_1: Coordinates of the closest point on the other fracture.
            init_distance: Distance between the two closest points.
            f_main_is_fracture: Boolean indicating if the main fracture is a fracture
                (as opposed to a boundary).

        Returns:
            List of points to be inserted on the main fracture.

        """
        # Insert mesh size control points based on the following approach: A
        # Cartesian-like coordinate system is layed out on the main fracture surface,
        # with center at the point on the main fracture which is closest to the other
        # fracture. One axis of the coordinate system is in the direction where the
        # distance increases the slowest, the other (since the fracture is plane) gives
        # the fastest distance increase.
        #
        # Then, starting with the closest point denoted i=0, j=0 (if the fracture is 2d)
        # as the candidate point:
        #
        # 1. The candidate point is considered as a mesh size control point. Whether it
        #    is inserted depends on the mesh size parameters and the angle between the
        #    two fractures (some details are given in the code below, but it may also be
        #    helpful to interpret the mesh size parameters individually to see adding a
        #    mesh size control point will impact the actual meshing algorithm).
        # 2. If a mesh size control point is inserted on (i, j),  all neighboring points
        #    along the grid lines (so, i+1, i-1 and, for 2d surfaces, j+1, j-1) become
        #    new candidate points. However, if we have already visited a given
        #    combination (i, j), it is not regarded again.
        # 3. A specific (i, j) combination can be encountered twice (for instance, go
        #    first left, then up, or first up, then left). Since it is not clear to EK
        #    that the two paths will render the same coordinate for the (i, j)
        #    combination (maybe it is trivially so for plane fractures), we do a check
        #    and pick one closest to (i=0, j=0).
        # 4. The candidate points are visited with priority closest to the origin (i=0,
        #    j=0), using a priority queue (heapq). This order of priority is to some
        #    degree motivated by a feeling it seems right, rather than deep insight into
        #    the algorithm.

        t_i, t_j = self._tangent_basis(f_main, f_other, Point(*cp_0), Point(*cp_1))

        def priority(ij):
            # The priority is given by the Manhattan distance from the origin. With a
            # min-heap as in heapq, this will give priority to points closer to the
            # origin.
            return abs(ij.i) + abs(ij.j)

        # Initialize the priority queue with the first candidate point.
        q: list[tuple[int, ij]] = []
        i = ij(0, 0)
        heapq.heappush(q, (priority(i), i))
        # Table mapping indices to candidate points, previous points, available
        # directions, and distance to the other fracture.

        # Search directions for new candidate control points - see point 2 in the
        # outline of the algorithm. For self.nd == 2, the fracture is a line, and we
        # only search left and right.
        direction = {
            Direction.WEST: True,
            Direction.EAST: True,
        }
        if self._nd == 3:
            direction.update(
                {
                    Direction.SOUTH: True,
                    Direction.NORTH: True,
                }
            )

        tab = {}
        # Seed the algorithm with the initial candidate point. Using the initial point
        # both as the original and the candidate point requires some special handling
        # below, but is needed to achieve a unified treatment.
        tab[i] = (Point(*cp_0), Point(*cp_0), direction, init_distance)
        # List of points to be added, as tuples of (gmsh_index, coordinates, distance).
        points_to_add = []
        # Set of indices that have been discarded.
        discarded_ijs = set()

        # Count the number of iterations. Used for safeguarding.
        iter_counter = 0

        while q:
            iter_counter += 1

            _, i = heapq.heappop(q)
            if i in discarded_ijs:
                continue
            p_cand, p_prev, dirs, _ = tab[i]

            if not gmsh.model.is_inside(self._nd - 1, f_main, p_cand):
                # We are outside the fracture, no need to proceed in this direction.
                discarded_ijs.add(i)
                tab.pop(i)
                continue

            # Gmsh index of the candidate point.
            gmsh_ind = gmsh.model.occ.add_point(*p_cand)
            # Distance from the candidate point to the other fracture.
            dist_other_fracture = gmsh.model.occ.get_distance(
                0, gmsh_ind, self._nd - 1, f_other
            )[0]
            # Distance between the candidate and previous points.
            dist_cand_prev = float(np.linalg.norm(np.array(p_cand) - np.array(p_prev)))

            # Mesh at the candidate point, as determined by the distance from the
            # previous point.
            if dist_cand_prev == 0:
                # This should only happen in the first iteration, when the previous and
                # candidate points have the same coordinates. There is no mesh size from
                # the previous point to compare with (see below if) and a point should
                # be added if the distance to the other fracture justifies it. Thus, we
                # set the mesh size from previous to positive inf to make sure it is not
                # less than the mesh size set according to the distance to the other
                # fracture, as this could have prevented adding the point.
                assert iter_counter == 1
                h_from_prev = np.inf
            else:
                # There is a previous point. Compute the mesh size at the candidate
                # point from the mesh size field centered at this point.
                h_from_prev = self._mesh_size_computer.size_at_distance(
                    dist_other_fracture,
                    dist_cand_prev,
                    f_main_is_fracture,
                    on_lower_dim=True,
                )

            # Check if the new point is so far away from the other surface that no more
            # points are needed, or if the mesh size resulting from inserting a point
            # here will be coarser than the mesh size obtained from the parent point.
            if (
                dist_other_fracture > self._mesh_size_computer.refinement_threshold()
                or self._mesh_size_computer.size_min(dist_other_fracture) > h_from_prev
            ):
                # No need to add more points in this direction.
                gmsh.model.occ.remove([(0, gmsh_ind)])
                discarded_ijs.add(i)
                tab.pop(i)
                continue

            # We have found a new mesh size control point. Register it.
            points_to_add.append((gmsh_ind, p_cand, dist_other_fracture))

            # Define the new candidate points that will have the newly added point as
            # its parent / previous point. The step size is set so that, for parallel
            # fractures, the control points are just close enough to ensure that the
            # mesh size is constant, i.e., we do not enter the transition zone towards
            # mesh sizes given by background conditions (see documentation of the
            # MeshSizeComputer for details). This can be estimated to 2 times the
            # distance between the two fractures at the current point (which will give
            # the correct estimate for parallel fractures, and possibly a somewhat too
            # long step for close to parallel fractures, but we cross our fingers this
            # will work out nicely).
            step_size = 2 * self._mesh_size_computer.dist_min(dist_other_fracture)

            # Loop over all directions, create new candidate points as needed.
            for direct, can_proceed in dirs.items():
                if not can_proceed:
                    continue

                dir_new = copy.copy(dirs)
                if direct == Direction.WEST:
                    # New index.
                    di = ij(i.i - 1, i.j)
                    # Step in the negative tangent direction.
                    delta = -t_i * step_size
                    # If this new candidate point makes it into a control point, it will
                    # be used to generate further candidate points. In that case, we
                    # should not proceed to the east of that point, since that would
                    # lead back to the current point.
                    dir_new[Direction.EAST] = False
                elif direct == Direction.EAST:
                    di = ij(i.i + 1, i.j)
                    delta = t_i * step_size
                    dir_new[Direction.WEST] = False
                elif direct == Direction.SOUTH:
                    assert t_j is not None
                    di = ij(i.i, i.j - 1)
                    delta = -t_j * step_size
                    dir_new[Direction.NORTH] = False
                elif direct == Direction.NORTH:
                    assert t_j is not None
                    di = ij(i.i, i.j + 1)
                    delta = t_j * step_size
                    dir_new[Direction.SOUTH] = False

                if di in discarded_ijs:
                    continue

                # If we reach here, we have a new candidate point.
                p_new = Point(*(np.array(p_cand) + delta))
                dist_new = dist_other_fracture

                if di in tab:
                    # There is already a candidate point in this ij-coordinate. Keep the
                    # one which is closest to the other fracture. EK: With planar
                    # fractures, my feeling is this should be superflouous, since the
                    # two points should have the same coordinates. But better safe than
                    # sorry.
                    p_new, dist_new = self._closest_point(cp_0, p_new, tab[di][0])
                    dir_new = self._direction_intersection(dir_new, tab[di][2])

                tab[di] = (p_new, p_cand, dir_new, dist_new)
                heapq.heappush(q, (priority(di), di))

            # Finished with this candidate point. Remove it from the table.
            discarded_ijs.add(i)
            tab.pop(i)

        return points_to_add

    def _closest_point(
        self, start: Point | list[float], cand_0: Point, cand_1: Point
    ) -> tuple[Point, float]:
        """Among two candidate points, return the one closest to a given start point.

        Parameters:
            start: The reference point.
            cand_0: First candidate point.
            cand_1: Second candidate point.

        Returns:
            The candidate point closest to the start point, and its distance to the
            start point.

        """
        vec_0 = np.array(cand_0) - np.array(start)
        vec_1 = np.array(cand_1) - np.array(start)

        dist_0 = float(np.linalg.norm(vec_0))
        dist_1 = float(np.linalg.norm(vec_1))

        if dist_0 < dist_1:
            return cand_0, dist_0
        else:
            return cand_1, dist_1

    def _point_inside_surface(self, point: Point, surface: int) -> bool:
        """Check if a point is inside another fracture surface.

        Parameters:
            point: Coordinates of the point to be checked.
            surface: Gmsh index of the other fracture surface.

        Returns:
            True if the point is inside the other fracture surface, False otherwise.

        """
        proj_pts, _ = gmsh.model.get_closest_point(self._nd - 1, surface, point)
        return gmsh.model.is_inside(self._nd - 1, surface, proj_pts)

    def _direction_intersection(
        self, dir_0: dict[Direction, bool], dir_1: dict[Direction, bool]
    ) -> dict[Direction, bool]:
        """Combine two direction dictionaries.

        For each direction, the combined dictionary will have True if both input
        dictionaries have True for that direction, and False otherwise.

        Parameters:
            dir_0: First direction dictionary.
            dir_1: Second direction dictionary.

        Returns:
            Combined direction dictionary.

        """
        match self._nd:
            case 2:
                return {
                    Direction.WEST: dir_0[Direction.WEST] and dir_1[Direction.WEST],
                    Direction.EAST: dir_0[Direction.EAST] and dir_1[Direction.EAST],
                }
            case 3:
                return {
                    Direction.WEST: dir_0[Direction.WEST] and dir_1[Direction.WEST],
                    Direction.EAST: dir_0[Direction.EAST] and dir_1[Direction.EAST],
                    Direction.SOUTH: dir_0[Direction.SOUTH] and dir_1[Direction.SOUTH],
                    Direction.NORTH: dir_0[Direction.NORTH] and dir_1[Direction.NORTH],
                }
            case _:
                raise ValueError("Invalid spatial dimension.")

    def _tangent_basis(
        self, f_main: int, f_other: int, cp_0: Point, cp_1: Point
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute tangent basis vectors on a fracture surface.

        Parameters:
            f_main: Gmsh index of the fracture where the basis is to be computed.
            f_other: Gmsh index of the other fracture.
            cp_0: Coordinates of the closest point on the main fracture.
            cp_1: Coordinates of the closest point on the other fracture.

        Returns:
            A tuple containing the tangent basis vectors. In 2D, both tangent vectors
            are returned, while in 1D, only one tangent vector is returned (the second
            element of the tuple is None).

        """
        if self._nd == 3:
            return self._tangent_basis_2d(f_main, f_other, cp_0, cp_1)
        else:
            return self._tangent_basis_1d(f_main)

    def _tangent_basis_1d(self, surface_tag: int) -> tuple[np.ndarray, None]:
        """Get a unit basis vector for the 1d line represented by surface_tag.

        The basis vector has an arbitrary positive direction.

        Parameters:
            surface_tag: Gmsh tag for the fracture where a basis vector is sought.

        Returns:
            A 2-tuple, with the first element containing the basis vector. The second
            element is None to mark that this is a 1d basis.

        """
        bnd = gmsh.model.get_parametrization_bounds(self._nd - 1, surface_tag)
        start = gmsh.model.get_value(self._nd - 1, surface_tag, bnd[0].tolist())
        end = gmsh.model.get_value(self._nd - 1, surface_tag, bnd[1].tolist())
        t_0 = np.array(end) - np.array(start)
        t_0 = t_0 / np.linalg.norm(t_0)
        return t_0, None

    def _tangent_basis_2d(
        self, f_main: int, f_other: int, cp_0: Point, cp_1: Point
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get a tangential basis for the 2d fracture surface f_main.

        The basis is constructed so that the first basis vector is aligned with the
        direction in which the distance between f_main and f_other increases the slowest
        (starting from the closest point on f_main, cp_0). The second basis vector is
        orthogonal and hence points in the direction of maximum increase (assuming both
        fractures are planar). The direction of the basis vectors is arbitrary, and
        there is no guarantee that they form a righ-hand or left-hand system (this is
        not needed by the calling method).

        Parameters:
            f_main: Gmsh tag of the fracture for which the tangential basis is sought.
            f_other: Gmsh tag of the other fracture.
            cp_0: Coordinate of the point on f_main which is closest to f_other.
            cp_1: Coordinate of the point on f_other which is closest to f_main.

        Returns:
            A two-tuple containing the first and second basis vectors.

        """
        n_0 = self._get_normal(f_main)
        vec = np.array(cp_1) - np.array(cp_0)
        nrm = np.linalg.norm(vec)
        # Detect (almost) identical points. How close the points can get depends on the
        # geometric tolerances used in the mesh size control algorithm in general. While
        # we could have hooked up on that tolerance (to the price of passing around the
        # relevant parameter), a simpler approach of detecting points that are so close
        # that arithmetic computations barely makes sense, and use a fallback that
        # should be equally fine in that case.
        if nrm < 1e-12:
            # If the control points are (almost) identical, we cannot use the
            # connecting vector to define a direction. Use the normal of the other
            # fracture instead, and take a cross product to define a direction.
            n_1 = self._get_normal(f_other)
            vec = np.cross(n_0, n_1)
            nrm = np.linalg.norm(vec)

        vec = vec / nrm

        proj_vec_0 = vec - np.dot(vec, n_0) * n_0
        if np.linalg.norm(proj_vec_0) < 1e-12:
            # The vector is (almost) aligned with the normal vector. Pick an
            # arbitrary perpendicular direction.
            if np.abs(n_0[0]) < 0.9:
                arbitrary_vec = np.array([1.0, 0.0, 0.0])
            else:
                arbitrary_vec = np.array([0.0, 1.0, 0.0])
            proj_vec_0 = np.cross(n_0, arbitrary_vec)
        # Tangent vector in f_0 in the direction of maximum increase of distance to
        # f_1.
        t_0_max = proj_vec_0 / np.linalg.norm(proj_vec_0)
        t_0_min = np.cross(n_0, t_0_max)
        # Tangent vector in f_0 in the direction of minimum increase of distance to
        # f_1.
        t_0_min = t_0_min / np.linalg.norm(t_0_min)

        return t_0_max, t_0_min

    def _get_normal(self, f):
        """Helper method to get the normal vector of a nd-1 object from Gmsh."""
        bnd = gmsh.model.get_parametrization_bounds(self._nd - 1, f)
        u_mid = 0.5 * (bnd[0][0] + bnd[1][0])
        v_mid = 0.5 * (bnd[0][1] + bnd[1][1])
        n = gmsh.model.getNormal(f, [u_mid, v_mid])
        return np.array(n)


class MeshSizeComputer:
    """Helper class to manage and compute mesh size parameters.

    This class provides a translation from the user-provided mesh size parameters as
    listed in the documentation of the __init__ method to the mesh size parameters that
    are used in the mesh size control algorithm.

    The methods h_{min, end, ...} provide (sometimes context-dependent) mesh size
    parameters. To understand the details of how this class is used, it is probably
    necassary to also study the wider mesh size control algorithm.

    """

    def __init__(self, mesh_args: dict) -> None:
        """Initialize mesh size computer.

        Parameters:
            mesh_args: Dictionary of mesh size parameters. Supported keys are:
                - "mesh_size_fracture": Fracture mesh size [m].
                - "mesh_size_boundary": Background mesh size [m]. If not provided, it is
                  set equal to the fracture mesh size.
                - "refinement_proximity_multiplier": Threshold for triggering refinement
                  around fractures (in units of fracture mesh size).
                - "refinement_size_multiplier": Buffer factor for mesh size around
                  fractures (in units of fracture mesh size).
                - "mesh_size_min": Minimum mesh size [m]. If not provided, it is set
                  equal to the fracture mesh size times the buffer factor. If set to a
                  value larger than the fracture mesh size, the fracture mesh size is
                  used.
                - "background_transition_multiplier": Factor controlling the distance
                  from fractures where the background mesh size is reached (in units of
                  background mesh size).

        Raises:
            ValueError: If required mesh size parameters are missing.

        """
        # Use typing ignore here, since this class is only accessed through the mesh
        # methods of the fracture network classes, where most relevant parameters are
        # given dimension-dependent default values.
        if "mesh_size_fracture" not in mesh_args:
            raise ValueError("mesh_size_fracture must be provided in mesh_args.")

        self._h_fracture: float = mesh_args.get("mesh_size_fracture")  # type: ignore
        self._h_background: float = mesh_args.get(  # type: ignore
            "mesh_size_boundary", self._h_fracture
        )
        self._threshold: float = mesh_args.get(  # type: ignore
            "refinement_proximity_multiplier"
        )
        self._refinement_size_multiplier: float = mesh_args.get(
            "refinement_size_multiplier"
        )  # type: ignore
        # By default, we let the minimum mesh size scale with the buffer and the
        # fracture mesh size.
        h_min = self._h_fracture * self._refinement_size_multiplier
        if "mesh_size_min" in mesh_args:
            # If the user has set a value, use this, but do not permit values less than
            # the fracture size.
            h_min = min(self._h_fracture, mesh_args.get("mesh_size_min"))  # type: ignore
        self._h_min: float = h_min
        self._farfield_transition: float = mesh_args.get(  # type: ignore
            "background_transition_multiplier", 1.0
        )

    def refinement_threshold(self) -> float:
        """Threshold for refinement around fractures [m].

        Objects that are farther away from a fracture than this threshold will not
        trigger mesh refinement.

        """
        return self._threshold * self._h_fracture

    def h_background(self) -> float:
        """Background mesh size [m]."""
        return self._h_background

    def h_fracture(self, is_boundary: bool = False) -> float:
        """Fracture size on fracture or boundary [m].

        Parameters:
            is_boundary: If ``True``, return the boundary mesh size.

        Returns:
            float: Mesh size. Will be equal to the user-provided fracture mesh size
                unless ``is_boundary = True``, in which case the background mesh size is
                returned.

        """
        if is_boundary:
            return self._h_background
        return self._h_fracture

    def h_min(self) -> float:
        """Minimum mesh size [m]. No smaller mesh sizes will be set anywhere in the
        domain. Gmsh may however decide to use smaller mesh sizes if the geometry
        requires it.

        Returns:
            float: Minimum mesh size.

        """
        return self._h_min

    def h_end(self, is_boundary: bool) -> float:
        """Mesh size at the end of the transition from refinement to 'standard'
        conditions [m].

        The 'standard' will be the fracture mesh size if ``is_boundary = False``, and
        the background mesh size if ``is_boundary = True``.

        """
        return self._h_background if is_boundary else self._h_fracture

    def dist_farfield(self, is_boundary: bool, on_lower_dim: bool) -> float:
        """Distance from fracture where background mesh size is reached [m].

        Parameters:
            is_boundary: If ``True``, return the distance for boundary mesh size.
            on_lower_dim: If ``True``, return the distance on a lower-dimensional
                object.

        Returns:
            float: Distance from fracture where background mesh size is reached [m].

        """
        if on_lower_dim:
            return self.h_end(is_boundary) * self._farfield_transition
        else:
            return self._h_fracture * self._farfield_transition

    def dist_min(self, dist: float) -> float:
        """Distance from a mesh size control point at which the transition from the
        minimal mesh size starts [m].

        Parameters:
            dist: Distance from the fracture.

        Returns:
            float: Distance from the fracture.

        """
        return self._min_size(dist)

    def size_min(self, dist: float) -> float:
        """Mesh size close to a mesh size control point [m].

        Parameters:
            dist: Distance from the fracture.

        Returns:
            float: Mesh size close to the fracture.

        """
        return self._min_size(dist) * self._refinement_size_multiplier

    def size_at_distance(
        self,
        distance_from_existing_point: float,
        fracture_distance_at_existing_point: float,
        is_boundary: bool,
        on_lower_dim: bool,
    ) -> float:
        """Compute the mesh size at a given distance from a mesh size control point.

        Given an existing mesh size control point, and a candidate new point at a
        distance 'dist', the function calculates the mesh size that the existing point
        will impose at this distance. The calling function can use this information to
        assess whether adding a new point will lead to a finer mesh than will then
        conditions imposed by the existing point (and hence adding the point will be
        superfluous).

        Parameters:
            distance_from_existing_point: Distance from the existing mesh size control
                point.
            fracture_distance_at_existing_point: Distance from the fracture at the
                existing mesh size control point.
            is_boundary: Whether the mesh size should be computed relative to the
                background mesh size (``True``) or the fracture mesh size
                (``False``).
            on_lower_dim: Whether the mesh size is to be computed on a lower-dimensional
                object (``True``) or in the surrounding domain (``False``).

        """
        # In the immediate vicinity of the existing mesh size control point, the mesh
        # size is proportional to the distance to other objects at that point, though
        # the distance is capped from below by a minimum distance. The mesh size in this
        # region is scaled by the factor buffer.
        end_near_old_region = self.dist_min(fracture_distance_at_existing_point)
        mesh_size_near_old = self.size_min(fracture_distance_at_existing_point)

        # The mesh size transits linearly from the size near the existing point to a
        # mesh size far away from the existing point. This mesh size is either the
        # fracture mesh size, if the existing control point is placed on a fracture
        # surface, and the mesh size is used for codimension meshing (i.e., we construct
        # the mesh on the fracture surface). Otherwise, the background mesh size is
        # used. The extent of the transition region is controlled by the
        # farfield_transition parameter and the mesh size.
        start_far_away_region = self.dist_farfield(
            is_boundary=is_boundary, on_lower_dim=on_lower_dim
        )
        size_far_away_region = self.h_end(is_boundary=is_boundary)

        if distance_from_existing_point >= start_far_away_region:
            return size_far_away_region
        elif distance_from_existing_point <= end_near_old_region:
            return mesh_size_near_old
        else:
            # Linear transition.
            h = mesh_size_near_old + (size_far_away_region - mesh_size_near_old) * (
                (distance_from_existing_point - end_near_old_region)
                / (start_far_away_region - end_near_old_region)
            )
            return h

    def _min_size(self, dist: float) -> float:
        """Compute the minimum mesh size at a given distance from the fracture.

        Parameters:
            dist: Distance from the fracture.

        Returns:
            float: Minimum mesh size at the given distance.

        """
        if dist == 0:
            # For intersecting fractures, we set the mesh size to the fracture mesh
            # size. This avoids execessive refinement at intersection lines in cases
            # where the intersection angle is large. For small angles, we should refine,
            # but experience indicates that the mesh size will be small anyway, partly
            # due to nearby mesh control points.
            return self._h_fracture
        else:
            return max(self._h_min, dist)


class GmshPointIdentifier:
    """Helper class to identify Gmsh point indices based on physical coordinates.

    On construction, the class retrieves all Gmsh point entities and their physical
    coordinates. The `index` method can then be used to find the Gmsh point index
    corresponding to a given physical coordinate, within a specified tolerance.

    The Gmsh points are assumed to be static after the construction of the class, i.e.,
    no new points are added to the Gmsh model after the class has been instantiated.

    """

    def __init__(self, tol: float = 1e-6) -> None:
        """Initialize the GmshPointIdentifier with a specified tolerance.

        Parameters:
            tol: Tolerance for matching physical coordinates to Gmsh points.

        """
        self._tol = tol
        phys_coord = []
        self._gmsh_point_ind = [ent[1] for ent in gmsh.model.get_entities(0)]
        for gmsh_ind in self._gmsh_point_ind:
            coord = gmsh.model.get_bounding_box(0, gmsh_ind)[:3]
            phys_coord.append(np.array(coord))
        self._phys_coord = np.array(phys_coord).T

    def index(self, point: np.ndarray) -> int:
        """Identify the Gmsh point index corresponding to a given physical coordinate.

        See class documentation for assumptions and usage.

        Parameters:
            point: Physical coordinate as a numpy array of shape (3,).

        Raises:
            ValueError: If the point is not found in the Gmsh model within the specified
                tolerance.

        Returns:
            The Gmsh point index corresponding to the given physical coordinate.

        """
        pd = np.linalg.norm(self._phys_coord - point.reshape(3, 1), axis=0)
        if np.all(pd > self._tol):
            raise ValueError("Point not found in Gmsh model.")
        return self._gmsh_point_ind[int(np.argmin(pd))]
