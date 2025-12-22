from __future__ import annotations

from abc import ABC, abstractmethod
import warnings
from typing import Optional, Union, cast, TYPE_CHECKING, Literal
import gmsh
import porepy as pp
from collections import namedtuple
from enum import Enum
import heapq
import numpy as np
from pathlib import Path
import copy
from itertools import combinations
import multiprocessing

# Custom typings
FractureList = Optional[
    list[pp.LineFracture] | list[pp.PlaneFracture | pp.EllipticFracture]
]


class FractureNetwork(ABC):
    """Abstract base class for fracture networks."""

    def __init__(
        self,
        nd: Literal[2, 3],
        fractures: Optional[FractureList] = None,
        domain: Optional[pp.Domain] = None,
        tol: float = 1e-8,
    ) -> None:
        self.nd = nd
        """Number of spatial dimensions (2 or 3)."""

        self.fractures = []
        """List of fractures forming the network."""
        if fractures is not None:
            for f in fractures:
                self.fractures.append(f)

        self.domain: Optional[pp.Domain] = domain
        """Domain specification for the fracture network."""

        self._tol = tol
        """Tolerance for geometric computations."""

    def num_frac(self) -> int:
        """Return the number of fractures in the network."""
        return len(self.fractures)

    @abstractmethod
    def domain_to_gmsh(self) -> None:
        """Define the domain in gmsh."""
        pass

    @abstractmethod
    def fractures_to_gmsh(self) -> None:
        """Define the fractures in gmsh."""
        pass

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

    def _prepare_mesh_inputs(
        self,
        file_name: Optional[Path],
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

        if constraints is None:
            constraints = np.array([], dtype=int)
        else:
            constraints = np.atleast_1d(constraints)
            constraints.sort()

        try:
            num_procs_available = multiprocessing.cpu_count() or 1
        except (NotImplementedError, AttributeError):
            num_procs_available = 1

        num_procs = kwargs.get("num_processors", max(num_procs_available - 2, 1))
        gmsh.option.setNumber("General.NumThreads", num_procs)

        if self.nd == 3:
            # Use HXT algorithm for 3d meshing by default. Note to self: It is important
            # to use Mesh.Algorithm3D, not Mesh3D.Algorithm, which triggers all sorts of
            # issues.
            meshing_algorithm = kwargs.get("meshing_algorithm_3d", 10)
        else:
            meshing_algorithm = kwargs.get("meshing_algorithm_2d", 5)
        gmsh.option.setNumber("Mesh.Algorithm3D", meshing_algorithm)

        return file_name, constraints

    def _entity_on_domain_boundary(self, target_dim: int, ind: list[int]) -> bool:
        """Helper function to determine if an entity lies on the domain boundary.

        The intended use is to determine if a line or set of points lie on the
        boundary of the domain, in which case it should not be considered an
        intersection line or point between fractures.

        The implementation could have been generalized in various ways, but is kept
        as it is, since it concerns very specific use cases that are covered by the
        current implementation.

        Known possible issue: If the domain is fully split by a fracture, the
        original boundary sides of the domain may have been split into multiple
        parts (this will manifest as the variable boundary_surfaces containing more
        surfaces than the original domain boundary, for instance more than six
        surfaces for a box domain). In this case, a line that extends over multiple
        of these 'sub-sides', but are really part of one side in the original
        boundary definition, may be misidentified as an intersection line. In EK's
        understanding, this should not happen, since that line will also have been
        split into multiple parts during fragmentation, and each part will be on one
        of the sub-sides. Still, Gmsh works in mysterious ways, so it was considered
        wise to take note of this possible issue.

        Parameters:
            dim: Dimension of the entity to check (0 for points, 1 for line). ind:
            List of Gmsh tags identifying the entity to check.

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

        # For each boundary surface of the domain, compute the distance to all
        # boundary points of the entity to check if they are all zero. Note that the
        # other way around (checking if each entity point is on any of the boundary
        # entities) risk false positives for a line extending across the domain
        # between two boundaries.
        for ent in boundary_entities:
            dist = [gmsh.model.occ.get_distance(*bp, *ent)[0] for bp in boundary_points]
            # EK: It is not 100% clear what an empty list of boundary points (i.e.,
            # len(dist) == 0) implies - the case arose while working with disc
            # fractures. However, it seems safest that the lack of boundary points does
            # not automatically leads to the fracture being classified as being on the
            # boundary, hence we rule out this case.
            if len(dist) > 0 and np.all(np.array(dist) < self._tol):
                return True
        # Having come this far, the entity is not on the domain boundary.
        return False

    def _insert_mesh_size_control_points(
        self, mesh_size_computer: MeshSizeComputer
    ) -> tuple[list[int], dict[int, list[tuple[np.ndarray, float]]]]:
        """Insert control points for mesh size specification on fractures and
        boundaries.

        The method identifies points on fracture surfaces and domain boundaries where
        mesh size control points should be inserted. Later in the meshing process Gmsh
        mesh size fields will be assigned based on the distances from these points to
        surrounding objects. For a detailed description of the approach, see the
        documentation of the MeshSizeComputer class.

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
        mesh_size_points = {}
        for f in fracture_tags | boundary_tags:
            mesh_size_points[f] = []

        control_points: list[int] = []

        # To avoid inserting the same point multiple times on the same line, and to
        # prune doubly defined points from the gmsh specification, we keep track of
        # which points have already been inserted where.

        # Coordinates of the inserted mesh size control points.
        inserted_points: list[np.ndarray] = []
        # Coordinates of the mesh size control points already inserted. Used to avoid
        # duplicates.
        inserted_on_entity: list[int] = []

        # Take note of the boundary points of all entities, to avoid inserting points
        # there (not doing so may confuse Gmsh).
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
                return False
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
        codim: bool,
        surface_lines: Optional[list[int]] = None,
    ) -> list:
        """Assign mesh size field based on distances from points to fractures.

        The mesh size field is either restricted to the entity itself (if
        ``codim=True``), or set in the surrounding domain (if ``codim=False``).

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
            is_boundary: ``True`` if the entity is on the domain boundary,
                ``False`` otherwise.
            codim: ``True`` if the mesh size is to be restricted to the entity itself
                (codimension 1), ``False`` if the mesh size is to be set in the
                surrounding domain.
            surface_lines: (3D only) List of Gmsh tags identifying lines on the
                surface, e.g., intersection lines with other fractures. The mesh size
                field will also be applied on these lines.

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
            # entity itself (codim=True) or the surrounding domain (codim=False).
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
            if codim:
                gmsh.model.mesh.field.setNumber(
                    threshold,
                    "DistMax",
                    mesh_size_computer.dist_farfield(is_boundary, on_codim=True),
                )
                gmsh.model.mesh.field.setNumber(
                    threshold,
                    "SizeMax",
                    mesh_size_computer.size_farfield(is_boundary),
                )
            else:
                gmsh.model.mesh.field.setNumber(
                    threshold,
                    "DistMax",
                    mesh_size_computer.dist_farfield(is_boundary, on_codim=False),
                )
                gmsh.model.mesh.field.setNumber(
                    threshold, "SizeMax", mesh_size_computer.h_farfield()
                )

            # Note to self: The order is important here - the restriction must refer
            # to the threshold field, not the other way around.
            restriction = gmsh.model.mesh.field.add("Restrict")
            gmsh.model.mesh.field.setNumber(restriction, "InField", threshold)
            if codim:
                if self.nd == 3 and surface_lines is not None:
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
        codim: bool,
    ) -> list:
        """Set uniform mesh size fields on the given entities.

        The mesh is either restricted to the entities themselves (if ``codim=True``), or
        in the surrounding domain (if ``codim=False``). In the former case, the mesh
        size is constant on the entities, while in the latter case, the mesh size
        transitions from a fine mesh size close to the entities to the far-field mesh
        size.

        Parameters:
            entities: Set of Gmsh tags identifying the entities where the mesh size
                field should be applied. The entities are of dimension self.nd - 1.
            mesh_size_computer: Instance of MeshSizeComputer providing the mesh size
                parameters.
            boundary_tags: Set of Gmsh tags identifying the boundary entities of the
                domain.
            codim: ``True`` if the mesh size is to be restricted to the entities
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
        if codim:
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
                # Set the mesh size outside the entities to the far-field size. Since we
                # will take the minimum over all mesh size fields later, this will in
                # practice not affect the mesh size outside the entities, but it seems
                # Gmsh requires that a value is set.
                gmsh.model.mesh.field.setNumber(
                    uniform_field, "VOut", mesh_size_computer.h_farfield()
                )
                gmsh_fields.append(uniform_field)
        else:
            # This will set a mesh size field in the surrounding domain, transitioning
            # from a fine mesh size close to the entities to the far-field mesh size.

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
                        ent in boundary_tags, on_codim=False
                    ),
                )
                gmsh.model.mesh.field.setNumber(
                    threshold,
                    "SizeMax",
                    mesh_size_computer.size_farfield(ent in boundary_tags),
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
        # other mesh size specifications.
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

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
                mesh_sizes = np.array(mesh_sizes)
                _, ind_map, inv_map = pp.array_operations.uniquify_point_set(
                    np.array(all_pts).T, tol=self._tol
                )
                min_size = np.empty(ind_map.size, dtype=float)

                # Loop over unique points, find minimum mesh size among duplicates.
                for i in range(ind_map.size):
                    inds = inv_map == i
                    min_size[i] = np.min(mesh_sizes[inds])

                # Map back to entities.
                for line_ind, pt_ind in enumerate(inv_map):
                    entity = entity_item_comb[line_ind][0]
                    item = entity_item_comb[line_ind][1]
                    mesh_size_points[entity][item] = (
                        mesh_size_points[entity][item][0],
                        min_size[pt_ind],
                    )


ij = namedtuple("Index", ["i", "j"])
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

    Attributes:
        fracture_tags: List of Gmsh tags corresponding to the fractures where points
            will be inserted.
        points_per_fracture: Dictionary mapping each fracture tag to a list of points
            (as numpy arrays) to be inserted on that fracture.

    """

    def __init__(self, nd: int, mesh_size_computer: MeshSizeComputer) -> None:
        self._nd = nd
        self._mesh_size_computer = mesh_size_computer

    def compute_points(
        self, f_0, f_1, cp_0, cp_1, distance, f_0_on_fracture, f_1_on_fracture
    ) -> tuple[list, list]:
        """Compute points to be inserted on the surfaces of two fractures.

        Given two fractures and their corresponding control points, this method
        computes the points that need to be inserted on the surfaces of both fractures
        to ensure proper mesh size control.

        Args:
            f_0: The first fracture object.
            f_1: The second fracture object.
            cp_0: Control point on the first fracture.
            cp_1: Control point on the second fracture.

        Returns:
            A tuple containing two lists:
                - List of points to be inserted on the first fracture.
                - List of points to be inserted on the second fracture.

        """
        # Implementation goes here
        points_0 = self._control_points(f_0, f_1, cp_0, cp_1, distance, f_0_on_fracture)
        points_1 = self._control_points(f_1, f_0, cp_1, cp_0, distance, f_1_on_fracture)
        return points_0, points_1

    def _control_points(
        self,
        f_main,
        f_other,
        cp_0,
        cp_1,
        init_distance,
        f_main_on_fracture,
    ):
        t_i, t_j = self._tangent_basis(f_main, f_other, cp_0, cp_1)

        def priority(ij):
            # The priority is given by the Manhattan distance from the origin. With a
            # min-heap as in heapq, this will give priority to points closer to the
            # origin.
            return abs(ij.i) + abs(ij.j)

        q = []
        tab = {}
        i = ij(0, 0)
        heapq.heappush(q, (priority(i), i))
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
        tab[i] = (Point(*cp_0), Point(*cp_0), direction, init_distance)
        points_to_add = []
        discarded_ijs = set()

        while q:
            _, i = heapq.heappop(q)
            if i in discarded_ijs:
                continue
            p_cand, p_prev, dirs, old_distance = tab[i]

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
            dist_prev_point = np.linalg.norm(np.array(p_cand) - np.array(p_prev))

            # Mesh at the candidate point, as determined by the distance from the
            # previous point.
            if dist_prev_point == 0:
                # This should only happen in the first iteration, when the previous and
                # candidate points have the same coordinates. There is no mesh size from
                # the previous point to compare with (see below if) and a point should
                # be added if the distance to the other fracture justifies it. Thus, we
                # set the mesh size from previous to positive inf to make sure it is not
                # less than the mesh size set according to the distance to the other
                # fracture, as this could have prevented adding the point.
                h_from_prev = np.inf
            else:
                # There is a previous point. Compute the mesh size at the candidate
                # point from the mesh size field centered at this point.
                h_from_prev = self._mesh_size_computer.size_at_distance(
                    dist_other_fracture,
                    dist_prev_point,
                    f_main_on_fracture,
                    on_codim=True,
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
            # mesh sizes given by far-field conditions (see documentation of the
            # MeshSizeComputer for details). This can be estimated to 2 times the
            # distance between the two fractures at the current point (which will give
            # the correct estimate for parallel fractures, and possibly a somewhat too
            # long step for close to parallel fractures, but we cross our fingers this
            # will work out nicely).
            step_size = 2 * self._mesh_size_computer.dist_min(dist_other_fracture)

            for direction, can_proceed in dirs.items():
                if not can_proceed:
                    continue

                dir_new = copy.copy(dirs)
                if direction == Direction.WEST:
                    di = ij(i.i - 1, i.j)
                    delta = -t_i * step_size
                    dir_new[Direction.EAST] = False
                elif direction == Direction.EAST:
                    di = ij(i.i + 1, i.j)
                    delta = t_i * step_size
                    dir_new[Direction.WEST] = False
                elif direction == Direction.SOUTH:
                    di = ij(i.i, i.j - 1)
                    delta = -t_j * step_size
                    dir_new[Direction.NORTH] = False
                elif direction == Direction.NORTH:
                    di = ij(i.i, i.j + 1)
                    delta = t_j * step_size
                    dir_new[Direction.SOUTH] = False

                if di in discarded_ijs:
                    continue

                p_new = Point(*(np.array(p_cand) + delta))
                dist_new = dist_other_fracture

                if di in tab:
                    p_new, dist_new = self._closest_point(
                        cp_0, p_new, dist_other_fracture, tab[di][0], tab[di][3]
                    )
                    dir_new = self._direction_union(dir_new, tab[di][2])

                tab[di] = (p_new, p_cand, dir_new, dist_new)
                heapq.heappush(q, (priority(di), di))
            discarded_ijs.add(i)
            tab.pop(i)

        return points_to_add

    def _closest_point(
        self, start: Point, cand_0: Point, dist_0, cand_1: Point, dist_1: float
    ) -> Point:
        vec_0 = np.array(cand_0) - np.array(start)
        vec_1 = np.array(cand_1) - np.array(start)

        dist_0 = np.linalg.norm(vec_0)
        dist_1 = np.linalg.norm(vec_1)

        if dist_0 < dist_1:
            return cand_0, dist_0
        else:
            return cand_1, dist_1

    def _point_inside_other_surface(self, point: Point, f_other) -> bool:
        proj_pts, _ = gmsh.model.get_closest_point(self._nd - 1, f_other, point)
        return gmsh.model.is_inside(self._nd - 1, f_other, proj_pts)

    def _direction_union(self, dir_0: Direction, dir_1: Direction) -> Direction:
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

    def _tangent_basis(self, f_main, f_other, cp_0, cp_1):
        if self._nd == 3:
            return self._tangent_basis_2d(f_main, f_other, cp_0, cp_1)
        else:
            return self._tangent_basis_1d(f_main, f_other, cp_0, cp_1)

    def _tangent_basis_1d(self, f_main, f_other, cp_0, cp_1):
        bnd = gmsh.model.get_parametrization_bounds(self._nd - 1, f_main)
        start = gmsh.model.get_value(self._nd - 1, f_main, bnd[0].tolist())
        end = gmsh.model.get_value(self._nd - 1, f_main, bnd[1].tolist())
        t_0 = np.array(end) - np.array(start)
        t_0 = t_0 / np.linalg.norm(t_0)
        return t_0, None

    def _tangent_basis_2d(self, f_main, f_other, cp_0, cp_1):
        n_0 = self._get_normal(f_main)
        vec = np.array(cp_1) - np.array(cp_0)
        nrm = np.linalg.norm(vec)
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
        bnd = gmsh.model.get_parametrization_bounds(self._nd - 1, f)
        u_mid = 0.5 * (bnd[0][0] + bnd[1][0])
        v_mid = 0.5 * (bnd[0][1] + bnd[1][1])
        n = gmsh.model.getNormal(f, [u_mid, v_mid])
        return np.array(n)


class MeshSizeComputer:
    """Helper class to manage and compute mesh size parameters.

    This provides a unified way to access mesh size parameters used in meshing.


    """

    def __init__(self, mesh_args: dict):
        self._hfarfield = mesh_args.get("mesh_size_bound")
        self._hfrac = mesh_args.get("mesh_size_frac")
        self._hmin = mesh_args.get("mesh_size_min", self._hfrac / 10)
        self._threshold = mesh_args.get("refinement_threshold", 1.0)
        self._buffer = mesh_args.get("refinement_buffer", 0.5)
        self._farfield_transition = mesh_args.get("farfield_transition", 10.0)

    def refinement_threshold(self) -> float:
        """Threshold for refinement around fractures [m].

        Objects that are farther away from a fracture than this threshold will not
        trigger mesh refinement.

        """
        return self._threshold * self._hfrac

    def h_farfield(self) -> float:
        """Far-field mesh size [m]."""
        return self._hfarfield

    def h_frac(self, is_boundary: bool = False) -> float:
        """Fracture size on fracture or boundary [m].

        Parameters:
            is_boundary: If ``True``, return the boundary mesh size.

        Returns:
            float: Mesh size. Will be equal to the user-provided fracture mesh size
                unless ``is_boundary = True``, in which case the far-field mesh size is
                returned.

        """
        if is_boundary:
            return self._hfarfield
        return self._hfrac

    def h_min(self) -> float:
        """Minimum mesh size [m]. No smaller mesh sizes will be set anywhere in the
        domain. Gmsh may however decide to use smaller mesh sizes if the geometry
        requires it.

        Returns:
            float: Minimum mesh size.

        """
        return self._hmin

    def h_end(self, is_boundary: bool) -> float:
        """Mesh size at the end of the transition from refinement to 'standard'
        conditions [m].

        The 'standard' will be the fracture mesh size if ``is_boundary = False``, and
        the far-field mesh size if ``is_boundary = True``.

        """
        return self._hfarfield if is_boundary else self._hfrac

    def dist_farfield(self, is_boundary: bool, on_codim: bool) -> float:
        """Distance from fracture where far-field mesh size is reached [m].

        TODO: Need better name, farfield can here imply both h_frac and h_bound.

        Parameters:
            is_boundary: If ``True``, return the distance for boundary mesh size.
            on_codim: If ``True``, return the distance on a lower-dimensional object.

        Returns:
            float: Distance from fracture where far-field mesh size is reached [m].

        """
        if on_codim:
            return self.h_end(is_boundary) * self._farfield_transition
        else:
            return self._hfarfield * self._farfield_transition

    def size_farfield(self, is_boundary: bool) -> float:
        """Far-field mesh size [m].

        Parameters:
            is_boundary: If ``True``, return the boundary mesh size.

        Returns:
            float: Mesh size. Will be equal to the user-provided fracture mesh size
                unless ``is_boundary = True``, in which case the far-field mesh size is
                returned.

        """
        return self.h_end(is_boundary)

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
        return self._min_size(dist) * self._buffer

    def size_at_distance(
        self, dist: float, old_distance: float, is_boundary: bool, on_codim: bool
    ) -> float:
        # In the immediate vicinity of the old point, the mesh size is proportional to
        # the distance to other objects at that point, though the distance is capped
        # from below by a minimum distance. The mesh size in this region is scaled by
        # the factor buffer.
        end_near_old_region = self.dist_min(old_distance)
        mesh_size_near_old = self.size_min(old_distance)

        # The mesh size transits linearly from the size near the old point to a mesh
        # size far away from the contol point. This mesh size is either the fracture
        # mesh size, if the control point is placed on a fracture surface, and the mesh
        # size is used for codimension meshing (i.e., we construct the mesh on the
        # fracture surface). Otherwise, the far-field mesh size is used. The extent of
        # the transition region is controlled by the farfield_transition parameter and
        # the mesh size.
        start_far_away_region = self.dist_farfield(
            is_boundary=is_boundary, on_codim=on_codim
        )
        size_far_away_region = self.size_farfield(is_boundary=is_boundary)

        if dist >= start_far_away_region:
            return size_far_away_region
        elif dist <= end_near_old_region:
            return mesh_size_near_old
        else:
            # Linear transition.
            h = mesh_size_near_old + (size_far_away_region - mesh_size_near_old) * (
                (dist - end_near_old_region)
                / (start_far_away_region - end_near_old_region)
            )
            return h

    def _min_size(self, dist: float) -> float:
        """Compute the minimum mesh size at a given distance from the fracture.

        Parameters:
            dist: Distance from the fracture.
        """
        return max(self._hmin, dist)


class GmshPointIdentifier:
    """Helper class to identify Gmsh point indices based on physical coordinates."""

    def __init__(self, tol=1e-6):
        self._tol = tol
        phys_coord = []
        self._gmsh_point_ind = [ent[1] for ent in gmsh.model.get_entities(0)]
        for gmsh_ind in self._gmsh_point_ind:
            coord = gmsh.model.get_bounding_box(0, gmsh_ind)[:3]
            phys_coord.append(np.array(coord))
        self._phys_coord = np.array(phys_coord).T

    def index(self, point: np.ndarray) -> int:
        """Identify the Gmsh point index corresponding to a given physical coordinate.

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
