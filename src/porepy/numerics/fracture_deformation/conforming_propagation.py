"""
Module indended for combining fracture propagation with complex multiphysics,
as represented in the model classes.

WARNING: This should be considered experimental code and should be used with
    extreme caution. In particular, the code likely contains bugs, possibly of a
    severe character. Moreover, simulation of fracture propagation may cause
    numerical stability issues that it will likely take case-specific adaptations
    to resolve.

WARNING 2: At the moment there are assumptions of purely tensional propagation.
    Violation of this assumption is not recommended with the current implementation.

Contains:
    ConformingFracturePropagation - class to be used together with a pp Model for
    propagation simulation.

    Required additional mechanical parameters:
        poisson_ratio (d_h)
        shear_modulus (d_h)
        SIFs_critical (d_l)

Literature:
    Thomas et al. 2020: Growth of three-dimensional fractures, arrays, and networks
    in brittle rocks under tension and compression
    Nejati et al, 2015: On the use of quarter-point tetrahedral finite elements in
    linear elastic fracture mechanics
    Richard et al. 2005: Theoretical crack path prediction

"""
import logging
import warnings
from typing import Any, Dict

import numpy as np
import scipy.sparse as sps

import porepy as pp

from .propagation_model import FracturePropagation

logger = logging.getLogger(__name__)
module_sections = ["models", "numerics"]


class ConformingFracturePropagation(FracturePropagation):
    """
    Class for fracture propagation along existing faces in the higher-dimensional
    grid.

    The propagation criteria is based on stress intensity factors, computed from
    a displacement correlation approach.

    Should be used in combination with a Model class, i.e. assumptions on methods
    and fields are made.

    WARNING: This should be considered experimental code and should be used with
        extreme caution. In particular, the code likely contains bugs, possibly of a
        severe character. Moreover, simulation of fracture propagation may cause
        numerical stability issues that it will likely take case-specific adaptations
        to resolve.

        The code structure for fracture propagation cannot be considered fixed, and it
        may be fundamentally restructured at unknown points in the future. If you use
        this functionality, please notify the maintainers (Eirik.Keilegavlen@uib.no),
        so that we may keep your usecases in mind if a major overhaul of the code is
        undertaken.

    """

    @pp.time_logger(sections=module_sections)
    def __init__(self, params):
        self.params = params
        # Tag for tensile propagation. This enforces SIF_II=SIF_III=0
        self._is_tensile = True

        # Declear variable for keeping track of whether a propagating fracture has_
        # been found. In practice, this is modified by self.evaluate_propagation()
        self.propagated_fracture: bool = False

    @pp.time_logger(sections=module_sections)
    def has_propagated(self) -> bool:
        if not hasattr(self, "propagated_fracture"):
            return False
        return self.propagated_fracture

    @pp.time_logger(sections=module_sections)
    def evaluate_propagation(self) -> None:
        """
        Evaluate propagation for all fractures based on the current solution.

        Computes SIFs using the Displacement Correlation method described in
        Nejati et al. based on the displacement jumps of the previous iterate.
        Then, propagation onset and angles are evaluated as described in Thomas
        et al.


        Returns
        -------
        None.

        """
        # IMPLEMENTATION NOTE: Warnings from debuggers relating to self not having
        # a gb is okay; this is provided by the Model class which this class should
        # be combined with.
        # It may not be the most pythonic approach, though.
        gb = self.gb

        if len(gb.grids_of_dimension(gb.dim_max() - 2)) > 0:
            warnings.warn(
                "Fracture propogation with intersecting fractures has not been tested"
            )

        face_list: Dict[pp.Grid, np.ndarray] = {}

        self.propagated_fracture = False

        # The propagation is implemented as a loop over interfaces, this gives access to
        # both higher and lower-dimensional grids (both of which may be modified).
        for e, d in gb.edges():
            g_l, g_h = gb.nodes_of_edge(e)

            # Only consider grids of co-dimension 1 for splitting.
            if g_h.dim == self._Nd:
                d_l: Dict = gb.node_props(g_l)
                d_h: Dict = gb.node_props(g_h)

                # Compute stress intensity factors for this fracture.
                self._displacement_correlation(g_l, d_h, d_l, d)

                # Determine whether the fracture should propagate based on computed SIFs,
                # tag faces in the lower-dimensional grid that should be split (that is,
                # find the parts of the fracture tips that should move).
                self._propagation_criterion(d_l)

                # Determine the propagation angle based on SIFs
                self._angle_criterion(d_l)

                # Determine faces to split in the higher-dimensional grid, that is, where
                # the fracture should grow.
                self._pick_propagation_faces(g_h, g_l, d_h, d_l, d)

            if d["propagation_face_map"].data.size > 0:
                # Find the faces in the lower-dimensional grid to split.
                _, col, _ = sps.find(d["propagation_face_map"])
                face_list.update({g_l: col})

                # We have propagated (at least) one fracture in this step
                self.propagated_fracture = True

            else:
                # No updates to the geometry.
                face_list.update({g_l: np.array([], dtype=int)})

        pp.propagate_fracture.propagate_fractures(gb, face_list)

    @pp.time_logger(sections=module_sections)
    def _displacement_correlation(
        self, g_l: pp.Grid, data_h: Dict, data_l: Dict, data_edge: Dict
    ) -> None:
        """
        Compute stress intensity factors by displacement correlation based on
        the solution of the previous iterate.

        Parameters
        ----------
        g_l : pp.Grid
            Fracture grid.
        data_h : Dict
            Matrix data. Assumed to contain the mechanical parameters "shear_modulus"
            and "poisson_ratio"
        data_l : Dict
            Fracture data. Will be updated with SIFs
        data_edge : Dict
            Interface data.

        Returns
        -------
        None.

        Stores the stress intensity factors in data_l under the name "SIFs". The value
        is an self._Nd times self.num_faces np.ndarray.

        """

        # NOTE: self.mechanics_parameter_key is expected to come from the model with which
        # this class is assumed to be combined.
        parameters_l: Dict[str, Any] = data_l[pp.PARAMETERS][
            self.mechanics_parameter_key  # type: ignore
        ]
        parameters_h: Dict[str, Any] = data_h[pp.PARAMETERS][
            self.mechanics_parameter_key  # type: ignore
        ]
        mg: pp.MortarGrid = data_edge["mortar_grid"]

        u_j: np.ndarray = data_edge[pp.STATE][pp.ITERATE][
            self.mortar_displacement_variable  # type: ignore
        ]

        # Only operate on tips
        tip_faces = g_l.tags["tip_faces"].nonzero()[0]
        # Cells in g_l on the fracture tips
        _, tip_cells = g_l.signs_and_cells_of_boundary_faces(tip_faces)

        # Project to fracture and apply jump operator
        u_l = (
            mg.mortar_to_secondary_avg(nd=self._Nd)
            * mg.sign_of_mortar_sides(nd=self._Nd)
            * u_j
        )
        # Jumps at the fracture tips
        u_l = u_l.reshape((self._Nd, g_l.num_cells), order="F")[:, tip_cells]

        # Pick out components of the tip displacement jump in the tip basis
        tip_bases = self._tip_bases(
            g_l, data_l["tangential_normal_projection"], tip_faces
        )
        d_u_tips = np.zeros((tip_bases.shape[1:3]))
        d_u_tips[0] = np.sum(u_l * tip_bases[0, :, :], axis=0)
        d_u_tips[1] = np.sum(u_l * tip_bases[1, :, :], axis=0)
        if self._Nd == 3:
            d_u_tips[2] = np.sum(u_l * tip_bases[2, :, :], axis=0)

        # Compute distance from face centers to cell centers:
        # Rather distance from cc to the face??
        fc_cc = g_l.face_centers[::, tip_faces] - g_l.cell_centers[::, tip_cells]
        dist_face_cell = np.linalg.norm(fc_cc, 2, axis=0)

        # The SIF vector has size equal to the number of faces in g_l, however,
        # only the tip values are non-zero.
        sifs = np.zeros((self._Nd, g_l.num_faces))
        sifs[:, tip_faces] = self._sifs_from_delta_u(
            d_u_tips, dist_face_cell, parameters_h
        )
        parameters_l["SIFs"] = sifs

    @pp.time_logger(sections=module_sections)
    def _sifs_from_delta_u(
        self, d_u: np.ndarray, rm: np.ndarray, parameters: Dict[str, Any]
    ) -> np.ndarray:
        """
        Compute the stress intensity factors from the relative displacements.

        See Eq. 19 in Nejati et al. Note that the pp [tangential, normal] convention
        for local coordinate systems is different from the [u, v, w] notation with
        v being the component normal to the fracture.

        Parameters:
            d_u (array): relative displacements, g_h.dim x n.
            rm (array): distance from correlation point to fracture tip.
            parameters (pp.Parameters): assumed to contain constant
                mu (array): Shear modulus.
                poisson_ratio (array): No, I'm not spelling it out!

        Returns:
            K (array): the displacement correlation stress intensity factor
            estimates.
        """
        mu: np.ndarray = parameters["shear_modulus"]
        poisson: np.ndarray = parameters["poisson_ratio"]
        kappa = 3 - 4 * poisson
        # kappa = 3 - poisson / (1 + poisson)

        (dim, n_points) = d_u.shape

        # Data structure for the SIFs
        K = np.zeros(d_u.shape)
        rm = rm.T

        # Compute SIF_I
        K[0] = np.sqrt(2 * np.pi / rm) * np.divide(mu, kappa + 1) * d_u[1, :]

        # Shortcut for tensile problems
        if self._is_tensile:
            return K

        # The computation of SIF_II can be numerically less stable than is SIF_I
        # The reason seems to be related to the MPSA solution not representing the
        # stress singularity at the fracture tips.
        # The SIFs can still be computed and used, however, they should not be
        # trusted blindly.
        logger.warning("Computing non-tensile SIFs, proceed with caution.")
        K[1] = np.sqrt(2 * np.pi / rm) * np.divide(mu, kappa + 1) * d_u[0, :]

        # For better values, it may be of interest to consider generalized approaches
        # to SIF calculation by
        # This is left in the code as a (potentially) useful code.
        # Generalised displacement correlation:
        # f = -2*(1-poisson)/np.sqrt(2*np.pi)/mu
        # f =2/(1+poisson)/np.sqrt(2*np.pi)/mu
        # for i in [1,0]:
        #     print((3*d_u[i,0]-d_u[i,1])/((3*np.sqrt(2)-np.sqrt(6))*np.sqrt(2*rm[0])*f))
        if dim == 3:
            K[2] = np.sqrt(2 * np.pi / rm) * np.divide(mu, 4) * d_u[2, :]
        return K

    @pp.time_logger(sections=module_sections)
    def _propagation_criterion(self, d: Dict) -> None:
        """
        Tag faces for propagation if the equivalent SIF exceeds a critical value.

        No checks on whether the faces are tips.
        See Eq. (4) in Thomas et al./(7) and (25) in Richard et al.

        Parameters
        ----------
        d : Dict
            Dictionary of a fracture. Assumed to contain facewise computed SIFs
            facewise or scalar critical SIFs, both with size self._Nd in first dimension.

        Returns
        -------
        None
            DESCRIPTION.

        Stores a boolean array identifying the faces to be propagated.
        """

        parameters: Dict[str, Any] = d[pp.PARAMETERS][
            self.mechanics_parameter_key  # type: ignore
        ]

        # Computed sifs
        K: np.ndarray = parameters["SIFs"]
        # Critical values
        K_crit: np.ndarray = parameters["SIFs_critical"]

        # Comparison
        a_1 = K_crit[0] / K_crit[1]
        shear_contribution = 4 * (a_1 * K[1]) ** 2
        if self._Nd == 3:
            a_2 = K_crit[0] / K_crit[2]
            shear_contribution += 4 * (a_2 * K[2]) ** 2
        K_equivalent = (K[0] + np.sqrt(K[0] ** 2 + shear_contribution)) / 2
        parameters["propagate_faces"] = K_equivalent >= K_crit[0]
        parameters["SIFs_equivalent"] = K_equivalent

    @pp.time_logger(sections=module_sections)
    def _angle_criterion(self, d: Dict) -> None:
        """
        Compute propagation angle based on SIFs.

        No checks on whether the faces are tips or whether they are tagged as
        propagating (see propagation_criterion).
        See Eq. (5) in Thomas et al/(8) and (23) in Richard et al.

        Parameters
        ----------
        d : Dict
            Dictionary of a fracture. Assumed to contain facewise computed SIFs
            facewise or scalar critical SIFs, both with size self._Nd in first dimension.

        Returns
        -------
        None
            DESCRIPTION.

        Stores an array of the faces to be propagated.
        """
        parameters: Dict[str, Any] = d[pp.PARAMETERS][
            self.mechanics_parameter_key  # type: ignore
        ]

        # Computed sifs
        K: np.ndarray = parameters["SIFs"]
        phi = np.zeros(K.shape[1])

        # Avoid division by zero: Find columns with non-zero values
        ind = np.any(K, axis=0)
        K = K[:, ind]

        A, B = np.radians(140), np.radians(-70)
        abs_K_1 = np.abs(K[1])
        denominator = K[0] + abs_K_1
        if self._Nd == 3:
            denominator += np.abs(K[2])

        phi[ind] = -np.sign(K[1]) * (
            A * abs_K_1 / denominator + B * (abs_K_1 / denominator) ** 2
        )
        # Mode II angle. This is the angle from the direction along the face normal
        # of the tip (e1) in the plane spanned by e1 and e3, the (local) fracture
        # normal on the j/+ side.
        parameters["propagation_angle_normal"] = phi
        # Could be expanded by Eq. (6) in Thomas for the tangent angle (in the
        # fracture plane). This does not make much sense with the current crude
        # picking of faces to propagate along.
        # parameters["propagation_angle_tangential"]

    @pp.time_logger(sections=module_sections)
    def _propagation_vector(self, g: pp.Grid, d: dict, face: int) -> np.ndarray:
        """


        Parameters
        ----------
        g : pp.Grid
            Fracture grid.
        d : dict
            The grid's data dictionary.
        face : int
            Index of the tip face for which the propagation vector is computed.

        Returns
        -------
        propagation_vector : np.ndarray
            Nd-dimensional propagation vector.

        """
        tip_basis = self._tip_bases(g, d["tangential_normal_projection"], face)[:, :, 0]
        angle = d[pp.PARAMETERS][self.mechanics_parameter_key][  # type: ignore
            "propagation_angle_normal"
        ][face]
        if self._Nd == 2:
            sign = np.cross(tip_basis[0], tip_basis[1])
            e2 = np.array([0, 0, sign])
        else:
            e2 = tip_basis[2]
        R = pp.map_geometry.rotation_matrix(angle, e2)[: self._Nd, : self._Nd]
        propagation_vector = np.dot(R, tip_basis[0])
        return propagation_vector

    @pp.time_logger(sections=module_sections)
    def _pick_propagation_faces(
        self, g_h: pp.Grid, g_l: pp.Grid, data_h: Dict, data_l: Dict, data_edge: Dict
    ) -> None:
        """
        Pick out which matrix faces to split based on which fracture faces are
        tagged as propagating and their propagation angles.

        Work flow:
            Pick out faces_l
            Identify the corresponding edges_h (= nodes if self._Nd==2)
            The edges' faces_h are candidates for propagation
            Pick the candidate based on the propagation angle

        Parameters
        ----------
        g_h : pp.Grid
            DESCRIPTION.
        g_l : pp.Grid
            DESCRIPTION.
        data_h : Dict
            DESCRIPTION.
        data_l : Dict
            DESCRIPTION.
        data_edge : Dict
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.
        Stores the matrix "propagation_faces_l_faces_h" identifying pairs of
        lower- and higherdimensional faces. During grid updates, the former will receive
        a new neighbour cell and the latter will be split.
        """
        nd: int = self._Nd
        parameters_l: Dict[str, Any] = data_l[pp.PARAMETERS][
            self.mechanics_parameter_key  # type: ignore
        ]

        # Faces in lower-dimensional grid to be split
        faces_l: np.ndarray = parameters_l["propagate_faces"].nonzero()[0]

        tip_bases = self._tip_bases(
            g_l, data_l["tangential_normal_projection"], faces_l
        )
        angles: np.ndarray = parameters_l["propagation_angle_normal"][faces_l]

        # Find the edges in lower-dimensional grid to be split. For 2d problems (1d
        # fractures) this will be a node, in 3d, this is two nodes.
        nodes_l, *_ = sps.find(g_l.face_nodes[:, faces_l])

        # Obtain the global index of all nodes.
        # NOTE: For algorithms that introduce new geometric points (not including points that
        # are split to represent a new fracture, but algorithms that do mesh adaptation will
        # be impacted), the global_point_ind must be kept updated so that it gives a unique
        # mapping between points that coincide in the geometry (e.g. the two sides of a
        # fracture, and the corresponding point on the fracture).
        global_nodes = g_l.global_point_ind[nodes_l]

        # Prepare for checking intersection. ind_l is used to reconstruct non-unique
        # nodes later.
        global_nodes, ind_l = np.unique(global_nodes, return_inverse=True)
        # Find g_h indices of unique global nodes
        _, nodes_h, *_ = np.intersect1d(
            g_h.global_point_ind, global_nodes, return_indices=True
        )

        # Reconstruct non-unique and reshape to edges (first dim is 2 if nd=3)
        edges_h = np.reshape(nodes_h[ind_l], (nd - 1, faces_l.size), order="F")

        # IMPLEMENTATION NOTE: No attempt at vectorization: Too many pitfalls. In particular,
        # the number of candidate faces is unknown and may differ between the nodes.

        # Data structure for storing which faces in g_h should be split
        faces_h = np.empty(faces_l.shape, dtype=int)
        for i, f in enumerate(faces_l):
            e = edges_h[:, i]
            candidate_faces_h = self._candidate_faces(g_h, e, g_l, f)

            ## Pick the right candidate:
            # Direction of h-dim face centers from the tip
            tip_coords = np.reshape(g_l.face_centers[:nd, faces_l[i]], (nd, 1))
            face_center_vecs = g_h.face_centers[:nd, candidate_faces_h] - tip_coords
            # normalization
            face_center_vecs = face_center_vecs / np.linalg.norm(
                face_center_vecs, axis=0
            )

            # Propagation vector, with sign assuring a positive orientation
            # of the basis
            if nd == 2:
                sign = np.cross(tip_bases[0, :, i], tip_bases[1, :, i])
                e2 = np.array([0, 0, sign])
            else:
                e2 = tip_bases[2, :, i]
            R = pp.map_geometry.rotation_matrix(angles[i], e2)[:nd, :nd]
            e0 = tip_bases[0, :, i]
            propagation_vector = np.dot(R, e0)
            # Pick the candidate closest to the propagation point,
            # i.e. smallest angle between propagation vector and face center vector
            distances = pp.geometry.distances.point_pointset(
                propagation_vector, face_center_vecs
            )
            ind = np.argsort(distances)
            # There might be no candidate faces left after imposition of restriction
            # of permissible candidates
            if candidate_faces_h.size > 0:
                faces_h[i] = candidate_faces_h[ind[0]]

        # inwards_normals = - g_h.face_normals[faces_h] * sign
        # Construct R_d vector along
        face_faces = sps.csr_matrix(
            (np.ones(faces_l.shape), (faces_l, faces_h)),
            shape=(g_l.num_faces, g_h.num_faces),
        )
        data_edge["propagation_face_map"] = face_faces
        vals = np.zeros(g_h.num_cells)
        cells = np.unique(g_h.cell_faces[faces_h].nonzero()[1])
        vals[cells] = 1
        data_h[pp.STATE]["neighbor_cells"] = vals

    @pp.time_logger(sections=module_sections)
    def _tip_bases(
        self,
        g: pp.Grid,
        projection: pp.TangentialNormalProjection,
        faces: np.ndarray,
    ) -> np.ndarray:
        """
        Construct local bases for tip faces of a fracture.

        Note: The orientation of a 2d basis may be found by
            np.cross(basis[0], basis[1])
        Parameters
        ----------
        g : grid.
        data_edge : dictionary

        faces : array
            The tip faces for which local bases are constructed.

        Returns
        -------
        basis : np.ndarray
            Basis vectors. nd x nd x nd. The first axis is for the basis vectors,
            the second is the dimension and the last for the tip faces. I.e.,
            basis vector i of tip face j is basis[i,:,j]. The ordering of the
            basis vectors is [e_{\perp}, e_n, e_{\parallel}], with the subscripts
            of the tangential vectors indicating that they are perpendicular and
            parallel to the fracture tip (face), respectively.
        """
        basis = np.empty((self._Nd, self._Nd, faces.size))
        signs, cells = g.signs_and_cells_of_boundary_faces(faces)

        basis[0, :, :] = np.reshape(
            g.face_normals[: self._Nd, faces] / g.face_areas[faces] * signs,
            ((self._Nd, faces.size)),
        )
        # Normals of the fracture plane
        if projection.normals.shape[1] == 1:
            basis[1, :, :] = projection.normals
        else:
            basis[1, :, :] = projection.normals[:, cells]
        if g.dim == 2:
            # e2 is parallel to the tip face
            basis[2, :, :] = np.cross(basis[0, :, :], basis[1, :, :], axis=0)
        return basis

    @pp.time_logger(sections=module_sections)
    def _candidate_faces(self, g_h: pp.Grid, edge_h, g_l: pp.Grid, face_l: np.ndarray):
        # TODO: Use identified_faces to avoid pathological cases arising through
        # propagation of multiple fractures within the same propagation step.

        @pp.time_logger(sections=module_sections)
        def faces_of_edge(g: pp.Grid, e: np.ndarray) -> np.ndarray:
            """
            Obtain indices of all faces sharing an edge.


            Parameters
            ----------
            g : pp.Grid
            e : np.ndarray
                The edge.

            Returns
            -------
            faces : np.ndarray
                Faces.
            """
            if g.dim == 1:
                faces = e
            elif g.dim == 2:
                faces = g.face_nodes[e].nonzero()[1]
            elif g.dim == 3:
                f_0 = g.face_nodes[e[0]].nonzero()[1]
                f_1 = g.face_nodes[e[1]].nonzero()[1]
                faces = np.intersect1d(f_0, f_1)
            else:
                raise ValueError("Grid dimension should be 1, 2 or 3")
            return faces

        # For an edge (corresponding to a fracture tip in g_l), find its neighboring
        # faces in g_h
        candidate_faces = faces_of_edge(g_h, edge_h)

        # Exclude faces that are on a fracture
        are_fracture = g_h.tags["fracture_faces"][candidate_faces]
        candidate_faces = candidate_faces[np.logical_not(are_fracture)]

        # Make sure splitting of a candidate does not lead to self-intersection.
        # This is done by checking that none of the face's edges is an "internal
        # fracture edge", i.e. that if it lies on a fracture, it is on a tip.
        #
        # IMPLEMENTATION NOTE: The below tests form an attempt to keep a reasonable fracture
        # geometry for general fractures. For general fracture geometries, this is difficult,
        # and the below code can not be trusted to give good results (and neither did other
        # attempts on implementing such quality checks). For such problems, the best option
        # may be remeshing.
        #
        # IMPLEMENTATION NOTE: For the special case of tensile fracturing along lines or
        # planes that are represented in the grid geometry, the below lines can be dropped
        # (but they should not do any harm either).
        for f in candidate_faces:
            # Obtain all edges:
            local_nodes = g_h.face_nodes[:, f].nonzero()[0]
            pts = g_h.nodes[:, local_nodes]

            # Faces are defined by one node in 1d and two in 2d. This requires
            # dimension dependent treatment:
            if g_h.dim == 3:
                # Sort nodes clockwise (!)
                # ASSUMPTION: This assumes that the new cell is star-shaped with respect to the
                # local cell center. This should be okay.
                map_to_sorted = pp.utils.sort_points.sort_point_plane(
                    pts, g_h.face_centers[:, f]
                )
                sorted_nodes = local_nodes[map_to_sorted]
                # Close the circle by appending the first node
                local_nodes = np.hstack((sorted_nodes, sorted_nodes[0]))

            # Loop over the edges of the candidate face
            for i in range(local_nodes.size - 1):
                e = [local_nodes[i + j] for j in range(g_l.dim)]

                if np.all(np.in1d(e, edge_h)):
                    # The edge which we are splitting
                    continue
                # Identify whether the edge is on a fracture. If so, remove the
                # candidate face unless the edge is a fracture tip. This is done
                # by going to g_l quantities by use of global_point_ind.
                if np.all(g_h.tags["fracture_nodes"][e]):
                    # To check
                    # Obtain the global index of all nodes
                    global_nodes = g_h.global_point_ind[e]
                    # Find g_h indices of unique global nodes
                    _, nodes_l, _ = np.intersect1d(
                        g_l.global_point_ind, global_nodes, return_indices=True
                    )

                    if g_l.dim == 1:
                        face_l = nodes_l
                    if g_l.dim == 2:
                        # Care has to be taken since we don't know whether nodes_l
                        # actually correspond to a face in g_l.
                        f_0 = g_l.face_nodes[nodes_l[0]].nonzero()[1]
                        f_1 = g_l.face_nodes[nodes_l[1]].nonzero()[1]
                        face_l = np.intersect1d(f_0, f_1)

                    if face_l.size == 1:
                        if g_l.tags["tip_faces"][face_l]:
                            continue
                    else:
                        # Not sure what has happened. Identify and deal with it!
                        assert face_l.size == 0

                    candidate_faces = np.setdiff1d(candidate_faces, f)

        return candidate_faces
