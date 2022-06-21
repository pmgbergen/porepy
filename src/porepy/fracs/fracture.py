"""A module containing the base class for all fractures."""

from __future__ import annotations

import abc
from typing import Generator, List, Optional, Tuple, Union

import numpy as np

import porepy as pp
from porepy.utils import setmembership


class Fracture(abc.ABC):
    """Abstract Base class for representing a single fracture.
    This base class provides general functionalities independent of the dimension of the
    fracture and the ambient dimension.
    It contains various utility methods,
    mainly intended for the use together with the FractureNetwork class.

    A fracture is defined by its `npt` vertices, stored in an `nd x npt` numpy-array,
    where `nd` is the assumed ambient dimension.
    The order/sorting of vertices has to be implemented explicitly.

    Dimension-dependent routines are denoted by abstract methods.

    PorePy currently only supports planar and convex fractures fully.
    As a work-around, the fracture can be split into convex parts.

    Attributes
    ----------
        p : numpy.ndarray(nd x npt)
            Fracture vertices. Will be stored in the implemented order
        orig_p : numpy.ndarray(nd x npt)
            Original fracture vertices, kept in case the fracture geometry is modified
            for some reason.
        center : numpy.ndarray(nd x 1)
            Fracture centroid.
        normal : numpy.ndarray(ndx 1)
            Normal vector
        index : int
            Index of fracture. Intended use in FractureNetwork.
            Exact use is not clear (several fractures can be given same index), use with care.
    """

    def __init__(
        self,
        points: np.ndarray,
        index: Optional[int] = None,
        check_convexity: bool = False,
        sort_points: bool = True,
    ):
        """The constructor sets the vertices of the fracture and sorts them optionally.
        Also computes the centroid and normals and checks for planarity and convexity.

        Parameters
        ----------
            points : numpy.ndarray(nd x npt) , or similar
                Vertices of new fracture. How many are necessary, depends on the dimension.
            index  : int , optional
                Index of fracture. Defaults to None.
            check_convexity : bool , optional
                Defaults to False.
                If True, check if the given points are convex.
                Can be skipped if it is known to be convex to save (substantial) time.
            sort_points : bool , optional
                Defaults to True.
                If True, sorts the points according to the implemented order.
                Convexity may play a role,
                thus if a fracture is known to be non-convex, the parameter should be False.

        Raises
        ------
        AssertionError
            Asserts the fracture is planar
            (PorePy currently only supports planar fractures).
        AssertionError
            (Optionally) Asserts the fracture is convex
            (PorePy currently only supports convex fractures).
        """

        self.p = np.asarray(points, dtype=np.float64)

        # Ensure the points are sorted
        if sort_points:
            self.sort_points()

        self.normal: np.ndarray = self.compute_normal()
        self.center: np.ndarray = self.compute_centroid()
        self.orig_p = self.p.copy()
        self.index = index

        assert self.is_planar(), "Points define non-planar fracture"

        if check_convexity:
            assert self.is_convex(), "Points form non-convex polygon"

    def __repr__(self) -> str:
        """Representation is same as str-representation."""
        return self.__str__()

    def __str__(self) -> str:
        """The str-representation displays the number of points and the centroid."""

        s = "Points: \n"
        s += str(self.p) + "\n"
        s += "Center: " + str(self.center)
        return s

    def __eq__(self, other) -> bool:
        """Equality is defined as two fractures having the same index.

        Notes
        -----
            There are issues with this, behavior may change in the future.
        """
        return self.index == other.index

    def set_index(self, i):
        """Set index of this fracture.
        
        TODO this is obsolete since the attribute `index` can be accessed directly
        
        Parameters
        ----------
            i : int
                Index.
        """
        self.index = i

    def points(self) -> Generator[np.ndarray, None, None]:
        """
        Generator over the vertices of the fracture.

        Yields
        ------
        numpy.ndarray(nd x 1)
            polygon vertex
        """
        for i in range(self.p.shape[1]):
            yield self.p[:, i].reshape((-1, 1))

    def segments(self) -> Generator[np.ndarray, None, None]:
        """
        Generator over the segments according to the currently applied order.

        Yields
        ------
        numpy.ndarray(nd x 2)
            polygon segment
        """

        sz = self.p.shape[1]
        for i in range(sz):
            yield self.p[:, np.array([i, i + 1]) % sz]

    def is_vertex(self, p, tol: float = 1e-4) -> Tuple[bool, Union[int, None]]:
        """Check whether a given point is a vertex of the fracture.

        Parameters
        ----------
            p : numpy.ndarray
                Point to check
            tol : float
                Defaults to 1e-4.
                Tolerance of point accuracy.

        Returns
        -------
        Tuple[bool, idx]
            The bool is True, if `p` is a vertex of this fracture, false otherwise.
            If `p` is a vertex of this fracture, `idx` is an integer,
            representing the of the identical vertex. Otherwise `idx` is None
        """
        p = p.reshape((-1, 1))
        ap = np.hstack((p, self.p))
        # TODO check if sqrt(3) is too much for other fractures than planes
        up, _, ind = setmembership.unique_columns_tol(ap, tol=tol * np.sqrt(3))

        # If the unique-operation did not remove any points, it is not a vertex.
        if up.shape[1] == ap.shape[1]:
            return False, None
        else:
            occurrences = np.where(ind == ind[0])[0]
            return True, (occurrences[1] - 1)

    def add_points(
        self,
        p: np.ndarray,
        check_convexity: bool = True,
        tol: float = 1e-4,
        enforce_pt_tol: Optional[float] = None,
    ) -> bool:
        """Add a point to the polygon with the implemented sorting enforced.

        Always run a test to check that the points are still planar.
        By default, a check of convexity is also performed, however, this can be
        turned off to speed up simulations (the test uses sympy, which turns
        out to be slow in many cases).

        Parameters:
            p : np.ndarray(nd x n)
                Points to be added
            check_convexity : bool , optional
                Defaults to true.
                Verify that the polygon is convex.
            tol : float , optional
                Defaults to 1e-4.
                Tolerance used to check if the point already exists.
            enforce_pt_tol : float , optional
                Defaults to None.
                If not None, enforces a maximal distance between points by removing points
                failing that criterion.

        Returns
        -------
        bool
            True, if the resulting polygon is planar, False otherwise.
            If the argument `check_convexity` is True, checks for planarity *AND* convexity
        """

        to_enforce = np.hstack(
            (
                np.zeros(self.p.shape[1], dtype=bool),
                np.ones(p.shape[1], dtype=bool),
            )
        )
        self.p = np.hstack((self.p, p))
        self.p, _, _ = setmembership.unique_columns_tol(self.p, tol=tol)

        # Sort points to ccw
        mask = self.sort_points()

        if enforce_pt_tol is not None:
            to_enforce = np.where(to_enforce[mask])[0]
            dist = pp.distances.pointset(self.p)
            dist /= np.amax(dist)
            np.fill_diagonal(dist, np.inf)
            mask = np.where(dist < enforce_pt_tol)[0]
            mask = np.setdiff1d(mask, to_enforce, assume_unique=True)
            self.remove_points(mask)

        if check_convexity:
            return self.is_convex() and self.is_planar(tol)
        else:
            return self.is_planar()

    def remove_points(
        self, ind: Union[np.ndarray, List[int]], keep_orig: bool = False
    ) -> None:
        """Remove points from the fracture definition.

        Parameters
        ----------
            ind : numpy.ndarray , or similar
                Indices of points to remove.
            keep_orig : bool , optional
                Defaults to False.
                If True, keeps the original points in the attribute :attr:`~Fracture.orig_p`.

        """

        self.p = np.delete(self.p, ind, axis=1)

        if not keep_orig:
            self.orig_p = self.p

    ###########################################################################################
    ### Dimension-dependent, abstract methods
    ###########################################################################################

    @abc.abstractmethod
    def sort_points(self) -> np.ndarray:
        """Abstract method to sort the vertices as needed for geometric algorithms.

        Returns
        -------
        numpy.ndarray(dtype=int)
            The indices corresponding to the sorting.
        """
        pass

    @abc.abstractmethod
    def local_coordinates(self) -> np.ndarray:
        """Abstract method for computing the vertex coordinates in a local system and its
        local dimension `d`.
        For now, `d` is expected to be `nd` - 1, i.e. the fracture to be of co-dimension 1.

        Returns
        -------
        numpy.ndarray((nd-1) x npt)
            The coordinates of the vertices in local dimensions.
        """
        pass

    @abc.abstractmethod
    def is_convex(self) -> bool:
        """Abstract method for implementing a convexity check.

        Returns
        -------
        bool
            True if the fracture is convex, False otherwise.
        """
        pass

    @abc.abstractmethod
    def is_planar(self, tol: float = 1e-4) -> bool:
        """Abstract method to check if the fracture is planar.

        Parameters
        ----------
            tol : float
                Tolerance for non-planarity. Treated as an absolute quantity
                (no scaling with fracture extent).

        Returns
        -------
        bool
            True if the fracture is planar, False otherwise.
        """
        pass

    @abc.abstractmethod
    def compute_centroid(self) -> np.ndarray:
        """Abstract method for computing and returning the centroid of the fracture.
        Convexity is assumed.
        """
        pass

    @abc.abstractmethod
    def compute_normal(self) -> np.ndarray:
        """Abstract method for computing and returning the normal vector."""
        pass
