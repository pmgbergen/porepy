"""Calculation script for Peng-Robinson roots."""
import csv
import itertools
import pathlib
from typing import Optional

import numpy as np

import porepy as pp

RESULTFILE: str = "roots.csv"
DELIMITER: str = ","

B_CRIT: float = pp.composite.peng_robinson.B_CRIT
A_CRIT: float = pp.composite.peng_robinson.A_CRIT
EPS: float = 1e-14
SMOOTHING_FACTOR: float = 5e-1

APPLY_SMOOTHING: bool = False

# define the upper limit for A and B by UPPER_LIM_FACTOR * A_CRIT (B_CRIT)
UPPER_LIM_FACTOR: float = 3
# Refinement between upper and lower limits
REFINEMENT: int = 300
# A flag to include A,B = 0
INCLUDE_ZERO: bool = True

A_LIMIT: list[float] = [-0, UPPER_LIM_FACTOR * A_CRIT]
# A_LIMIT: list[float] = [-A_CRIT, 3]
# A_LIMIT: list[float] = [-25, 25]
B_LIMIT: list[float] = [-0, UPPER_LIM_FACTOR * B_CRIT]
# B_LIMIT: list[float] = [-B_CRIT, 1]
# B_LIMIT: list[float] = [-25, 25]


REGION_ENCODING: list[int] = [
    0,  # triple-root-region
    1,  # 1-real-root region
    2,  # 2-root-region with multiplicity
    3,  # 3-root-region
]

HEADERS: list[str] = ["A", "B", "root case", "Z_L", "Z_G", "extended"]


def path():
    """Returns path to script calling this function as string."""
    return str(pathlib.Path(__file__).parent.resolve())


def read_root_results(
    filename: str,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[float, float], list]]:
    """Reads the results created by this module and returns

    1. An ordered vector of values for A
    2. An ordered vector of values for B
    3. A map
       (A, B) -> [root cases (int), list of roots (Liq-Intermediate-Gas), extended flag]
       where the extended flag is either 0 or 2,
       indicating that the Liq or Gas root are extended respectively.

    """
    file_path = f"{path()}/{filename}"
    A: list[float] = list()
    B: list[float] = list()
    ab_map: dict[tuple[float, float], list] = dict()
    print(f"Reading root results: file {file_path}", flush=True)
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        _ = next(reader)  # get rid of header

        for row in reader:
            A_ = float(row[0])
            B_ = float(row[1])
            reg = int(row[2])
            Z_L = float(row[3])
            # Z_I = float(row[4])
            Z_G = float(row[4])

            if row[5] == "None" or row[5] is None:
                is_extended = None
            else:
                is_extended = int(row[5])

            A.append(A_)
            B.append(B_)

            ab_map.update({(A_, B_): [reg, Z_L, Z_G, is_extended]})
    print(f"Reading root results: done", flush=True)

    A_vec = np.array(A)
    B_vec = np.array(B)
    A_vec = np.sort(np.unique(A_vec))
    B_vec = np.sort(np.unique(B_vec))

    return A_vec, B_vec, ab_map


def Z_I_smoother(
    Z_L: np.ndarray, Z_I: np.ndarray, Z_G: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothing procedure on boundaries of three-root-region.

    Smooths the value and Jacobian rows of the liquid and gas root close to
    phase boundaries.

    See Also:
        `Vu et al. (2021), Section 6.
        <https://doi.org/10.1016/j.matcom.2021.07.015>`_

    Parameters:
        Z_L: liquid-like root.
        Z_I: intermediate root.
        Z_G: gas-like root.

    Returns:
        A tuple containing the smoothed liquid and gas root as AD arrays

    """
    # proximity:
    # If close to 1, intermediate root is close to gas root.
    # If close to 0, intermediate root is close to liquid root.
    # values bound by [0,1]
    proximity: np.ndarray = (Z_I - Z_L) / (Z_G - Z_L)

    # average intermediate and gas root for gas root smoothing
    W_G = (Z_I + Z_G) / 2
    # analogously for liquid root
    W_L = (Z_I + Z_L) / 2

    v_G = _gas_smoother(proximity)
    v_L = _liquid_smoother(proximity)

    # smoothing values with convex combination
    Z_G_val = (1 - v_G) * Z_G + v_G * W_G
    Z_L_val = (1 - v_L) * Z_L + v_L * W_L

    return Z_L_val, Z_G_val


def _gas_smoother(proximity: np.ndarray) -> np.ndarray:
    """Smoothing function for three-root-region where the intermediate root comes
    close to the gas root."""
    # smoother starts with zero values
    smoother = np.zeros(proximity.shape[0])
    # values around smoothing parameter are constructed according to Vu
    upper_bound = proximity < 1 - SMOOTHING_FACTOR
    lower_bound = (1 - 2 * SMOOTHING_FACTOR) < proximity
    bound = np.logical_and(upper_bound, lower_bound)

    bound_smoother = (proximity[bound] - (1 - 2 * SMOOTHING_FACTOR)) / SMOOTHING_FACTOR
    bound_smoother = bound_smoother**2 * (3 - 2 * bound_smoother)

    smoother[bound] = bound_smoother
    # where proximity is close to one, set value of one
    smoother[proximity >= 1 - SMOOTHING_FACTOR] = 1.0

    return smoother


def _liquid_smoother(proximity: np.ndarray) -> np.ndarray:
    """Smoothing function for three-root-region where the intermediate root comes
    close to the liquid root."""
    # smoother starts with zero values
    smoother = np.zeros(proximity.shape[0])
    # values around smoothing parameter are constructed according to Vu
    upper_bound = proximity < 2 * SMOOTHING_FACTOR
    lower_bound = SMOOTHING_FACTOR < proximity
    bound = np.logical_and(upper_bound, lower_bound)

    bound_smoother = (proximity[bound] - SMOOTHING_FACTOR) / SMOOTHING_FACTOR
    bound_smoother = (-1) * bound_smoother**2 * (3 - 2 * bound_smoother) + 1

    smoother[bound] = bound_smoother
    # where proximity is close to zero, set value of one
    smoother[proximity <= SMOOTHING_FACTOR] = 1.0

    return smoother


def calc_Z(
    A: np.ndarray,
    B: np.ndarray,
    apply_W_correction: bool = False,
    apply_smoother: bool = False,
) -> tuple[int, list[np.ndarray], Optional[int]]:
    """Algorithm for calculating (extended roots) of the compressibility polynomial.

    Parameters:
        A: Dimensionless cohesion
        B: Dimensionless covolume
        apply_W_correction: In the case of 1 real root, apply corrective algorithm.
        apply_smoother: In the case of 3 real roots, apply smoothing procedure

    Returns:
        A tuple containing the polynomial root case, the roots (liquid- intermediate - gas),
        and an optional integer, saying which one is extended, in the case of 1 real root.

        The intermediate root is zero in all cases except in the 3-root case
    """
    # the coefficients of the compressibility polynomial
    c0 = np.power(B, 3) + np.power(B, 2) - A * B
    c1 = A - 2 * B - 3 * np.power(B, 2)
    c2 = B - 1

    # the coefficients of the reduced polynomial (elimination of 2-monomial)
    r = c1 - np.power(c2, 2) / 3
    q = 2 / 27 * np.power(c2, 3) - c2 * c1 / 3 + c0

    # discriminant to determine the number of roots
    delta = np.power(q, 2) / 4 + np.power(r, 3) / 27

    # prepare storage for roots
    # storage for roots
    nc = len(delta)
    Z_L = np.zeros(nc)
    Z_I = np.zeros(nc)
    Z_G = np.zeros(nc)

    ### CLASSIFYING REGIONS
    # identify super-critical region and store information
    # limited physical insight into what this means with this EoS
    is_supercritical = B >= B_CRIT / A_CRIT * A
    # rectangle with upper right corner at (Ac,Bc)
    acbc_rect = np.logical_and(A < A_CRIT, B < B_CRIT)
    # subcritical triangle in the acbc rectangle
    subc_triang = np.logical_and(np.logical_not(is_supercritical), B < B_CRIT)
    # supercritical triangle in the acbc rectangle
    sc_triang = np.logical_and(acbc_rect, np.logical_not(subc_triang))
    # trapezoid above the acbc recangle, bound supercritical slope (supercrit)
    B_trapez = np.logical_and(is_supercritical, B >= B_CRIT)
    # trapezoid right of the acbc rectangle, bound by supercritical slope (sub-crit)
    A_trapez = np.logical_and(A > A_CRIT, np.logical_not(is_supercritical))

    # discriminant of zero indicates triple or two real roots with multiplicity
    degenerate_region = np.isclose(delta, 0.0, atol=EPS)

    double_root_region = np.logical_and(degenerate_region, np.abs(r) > EPS)
    triple_root_region = np.logical_and(degenerate_region, np.isclose(r, 0.0, atol=EPS))

    one_root_region = delta > EPS
    three_root_region = delta < -EPS

    is_extended: Optional[int] = None

    # sanity check that every cell/case covered
    assert np.all(
        np.logical_or.reduce(
            [
                one_root_region,
                triple_root_region,
                double_root_region,
                three_root_region,
            ]
        )
    ), "Uncovered region for polynomial root cases."

    # sanity check that the regions are mutually exclusive
    # this array must have 1 in every entry for the test to pass
    trues_per_row = np.vstack(
        [one_root_region, triple_root_region, double_root_region, three_root_region]
    ).sum(axis=0)
    # TODO assert dtype does not compromise test
    trues_check = np.ones(nc, dtype=trues_per_row.dtype)
    assert np.all(
        trues_check == trues_per_row
    ), "Regions with different root scenarios overlap."

    ### COMPUTATIONS IN THE ONE-ROOT-REGION
    # Missing real root is replaced with conjugated imaginary roots
    if np.all(one_root_region):

        # delta has only positive values in this case by logic
        t_1 = -q / 2 + np.sqrt(delta)

        # t_1 might be negative, in this case we must choose the real cubic root
        # by extracting cbrt(-1), where -1 is the real cubic root.
        im_cube = t_1 < 0.0
        if np.any(im_cube):
            t_1[im_cube] *= -1

            u_1 = np.cbrt(t_1)

            u_1[im_cube] *= -1
        else:
            u_1 = np.cbrt(t_1)

        ## Relevant roots
        # only real root, Cardano formula, positive discriminant
        # TODO If u really tiny, this yields infinity
        real_part = u_1 - r / (u_1 * 3)
        z_1 = real_part - c2 / 3
        # real part of the conjugate imaginary roots
        # used for extension of vanished roots
        w_1 = (1 - B - z_1) / 2
        # w_1 = -real_part / 2 - c2_1 / 3

        ## simplified labeling, Vu et al. (2021), equ. 4.24
        gas_region = w_1 < z_1
        liquid_region = z_1 < w_1
        # assert the whole one-root-region is covered
        assert np.all(
            np.logical_or(gas_region, liquid_region)
        ), "Phase labeling does not cover whole one-root-region."
        # assert mutual exclusion to check sanity
        assert np.all(
            np.logical_not(np.logical_and(gas_region, liquid_region))
        ), "Labeled subregions in one-root-region overlap."

        # # Correction
        if apply_W_correction:
            correction = np.logical_or(
                np.logical_not(acbc_rect[one_root_region]),
                w_1 <= B,
            )
            # W must be greater than B
            phys_bound = w_1 <= B
            w_1[correction] = z_1[correction]

        # store values in one-root-region
        nc_1 = np.count_nonzero(one_root_region)
        Z_L_val_1 = np.zeros(nc_1)
        Z_G_val_1 = np.zeros(nc_1)

        # store gas root where actual gas, use extension where liquid
        Z_G_val_1[gas_region] = z_1[gas_region]
        Z_G_val_1[liquid_region] = w_1[liquid_region]

        # store liquid where actual liquid, use extension where gas
        Z_L_val_1[liquid_region] = z_1[liquid_region]
        Z_L_val_1[gas_region] = w_1[gas_region]

        # store values in global root structure
        Z_L[one_root_region] = Z_L_val_1
        Z_G[one_root_region] = Z_G_val_1

        if np.all(gas_region):
            is_extended = 0
        else:
            if np.all(liquid_region):
                is_extended = 2
            else:
                raise RuntimeError(
                    "Something went wrong with labeling which is extended."
                )
        assert is_extended is not None, "is_extended is still None"
        return REGION_ENCODING[0], [Z_L, Z_I, Z_G], is_extended

    ### COMPUTATIONS IN TRIPLE ROOT REGION
    # the single real root is returned.
    if np.all(triple_root_region):

        z_triple = -c2 / 3

        # store root where it belongs
        Z_L[triple_root_region] = z_triple
        Z_G[triple_root_region] = z_triple

        return REGION_ENCODING[1], [Z_L, Z_I, Z_G], None

    ### COMPUTATIONS IN DOUBLE ROOT REGION
    # compute both roots and label the bigger one as the gas root
    if np.any(double_root_region):

        u = 3 / 2 * q / r

        z_1 = 2 * u - c2 / 3
        z_23 = -u - c2 / 3

        # choose bigger root as gas like
        # theoretically they should strictly be different, otherwise it would be
        # the three root case
        double_is_gaslike = z_23 > z_1
        double_is_liquidlike = z_23 < z_1
        assert np.all(
            np.logical_or(double_is_gaslike, double_is_liquidlike)
        ), "Double root is equal other."

        # store values in double-root-region
        nc_d = np.count_nonzero(double_root_region)
        Z_L_val_d = np.zeros(nc_d)
        Z_G_val_d = np.zeros(nc_d)

        # store bigger as gas root, smaller as liquid root
        Z_G_val_d[double_is_gaslike] = z_23[double_is_gaslike]
        Z_G_val_d[double_is_liquidlike] = z_1[double_is_liquidlike]
        # store liquid where actual liquid, use extension where gas
        Z_L_val_d[double_is_gaslike] = z_1[double_is_gaslike]
        Z_L_val_d[double_is_liquidlike] = z_23[double_is_liquidlike]

        # store values in global root structure
        Z_L[double_root_region] = Z_L_val_d
        Z_G[double_root_region] = Z_G_val_d

        return REGION_ENCODING[2], [Z_L, Z_I, Z_G], None

    ### COMPUTATIONS IN THE THREE-ROOT-REGION
    # compute all three roots, label them (smallest=liquid, biggest=gas)
    # optionally smooth them
    if np.all(three_root_region):

        # compute roots in three-root-region using Cardano formula,
        # Casus Irreducibilis
        t_2 = np.arccos(-q / 2 * np.sqrt(-27 * np.power(r, -3))) / 3
        t_1 = np.sqrt(-4 / 3 * r)

        z3_3 = t_1 * np.cos(t_2) - c2 / 3
        z2_3 = -t_1 * np.cos(t_2 + np.pi / 3) - c2 / 3
        z1_3 = -t_1 * np.cos(t_2 - np.pi / 3) - c2 / 3

        # assert roots are ordered by size
        assert np.all(z1_3 <= z2_3) and np.all(
            z2_3 <= z3_3
        ), "Roots in three-root-region improperly ordered."

        ## Smoothing of roots close to double-real-root case
        # this happens when the phase changes, at the phase border the polynomial
        # can have a real double root.
        if apply_smoother:
            Z_L_3, Z_G_3 = Z_I_smoother(z1_3, z2_3, z3_3)
        else:
            Z_L_3, Z_G_3 = (z1_3, z3_3)

        ## Labeling in the three-root-region follows topological patterns
        # biggest root belongs to gas phase
        # smallest root belongs to liquid phase
        Z_L[three_root_region] = Z_L_3
        Z_I[three_root_region] = z2_3
        Z_G[three_root_region] = Z_G_3

        return REGION_ENCODING[3], [Z_L, Z_I, Z_G], None

    raise RuntimeError("No region case was triggered.")


def calc_Z_mod(
    A: np.ndarray,
    B: np.ndarray,
    apply_W_correction: bool = False,
    apply_smoother: bool = False,
) -> tuple[int, list[np.ndarray], Optional[int]]:
    """Algorithm for calculating (extended roots) of the compressibility polynomial.

    Parameters:
        A: Dimensionless cohesion
        B: Dimensionless covolume
        apply_W_correction: In the case of 1 real root, apply corrective algorithm.
        apply_smoother: In the case of 3 real roots, apply smoothing procedure

    Returns:
        A tuple containing the polynomial root case, the roots (liquid- intermediate - gas),
        and an optional integer, saying which one is extended, in the case of 1 real root.

        The intermediate root is zero in all cases except in the 3-root case
    """
    # the coefficients of the compressibility polynomial
    c0 = np.power(B, 3) + np.power(B, 2) - A * B
    c1 = A - 2 * B - 3 * np.power(B, 2)
    c2 = B - 1

    # the coefficients of the reduced polynomial (elimination of 2-monomial)
    r = c1 - np.power(c2, 2) / 3
    q = 2 / 27 * np.power(c2, 3) - c2 * c1 / 3 + c0

    # discriminant to determine the number of roots
    delta = np.power(q, 2) / 4 + np.power(r, 3) / 27

    # prepare storage for roots
    # storage for roots
    nc = len(delta)
    Z_L = np.zeros(nc)
    Z_I = np.zeros(nc)
    Z_G = np.zeros(nc)

    ### CLASSIFYING REGIONS
    # identify super-critical region and store information
    # limited physical insight into what this means with this EoS
    is_supercritical = B >= B_CRIT / A_CRIT * A
    # rectangle with upper right corner at (Ac,Bc)
    acbc_rect = np.logical_and(
        np.logical_and(EPS < A, A < A_CRIT - EPS),
        np.logical_and(EPS < B, B < B_CRIT - EPS),
    )
    # This can happen if some fraction is slightly negative and pushed B outside
    A_neg_halfplain = A <= EPS
    # subcritical triangle in the acbc rectangle
    subc_triang = np.logical_and(np.logical_not(is_supercritical), acbc_rect)
    # supercritical triangle in the acbc rectangle
    sc_triang = np.logical_and(acbc_rect, np.logical_not(subc_triang))
    # this is the larger, outer super-critical region, in the positive A half-plain
    # it includes the positive A axis, where the smaller (extended) root possibly
    # violates the lower bound B
    outer_region = np.logical_and(
        np.logical_not(A_neg_halfplain), np.logical_not(acbc_rect)
    )
    # At A,B=0 we have 2 real roots, one with multiplicity 2
    zero_point = np.logical_and(np.isclose(A, 0, atol=EPS), np.isclose(B, 0, atol=EPS))
    # The critical point is known to be a triple-point
    critical_point = np.logical_and(
        np.isclose(A, A_CRIT, rtol=0, atol=EPS), np.isclose(B, B_CRIT, rtol=0, atol=EPS)
    )

    # discriminant of zero indicates triple or two real roots with multiplicity
    degenerate_region = np.isclose(delta, 0.0, atol=EPS)

    double_root_region = np.logical_and(degenerate_region, np.abs(r) > EPS)
    triple_root_region = np.logical_and(degenerate_region, np.isclose(r, 0.0, atol=EPS))

    one_root_region = delta > EPS
    three_root_region = delta < -EPS

    is_extended: Optional[int] = None

    # sanity check that every cell/case covered
    assert np.all(
        np.logical_or.reduce(
            [
                one_root_region,
                triple_root_region,
                double_root_region,
                three_root_region,
            ]
        )
    ), "Uncovered region for polynomial root cases."

    # sanity check that the regions are mutually exclusive
    # this array must have 1 in every entry for the test to pass
    trues_per_row = np.vstack(
        [one_root_region, triple_root_region, double_root_region, three_root_region]
    ).sum(axis=0)
    # TODO assert dtype does not compromise test
    trues_check = np.ones(nc, dtype=trues_per_row.dtype)
    assert np.all(
        trues_check == trues_per_row
    ), "Regions with different root scenarios overlap."

    ### COMPUTATIONS IN THE ONE-ROOT-REGION
    # Missing real root is replaced with conjugated imaginary roots
    region = np.logical_and(one_root_region, subc_triang)
    if np.all(region):

        # delta has only positive values in this case by logic
        t_1 = -q / 2 + np.sqrt(delta)

        # t_1 might be negative, in this case we must choose the real cubic root
        # by extracting cbrt(-1), where -1 is the real cubic root.
        im_cube = t_1 < 0.0
        if np.any(im_cube):
            t_1[im_cube] *= -1

            u_1 = np.cbrt(t_1)

            u_1[im_cube] *= -1
        else:
            u_1 = np.cbrt(t_1)

        real_part = u_1 - r / (u_1 * 3)

        z_1 = real_part - c2 / 3

        w_1 = (1 - B - z_1) / 2
        # w_1 = -real_part / 2 - c2_1 / 3

        assert np.all(w_1 > B), "Extended root violates B-bound in subcrit triangle."
        assert np.all(z_1 > B), "Single real root violates B-bound in subcrit triangle."

        extension_is_bigger = w_1 > z_1
        extension_is_smaller = np.logical_not(extension_is_bigger)

        big_root = z_1.copy()
        big_root[extension_is_bigger] = w_1[extension_is_bigger]

        small_root = z_1.copy()
        small_root[extension_is_smaller] = w_1[extension_is_smaller]

        assert np.all(small_root <= big_root), "Roots are not ordered by size."

        # store values in global root structure
        Z_L[region] = small_root
        Z_G[region] = big_root

        if np.all(extension_is_smaller):
            is_extended = 0
        else:
            if np.all(extension_is_bigger):
                is_extended = 2
            else:
                raise RuntimeError(
                    "Something went wrong with labeling which is extended."
                )
        assert is_extended is not None, "is_extended is still None"
        return REGION_ENCODING[0], [Z_L, Z_I, Z_G], is_extended

    region = np.logical_and(one_root_region, np.logical_not(subc_triang))
    if np.all(region):

        # delta has only positive values in this case by logic
        t_1 = -q / 2 + np.sqrt(delta)

        # t_1 might be negative, in this case we must choose the real cubic root
        # by extracting cbrt(-1), where -1 is the real cubic root.
        im_cube = t_1 < 0.0
        if np.any(im_cube):
            t_1[im_cube] *= -1

            u_1 = np.cbrt(t_1)

            u_1[im_cube] *= -1
        else:
            u_1 = np.cbrt(t_1)

        real_part = u_1 - r / (u_1 * 3)

        z_1 = real_part - c2 / 3

        w_1 = (1 - B - z_1) / 2
        # w_1 = -real_part / 2 - c2_1 / 3

        extension_is_bigger = w_1 > z_1
        extension_is_smaller = np.logical_not(extension_is_bigger)

        big_root = z_1.copy()
        big_root[extension_is_bigger] = w_1[extension_is_bigger]

        small_root = z_1.copy()
        small_root[extension_is_smaller] = w_1[extension_is_smaller]

        assert np.all(
            big_root > B
        ), "Bigger root violates B-bound in 1-root-region outside of subcrit triangle."

        correction = small_root <= B
        if np.any(correction):
            c = B + EPS
            over_corrected = c >= big_root
            if np.any(over_corrected):
                c = B + (big_root - B) / 2
            small_root = c

        # store values in global root structure
        Z_L[region] = small_root
        Z_G[region] = big_root

        if np.all(extension_is_smaller):
            is_extended = 0
        else:
            if np.all(extension_is_bigger):
                is_extended = 2
            else:
                raise RuntimeError(
                    "Something went wrong with labeling which is extended."
                )
        assert is_extended is not None, "is_extended is still None"
        return REGION_ENCODING[0], [Z_L, Z_I, Z_G], is_extended

    ### COMPUTATIONS IN THE THREE-ROOT-REGION
    # compute all three roots, label them (smallest=liquid, biggest=gas)
    # optionally smooth them
    region = np.logical_and(three_root_region, subc_triang)
    if np.all(region):

        # compute roots in three-root-region using Cardano formula,
        # Casus Irreducibilis
        t_2 = np.arccos(-q / 2 * np.sqrt(-27 * np.power(r, -3))) / 3
        t_1 = np.sqrt(-4 / 3 * r)

        z3_3 = t_1 * np.cos(t_2) - c2 / 3
        z2_3 = -t_1 * np.cos(t_2 + np.pi / 3) - c2 / 3
        z1_3 = -t_1 * np.cos(t_2 - np.pi / 3) - c2 / 3

        # assert roots are ordered by size
        assert np.all(z1_3 <= z2_3) and np.all(
            z2_3 <= z3_3
        ), "Roots in three-root-region improperly ordered."
        assert np.all(
            z1_3 > B
        ), "Sub-critical three-root-region violates lower bound by covolume"

        ## Smoothing of roots close to double-real-root case
        # this happens when the phase changes, at the phase border the polynomial
        # can have a real double root.
        if apply_smoother:
            Z_L_3, Z_G_3 = Z_I_smoother(z1_3, z2_3, z3_3)
        else:
            Z_L_3, Z_G_3 = (z1_3, z3_3)

        ## Labeling in the three-root-region follows topological patterns
        # biggest root belongs to gas phase
        # smallest root belongs to liquid phase
        Z_L[region] = Z_L_3
        Z_I[region] = z2_3
        Z_G[region] = Z_G_3

        return REGION_ENCODING[3], [Z_L, Z_I, Z_G], None

        ### COMPUTATIONS IN TRIPLE ROOT REGION

    region = np.logical_and(three_root_region, np.logical_not(subc_triang))
    if np.all(region):

        # compute roots in three-root-region using Cardano formula,
        # Casus Irreducibilis
        t_2 = np.arccos(-q / 2 * np.sqrt(-27 * np.power(r, -3))) / 3
        t_1 = np.sqrt(-4 / 3 * r)

        z3_3 = t_1 * np.cos(t_2) - c2 / 3
        z2_3 = -t_1 * np.cos(t_2 + np.pi / 3) - c2 / 3
        z1_3 = -t_1 * np.cos(t_2 - np.pi / 3) - c2 / 3

        # assert roots are ordered by size
        assert np.all(z1_3 <= z2_3) and np.all(
            z2_3 <= z3_3
        ), "Roots in three-root-region improperly ordered."
        assert np.all(
            z3_3 > B
        ), "Bigger root violates B bound outside of subcrit triangle."

        correction = z1_3 <= B
        if np.any(correction):
            c = B + EPS
            over_corrected = c >= z3_3
            if np.any(over_corrected):
                c = B + (z3_3 - B) / 2

            z1_3[correction] = c

        ## Labeling in the three-root-region follows topological patterns
        # biggest root belongs to gas phase
        # smallest root belongs to liquid phase
        Z_L[region] = z1_3
        Z_I[region] = z2_3
        Z_G[region] = z3_3

        return REGION_ENCODING[3], [Z_L, Z_I, Z_G], None

    ### COMPUTATION IN THE TRIPLE-ROOT-REGIOn
    region = np.logical_or(triple_root_region, critical_point)
    if np.all(region):

        z_triple = -c2 / 3

        # store root where it belongs
        Z_L[region] = z_triple
        Z_G[region] = z_triple

        return REGION_ENCODING[1], [Z_L, Z_I, Z_G], None

    ### COMPUTATIONS IN DOUBLE ROOT REGION
    region = np.logical_or(double_root_region, zero_point)
    if np.any(region):

        u = 3 / 2 * q / r

        z_1 = 2 * u - c2 / 3
        z_23 = -u - c2 / 3

        # choose bigger root as gas like
        # theoretically they should strictly be different, otherwise it would be
        # the three root case
        double_is_bigger = z_23 > z_1
        double_is_smaller = np.logical_not(double_is_bigger)

        # store values in double-root-region
        nc_d = np.count_nonzero(region)
        small_root = np.zeros(nc_d)
        big_root = np.zeros(nc_d)

        big_root[double_is_bigger] = z_23[double_is_bigger]
        big_root[double_is_smaller] = z_1[double_is_smaller]

        small_root[double_is_bigger] = z_1[double_is_bigger]
        small_root[double_is_smaller] = z_23[double_is_smaller]

        correction = small_root <= B
        if np.any(correction):
            c = B + EPS
            over_corrected = c >= big_root
            if np.any(over_corrected):
                c[over_corrected] = (
                    B + (big_root[over_corrected] - B[over_corrected]) / 2
                )

            small_root[correction] = c

        # store values in global root structure
        Z_L[region] = small_root
        Z_G[region] = big_root

        return REGION_ENCODING[2], [Z_L, Z_I, Z_G], None

    raise RuntimeError("No region case was triggered.")


if __name__ == "__main__":

    # TODO this is a problematic A,B combo were there is a division by zero error in
    # one root region
    # test = calc_Z(np.array([0.3620392380873223]), np.array([-0.4204815080014268]))

    A = np.linspace(A_LIMIT[0], A_LIMIT[1], REFINEMENT)
    B = np.linspace(B_LIMIT[0], B_LIMIT[1], REFINEMENT)

    if INCLUDE_ZERO:
        A = np.hstack([A, np.array([0.0])])
        A = np.sort(np.unique(A))
        B = np.hstack([B, np.array([0.0])])
        B = np.sort(np.unique(B))

    A_mesh, B_mesh = np.meshgrid(A, B)

    n, m = A_mesh.shape
    nm = n * m
    file_path = f"{path()}/{RESULTFILE}"

    # compute how far the root non-extended roots deviate from being a zero
    rows: list = []
    rows.append(HEADERS)

    print("Calculating roots: ...", end="", flush=True)
    counter: int = 1
    eos = pp.composite.peng_robinson.PengRobinson(
        False, smoothing_factor=SMOOTHING_FACTOR, eps=EPS
    )
    func = eos._Z
    # func = calc_Z_mod
    for i in range(n):
        for j in range(m):
            A_ = A_mesh[i, j]
            B_ = B_mesh[i, j]

            A_ij = np.array([A_])
            B_ij = np.array([B_])

            Z_L, Z_G, reg = func(A_ij, B_ij, apply_smoother=APPLY_SMOOTHING)
            is_extended = int(eos.is_extended[0])
            Z_L = float(Z_L[0])
            Z_G = float(Z_G[0])
            # Z_L = roots[0][0]
            # Z_I = roots[1][0]
            # Z_G = roots[2][0]

            rows.append([A_, B_, reg, Z_L, Z_G, is_extended])

            print(f"\rCalculating roots: {counter}/{nm}", end="", flush=True)
            counter += 1
    print("", flush=True)

    print(f"Writing results to: {file_path}", flush=True)
    with open(file_path, "w") as file:
        writer = csv.writer(file, delimiter=DELIMITER)
        for row in rows:
            writer.writerow(row)

    # Sanity check
    print("Sanity check ...", flush=True)
    A_vec, B_vec, ab_map = read_root_results(RESULTFILE)
    row: int = 1
    for a, b in itertools.product(A_vec, B_vec):
        r = ab_map[(a, b)]
        assert isinstance(r[0], int), f"Root region not readable as int: row {row}"
        assert isinstance(r[1], float), f"Liquid root not readable as float: row {row}"
        # assert isinstance(r[2], float), f"Intermediate not readable as float: row {row}"
        assert isinstance(r[2], float), f"Gas root not readable as float: row {row}"
        assert isinstance(
            r[3], int
        ), f"Extension flag not readable as int or None: row {row}"
    print("Done", flush=True)
