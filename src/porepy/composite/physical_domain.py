"""Contains a class representing physical properties."""

from __future__ import annotations

from typing import Dict

import numpy as np

import porepy as pp

from .model_solids import UnitSolid

__all__ = ["PhysicalSubdomain", "PhysicalDomain"]


class PhysicalSubdomain:
    """
    Class representing a physical wrapper of :class:`~porepy.Grid`.

    It combines the physical properties of substances with
    the space discretization of the geometry.

    In future, this class can also serve as the single point of implementation
    for heuristic laws.

    NOTE: We assume currently only a single substance.

    Currently supported heuristic laws for RELATIVE PERMEABILITY:
        - 'quadratic': quadratic power law for saturation

    Currently supported heuristic laws for POROSITY:
        - 'linear_pressure': linear pressure law
    """

    def __init__(
        self,
        grid: pp.Grid,
        substances: pp.composite.SolidComponent,
        rel_perm_law: str,
    ) -> None:
        """Constructor using geometrical and substance information.

        :param grid: the discretization for which variables and substance parameter arrays
            should be provided
        :type grid: :class:`porepy.Grid`
        :param substance: substance representative with access to physical values
        :type substance: :class:`~porepy.composite.SolidSubstance`
        """

        self.grid: pp.Grid = grid
        self.substance: pp.composite.SolidComponent = substances

        self.rel_perm_law = str(rel_perm_law)

    def __str__(self) -> str:
        """String representation combining information about geometry and material."""
        out = "Physical subdomain made of %s on grid:\n" % (self.substance.name)
        return out + str(self.grid)

    def base_porosity(self) -> pp.ad.Operator:
        """
        :return: AD representation of the base porosity
        :rtype: :class:`~porepy.numerics.ad.operators.Array`
        """
        arr = np.ones(self.grid.num_cells) * self.substance.base_porosity()

        return pp.ad.Array(arr)

    def base_permeability(self) -> pp.SecondOrderTensor:
        """
        :return: AD representation of the base permeability
        :rtype: :class:`~porepy.numerics.ad.operators.SecondOrderTensorAd`
        """
        # arr = np.ones(self.grid.num_cells) * self.substance.base_permeability()
        return pp.SecondOrderTensor(self.substance.base_permeability())

    # ------------------------------------------------------------------------------
    ### HEURISTIC LAWS NOTE all heuristic laws can be modularized somewhere and referenced here
    # ------------------------------------------------------------------------------

    def porosity(
        self, pressure: pp.ad.Variable, enthalpy: pp.ad.Variable, law: str
    ) -> pp.ad.Operator:
        """
        Currently supported heuristic laws (values for 'law'):
            - 'pressure':      linear model using base porosity and reference pressure

        Inherit this class and overwrite this method if you want to implement special models
        for the phase density.
        Use keyword arguments 'kwargs' to provide arguments for the heuristic law.

        Math. Dimension:        scalar
        Phys. Dimension:        [-] (fractional)

        :return: Ad object representing the porosity
        :rtype: :class:`porepy.ad.Operator`
        """

        law = str(law)
        if law == "pressure":

            p_ref = self.substance.poro_reference_pressure()
            phi_0 = self.substance.base_porosity()
            poro_law = lambda p: phi_0 * (p - p_ref)

            return pp.ad.Function(
                poro_law,
                "porosity-%s-%s" % (law, self.substance.name),
            )(pressure)
        else:
            raise NotImplementedError(
                "Unknown 'law' keyword for subdomain porosity.: %s \n" % (law)
                + "Available: 'pressure,'"
            )

    def relative_permeability(
        self, saturation: pp.ad.Variable, law: str
    ) -> pp.ad.Operator:
        """
        Currently supported heuristic laws (values for 'law'):
            - 'quadratic':      squared saturation values

        Math. Dimension:        scalar
        Phys. Dimension:        [-] (fractional)

        :return: relative permeability using the law specified at instantiation
        :rtype: :class:`~porepy.ad.operators.Operator`
        """
        law = str(law)
        if law == "quadratic":
            return pp.ad.Function(
                lambda S: S**2, "rel-perm-%s-%s" % (law, self.substance.name)
            )(saturation)
        else:
            raise NotImplementedError(
                "Unknown 'law' keyword for rel.Perm.: %s \n" % (law)
                + "Available: 'quadratic,'"
            )


class PhysicalDomain:
    """Physical representation of the domain. Provides functionalities to combine
    the AD framework and physical properties of respective subdomains.

    Assign :class:`~porepy.composite.UnitSolid` as default material for each subdomain at
    instantiation.

    Combines permeability and porosity properties into global entities.
    """

    def __init__(self, gb: "pp.GridBucket") -> None:
        """
        :param gb: geometric representation of the computational domain
        :type gb: :class:`~porepy.GridBucket`
        """
        ### PUBLIC
        self.gb: "pp.GridBucket" = gb
        # key: grid, value: MaterialSubdomain
        self._physical_subdomains: Dict["pp.Grid", PhysicalSubdomain] = dict()

        for grid, _ in self.gb:
            self._physical_subdomains.update(
                {grid: PhysicalSubdomain(grid, UnitSolid(self.gb))}
            )

    def __str__(self) -> str:
        """Adds information to the string representation of the gridbucket."""

        out = "Physical subdomain made of \n"
        materials = [
            subdomain.substance for subdomain in self._physical_subdomains.values()
        ]
        out += ", ".join(set(materials))
        out += "\n on grid bucker:\n" + str(self.gb)
        return out

    def get_physical_subdomain(self, grid: "pp.Grid") -> PhysicalSubdomain:
        """Returns the physical representation of a subdomain.

        :param grid: grid in this domain's grid bucket
        :type: :class:`~porepy.grids.grid.Grid`

        :return: assigned physical subdomain
        :rtype: :class:`porepy.composite.PhysicalSubdomain`
        """
        return self._physical_subdomains[grid]

    def assign_material_to_grid(
        self, grid: "pp.Grid", substance: "pp.composite.SolidSubstance"
    ) -> None:
        """
        Assigns a material to a grid i.e., creates an instance of
        :class:`~porepy.composite.PhysicalSubdomain`
        Replaces the previous instance.

        :param grid: a sub grid present in the grid bucket passed at instantiation
        :type grid: :class:`~porepy.grids.grid.Grid`

        :param substance: the substance to be associated with the subdomain
        :type substance: :class:`~porepy.composite.substance.SolidSubstance`
        """
        if grid in self.gb.get_grids():
            self._physical_subdomains.update({grid: PhysicalSubdomain(grid, substance)})
        else:
            raise KeyError("Argument 'grid' not among grids in GridBucket.")
