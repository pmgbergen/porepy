import numpy as np

import porepy as pp


class ApertureMixin:
    """Aperture by subdomain type: matrix, fractures and intersections, wells."""
    
    @pp.ad.cached_method
    def aperture(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        if len(subdomains) == 0:
            b0 = super().aperture(subdomains)  # type:ignore[misc]
            return b0
        well_tags = {"injection_well", "production_well"}
        fractures_and_intersections = [
            sd for sd in subdomains if sd.dim < self.nd and not well_tags.intersection(sd.tags)
        ]
        wells_grids = [sd for sd in subdomains if well_tags.intersection(sd.tags)]
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)
        for i, sd in enumerate(subdomains):
            if sd.dim == self.nd:
                a_loc = super().aperture([sd])
            elif sd in fractures_and_intersections:
                a_loc = self.fracture_or_intersection_aperture([sd])
            elif sd in wells_grids:
                a_loc = np.ones(sd.num_cells)*self._well_radius
            else:
                a_loc = super().apperture([sd])
            a_glob = projection.cell_prolongation([sd]) @ a_loc
            if i == 0:
                aperture = a_glob
            else:
                aperture += a_glob
        aperture.set_name("aperture")
        return aperture
    

class PorosityMixin(pp.PorePyModel):
    """Porosity by subdomain type: matrix, fractures and intersections, wells."""

    def porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Porosity on all subdomains, assembled from the matrix, fracture and well laws.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise porosity operator.

        """
        well_tags = {"injection_well", "production_well"}
        subdomains_nd = [sd for sd in subdomains if sd.dim == self.nd]
        subdomains_lower = [sd for sd in subdomains if sd.dim < self.nd and not well_tags.intersection(sd.tags)]
        subdomains_wells = [sd for sd in subdomains if well_tags.intersection(sd.tags)]
        projection = pp.ad.SubdomainProjections(subdomains, dim=1)

        # Constant unitary porosity in fractures and intersections
        phi_nd = projection.cell_prolongation(subdomains_nd) @ self.porosity_matrix(
            subdomains_nd
        )
        
        phi_lower = projection.cell_prolongation(subdomains_lower) @ self.porosity_fracture_and_intersection(
            subdomains_lower
        )
        phi_wells = projection.cell_prolongation(subdomains_wells) @ self.porosity_wells(
            subdomains_wells
        )
        phi = phi_nd + phi_lower + phi_wells
        phi.set_name("porosity")
        
        return phi
    
    def porosity_fracture_and_intersection(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        # Sanity check
        """Porosity in fractures and intersections; the matrix law by default.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise porosity operator.

        """
        return self.porosity_matrix(subdomains)
    
    def porosity_wells(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        # Sanity check
        """Porosity in well grids; the matrix law by default.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise porosity operator.

        """
        return self.porosity_matrix(subdomains)


class PermeabilityMixin(pp.PorePyModel):
    
    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability on all subdomains, assembled from the matrix, well, fracture and intersection laws.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        well_tags = {"injection_well", "production_well"}
        projection = pp.ad.SubdomainProjections(subdomains, dim=9)
        matrix = [sd for sd in subdomains if sd.dim == self.nd]
        fractures = [sd for sd in subdomains if sd.dim == self.nd - 1]
        intersections = [
            sd for sd in subdomains if sd.dim == self.nd - 2 and not well_tags.intersection(sd.tags)
        ]
        wells = [sd for sd in subdomains if well_tags.intersection(sd.tags)]
        permeability = (
            projection.cell_prolongation(matrix)
            @ self.matrix_permeability(matrix)
            + projection.cell_prolongation(wells)
            @ self.well_permeability(wells)
            + projection.cell_prolongation(fractures)
            @ self.fracture_permeability(fractures)
            + projection.cell_prolongation(intersections)
            @ self.intersection_permeability(intersections)
        )
        permeability.set_name("permeability")
        return permeability
    
    def well_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return self.matrix_permeability(subdomains)
    
    def fracture_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Cubic-law permeability of the fractures.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        return self.cubic_law_permeability(subdomains)

    def intersection_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Cubic-law permeability of the intersections.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        return self.cubic_law_permeability(subdomains)
    
    def cubic_law_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Cubic-law permeability of fractures or intersections.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise permeability operator.

        """
        if len(subdomains) == 0:
            size = 0
            return pp.wrap_as_dense_ad_array(
                np.zeros((size,)), name="cubic_law_permeability"
            )
        aperture = self.aperture(subdomains)
        permeability = (aperture ** pp.ad.Scalar(2)) / pp.ad.Scalar(12)
        
        return self.isotropic_second_order_tensor(subdomains, permeability)