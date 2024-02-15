"""
Module containing a mixin class to prescribe FV flux discretization schemes in models.
"""

import porepy as pp


class FluxDiscretization:
    """Helper class with a method to set the Darcy flux variable."""

    params: dict
    """Dictionary specifying the model parameters."""

    darcy_keyword: str
    """Keyword for the Darcy (flow) problem."""

    fourier_keyword: str
    """Keyword for the Fourier (energy) problem."""

    def darcy_flux_discretization(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.MpfaAd | pp.ad.TpfaAd:
        """Discretization object for the Darcy flux term.

        Parameters:
            subdomains: List of subdomains where the Darcy flux is defined.

        Returns:
            Discretization of the Darcy flux.

        """
        scheme = self.params.get("darcy_flux_discretization", "mpfa")
        if scheme.lower() == "mpfa":
            return pp.ad.MpfaAd(self.darcy_keyword, subdomains)
        elif scheme.lower() == "tpfa":
            return pp.ad.TpfaAd(self.darcy_keyword, subdomains)
        else:
            msg = f"{scheme} is not a valid Darcy flux discretization scheme. "
            msg += "Use either 'tpfa' or 'mpfa'."
            raise NotImplementedError(msg)

    def fourier_flux_discretization(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.MpfaAd | pp.ad.TpfaAd:
        """Discretization object for the Fourier flux term.

        Parameters:
            subdomains: List of subdomains where the Fourier flux is defined.

        Returns:
            Discretization of the Fourier flux.

        """
        scheme = self.params.get("fourier_flux_discretization", "mpfa")
        if scheme.lower() == "mpfa":
            return pp.ad.MpfaAd(self.fourier_keyword, subdomains)
        elif scheme.lower() == "tpfa":
            return pp.ad.TpfaAd(self.fourier_keyword, subdomains)
        else:
            msg = f"{scheme} is not a valid Fourier flux discretization scheme. "
            msg += "Use either 'tpfa' or 'mpfa'."
            raise NotImplementedError(msg)
