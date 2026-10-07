"""Shared plot style for the H2O-CO2 OBL figures: the paper colormap (seaborn `vlag`).

Matches the convention used across the paper figures (subsection_4_1 plot_reference._cmap):
vlag diverging (heavy = blue), used for saturations/densities and for non-negative difference
plots with vmin=0. Falls back to matplotlib coolwarm when seaborn is unavailable.
"""
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    def paper_cmap(name="vlag"):
        return sns.color_palette(name, as_cmap=True)
except Exception:                      # seaborn not installed
    def paper_cmap(name="vlag"):
        return plt.get_cmap("coolwarm" if name == "vlag" else "viridis")
