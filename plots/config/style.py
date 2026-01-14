"""
Plot Style Configuration

Provides unified styling for all thesis plots, ensuring visual consistency
across figures. Meets NeurIPS-grade publication standards:
- Color-blind safe palettes
- High DPI for print (300+)
- Consistent typography readable at single-column width
- Grayscale legibility preserved
- Clean layout without chart junk
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

PALETTES: dict[str, list[str]] = {
    "colorblind": [
        "#0173B2",
        "#DE8F05",
        "#029E73",
        "#CC78BC",
        "#ECE133",
        "#56B4E9",
    ],
    "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    "vibrant": ["#EE7733", "#0077BB", "#33BBEE", "#EE3377", "#CC3311", "#009988"],
    "muted": ["#88CCEE", "#44AA99", "#117733", "#332288", "#DDCC77", "#999933"],
    "dark": ["#222255", "#225555", "#225522", "#666633", "#663333", "#555555"],
}

MARKERS: list[str] = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES: list[str] = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

FIGURE_SIZES: dict[str, tuple[float, float]] = {
    "single_column": (3.5, 2.8),
    "double_column": (7.0, 4.0),
    "full_width": (7.5, 5.0),
    "full_page": (7.5, 9.0),
    "square": (4.0, 4.0),
    "wide": (7.5, 3.5),
}


@dataclass
class PlotStyle:
    """Base styling configuration for plots."""

    palette: str = "colorblind"
    theme: str = "whitegrid"
    font_family: str = "sans-serif"
    font_size: int = 11
    title_size: int = 14
    label_size: int = 11
    legend_size: int = 9
    tick_size: int = 9
    dpi: int = 150
    figsize: tuple[float, float] = (15.0, 10.0)
    alpha: float = 0.8
    linewidth: float = 2.0
    markersize: float = 8.0

    def apply(self) -> None:
        """Apply style settings to matplotlib rcParams."""
        sns.set_style(self.theme)
        plt.rcParams.update(
            {
                "font.family": self.font_family,
                "font.size": self.font_size,
                "axes.titlesize": self.title_size,
                "axes.labelsize": self.label_size,
                "legend.fontsize": self.legend_size,
                "xtick.labelsize": self.tick_size,
                "ytick.labelsize": self.tick_size,
                "figure.dpi": self.dpi,
            }
        )

    def get_colors(self, n: int) -> list[str]:
        """
        Get n colors from selected palette.

        If n exceeds palette size, cycles through palette to extend.

        Args:
            n: Number of colors needed

        Returns:
            List of hex color strings
        """
        base_colors = PALETTES.get(self.palette, PALETTES["colorblind"])

        if n <= len(base_colors):
            return base_colors[:n]

        extended = []
        while len(extended) < n:
            extended.extend(base_colors)
        return extended[:n]


@dataclass
class ThesisStyle(PlotStyle):
    """
    NeurIPS-grade publication styling for thesis figures.

    Standards:
    - Vector formats first (PDF, SVG), raster as secondary
    - High DPI for raster (300+)
    - Color-blind safe palettes with marker/linestyle distinction
    - Clean typography readable at single-column width
    - No chart junk (top/right spines removed)
    - Subtle grid usage
    - Consistent margins and aspect ratios
    """

    palette: str = "colorblind"
    theme: str = "white"
    font_family: str = "serif"
    font_size: int = 9
    title_size: int = 10
    label_size: int = 9
    legend_size: int = 8
    tick_size: int = 8
    display_dpi: int = 100
    savefig_dpi: int = 300
    figsize: tuple[float, float] = (7.0, 4.5)
    alpha: float = 0.85
    linewidth: float = 1.5
    markersize: float = 5.0
    grid_alpha: float = 0.25
    ci_alpha: float = 0.2
    edge_color: str = "black"
    edge_width: float = 0.5
    capsize: float = 3.0
    default_formats: list[str] = field(default_factory=lambda: ["pdf", "png"])

    def apply(self) -> None:
        """Apply NeurIPS-grade style settings."""
        plt.rcParams.update(
            {
                "font.family": self.font_family,
                "font.size": self.font_size,
                "axes.titlesize": self.title_size,
                "axes.labelsize": self.label_size,
                "axes.titleweight": "bold",
                "axes.labelweight": "normal",
                "legend.fontsize": self.legend_size,
                "legend.framealpha": 0.9,
                "legend.edgecolor": "0.8",
                "xtick.labelsize": self.tick_size,
                "ytick.labelsize": self.tick_size,
                "xtick.major.size": 3,
                "ytick.major.size": 3,
                "xtick.minor.size": 1.5,
                "ytick.minor.size": 1.5,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "figure.dpi": self.display_dpi,
                "savefig.dpi": self.savefig_dpi,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.05,
                "axes.grid": True,
                "grid.alpha": self.grid_alpha,
                "grid.linewidth": 0.5,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": 0.8,
                "lines.linewidth": self.linewidth,
                "lines.markersize": self.markersize,
                "errorbar.capsize": self.capsize,
                "figure.constrained_layout.use": False,
                "figure.autolayout": False,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        sns.set_style(
            "white",
            {
                "axes.edgecolor": "0.2",
                "axes.grid": True,
                "grid.color": "0.85",
            },
        )
        plt.rcParams.update(
            {
                "axes.spines.top": False,
                "axes.spines.right": False,
            }
        )

    def get_marker(self, idx: int) -> str:
        """Get marker for index, cycling through available markers."""
        return MARKERS[idx % len(MARKERS)]

    def get_linestyle(self, idx: int) -> str:
        """Get linestyle for index, cycling through available styles."""
        return LINESTYLES[idx % len(LINESTYLES)]

    def get_distinguishable_style(self, idx: int) -> dict:
        """
        Get a combination of color, marker, and linestyle for index.

        Ensures curves are distinguishable even in grayscale.
        """
        colors = PALETTES[self.palette]
        return {
            "color": colors[idx % len(colors)],
            "marker": MARKERS[idx % len(MARKERS)],
            "linestyle": LINESTYLES[idx % len(LINESTYLES)],
            "linewidth": self.linewidth,
            "markersize": self.markersize,
        }

    def get_figsize(self, size_name: str) -> tuple[float, float]:
        """Get standardized figure size by name."""
        return FIGURE_SIZES.get(size_name, self.figsize)


def save_figure_thesis(
    fig: plt.Figure,
    path: Path | str,
    formats: list[str] | None = None,
    close: bool = True,
) -> list[Path]:
    """
    Save figure in thesis-quality formats.

    Vector formats (PDF, SVG) are prioritized for publication.
    PNG is generated as secondary raster format.

    Args:
        fig: Matplotlib figure to save
        path: Base path without extension
        formats: List of formats (default: ['pdf', 'png'])
        close: Whether to close figure after saving

    Returns:
        List of saved file paths
    """
    if formats is None:
        formats = ["pdf", "png"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for fmt in formats:
        out_path = path.with_suffix(f".{fmt}")
        fig.savefig(out_path, format=fmt, bbox_inches="tight", pad_inches=0.05)
        saved_paths.append(out_path)

    if close:
        plt.close(fig)

    return saved_paths
