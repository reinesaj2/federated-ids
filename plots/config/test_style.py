"""Tests for plotting style configuration."""

import matplotlib.pyplot as plt

from plots.config.style import PALETTES, PlotStyle, ThesisStyle


class TestPalettes:
    def test_colorblind_palette_has_minimum_colors(self):
        assert len(PALETTES["colorblind"]) >= 5

    def test_all_palettes_have_valid_hex_colors(self):
        for palette_name, colors in PALETTES.items():
            for color in colors:
                assert color.startswith("#"), f"{palette_name}: {color} not hex"
                assert len(color) == 7, f"{palette_name}: {color} wrong length"

    def test_palettes_have_required_keys(self):
        required = {"colorblind", "default", "vibrant", "muted", "dark"}
        assert required.issubset(set(PALETTES.keys()))


class TestPlotStyle:
    def test_default_values(self):
        style = PlotStyle()
        assert style.palette == "colorblind"
        assert style.font_size > 0
        assert style.dpi > 0

    def test_get_colors_returns_requested_count(self):
        style = PlotStyle()
        colors = style.get_colors(3)
        assert len(colors) == 3

    def test_get_colors_extends_palette_when_needed(self):
        style = PlotStyle()
        base_count = len(PALETTES[style.palette])
        colors = style.get_colors(base_count + 5)
        assert len(colors) == base_count + 5

    def test_get_colors_returns_valid_hex(self):
        style = PlotStyle()
        colors = style.get_colors(10)
        for color in colors:
            assert color.startswith("#")

    def test_apply_modifies_rcparams(self):
        style = PlotStyle(font_size=14, dpi=200)
        original_font_size = plt.rcParams["font.size"]
        style.apply()
        assert plt.rcParams["font.size"] == 14
        assert plt.rcParams["figure.dpi"] == 200
        plt.rcParams["font.size"] = original_font_size

    def test_custom_palette_selection(self):
        style = PlotStyle(palette="vibrant")
        colors = style.get_colors(3)
        assert colors == PALETTES["vibrant"][:3]


class TestThesisStyle:
    def test_thesis_style_has_publication_defaults(self):
        style = ThesisStyle()
        assert style.font_family == "serif"
        assert style.savefig_dpi == 300
        assert style.display_dpi == 100

    def test_thesis_style_apply_sets_grid(self):
        style = ThesisStyle()
        style.apply()
        assert plt.rcParams["axes.grid"] is True
        assert plt.rcParams["grid.alpha"] == style.grid_alpha

    def test_thesis_style_removes_top_right_spines(self):
        style = ThesisStyle()
        style.apply()
        assert plt.rcParams["axes.spines.top"] is False
        assert plt.rcParams["axes.spines.right"] is False
