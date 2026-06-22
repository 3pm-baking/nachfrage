"""Smoke tests for nachfrage.plot — matplotlib-based plotting."""

import numpy as np
import pytest


@pytest.fixture
def ppd(rng):
    """3 products, 1000 draws."""
    return rng.negative_binomial(5, 0.2, size=(1000, 3))


@pytest.fixture
def product_keys():
    return ["Cheese Cake (slice)", "Apple Strudel (piece)", "Kolache (each)"]


@pytest.fixture
def decisions():
    """Mock decisions list matching demand_optimizer output."""
    return [
        {
            "product_key": "Cheese Cake (slice)",
            "price": 5.0,
            "cost": 2.0,
            "batch_size": 1,
            "opt_qty": 10,
            "current_qty": 12,
            "opt_profit": 30.0,
            "opt_sellout": 0.5,
            "opt_leftovers": 2.0,
            "opt_sales": 9.0,
        },
        {
            "product_key": "Apple Strudel (piece)",
            "price": 4.0,
            "cost": 1.5,
            "batch_size": 1,
            "opt_qty": 8,
            "current_qty": 10,
            "opt_profit": 20.0,
            "opt_sellout": 0.6,
            "opt_leftovers": 1.5,
            "opt_sales": 7.0,
        },
    ]


@pytest.fixture
def raw_df():
    """Mock DataFrame for calibration tests."""
    import pandas as pd

    return pd.DataFrame({
        "product_key": ["Cheese Cake (slice)", "Apple Strudel (piece)"] * 5,
        "prepared": [12, 10, 12, 10, 12, 10, 12, 10, 12, 10],
        "sold_out": [True, False, False, True, False, False, True, False, False, False],
    })


class TestPlotForest:
    def test_saves_file(self, ppd, product_keys, tmp_path):
        """plot_forest writes a PNG file."""
        from nachfrage.plot import plot_forest

        path = tmp_path / "forest.png"
        plot_forest(ppd, product_keys, "Test", path)

        assert path.exists()
        assert path.stat().st_size > 0


class TestPlotTopDensities:
    def test_saves_file(self, ppd, product_keys, tmp_path):
        """plot_top_densities writes a PNG file."""
        from nachfrage.plot import plot_top_densities

        top = product_keys[:2]
        path = tmp_path / "densities.png"
        plot_top_densities(ppd, product_keys, top, "Test", path)

        assert path.exists()
        assert path.stat().st_size > 0


class TestPlotSelloutCurves:
    def test_saves_file(self, ppd, product_keys, tmp_path):
        """plot_sellout_curves writes a PNG file."""
        from nachfrage.plot import plot_sellout_curves

        top = product_keys[:2]
        path = tmp_path / "sellout.png"
        plot_sellout_curves(ppd, product_keys, top, "Test", path)

        assert path.exists()
        assert path.stat().st_size > 0


class TestPlotCalibration:
    def test_saves_file(self, ppd, product_keys, raw_df, tmp_path):
        """plot_calibration writes a PNG file."""
        from nachfrage.plot import plot_calibration

        path = tmp_path / "calibration.png"
        plot_calibration(raw_df, ppd, product_keys, "Test", path)

        assert path.exists()
        assert path.stat().st_size > 0


class TestPlotProfitCurves:
    def test_saves_file(self, ppd, product_keys, decisions, tmp_path):
        """plot_profit_curves writes a PNG file."""
        from nachfrage.plot import plot_profit_curves

        path = tmp_path / "profit_curves.png"
        plot_profit_curves(decisions, ppd, product_keys, "Test", path)

        assert path.exists()
        assert path.stat().st_size > 0


class TestPlotOptimalGrid:
    def test_saves_file(self, decisions, tmp_path):
        """plot_optimal_grid writes a PNG file."""
        from nachfrage.plot import plot_optimal_grid

        path = tmp_path / "optimal_grid.png"
        plot_optimal_grid(decisions, "Test", path)

        assert path.exists()
        assert path.stat().st_size > 0


class TestPlotWasteSensitivity:
    def test_saves_file(self, ppd, product_keys, decisions, tmp_path):
        """plot_waste_sensitivity writes a PNG file."""
        from nachfrage.plot import plot_waste_sensitivity

        path = tmp_path / "waste_sensitivity.png"
        plot_waste_sensitivity(decisions, ppd, product_keys, "Test", path)

        assert path.exists()
        assert path.stat().st_size > 0
