"""Tests for nachfrage.analysis — text table formatters."""

import pytest


class TestFormatScenarios:
    """Tests for format_scenarios()."""

    @pytest.fixture
    def ppd(self, rng):
        """3 products, 10000 draws of demand."""
        return rng.negative_binomial(5, 0.2, size=(10000, 3))

    @pytest.fixture
    def product_keys(self):
        return ["Cheese Cake (slice)", "Apple Strudel (piece)", "Kolache (each)"]

    def test_contains_headers(self, ppd, product_keys):
        """Output contains column headers."""
        from nachfrage.analysis import format_scenarios

        output = format_scenarios(ppd, product_keys)

        assert "Product" in output
        assert "Mean" in output
        assert "P10" in output
        assert "P50" in output
        assert "P90" in output

    def test_contains_product_names(self, ppd, product_keys):
        """Output contains each product name."""
        from nachfrage.analysis import format_scenarios

        output = format_scenarios(ppd, product_keys)

        for key in product_keys:
            assert key in output

    def test_empty_product_keys(self, ppd):
        """Returns header-only string for empty keys."""
        from nachfrage.analysis import format_scenarios

        output = format_scenarios(ppd[:0, :0], [])

        assert "Scenario" in output or "Product" in output
        assert isinstance(output, str)

    def test_single_product(self, ppd):
        """Works with a single product."""
        from nachfrage.analysis import format_scenarios

        output = format_scenarios(ppd[:, :1], ["Single Cake (slice)"])

        assert "Single Cake (slice)" in output
        assert isinstance(output, str)
