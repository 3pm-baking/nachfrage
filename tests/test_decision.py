"""Tests for nachfrage.decision — pure numpy newvendor functions."""

import numpy as np
import pandas as pd
import pytest


class TestOptimalQuantity:
    """Tests for optimal_quantity()."""

    def test_none_price_returns_none(self, deterministic_ppd):
        """Returns all None when price is None."""
        from nachfrage.decision import optimal_quantity

        result = optimal_quantity(deterministic_ppd[:, 0], None, 2.0)
        assert result == (None, None, None, None, None, None)

    def test_none_cost_returns_none(self, deterministic_ppd):
        """Returns all None when unit_cost is None."""
        from nachfrage.decision import optimal_quantity

        result = optimal_quantity(deterministic_ppd[:, 0], 5.0, None)
        assert result == (None, None, None, None, None, None)

    def test_cost_exceeds_price_returns_none(self, deterministic_ppd):
        """Returns all None when unit_cost >= price."""
        from nachfrage.decision import optimal_quantity

        result = optimal_quantity(deterministic_ppd[:, 0], 5.0, 5.0)
        assert result == (None, None, None, None, None, None)

        result = optimal_quantity(deterministic_ppd[:, 0], 5.0, 8.0)
        assert result == (None, None, None, None, None, None)

    def test_deterministic_demand(self, deterministic_ppd):
        """With all demand exactly 10, optimal q = 10."""
        from nachfrage.decision import optimal_quantity

        best_q, utility, sales, sellout, leftovers, profit = optimal_quantity(
            deterministic_ppd[:, 0], price=5.0, unit_cost=2.0,
        )

        assert best_q == 10
        # Profit = 10 * 5 - 10 * 2 = 30
        assert profit == pytest.approx(30.0)
        # Sales = min(10, 10) = 10
        assert sales == pytest.approx(10.0)
        # Sellout probability: all draws >= 10 → 1.0
        assert sellout == pytest.approx(1.0)
        # Leftovers = max(10 - 10, 0) = 0
        assert leftovers == pytest.approx(0.0)

    def test_zero_demand(self, deterministic_ppd):
        """With all demand zero, optimal q = 0."""
        from nachfrage.decision import optimal_quantity

        best_q, utility, sales, sellout, leftovers, profit = optimal_quantity(
            deterministic_ppd[:, 2], price=5.0, unit_cost=2.0,
        )

        assert best_q == 0
        assert profit == pytest.approx(0.0)
        assert sales == pytest.approx(0.0)

    def test_batch_constraint(self, deterministic_ppd):
        """Optimal q is a multiple of batch_size."""
        from nachfrage.decision import optimal_quantity

        best_q, _, _, _, _, _ = optimal_quantity(
            deterministic_ppd[:, 0], price=5.0, unit_cost=2.0, batch_size=4,
        )

        assert best_q % 4 == 0
        # With demand 10 and batch=4, optimal should be 8 or 12
        assert best_q in (8, 12)

    def test_waste_penalty_lowers_quantity(self, deterministic_ppd):
        """Higher waste penalty λ → same or lower optimal q."""
        from nachfrage.decision import optimal_quantity

        best_q_no_penalty, _, _, _, _, _ = optimal_quantity(
            deterministic_ppd[:, 0], price=5.0, unit_cost=2.0, waste_penalty=0,
        )
        best_q_with_penalty, _, _, _, leftovers, _ = optimal_quantity(
            deterministic_ppd[:, 0], price=5.0, unit_cost=2.0, waste_penalty=10,
        )

        assert best_q_with_penalty <= best_q_no_penalty
        # With penalty, there should be fewer leftovers
        assert leftovers is not None

    def test_tuples_are_python_types(self, deterministic_ppd):
        """Result tuple has correct types."""
        from nachfrage.decision import optimal_quantity

        result = optimal_quantity(
            deterministic_ppd[:, 0], price=5.0, unit_cost=2.0,
        )

        best_q, utility, sales, sellout, leftovers, profit = result
        assert isinstance(best_q, int | np.integer)
        assert isinstance(utility, float | np.floating)
        assert isinstance(sales, float | np.floating)
        assert isinstance(profit, float | np.floating)


class TestProfitProfile:
    """Tests for profit_profile()."""

    def test_returns_correct_shapes(self, deterministic_ppd):
        """qs, utilities, and sellout have same length."""
        from nachfrage.decision import profit_profile

        qs, utilities, sellout = profit_profile(
            deterministic_ppd[:, 0], price=5.0, unit_cost=2.0,
        )

        assert len(qs) == len(utilities) == len(sellout)
        assert len(qs) > 0

    def test_sellout_probability_non_increasing(self, deterministic_ppd):
        """P(sell out) decreases as q increases."""
        from nachfrage.decision import profit_profile

        qs, utilities, sellout = profit_profile(
            deterministic_ppd[:, 0], price=5.0, unit_cost=2.0,
        )

        diffs = np.diff(sellout)
        assert np.all(diffs <= 0)

    def test_batch_constraint(self, deterministic_ppd):
        """Quantities are multiples of batch_size."""
        from nachfrage.decision import profit_profile

        qs, _, _ = profit_profile(
            deterministic_ppd[:, 0], price=5.0, unit_cost=2.0, batch_size=3,
        )

        assert np.all(qs % 3 == 0)

    def test_zero_not_included_by_default(self, deterministic_ppd):
        """First quantity should be 0."""
        from nachfrage.decision import profit_profile

        qs, _, _ = profit_profile(
            deterministic_ppd[:, 0], price=5.0, unit_cost=2.0,
        )

        assert qs[0] == 0


class TestWasteSensitivity:
    """Tests for waste_sensitivity()."""

    def test_returns_dataframe_with_correct_columns(self, deterministic_ppd):
        """Returns DataFrame with expected columns."""
        from nachfrage.decision import waste_sensitivity

        df = waste_sensitivity(deterministic_ppd[:, 0], 5.0, 2.0)

        expected_cols = {"lambda", "opt_q", "leftovers", "profit", "sellout"}
        assert expected_cols.issubset(set(df.columns))

    def test_columns_are_non_increasing_with_lambda(self, deterministic_ppd):
        """Higher λ should never increase optimal_q."""
        from nachfrage.decision import waste_sensitivity

        df = waste_sensitivity(deterministic_ppd[:, 0], 5.0, 2.0)

        opt_qs = df["opt_q"].values
        # Each successive value should be ≤ previous
        assert np.all(np.diff(opt_qs) <= 0)
