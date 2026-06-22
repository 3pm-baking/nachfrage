"""Newsvendor decision functions — optimal_quantity, profit_profile, waste_sensitivity.

Pure NumPy functions: no file I/O, no database lookups, no PyMC imports.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


class NewsvendorResult(NamedTuple):
    """Result from optimal_quantity().

    Immutable, with named field access and backward-compatible tuple unpacking.

    Fields:
        best_q: Optimal quantity (multiple of batch_size).
        utility: Expected utility at best_q.
        expected_sales: Expected units sold at best_q.
        sellout_prob: P(sell out) at best_q.
        expected_leftovers: Expected leftover units at best_q.
        profit: Expected profit at best_q.
    """

    best_q: int | None
    utility: float | None
    expected_sales: float | None
    sellout_prob: float | None
    expected_leftovers: float | None
    profit: float | None


class ProfitProfile(NamedTuple):
    """Result from profit_profile().

    Immutable, with named field access and backward-compatible tuple unpacking.

    Fields:
        quantities: Array of prep quantities evaluated.
        utilities: Expected utility at each quantity.
        sellout_probs: P(sell out) at each quantity.
    """

    quantities: np.ndarray
    utilities: np.ndarray
    sellout_probs: np.ndarray


def optimal_quantity(
    ppd_col: np.ndarray,
    price: float | None,
    unit_cost: float | None,
    batch_size: int = 1,
    waste_penalty: float = 0,
) -> NewsvendorResult:
    """Find the quantity Q that maximizes expected utility, constrained to batch multiples.

    Expected Utility(Q) = E[min(D, Q)] * price - Q * unit_cost
                         - E[max(Q - D, 0)] * waste_penalty

    where D is demand from the posterior predictive and waste_penalty is
    the subjective cost per leftover unit.

    Args:
        ppd_col: Posterior predictive draws for a single product (1D array).
        price: Sale price per unit.
        unit_cost: Production cost per unit.
        batch_size: Quantities must be multiples of batch_size.
        waste_penalty: Subjective cost per leftover unit (λ).

    Returns:
        NewsvendorResult with fields: best_q, utility, expected_sales,
        sellout_prob, expected_leftovers, profit.

    Raises:
        ValueError: If price or unit_cost is None, or unit_cost >= price.
    """
    if price is None:
        raise ValueError("price must be a number, got None")
    if unit_cost is None:
        raise ValueError("unit_cost must be a number, got None")
    if unit_cost >= price:
        raise ValueError(f"unit_cost ({unit_cost}) must be less than price ({price})")

    max_q = int(np.percentile(ppd_col, 99)) + 5
    max_q = min(max_q, 200)
    max_q = ((max_q + batch_size - 1) // batch_size) * batch_size

    qs = np.arange(0, max_q + 1, batch_size)
    sales = np.minimum(ppd_col[:, None], qs)
    waste = np.maximum(qs - ppd_col[:, None], 0)
    utility = (
        sales.mean(axis=0) * price - qs * unit_cost - waste.mean(axis=0) * waste_penalty
    )

    best_idx = int(np.argmax(utility))
    best_q = int(qs[best_idx])
    best_utility = float(utility[best_idx])

    sales_at_best = float(np.minimum(ppd_col, best_q).mean())
    waste_at_best = float(np.maximum(best_q - ppd_col, 0).mean())
    sellout_prob = float((ppd_col >= best_q).mean())
    profit_at_best = float(sales_at_best * price - best_q * unit_cost)

    return NewsvendorResult(
        best_q,
        best_utility,
        sales_at_best,
        sellout_prob,
        waste_at_best,
        profit_at_best,
    )


def profit_profile(
    ppd_col: np.ndarray,
    price: float,
    unit_cost: float,
    max_q: int | None = None,
    batch_size: int = 1,
    waste_penalty: float = 0,
) -> ProfitProfile:
    """Compute expected utility and P(sell out) for every batch-constrained quantity.

    Args:
        ppd_col: Posterior predictive draws for a single product (1D array).
        price: Sale price per unit.
        unit_cost: Production cost per unit.
        max_q: Maximum quantity to evaluate (defaults to 99th percentile + 5).
        batch_size: Step size between quantities.
        waste_penalty: Subjective cost per leftover unit.

    Returns:
        ProfitProfile with fields: quantities, utilities, sellout_probs.
    """
    if max_q is None:
        max_q = min(int(np.percentile(ppd_col, 99)) + 5, 200)
        max_q = ((max_q + batch_size - 1) // batch_size) * batch_size

    qs = np.arange(0, max_q + 1, batch_size)
    utilities = np.array(
        [
            (
                np.minimum(ppd_col, q).mean() * price
                - q * unit_cost
                - np.maximum(q - ppd_col, 0).mean() * waste_penalty
            )
            for q in qs
        ]
    )
    sellout = np.array([float((ppd_col >= q).mean()) for q in qs])
    return ProfitProfile(qs, utilities, sellout)


def waste_sensitivity(
    ppd_col: np.ndarray,
    price: float,
    unit_cost: float,
    batch_size: int = 1,
) -> pd.DataFrame:
    """Compute optimal quantity and expected leftovers across waste-penalty values.

    Args:
        ppd_col: Posterior predictive draws for a single product (1D array).
        price: Sale price per unit.
        unit_cost: Production cost per unit.
        batch_size: Quantities must be multiples of batch_size.

    Returns:
        DataFrame with columns: lambda, opt_q, leftovers, profit, sellout.
    """
    lambdas = [0, 1, 2, 3, 5, 10]
    results = []
    for lam in lambdas:
        result = optimal_quantity(
            ppd_col,
            price,
            unit_cost,
            batch_size,
            waste_penalty=lam,
        )
        if result.best_q is None:
            continue
        results.append(
            {
                "lambda": lam,
                "opt_q": result.best_q,
                "leftovers": result.expected_leftovers,
                "profit": result.profit,
                "sellout": result.sellout_prob,
            }
        )
    return pd.DataFrame(results)
