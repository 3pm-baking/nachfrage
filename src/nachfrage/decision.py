"""Newsvendor decision functions — optimal_quantity, profit_profile, waste_sensitivity.

Pure NumPy functions: no file I/O, no database lookups, no PyMC imports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def optimal_quantity(
    ppd_col: np.ndarray,
    price: float | None,
    unit_cost: float | None,
    batch_size: int = 1,
    waste_penalty: float = 0,
) -> tuple[int | None, float | None, float | None, float | None, float | None, float | None]:
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
        tuple of (best_q, expected_utility, expected_sales, sellout_prob,
                  expected_leftovers, expected_profit).
        Returns all None when price or cost are invalid.
    """
    if price is None or unit_cost is None or unit_cost >= price:
        return None, None, None, None, None, None

    max_q = int(np.percentile(ppd_col, 99)) + 5
    max_q = min(max_q, 200)
    max_q = ((max_q + batch_size - 1) // batch_size) * batch_size

    qs = np.arange(0, max_q + 1, batch_size)
    utility = np.zeros_like(qs, dtype=float)

    for j, q in enumerate(qs):
        sales = np.minimum(ppd_col, q)
        waste = np.maximum(q - ppd_col, 0)
        utility[j] = sales.mean() * price - q * unit_cost - waste.mean() * waste_penalty

    best_idx = int(np.argmax(utility))
    best_q = int(qs[best_idx])
    best_utility = float(utility[best_idx])

    sales_at_best = float(np.minimum(ppd_col, best_q).mean())
    waste_at_best = float(np.maximum(best_q - ppd_col, 0).mean())
    sellout_prob = float((ppd_col >= best_q).mean())
    profit_at_best = float(sales_at_best * price - best_q * unit_cost)

    return best_q, best_utility, sales_at_best, sellout_prob, waste_at_best, profit_at_best


def profit_profile(
    ppd_col: np.ndarray,
    price: float,
    unit_cost: float,
    max_q: int | None = None,
    batch_size: int = 1,
    waste_penalty: float = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute expected utility and P(sell out) for every batch-constrained quantity.

    Args:
        ppd_col: Posterior predictive draws for a single product (1D array).
        price: Sale price per unit.
        unit_cost: Production cost per unit.
        max_q: Maximum quantity to evaluate (defaults to 99th percentile + 5).
        batch_size: Step size between quantities.
        waste_penalty: Subjective cost per leftover unit.

    Returns:
        tuple of (quantities, expected_utilities, sellout_probabilities).
    """
    if max_q is None:
        max_q = min(int(np.percentile(ppd_col, 99)) + 5, 200)
        max_q = ((max_q + batch_size - 1) // batch_size) * batch_size

    qs = np.arange(0, max_q + 1, batch_size)
    utilities = np.array([
        (np.minimum(ppd_col, q).mean() * price
         - q * unit_cost
         - np.maximum(q - ppd_col, 0).mean() * waste_penalty)
        for q in qs
    ])
    sellout = np.array([float((ppd_col >= q).mean()) for q in qs])
    return qs, utilities, sellout


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
        best_q, utility, sales, sellout, leftovers, profit = optimal_quantity(
            ppd_col, price, unit_cost, batch_size, waste_penalty=lam,
        )
        if best_q is None:
            continue
        results.append({
            "lambda": lam,
            "opt_q": best_q,
            "leftovers": leftovers,
            "profit": profit,
            "sellout": sellout,
        })
    return pd.DataFrame(results)
