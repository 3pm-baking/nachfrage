"""Newsvendor decision functions — optimal_quantity, profit_profile, waste_sensitivity.

Pure NumPy functions: no file I/O, no database lookups, no PyMC imports.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

#: Posterior draws for a product backed by fewer observations than this are
#: prior-dominated: the predictive tail reflects the model's beliefs, not the
#: item's observed sales record. Quantities optimized from such a tail can be
#: wildly oversized (a real dogfooding scenario: one observation, a large
#: margin, and the optimizer returned 43 whole cakes). Below the threshold,
#: callers should treat ``confidence == "low"`` results as planning hints
#: and sanity-check them against the raw sales history.
MIN_OBSERVATIONS_FOR_CONFIDENCE = 3


class NewsvendorResult(NamedTuple):
    """Result from optimal_quantity().

    Immutable, with named field access.

    Fields:
        best_q: Optimal quantity (multiple of batch_size).
        utility: Expected utility at best_q.
        expected_sales: Expected units sold at best_q.
        sellout_prob: P(sell out) at best_q.
        expected_leftovers: Expected leftover units at best_q.
        profit: Expected profit at best_q.
        confidence: "supported" when ``n_observations`` >=
            MIN_OBSERVATIONS_FOR_CONFIDENCE, "low" below it, and ``None``
            when the caller did not report an observation count. The
            posterior itself does not know how many real observations feed
            each level — only the caller does.
    """

    best_q: int | None
    utility: float | None
    expected_sales: float | None
    sellout_prob: float | None
    expected_leftovers: float | None
    profit: float | None
    confidence: str | None = None


class ProfitProfile(NamedTuple):
    """Result from profit_profile().

    Immutable, with named field access.

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
    n_observations: int | None = None,
) -> NewsvendorResult:
    """Find the quantity Q that maximizes expected utility under batch limits.

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
        n_observations: How many real observations of this product the
            posterior was fit on. The posterior draws themselves cannot tell
            (one draw is not one market day); without this the result's
            ``confidence`` stays ``None`` and the caller loses the thin-data
            signal.

    Returns:
        NewsvendorResult with fields: best_q, utility, expected_sales,
        sellout_prob, expected_leftovers, profit, confidence.

    Raises:
        ValueError: If price or unit_cost is None, or unit_cost >= price.
    """
    if price is None:
        raise ValueError("price must be a number, got None")
    if unit_cost is None:
        raise ValueError("unit_cost must be a number, got None")
    if unit_cost >= price:
        raise ValueError(f"unit_cost ({unit_cost}) must be less than price ({price})")
    if n_observations is not None and n_observations < 0:
        raise ValueError("n_observations must be >= 0")

    if n_observations is None:
        confidence = None
    elif n_observations >= MIN_OBSERVATIONS_FOR_CONFIDENCE:
        confidence = "supported"
    else:
        confidence = "low"

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
        confidence,
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
