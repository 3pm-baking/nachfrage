"""Text table formatters for results and scenarios."""

from __future__ import annotations

import numpy as np


def format_scenarios(
    ppd: np.ndarray,
    product_keys: list[str],
) -> str:
    """Format a scenario table from posterior predictive samples.

    Args:
        ppd: Array of shape (n_samples, n_products).
        product_keys: List of product names.

    Returns:
        Multi-line string with scenario table.
    """
    if ppd.size == 0 or len(product_keys) == 0:
        return "Scenario Table — Posterior Predictive Demand\nNo products."

    mu = ppd.mean(axis=0)
    q = np.percentile(ppd, [10, 25, 50, 75, 90], axis=0)
    lines = ["Scenario Table — Posterior Predictive Demand"]
    lines.append("=" * 78)
    header = f"{'#':>3} {'Product':<52} {'Mean':>5} {'P10':>4} {'P25':>4} {'P50':>4} {'P75':>4} {'P90':>4}"
    lines.append(header)
    lines.append("-" * 78)

    sort_idx = np.argsort(mu)[::-1]
    for rank, pid in enumerate(sort_idx, 1):
        line = (
            f"{rank:>3} {product_keys[pid]:<52} "
            f"{mu[pid]:>5.0f} "
            f"{q[0, pid]:>4.0f} "
            f"{q[1, pid]:>4.0f} "
            f"{q[2, pid]:>4.0f} "
            f"{q[3, pid]:>4.0f} "
            f"{q[4, pid]:>4.0f}"
        )
        lines.append(line)

    return "\n".join(lines)


def format_results_table(
    idata,
    df,
    product_keys: list[str],
    label: str = "ALL",
) -> str:
    """Format a detailed results table from inference data.

    Args:
        idata: ArviZ InferenceData from a fitted model.
        df: DataFrame of observations.
        product_keys: List of unique product names.
        label: Label for the table header.

    Returns:
        Multi-line string with the results table.
    """
    try:
        import arviz as az
        import numpy as np
        import pandas as pd

        mu_samples = idata.posterior["mu_product"].values
        alpha_samples = idata.posterior["demand_alpha"].values
        mu_samples = mu_samples.reshape(-1, mu_samples.shape[-1])
        alpha_samples = alpha_samples.reshape(-1)
    except Exception:
        return "Results unavailable — idata missing expected variables."

    lines = []
    lines.append("=" * 78)
    lines.append(f"DEMAND MODEL — {label}".center(78))
    lines.append("=" * 78)
    lines.append(f"Observations:          {len(df)}")
    lines.append(f"Unique products:       {len(product_keys)}")

    if "is_censored" in df.columns:
        n_cens = df["is_censored"].sum()
    else:
        n_cens = df.get("sold_out", pd.Series([False] * len(df))).sum()

    n_obs = len(df)
    lines.append(f"Censored (sellout):    {n_cens} ({n_cens / n_obs * 100:.0f}%)")
    lines.append(
        f"Uncensored (leftover): {n_obs - n_cens} "
        f"({(n_obs - n_cens) / n_obs * 100:.0f}%)"
    )

    lines.append("")
    mu_mean = mu_samples.mean(axis=0)
    mu_hdi = az.hdi(mu_samples.T, prob=0.94)

    lines.append(
        f"{'Product':<52} {'Obs':>3} {'Cens':>3} "
        f"{'Mean':>7} {'Lo 94%':>8} {'Hi 94%':>8}"
    )
    lines.append("-" * 81)

    order = sorted(
        product_keys,
        key=lambda k: mu_mean[list(product_keys).index(k)],
        reverse=True,
    )
    for pk in order:
        pid = list(product_keys).index(pk)
        sub = df[df.get("product_key", pd.Series(dtype=str)) == pk]
        if len(sub) > 0 and "is_censored" in sub.columns:
            cens = sub["is_censored"].sum()
        elif len(sub) > 0 and "sold_out" in sub.columns:
            cens = sub["sold_out"].sum()
        else:
            cens = 0
        obs = len(sub)
        m = mu_mean[pid]
        lo, hi = mu_hdi[pid]
        lines.append(f"{pk:<52} {obs:>3} {cens:>3} {m:>7.1f} {lo:>8.1f} {hi:>8.1f}")

    lines.append("")
    α_mean = float(alpha_samples.mean())
    α_hdi = az.hdi(alpha_samples, prob=0.94)
    lines.append(
        f"Overdispersion α = {α_mean:.2f}  (94% HDI: [{α_hdi[0]:.2f}, {α_hdi[1]:.2f}])"
    )
    lines.append("  α → ∞ = Poisson (no overdispersion); lower α = more overdispersion")

    ess = az.ess(
        idata, var_names=["mu_global", "sigma_product", "demand_alpha", "mu_product"]
    )
    rhat = az.rhat(
        idata, var_names=["mu_global", "sigma_product", "demand_alpha", "mu_product"]
    )
    ess_vals = [np.asarray(v).min() for v in ess.values()]
    rhat_vals = [np.asarray(v).max() for v in rhat.values()]
    lines.append("\nConvergence:")
    lines.append(f"  Min ESS:   {min(ess_vals):.0f}")
    lines.append(f"  Max r_hat: {max(rhat_vals):.3f}")

    return "\n".join(lines)
