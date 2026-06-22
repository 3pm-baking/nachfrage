"""Matplotlib-based plotting for demand models.

Requires matplotlib (install with `pip install nachfrage[plot]`).
All functions accept numpy arrays and write to file paths.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# Shared color palette
COLORS = np.array([
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
])


def _sensible_max(ppd: np.ndarray, product_ids: list[int], percentile: int = 99, cap: int = 100) -> int:
    """Compute a reasonable x-axis maximum for demand histograms."""
    vals = ppd[:, product_ids].ravel()
    bound = int(np.percentile(vals, percentile)) + 5
    return min(bound, cap)


def plot_forest(
    ppd: np.ndarray,
    products: list[str],
    label: str,
    path: str | Path,
) -> None:
    """Forest plot of posterior predictive means with 94% HDI intervals.

    Args:
        ppd: Array of shape (n_samples, n_products).
        products: List of product names.
        label: Title label.
        path: Output file path (PNG).
    """
    import arviz as az
    import matplotlib.pyplot as plt

    if len(products) == 0 or ppd.size == 0:
        return

    mu_mean = ppd.mean(axis=0)
    mu_hdi = az.hdi(ppd.T, prob=0.94)
    sort_idx = np.argsort(mu_mean)[::-1]
    sorted_prods = [products[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(10, max(6, len(products) * 0.35)))
    y_pos = np.arange(len(sorted_prods))

    for i, pid in enumerate(sort_idx):
        lo, hi = mu_hdi[pid]
        m = mu_mean[pid]
        ax.hlines(i, lo, hi, color="gray", linewidth=1.5)
        ax.plot(m, i, "o", color="steelblue", markersize=6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_prods, fontsize=9)
    ax.set_xlabel("Expected demand (94% HDI)", fontsize=12)
    ax.set_title(f"Demand Estimates — {label}", fontsize=14, fontweight="bold")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_densities(
    ppd: np.ndarray,
    products: list[str],
    top: list[str],
    label: str,
    path: str | Path,
) -> None:
    """Overlaid density histograms for top products' posterior predictive.

    Args:
        ppd: Array of shape (n_samples, n_products).
        products: All product names (for lookup).
        top: Names of top products to highlight.
        label: Title label.
        path: Output file path (PNG).
    """
    import matplotlib.pyplot as plt

    top_ids = [products.index(p) for p in top if p in products]
    if not top_ids:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sensible_cap = _sensible_max(ppd, top_ids, percentile=99.5, cap=100)

    for i, pid in enumerate(top_ids):
        color = COLORS[i % len(COLORS)]
        label_text = products[pid]
        prod_max = min(int(np.percentile(ppd[:, pid], 99.5)) + 2, sensible_cap)
        ax.hist(
            ppd[:, pid],
            bins=np.arange(0, prod_max + 2) - 0.5,
            density=True,
            alpha=0.5,
            color=color,
            label=label_text,
            histtype="stepfilled",
        )

    ax.set_xlabel("Demand (units)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Posterior Predictive — {label}", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.5, sensible_cap)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sellout_curves(
    ppd: np.ndarray,
    products: list[str],
    top: list[str],
    label: str,
    path: str | Path,
) -> None:
    """Probability of selling out at each prep quantity for top products.

    Args:
        ppd: Array of shape (n_samples, n_products).
        products: All product names (for lookup).
        top: Names of top products to plot.
        label: Title label.
        path: Output file path (PNG).
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    top_ids = [products.index(p) for p in top if p in products]
    if not top_ids:
        return

    max_n = int(ppd[:, top_ids].max()) + 5
    n_range = np.arange(0, max_n + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, pid in enumerate(top_ids):
        color = COLORS[i % len(COLORS)]
        label_text = products[pid]
        sellout_probs = np.array([(ppd[:, pid] >= n).mean() for n in n_range])
        ax.plot(n_range, sellout_probs, color=color, label=label_text, linewidth=2)

    sensible_cap = _sensible_max(ppd, top_ids, percentile=99, cap=100)
    ax.set_xlabel("Quantity brought", fontsize=12)
    ax.set_ylabel("P(sell out)", fontsize=12)
    ax.set_title(f"Sellout Probability — {label}", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.5, sensible_cap)
    ax.legend(fontsize=9, loc="lower left")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.2)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(
    df: pd.DataFrame,
    ppd: np.ndarray,
    product_keys: list[str],
    label: str,
    path: str | Path,
) -> None:
    """Calibration plot: predicted vs actual sellout rate.

    Args:
        df: DataFrame with columns 'product_key', 'prepared', 'sold_out'.
        ppd: Posterior predictive array of shape (n_samples, n_products).
        product_keys: List of product names.
        label: Title label.
        path: Output file path (PNG).
    """
    import matplotlib.pyplot as plt

    df = df.copy()
    key_to_idx = {k: i for i, k in enumerate(product_keys)}
    df["p_pred"] = df.apply(
        lambda r: (ppd[:, key_to_idx.get(r["product_key"], 0)] >= r["prepared"]).mean()
        if r["product_key"] in key_to_idx else np.nan,
        axis=1,
    )

    bins = np.arange(0, 1.05, 0.1)
    bin_labels = [f"{b*100:.0f}-{(b+0.1)*100:.0f}%" for b in bins[:-1]]
    df["p_bin"] = pd.cut(df["p_pred"], bins=bins, labels=bin_labels, right=False)
    cal = (
        df.groupby("p_bin", observed=True)
        .agg(
            actual_rate=("sold_out", "mean"),
            mean_pred=("p_pred", "mean"),
            n=("sold_out", "count"),
        )
        .reset_index()
    )
    cal["actual_lo"] = cal.apply(
        lambda r: ((r["actual_rate"] * (1 - r["actual_rate"])) / max(r["n"], 1)) ** 0.5 * 1.96,
        axis=1,
    )
    cal["actual_lo"] = (cal["actual_rate"] - cal["actual_lo"]).clip(0)
    cal["actual_hi"] = cal.apply(
        lambda r: ((r["actual_rate"] * (1 - r["actual_rate"])) / max(r["n"], 1)) ** 0.5 * 1.96,
        axis=1,
    )
    cal["actual_hi"] = (cal["actual_rate"] + cal["actual_hi"]).clip(upper=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
    for _, row in cal.iterrows():
        if row["n"] == 0:
            continue
        ax1.errorbar(
            row["mean_pred"], row["actual_rate"],
            yerr=[[row["actual_rate"] - row["actual_lo"]], [row["actual_hi"] - row["actual_rate"]]],
            fmt="o", color="steelblue", markersize=8, capsize=3, ecolor="gray",
            elinewidth=1, alpha=0.8,
        )
        ax1.annotate(
            f"n={int(row['n'])}", (row["mean_pred"], row["actual_rate"]),
            textcoords="offset points", xytext=(5, -10), fontsize=8,
        )
    ax1.set_xlabel("Mean predicted P(sell out)", fontsize=12)
    ax1.set_ylabel("Actual sellout rate", fontsize=12)
    ax1.set_title(f"Calibration — {label}", fontsize=14, fontweight="bold")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(alpha=0.2)
    ax1.set_aspect("equal")

    off = cal["actual_rate"] < cal["mean_pred"]
    colors = ["#d65f5f" if o else "#5fd65f" for o in off]
    ax2.barh(range(len(cal)), cal["n"], color=colors, alpha=0.7, edgecolor="gray", linewidth=0.5)
    ax2.set_yticks(range(len(cal)))
    ax2.set_yticklabels(cal["p_bin"], fontsize=8)
    ax2.set_xlabel("Observations", fontsize=12)
    ax2.set_title("Obs per bin (red=overconfident)", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", alpha=0.2)
    for i, (_, row) in enumerate(cal.iterrows()):
        if row["n"] > 0:
            ax2.text(row["n"] + 0.3, i, f"{row['n']:.0f}", va="center", fontsize=9)

    cal_summary = cal[cal["n"] > 0].copy()
    cal_summary["diff"] = (cal_summary["actual_rate"] - cal_summary["mean_pred"]).abs()
    mae = cal_summary["diff"].mean()
    fig.suptitle(f"MAE = {mae:.3f}", fontsize=11, y=1.02)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_profit_curves(
    decisions: list[dict],
    ppd: np.ndarray,
    product_keys: list[str],
    label: str,
    path: str | Path,
    waste_penalty: float = 0,
) -> None:
    """Profit curves for top-N products with optimal quantity marked.

    Args:
        decisions: List of decision dicts from the optimizer.
        ppd: PPD array of shape (n_samples, n_products).
        product_keys: List of product names.
        label: Title label.
        path: Output file path (PNG).
        waste_penalty: Waste aversion lambda for label.
    """
    from nachfrage.decision import profit_profile

    import matplotlib.pyplot as plt

    top = sorted(
        [r for r in decisions if r.get("opt_profit") is not None],
        key=lambda r: r["opt_profit"],
    )[-8:]
    if not top:
        return

    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.ravel()
    key_to_idx = {k: i for i, k in enumerate(product_keys)}

    for ax, row in zip(axes, top):
        pk = row["product_key"]
        pid = key_to_idx.get(pk)
        if pid is None:
            ax.set_visible(False)
            continue
        if (
            row.get("cost") is None
            or row.get("price") is None
            or row["cost"] >= row["price"]
        ):
            ax.text(0.5, 0.5, "No valid cost/price", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        batch_size = row.get("batch_size", 1)
        qs, utilities, sellout = profit_profile(
            ppd[:, pid], row["price"], row["cost"],
            batch_size=batch_size, waste_penalty=waste_penalty,
        )
        opt_q = int(row["opt_qty"])
        current_q = int(row.get("current_qty", 0))
        y_label = "Expected utility ($)" if waste_penalty > 0 else "Expected profit ($)"

        ax.plot(qs, utilities, color="steelblue", linewidth=2)
        ax.axvline(opt_q, color="green", linestyle="--", alpha=0.7,
                   label=f"Optimal ({opt_q})")
        if current_q and current_q != opt_q:
            ax.axvline(current_q, color="red", linestyle=":", alpha=0.5,
                       label=f"Current ({current_q})")

        ax.scatter([opt_q], [utilities[qs == opt_q][0]], color="green", s=50, zorder=5)
        ax.set_title(f"{pk.split(' (')[0]} ({pk.split('(')[-1]}", fontsize=9)
        ax.set_xlabel("Prep quantity", fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.2)
        y_max = max(utilities[qs <= 120])
        ax.set_ylim(-row["price"] * 2, y_max * 1.15)
        ax.set_xlim(-1, min(120, qs[-1] + 1))

    for j in range(len(top), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Expected Profit Curves — {label}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_optimal_grid(
    decisions: list[dict],
    label: str,
    path: str | Path,
) -> None:
    """Compare current vs optimal prep quantities as a horizontal bar chart.

    Args:
        decisions: List of decision dicts with 'opt_qty', 'current_qty', 'cost', 'price'.
        label: Title label.
        path: Output file path (PNG).
    """
    import matplotlib.pyplot as plt

    valid = [
        r for r in decisions
        if r.get("opt_qty") is not None
        and r.get("price")
        and r.get("cost")
        and r["cost"] < r["price"]
    ]
    if not valid:
        return

    df = pd.DataFrame(valid)
    df["name"] = df["product_key"].str.replace(r" \(.*\)", "", regex=True)
    df = df.sort_values("opt_qty", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)))
    y = np.arange(len(df))
    height = 0.35

    ax.barh(y + height / 2, df["current_qty"], height, label="Current",
            color="lightcoral", alpha=0.7)
    ax.barh(y - height / 2, df["opt_qty"], height, label="Optimal",
            color="steelblue", alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(df["name"], fontsize=8)
    ax.set_xlabel("Prep quantity", fontsize=11)
    ax.set_title(f"Current vs Optimal Prep — {label}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_waste_sensitivity(
    decisions: list[dict],
    ppd: np.ndarray,
    product_keys: list[str],
    label: str,
    path: str | Path,
) -> None:
    """Optimal quantity vs waste-aversion lambda for top products.

    Args:
        decisions: List of decision dicts.
        ppd: PPD array of shape (n_samples, n_products).
        product_keys: List of product names.
        label: Title label.
        path: Output file path (PNG).
    """
    from nachfrage.decision import optimal_quantity

    import matplotlib.pyplot as plt

    key_to_idx = {k: i for i, k in enumerate(product_keys)}
    high_cost = sorted(
        [r for r in decisions if r.get("cost") and r.get("price") and r["cost"] < r["price"]],
        key=lambda r: r["opt_profit"],
    )[-8:]

    lambdas = [0, 1, 2, 3, 5, 10]
    fig, ax = plt.subplots(figsize=(10, 6))

    for row in high_cost:
        pk = row["product_key"]
        pid = key_to_idx.get(pk)
        if pid is None:
            continue

        opt_qs = []
        for lam in lambdas:
            best_q, _, _, _, _, _ = optimal_quantity(
                ppd[:, pid], row["price"], row["cost"],
                row.get("batch_size", 1), waste_penalty=lam,
            )
            opt_qs.append(best_q if best_q is not None else 0)

        short_name = pk.split(" (")[0][:30]
        ax.plot(lambdas, opt_qs, "o-", linewidth=2, markersize=5, label=short_name)

    ax.set_xlabel("Waste-aversion λ ($/leftover unit)", fontsize=12)
    ax.set_ylabel("Optimal prep quantity", fontsize=12)
    ax.set_title(f"Waste-Aversion Sensitivity — {label}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.2)
    ax.set_xticks(lambdas)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
