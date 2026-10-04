"""Exposure offsets: demand modeling when observation windows differ.

Every observation in a ``DemandModel`` implicitly covers a full market
window. Real life does not guarantee that: markets close early, weather
cuts a day short, and — the motivating case for this term — an item sells
out, after which the rest of the window tells you nothing.

The ``ExposureOffset`` term takes each observation's covered opportunity,
an ``exposure`` column normalized against a fixed reference window
(``hours_observed / REFERENCE_HOURS`` — a full 2-hour market is 2/3 against
a 3-hour reference), and adds ``log(exposure)`` to the linear predictor.
The shared ``exp`` link turns that into the standard count-regression
identity::

    sold ~ NB(rate * exposure, alpha)

so two rows with the same ``sold`` are NOT the same measurement when one
covered the full window and the other covered half of it.

Run:

    uv run python examples/exposure_demo.py

Part A shows the term on partial-observation *uncensored* data: two
products with the same underlying demand, one observed only on short
markets. Part B drives the sellout-time story (the data source that
motivated the term) and points at the open boundary-likelihood follow-up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm  # noqa: F401 — imported so init cost is paid once, once
from pandas import DataFrame
from pymc_extras.prior import Prior
from pymc_marketing.terms import Intercept

from nachfrage.models import DemandModel
from nachfrage.terms import ExposureOffset, GroupContribution

RNG = np.random.default_rng(3)
ALPHA = 6.0  # NB dispersion: CV shrinks as alpha grows


def _draw(alpha: float, mu: float) -> float:
    """One NegativeBinomial(mu, alpha) observation (discrete, count scale)."""
    return float(RNG.negative_binomial(alpha, alpha / (alpha + mu)))


# ---------------------------------------------------------------------------
# Part A — partial windows, uncensored: the term in its cleanest form
#
# Two products with the same underlying demand rate; one only ever observed
# on markets cut short (weather, early close). No sellouts, so the data is
# not censored and the comparison isolates the exposure mechanism.
# ---------------------------------------------------------------------------

PRODUCTS_A = {
    "Full-window markets only": (13.0, 1.00),
    "Short markets only": (13.0, 0.55),
}


def part_a_data() -> tuple[pd.DataFrame, dict[str, float]]:
    n, prepared = 60, 999.0  # prepared never binds: no censoring
    rows, truth = [], {}
    for label, (rate, window) in PRODUCTS_A.items():
        truth[label] = rate
        for _ in range(n):
            sold = _draw(ALPHA, rate * window)
            rows.append(
                {
                    "product": label,
                    "sold": sold,
                    "prepared": prepared,
                    "exposure": window,
                }
            )
    return DataFrame(rows), truth


def fit(df: pd.DataFrame, *, with_exposure: bool, draws: int = 500) -> DemandModel:
    terms: list = [
        Intercept(prior=Prior("Normal", mu=np.log(13.0), sigma=0.7)),
        GroupContribution(
            data_source="product",
            prior=Prior(
                "Normal",
                mu=0.0,
                sigma=Prior("HalfNormal", sigma=0.5),
                dims="product",
            ),
        ),
    ]
    if with_exposure:
        terms = terms + [ExposureOffset()]

    data = df if with_exposure else df.drop(columns="exposure")
    model = DemandModel(terms=terms)
    model.build(data)
    model.fit(draws=draws, tune=draws, chains=2, random_seed=1, progressbar=False)
    return model


def posterior_means(model: DemandModel) -> dict[str, float]:
    ppd = model.sample_product_predictive()
    means = ppd.mean("sample")
    return {str(p): float(means.sel(product=p)) for p in means.coords["product"].values}


def part_a() -> None:
    print("Part A — partial observation windows, uncensored")
    print("=" * 74)
    df, truth = part_a_data()

    shown = df.groupby("product", sort=False)[["sold", "exposure"]].mean()
    for label, row in shown.iterrows():
        print(
            f"  observed {label:28s} sold {row['sold']:5.2f}"
            f" per exposure {row['exposure']:.2f}"
        )

    for with_exp, tag in (
        (False, "without ExposureOffset"),
        (True, "with ExposureOffset   "),
    ):
        result = posterior_means(fit(df, with_exposure=with_exp))
        print(f"  fitted demand rate {tag}:")
        for label, mean in result.items():
            print(f"    {label:28s} {mean:6.2f}   (true rate {truth[label]:.1f})")
    print(
        "  reading: without the term the short-market product is misread\n"
        "  as genuinely weaker demand; with it both products collapse to\n"
        "  their shared full-window rate.\n"
    )


# ---------------------------------------------------------------------------
# Part B — sellout times (the data source that motivated the term)
#
# A sellout at ``t`` of a window of length W contributes exposure = t / W:
# a 12-of-12 row that happened at closing hour is NOT the same measurement
# as a 12-of-12 row that happened half way through the market. With the
# constant-rate assumption the exposure a seller records is
# prepared / latent_demand — unknown to the seller but bounded by the clock.
#
# OPEN FOLLOW-UP: with sellout rows the demand likelihood is *censored* at
# the prepared quantity,DemandModel's boundary handling for discrete
# distributions stacks an ever-growing mass at the boundary: with
# sellout-heavy synthetic data the posterior becomes unstable (see
# tests/test_exposure.py). Real sellout data rides on plenty of uncensored
# days, so record ``sold_out`` times today and revisit the boundary
# handling before acting on sellout-time fits.
# ---------------------------------------------------------------------------


def part_b_data() -> tuple[pd.DataFrame, dict[str, float]]:
    prepared, n = 12.0, 90
    products = {
        "Sells out at closing bell (true 13)": (13.0, 0.25),
        "Sells out half way through (true 26)": (26.0, 0.25),
    }
    rows, truth = [], {}
    for label, (rate, sellout_share) in products.items():
        truth[label] = rate
        n_days = int(n / max(sellout_share, 1e-9))
        for _ in range(n_days):
            latent = _draw(ALPHA, rate)
            if latent >= prepared:
                # sellout: everything is gone by t = prepared/latent of the
                # window (constant-rate clock reading)
                rows.append(
                    {
                        "product": label,
                        "sold": prepared,
                        "prepared": prepared,
                        "exposure": np.clip(
                            prepared / max(latent, prepared), 0.05, 1.0
                        ),
                    }
                )
            else:
                rows.append(
                    {
                        "product": label,
                        "sold": latent,
                        "prepared": prepared,
                        "exposure": 1.0,
                    }
                )
    df = DataFrame(rows)
    return df, truth


def part_b() -> None:
    print("Part B — sellout times (censored rows; experimental)")
    print("=" * 74)
    df, truth = part_b_data()
    shown = df.groupby("product", sort=False)[["sold", "exposure"]].mean()
    for label, row in shown.iterrows():
        print(
            f"  observed {label:38s} sold {row['sold']:5.2f}"
            f" mean exposure {row['exposure']:.2f}"
        )

    try:
        result = posterior_means(fit(df, with_exposure=True, draws=300))
    except RuntimeError as e:
        print(f"  fit could not initialise: {type(e).__name__}")
        print(
            "  the censored-boundary interaction with stacked sellout rows\n"
            "  is unstable — this is the OPEN FOLLOW-UP, not the term's.\n"
            "  Exposure data from real markets (with plentiful uncensored\n"
            "  days) should be re-quantified against this.\n"
        )
        return
    print("  exposure-aware fit:")
    for label, mean in result.items():
        print(f"    {label:38s} {mean:6.2f}   (true rate {truth[label]:.1f})")
    print(
        "  Experimental: the censored-boundary interaction is unstable on\n"
        "  synthetic data; treat these numbers as an illustration of the\n"
        "  term, not a validated posterior. See the OPEN FOLLOW-UP note.\n"
    )


def main() -> None:
    part_a()
    part_b()


if __name__ == "__main__":
    main()
