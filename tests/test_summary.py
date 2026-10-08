"""Tests for nachfrage.summary — tidy posterior demand summaries."""

import warnings

import numpy as np
import pandas as pd
import pytest
from pymc_extras.prior import Prior
from pymc_marketing.terms import Intercept

warnings.filterwarnings("ignore", category=FutureWarning)

GROUPS = ["north", "south"]
PRODUCTS = ["Cake A", "Cake B"]


def _make_data(rng, n_sellouts: int = 0) -> pd.DataFrame:
    """Two products across two regions; some rows forced to sold == prepared."""
    alpha = 5.0
    rows = []
    for product in PRODUCTS:
        mu = 10.0 if product == "Cake A" else 6.0
        for region in GROUPS:
            scale = 1.2 if region == "north" else 0.7
            demand = rng.negative_binomial(alpha, alpha / (alpha + mu * scale), size=4)
            prepared = np.ceil(demand * 1.3).astype(int)
            sold = np.minimum(demand, prepared)
            for s, p in zip(sold, prepared, strict=True):
                rows.append(
                    {
                        "sold": float(s),
                        "prepared": float(p),
                        "product": product,
                        "region": region,
                    }
                )
    df = pd.DataFrame(rows)
    # Force deterministic sellouts (definition: sold >= prepared).
    for i in range(n_sellouts):
        df.loc[i, "sold"] = df.loc[i, "prepared"]
    return df


def _fit(df, *, draws: int = 20, tune: int = 20):
    from nachfrage.models import DemandModel
    from nachfrage.terms import GroupContribution

    terms = [
        Intercept(prior=Prior("Normal", mu=np.log(8), sigma=0.5)),
        GroupContribution(
            data_source="region",
            prior=Prior(
                "Normal", mu=0, sigma=Prior("HalfNormal", sigma=0.5), dims="region"
            ),
        ),
        GroupContribution(
            data_source="product",
            prior=Prior(
                "Normal", mu=0, sigma=Prior("HalfNormal", sigma=0.5), dims="product"
            ),
        ),
    ]
    model = DemandModel(terms=terms)
    model.build(df)
    model.fit(draws=draws, tune=tune, chains=1, random_seed=0, progressbar=False)
    return model


@pytest.fixture(scope="module")
def fitted_model():
    rng = np.random.default_rng(0)
    return _fit(_make_data(rng, n_sellouts=5))


@pytest.fixture(scope="module")
def fitted_censored_model():
    rng = np.random.default_rng(1)
    return _fit(_make_data(rng, n_sellouts=5))


class TestSummarizeDemand:
    def test_rows_cover_the_grid(self, fitted_model):
        from nachfrage.summary import summarize_demand

        summary = summarize_demand(fitted_model)

        assert len(summary) == len(PRODUCTS) * len(GROUPS)
        assert set(summary.columns) >= {
            "product",
            "region",
            "demand_mean",
            "q05",
            "q50",
            "q95",
            "n_obs",
            "n_censored",
        }
        assert set(summary["product"]) == set(PRODUCTS)
        assert set(summary["region"]) == set(GROUPS)
        assert (summary["demand_mean"] > 0).all()
        # Quantiles are ordered around the mean sensibly.
        assert (summary["q05"] <= summary["q50"]).all()
        assert (summary["q50"] <= summary["q95"]).all()

    def test_quantile_columns_follow_requested(self, fitted_model):
        from nachfrage.summary import summarize_demand

        summary = summarize_demand(fitted_model, quantiles=(0.10, 0.90))

        assert {"q10", "q90"} <= set(summary.columns)
        assert "q05" not in summary.columns

    def test_conditions_restrict_grid(self, fitted_model):
        from nachfrage.summary import summarize_demand

        summary = summarize_demand(fitted_model, conditions={"region": "north"})

        assert len(summary) == len(PRODUCTS)
        assert set(summary["region"]) == {"north"}

    def test_obs_and_censored_counts_match_training(self, fitted_censored_model):
        from nachfrage.summary import summarize_demand

        summary = summarize_demand(fitted_censored_model)
        training = fitted_censored_model._training_ds

        sold = np.asarray(training["sold"].values)
        prepared = np.asarray(training["prepared"].values)
        expected_total_censored = int((sold >= prepared).sum())

        assert summary["n_obs"].sum() == len(sold)
        assert summary["n_censored"].sum() == expected_total_censored
        assert (summary["n_censored"] <= summary["n_obs"]).all()

    def test_cells_without_training_rows_are_zero(self, fitted_model):
        """A grid cell never observed gets a posterior but zero counts."""
        from nachfrage.models import DemandModel
        from nachfrage.terms import GroupContribution

        # Drop every Cake B / north row, so that cell is absent in training.
        trimmed = fitted_model._require_training_ds().to_dataframe()
        no_cell = trimmed[
            ~((trimmed["product"] == "Cake B") & (trimmed["region"] == "north"))
        ]

        # Refit on the trimmed frame with the same tiny budget.
        terms = [
            Intercept(prior=Prior("Normal", mu=np.log(8), sigma=0.5)),
            GroupContribution(
                data_source="region",
                prior=Prior(
                    "Normal",
                    mu=0,
                    sigma=Prior("HalfNormal", sigma=0.5),
                    dims="region",
                ),
            ),
            GroupContribution(
                data_source="product",
                prior=Prior(
                    "Normal",
                    mu=0,
                    sigma=Prior("HalfNormal", sigma=0.5),
                    dims="product",
                ),
            ),
        ]
        model = DemandModel(terms=terms)
        model.build(no_cell[["sold", "prepared", "product", "region"]])
        model.fit(draws=20, tune=20, chains=1, random_seed=0, progressbar=False)

        from nachfrage.summary import summarize_demand

        summary = summarize_demand(model)
        missing = summary[
            (summary["product"] == "Cake B") & (summary["region"] == "north")
        ]
        assert len(missing) == 1
        assert missing["n_obs"].iloc[0] == 0
        assert missing["n_censored"].iloc[0] == 0
        assert missing["demand_mean"].iloc[0] > 0
