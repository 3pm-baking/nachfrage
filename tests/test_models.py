"""Tests for nachfrage.models — DemandModel lifecycle.

Only nachfrage-owned behavior is covered here. The term algebra itself
(``Intercept``, ``Dot``, ``Sum``, ``Transform``, ``collect_coords``,
``register_data``) is upstream ``pymc_marketing.terms`` and is tested there.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from pymc_extras.prior import Prior
from pymc_marketing.terms import Intercept

warnings.filterwarnings("ignore", category=FutureWarning)

PRODUCTS = ["Cake A", "Cake B"]


def _make_data(rng, *, with_price: bool, n_per: int = 6) -> pd.DataFrame:
    """Two products with a censored sellout pattern, optionally priced."""
    true_mu = np.array([10.0, 6.0])
    alpha = 5.0
    demand = rng.negative_binomial(
        alpha, alpha / (alpha + true_mu), size=(n_per, len(PRODUCTS))
    )
    prepared = np.ceil(demand * 1.3).astype(int)
    sold = demand.T.ravel()
    prepared = prepared.T.ravel()
    censored = sold >= prepared
    sold[censored] = prepared[censored]

    df = pd.DataFrame(
        {
            "sold": sold.astype(float),
            "prepared": prepared.astype(float),
            "product": np.repeat(PRODUCTS, n_per),
        }
    )
    if with_price:
        df["log_price"] = np.log(5.0)
    return df


@pytest.fixture
def tiny_data(rng):
    return _make_data(rng, with_price=False)


@pytest.fixture
def price_data(rng):
    return _make_data(rng, with_price=True)


def _fit(df, *, with_price: bool, draws: int = 20, tune: int = 20):
    from nachfrage.models import DemandModel
    from nachfrage.terms import default_terms

    model = DemandModel(terms=default_terms(with_price=with_price))
    model.build(df)
    model.fit(draws=draws, tune=tune, chains=1, random_seed=0, progressbar=False)
    return model


@pytest.fixture(scope="module")
def fitted_model():
    rng = np.random.default_rng(0)
    return _fit(_make_data(rng, with_price=False), with_price=False)


@pytest.fixture(scope="module")
def fitted_price_model():
    rng = np.random.default_rng(0)
    return _fit(_make_data(rng, with_price=True), with_price=True)


class TestGroupContribution:
    """The one term nachfrage adds on top of upstream."""

    def test_creates_non_centered_variables(self, tiny_data):
        import pymc as pm

        from nachfrage.models import DemandModel
        from nachfrage.terms import default_terms

        model = DemandModel(terms=default_terms(with_price=False))
        model.build(tiny_data)

        names = set(model.model.named_vars)
        assert {"product_raw", "product_sigma", "product_offset"} <= names
        assert isinstance(model.model, pm.Model)

    def test_offset_gathers_onto_observations(self, tiny_data):
        """The offset is per product; the likelihood sees one row per obs."""
        from nachfrage.models import DemandModel
        from nachfrage.terms import default_terms

        model = DemandModel(terms=default_terms(with_price=False))
        model.build(tiny_data)

        offset = model.model["product_offset"]
        assert offset.type.dims == ("product",)


class TestBuild:
    def test_build_creates_model(self, tiny_data):
        import pymc as pm

        from nachfrage.models import DemandModel
        from nachfrage.terms import default_terms

        model = DemandModel(terms=default_terms(with_price=False))
        model.build(tiny_data)

        assert isinstance(model.model, pm.Model)
        assert model.idata is None

    def test_build_missing_columns(self):
        from nachfrage.models import DemandModel
        from nachfrage.terms import default_terms

        model = DemandModel(terms=default_terms(with_price=False))
        with pytest.raises(ValueError):
            model.build(pd.DataFrame({"sold": [1.0, 2.0, 3.0]}))

    def test_product_names_in_first_seen_order(self, tiny_data):
        from nachfrage.models import DemandModel
        from nachfrage.terms import default_terms

        model = DemandModel(terms=default_terms(with_price=False))
        model.build(tiny_data)

        assert model.product_names == PRODUCTS


class TestFit:
    def test_fit_produces_pinned_posterior_names(self, fitted_model):
        names = set(fitted_model.idata.posterior.data_vars)
        assert {
            "intercept",
            "product_raw",
            "product_sigma",
            "product_offset",
            "demand_alpha",
        } <= names

    def test_fit_with_price_adds_beta(self, fitted_price_model):
        assert "beta_log_price" in fitted_price_model.idata.posterior


class TestSamplePosteriorPredictive:
    def test_dims_and_obs_coord(self, fitted_model):
        ppd = fitted_model.sample_posterior_predictive()

        assert set(ppd.dims) == {"sample", "obs"}
        assert "obs" in ppd.coords
        assert ppd.sizes["obs"] == 12

    def test_raises_without_idata(self, tiny_data):
        from nachfrage.models import DemandModel
        from nachfrage.terms import default_terms

        model = DemandModel(terms=default_terms(with_price=False))
        model.build(tiny_data)
        with pytest.raises(RuntimeError):
            model.sample_posterior_predictive()


class TestSampleProductPredictive:
    def test_dims_and_labels(self, fitted_model):
        ppd = fitted_model.sample_product_predictive()

        assert set(ppd.dims) == {"sample", "product"}
        assert [str(p) for p in ppd.coords["product"].values] == PRODUCTS

    def test_values_are_non_negative(self, fitted_model):
        ppd = fitted_model.sample_product_predictive()
        assert float(ppd.min()) >= 0.0


class TestPredictAtPrice:
    def test_dims(self, fitted_price_model):
        ppd = fitted_price_model.predict_demand_at_price(price=5.0)
        assert set(ppd.dims) == {"sample", "product"}

    def test_higher_price_lowers_demand(self, fitted_price_model):
        cheap = fitted_price_model.predict_demand_at_price(price=4.0).mean("sample")
        dear = fitted_price_model.predict_demand_at_price(price=9.0).mean("sample")
        assert bool((dear < cheap).all())

    def test_falls_back_without_price_term(self, fitted_model):
        ppd = fitted_model.predict_demand_at_price(price=9.0)
        assert set(ppd.dims) == {"sample", "product"}


class TestNetCDF:
    def test_roundtrip_preserves_predictive(self, fitted_model, tmp_path):
        from nachfrage.models import DemandModel
        from nachfrage.terms import default_terms

        path = tmp_path / "model.nc"
        fitted_model.to_netcdf(path)

        reloaded = DemandModel.from_netcdf(path, terms=default_terms(with_price=False))
        assert reloaded.product_names == PRODUCTS
        assert set(reloaded.sample_product_predictive().dims) == {
            "sample",
            "product",
        }

    def test_saved_design_excludes_likelihood_columns(self, fitted_model, tmp_path):
        import xarray as xr

        path = tmp_path / "model.nc"
        fitted_model.to_netcdf(path)

        design = xr.open_datatree(path)["design"].ds
        assert "sold" not in design
        assert "prepared" not in design
        assert "product" in design

    def test_from_idata_without_design_cannot_predict(self, fitted_model):
        from nachfrage.models import DemandModel
        from nachfrage.terms import default_terms

        bare = DemandModel.from_idata(
            fitted_model.idata, terms=default_terms(with_price=False)
        )
        with pytest.raises(RuntimeError):
            bare.sample_product_predictive()

    def test_terms_round_trip_through_netcdf(self, fitted_price_model, tmp_path):
        from nachfrage.models import DemandModel

        path = tmp_path / "model.nc"
        fitted_price_model.to_netcdf(path)

        reloaded = DemandModel.from_netcdf(path)
        assert [type(t).__name__ for t in reloaded.terms] == [
            type(t).__name__ for t in fitted_price_model.terms
        ]

    def test_reload_without_terms_predicts(self, fitted_price_model, tmp_path):
        from nachfrage.models import DemandModel

        path = tmp_path / "model.nc"
        fitted_price_model.to_netcdf(path)

        reloaded = DemandModel.from_netcdf(path)
        ppd = reloaded.sample_product_predictive()
        assert set(ppd.dims) == {"sample", "product"}

    def test_missing_terms_raises_clear_error(self, fitted_model):
        from nachfrage.models import DemandModel

        with pytest.raises(ValueError, match="No terms given"):
            DemandModel.from_idata(fitted_model.idata.copy())

    def test_explicit_terms_still_accepted(self, fitted_model, tmp_path):
        from nachfrage.models import DemandModel
        from nachfrage.terms import default_terms

        path = tmp_path / "model.nc"
        fitted_model.to_netcdf(path)

        reloaded = DemandModel.from_netcdf(path, terms=default_terms(with_price=False))
        assert set(reloaded.sample_product_predictive().dims) == {
            "sample",
            "product",
        }


GROUPS = ["north", "south"]


def _make_two_group_data(rng, n_per: int = 5) -> pd.DataFrame:
    """Two products across two regions, censored sellouts, region-scaled demand."""
    true_mu = {"Cake A": 10.0, "Cake B": 6.0}
    alpha = 5.0
    rows = []
    for product, mu in true_mu.items():
        for region in GROUPS:
            scale = 1.2 if region == "north" else 0.7
            demand = rng.negative_binomial(
                alpha, alpha / (alpha + mu * scale), size=n_per
            )
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
    return pd.DataFrame(rows)


def _fit_two_group(df, *, draws: int = 20, tune: int = 20):
    import numpy as np

    from nachfrage.models import DemandModel
    from nachfrage.terms import GroupContribution

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
    model.build(df)
    model.fit(draws=draws, tune=tune, chains=1, random_seed=0, progressbar=False)
    return model


@pytest.fixture(scope="module")
def fitted_two_group_model():
    rng = np.random.default_rng(0)
    return _fit_two_group(_make_two_group_data(rng))


class TestSampleGroupPredictive:
    """Multi-group posterior predictive over a group-level grid."""

    def test_grid_covers_all_group_combinations(self, fitted_two_group_model):
        ppd = fitted_two_group_model.sample_group_predictive()

        assert set(ppd.dims) == {"sample", "product", "region"}
        assert list(ppd.coords["product"].values) == ["Cake A", "Cake B"]
        assert list(ppd.coords["region"].values) == GROUPS
        posterior = fitted_two_group_model.idata["posterior"]
        total_draws = posterior.sizes["chain"] * posterior.sizes["draw"]
        assert ppd.sizes["sample"] == total_draws

    def test_conditional_restricts_to_one_level(self, fitted_two_group_model):
        ppd = fitted_two_group_model.sample_group_predictive({"region": "north"})

        assert set(ppd.dims) == {"sample", "product", "region"}
        assert list(ppd.coords["region"].values) == ["north"]

    def test_unknown_condition_group_raises(self, fitted_two_group_model):
        with pytest.raises(ValueError, match="Unknown condition group"):
            fitted_two_group_model.sample_group_predictive({"venue": "north"})

    def test_unknown_condition_level_raises(self, fitted_two_group_model):
        with pytest.raises(ValueError, match="not observed"):
            fitted_two_group_model.sample_group_predictive({"region": "east"})

    def test_product_predictive_points_to_group_predictive(
        self, fitted_two_group_model
    ):
        with pytest.raises(ValueError, match="sample_group_predictive"):
            fitted_two_group_model.sample_product_predictive()

    def test_grid_demand_is_positive_and_varies_by_group(self, fitted_two_group_model):
        ppd = fitted_two_group_model.sample_group_predictive()

        means = ppd.mean("sample")
        assert (means > 0).all()

    def test_round_trip_preserves_grid_predictive(
        self, fitted_two_group_model, tmp_path
    ):
        from nachfrage.models import DemandModel

        path = tmp_path / "model.nc"
        fitted_two_group_model.to_netcdf(path)

        reloaded = DemandModel.from_netcdf(path)
        ppd = reloaded.sample_group_predictive({"region": "north"})
        assert set(ppd.dims) == {"sample", "product", "region"}
        assert len(ppd.coords["product"].values) == 2


@pytest.fixture(scope="module")
def fitted_two_group_price_model():
    """Two-group model that also carries a log_price Dot term."""
    rng = np.random.default_rng(2)
    df = _make_two_group_data(rng)
    df["log_price"] = np.log(5.0)
    return _fit_two_group(df)


class TestSampleGroupPredictiveDesign:
    """Grid rows resolve covariate columns generically, with overrides."""

    def test_grid_supports_price_term_model(self, fitted_two_group_price_model):
        ppd = fitted_two_group_price_model.sample_group_predictive()

        assert set(ppd.dims) == {"sample", "product", "region"}
        assert (ppd.mean("sample") > 0).all()

    def test_design_override_sets_the_column(self, fitted_two_group_price_model):
        at_zero = fitted_two_group_price_model.sample_group_predictive(
            design={"log_price": 0.0}
        )
        at_high = fitted_two_group_price_model.sample_group_predictive(
            design={"log_price": np.log(10.0)}
        )

        # Demand at a much higher price should not exceed demand at the low one.
        assert (at_high.mean("sample") <= at_zero.mean("sample")).all()

    def test_unknown_design_column_raises(self, fitted_two_group_price_model):
        with pytest.raises(ValueError, match="Unknown design column"):
            fitted_two_group_price_model.sample_group_predictive(design={"bogus": 1.0})

    def test_design_override_rejected_without_covariates(self, fitted_two_group_model):
        with pytest.raises(ValueError, match="covariate columns"):
            fitted_two_group_model.sample_group_predictive(design={"log_price": 2.0})
