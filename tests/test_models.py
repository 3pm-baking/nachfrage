"""Tests for nachfrage.models — DemandModel lifecycle.

Only nachfrage-owned behavior is covered here. The term algebra itself
(``Intercept``, ``Dot``, ``Sum``, ``Transform``, ``collect_coords``,
``register_data``) is upstream ``pymc_marketing.terms`` and is tested there.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

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
