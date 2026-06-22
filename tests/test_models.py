"""Tests for nachfrage.models — DemandModel lifecycle."""

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture
def model_config():
    """Default model config with Prior objects."""
    from pymc_extras.prior import Censored, Prior

    return {
        "likelihood": Censored(
            Prior("NegativeBinomial", alpha=Prior("HalfNormal", sigma=5.0)),
        ),
        "mu_global": Prior("Normal", mu=np.log(12), sigma=0.5),
        "sigma_product": Prior("HalfNormal", sigma=0.5),
        "mu_product_raw": Prior("Normal", sigma=1.0, dims="product"),
    }


@pytest.fixture
def small_data(rng):
    """Small realistic dataset: 3 products, 10 obs each = 30 obs."""
    n_products = 3
    n_per = 10
    product_names = ["Cheese Cake (slice)", "Apple Strudel (piece)", "Kolache (each)"]

    true_mu = np.array([12.0, 8.0, 5.0])
    alpha = 5.0

    demand = rng.negative_binomial(
        alpha, alpha / (alpha + true_mu), size=(n_per, n_products)
    )
    prepared = np.ceil(demand * 1.2).astype(int)

    sold = demand.T.ravel()
    prepared = prepared.T.ravel()
    censored = (sold >= prepared).astype(bool)
    sold[censored] = prepared[censored]

    return pd.DataFrame(
        {
            "sold": sold,
            "prepared": prepared,
            "product": np.repeat(product_names, n_per),
        }
    )


@pytest.fixture
def tiny_data(rng):
    """Tiny dataset for fast integration tests: 2 products, 5 obs each."""
    n_products = 2
    n_per = 5
    product_names = ["Cake A", "Cake B"]
    true_mu = np.array([10.0, 6.0])
    alpha = 5.0
    demand = rng.negative_binomial(
        alpha, alpha / (alpha + true_mu), size=(n_per, n_products)
    )
    prepared = np.ceil(demand * 1.3).astype(int)
    sold = demand.T.ravel()
    prepared = prepared.T.ravel()
    censored = (sold >= prepared).astype(bool)
    sold[censored] = prepared[censored]
    return pd.DataFrame(
        {
            "sold": sold,
            "prepared": prepared,
            "product": np.repeat(product_names, n_per),
        }
    )


class TestDemandModelInit:
    """Tests for DemandModel.__init__()."""

    def test_init_with_no_args(self):
        """Default model_config is used when no args passed."""
        from nachfrage.models import DemandModel

        model = DemandModel()
        assert model.model_config is not None
        assert "likelihood" in model.model_config
        assert "mu_global" in model.model_config
        assert model.model is None
        assert model.idata is None

    def test_init_with_partial_override(self, model_config):
        """Custom config merges with defaults."""
        from pymc_extras.prior import Prior

        from nachfrage.models import DemandModel

        custom = DemandModel({"mu_global": Prior("Normal", mu=np.log(20), sigma=1.0)})
        assert custom.model_config is not None
        # Should still have the other default keys
        assert "likelihood" in custom.model_config
        assert "sigma_product" in custom.model_config

    def test_init_model_and_idata_start_none(self):
        """New instance has model=None and idata=None."""
        from nachfrage.models import DemandModel

        model = DemandModel()
        assert model.model is None
        assert model.idata is None


class TestDemandModelBuild:
    """Tests for DemandModel.build()."""

    def test_build_creates_model(self, small_data, model_config):
        """build() creates a pm.Model."""
        from nachfrage.models import DemandModel

        dm = DemandModel(model_config)
        dm.build(small_data)

        import pymc as pm

        assert dm.model is not None
        assert isinstance(dm.model, pm.Model)

    def test_build_missing_columns(self, model_config):
        """Raises ValueError when required columns are missing."""
        from nachfrage.models import DemandModel

        dm = DemandModel(model_config)
        bad_df = pd.DataFrame({"sold": [1, 2, 3]})
        with pytest.raises(ValueError):
            dm.build(bad_df)

    def test_build_stores_product_names(self, small_data, model_config):
        """product_names are stored after build (derived from product column)."""
        from nachfrage.models import DemandModel

        dm = DemandModel(model_config)
        dm.build(small_data)

        expected = ["Cheese Cake (slice)", "Apple Strudel (piece)", "Kolache (each)"]
        assert dm.product_names == expected

    def test_build_with_single_product(self, model_config, rng):
        """Works with only one product."""
        from nachfrage.models import DemandModel

        n = 10
        sold = rng.poisson(8, size=n)
        prepared = (sold * 1.3).astype(int)
        censored = (sold >= prepared).astype(bool)
        sold[censored] = prepared[censored]

        df = pd.DataFrame(
            {
                "sold": sold,
                "prepared": prepared,
                "product": ["Only Cake"] * n,
            }
        )

        dm = DemandModel(model_config)
        dm.build(df)

        assert dm.model is not None


class TestDemandModelFit:
    """Tests for DemandModel.fit() (integration — requires actual sampling)."""

    def test_fit_produces_idata(self, tiny_data, model_config):
        """fit() stores an InferenceData object with expected variables."""
        from nachfrage.models import DemandModel

        dm = DemandModel(model_config)
        dm.build(tiny_data)
        dm.fit(
            draws=5,
            tune=5,
            chains=1,
            random_seed=42,
            progressbar=False,
        )

        import xarray as xr

        assert dm.idata is not None
        assert isinstance(dm.idata, xr.DataTree)
        assert "/posterior" in dm.idata.groups

        posterior = dm.idata.posterior
        assert "mu_global" in posterior.data_vars
        assert "sigma_product" in posterior.data_vars
        assert "demand_alpha" in posterior.data_vars
        assert "mu_product" in posterior.data_vars

    def test_fit_with_nutpie_sampler(self, tiny_data, model_config):
        """fit() works with nutpie sampler (default)."""
        from nachfrage.models import DemandModel

        dm = DemandModel(model_config)
        dm.build(tiny_data)
        dm.fit(
            draws=5,
            tune=5,
            chains=2,
            nuts_sampler="nutpie",
            random_seed=42,
            progressbar=False,
        )

        assert dm.idata is not None


class TestDemandModelSamplePPD:
    """Tests for DemandModel.sample_posterior_predictive()."""

    @pytest.fixture
    def fitted_model(self, tiny_data, model_config):
        """A fitted DemandModel instance."""
        from nachfrage.models import DemandModel

        dm = DemandModel(model_config)
        dm.build(tiny_data)
        dm.fit(draws=5, tune=5, chains=1, random_seed=42, progressbar=False)
        return dm

    def test_returns_xarray_dataarray(self, fitted_model):
        """Returns xr.DataArray."""
        ppd = fitted_model.sample_posterior_predictive(n_samples=100)

        import xarray as xr

        assert isinstance(ppd, xr.DataArray)

    def test_correct_dims(self, fitted_model):
        """PPD has dims (sample, product)."""
        ppd = fitted_model.sample_posterior_predictive(n_samples=100)

        assert set(ppd.dims) == {"sample", "product"}
        assert ppd.sizes["sample"] == 100
        assert ppd.sizes["product"] == 2

    def test_product_coords_preserved(self, fitted_model):
        """PPD product coordinate matches the names from build()."""
        ppd = fitted_model.sample_posterior_predictive(n_samples=100)

        assert list(ppd.coords["product"].values) == ["Cake A", "Cake B"]

    def test_raises_without_idata(self, model_config):
        """Raises RuntimeError if called before fit."""
        from nachfrage.models import DemandModel

        dm = DemandModel(model_config)
        with pytest.raises(RuntimeError):
            dm.sample_posterior_predictive()


class TestDemandModelNewProductPPD:
    """Tests for DemandModel.sample_new_product_predictive()."""

    @pytest.fixture
    def fitted_model(self, tiny_data, model_config):
        """A fitted DemandModel instance."""
        from nachfrage.models import DemandModel

        dm = DemandModel(model_config)
        dm.build(tiny_data)
        dm.fit(draws=5, tune=5, chains=1, random_seed=42, progressbar=False)
        return dm

    def test_returns_xarray_dataarray(self, fitted_model):
        """Returns xr.DataArray."""
        ppd = fitted_model.sample_new_product_predictive()

        import xarray as xr

        assert isinstance(ppd, xr.DataArray)

    def test_correct_dims_single_product(self, fitted_model):
        """Default single product gives dims (chain, draw, product) with 1 product."""
        ppd = fitted_model.sample_new_product_predictive()

        assert set(ppd.dims) == {"chain", "draw", "product"}
        assert ppd.sizes["product"] == 1

    def test_multiple_products(self, fitted_model):
        """n_products controls the product dimension."""
        ppd = fitted_model.sample_new_product_predictive(n_products=3)

        assert ppd.sizes["product"] == 3
        assert list(ppd.coords["product"].values) == ["new_0", "new_1", "new_2"]

    def test_non_negative_integer_values(self, fitted_model):
        """All PPD values are non-negative integers."""
        ppd = fitted_model.sample_new_product_predictive()

        assert ppd.dtype.kind == "i"
        assert np.all(ppd.values >= 0)


class TestDemandModelNetCDF:
    """Tests for to_netcdf() / from_netcdf() roundtrip."""

    @pytest.fixture
    def fitted_model(self, tiny_data, model_config):
        """A fitted DemandModel instance."""
        from nachfrage.models import DemandModel

        dm = DemandModel(model_config)
        dm.build(tiny_data)
        dm.fit(draws=5, tune=5, chains=1, random_seed=42, progressbar=False)
        return dm

    def test_roundtrip_preserves_ppd(self, fitted_model, tmp_path):
        """After to_netcdf + from_netcdf, sample_posterior_predictive gives same results."""
        from nachfrage.models import DemandModel

        ppd_orig = fitted_model.sample_posterior_predictive(n_samples=100, seed=42)

        path = tmp_path / "test_posterior.nc"
        fitted_model.to_netcdf(path)
        assert path.exists()

        loaded = DemandModel.from_netcdf(path)
        assert loaded.idata is not None
        assert loaded.product_names == fitted_model.product_names

        # PPD with same seed should be identical
        ppd_loaded = loaded.sample_posterior_predictive(n_samples=100, seed=42)
        np.testing.assert_array_equal(ppd_orig.values, ppd_loaded.values)

    def test_roundtrip_preserves_product_names(self, fitted_model, tmp_path):
        """Product names survive roundtrip."""
        from nachfrage.models import DemandModel

        path = tmp_path / "test_posterior.nc"
        fitted_model.to_netcdf(path)

        loaded = DemandModel.from_netcdf(path)
        assert loaded.product_names == fitted_model.product_names

    def test_from_netcdf_with_model_config(self, fitted_model, tmp_path):
        """from_netcdf accepts optional model_config."""
        from nachfrage.models import DemandModel

        path = tmp_path / "test_posterior.nc"
        fitted_model.to_netcdf(path)

        loaded = DemandModel.from_netcdf(path, model_config={"mu_global": "override"})
        assert loaded.model_config["mu_global"] == "override"
