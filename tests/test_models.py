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
        ppd = fitted_model.sample_posterior_predictive()

        import xarray as xr

        assert isinstance(ppd, xr.DataArray)

    def test_correct_dims(self, fitted_model):
        """PPD has dims (sample, product)."""
        ppd = fitted_model.sample_posterior_predictive()

        assert set(ppd.dims) == {"sample", "product"}
        assert ppd.sizes["product"] == 2

    def test_product_coords_preserved(self, fitted_model):
        """PPD product coordinate matches the names from build()."""
        ppd = fitted_model.sample_posterior_predictive()

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
        """Default single product gives dims (sample, product) with 1 product."""
        ppd = fitted_model.sample_new_product_predictive()

        assert set(ppd.dims) == {"sample", "product"}
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


class TestDemandModelBuildPrice:
    """Tests for DemandModel.build() with price column."""

    @pytest.fixture
    def price_data_tiny(self, rng):
        """Tiny dataset with price: 2 products, 6 obs each, 2 price levels.

        True elasticity = -1.0 (unit elastic): doubling price halves demand.
        """
        product_names = ["Cake A", "Cake B"]
        true_beta = -1.0
        alpha = 5.0

        price_pairs = [(5.0, 10.0), (6.0, 12.0)]
        ref_mus = [10.0, 6.0]

        prices_list: list[float] = []
        true_mus: list[float] = []
        products: list[str] = []

        for i, (p_low, p_high) in enumerate(price_pairs):
            geo_mean = np.sqrt(p_low * p_high)
            for p in [p_low, p_high]:
                for _ in range(3):
                    mu = ref_mus[i] * np.exp(true_beta * (np.log(p) - np.log(geo_mean)))
                    prices_list.append(p)
                    true_mus.append(mu)
                    products.append(product_names[i])

        true_mus_arr = np.array(true_mus)
        demand = rng.negative_binomial(alpha, alpha / (alpha + true_mus_arr))
        prepared = np.ceil(demand * 1.3).astype(int)
        sold = demand.copy().astype(float)
        censored = (sold >= prepared).astype(bool)
        sold[censored] = prepared[censored]

        return pd.DataFrame({
            "sold": sold,
            "prepared": prepared,
            "product": products,
            "price": prices_list,
        })

    def test_build_with_price_sets_flag(self, price_data_tiny, model_config):
        """_has_price is True when price column present."""
        from nachfrage.models import DemandModel

        dm = DemandModel(model_config)
        dm.build(price_data_tiny)

        assert dm._has_price is True
        assert dm._mean_log_price is not None
        assert "Cake A" in dm._mean_log_price
        assert "Cake B" in dm._mean_log_price

    def test_build_with_price_creates_beta_price(self, price_data_tiny, model_config):
        """PyMC model includes beta_price variable."""
        from nachfrage.models import DemandModel

        dm = DemandModel(model_config)
        dm.build(price_data_tiny)

        assert dm.model is not None
        assert "beta_price" in dm.model.named_vars

    def test_build_without_price_no_beta(self, tiny_data, model_config):
        """Without price column, no beta_price in model."""
        from nachfrage.models import DemandModel

        dm = DemandModel(model_config)
        dm.build(tiny_data)

        assert dm._has_price is False
        assert dm._mean_log_price is None
        assert "beta_price" not in dm.model.named_vars

    def test_build_price_log_price_is_stored(self, price_data_tiny, model_config):
        """_mean_log_price stores geometric mean of price per product."""
        from nachfrage.models import DemandModel

        dm = DemandModel(model_config)
        dm.build(price_data_tiny)

        # Cake A at prices 5 and 10: geo_mean = sqrt(50) ≈ 7.07, log ≈ 1.955
        # Cake B at prices 6 and 12: geo_mean = sqrt(72) ≈ 8.49, log ≈ 2.139
        assert dm._mean_log_price is not None
        np.testing.assert_allclose(dm._mean_log_price["Cake A"], np.log(np.sqrt(50)), rtol=1e-5)
        np.testing.assert_allclose(dm._mean_log_price["Cake B"], np.log(np.sqrt(72)), rtol=1e-5)


class TestDemandModelFitPrice:
    """Integration tests: fit model with price covariate."""

    def test_fit_produces_beta_price_posterior(self, tiny_data, model_config):
        """Fitting with price gives beta_price in posterior."""
        from nachfrage.models import DemandModel

        # Add price column to tiny_data (constant price for each product)
        df = tiny_data.copy()
        df["price"] = np.where(df["product"] == "Cake A", 5.0, 10.0)

        dm = DemandModel(model_config)
        dm.build(df)
        dm.fit(draws=5, tune=5, chains=1, random_seed=42, progressbar=False)

        assert dm.idata is not None
        posterior = dm.idata.posterior
        assert "beta_price" in posterior.data_vars
        assert posterior["beta_price"].values.shape is not None


class TestDemandModelPredictAtPrice:
    """Tests for predict_demand_at_price()."""

    @pytest.fixture
    def fitted_price_model(self, tiny_data, model_config):
        """A fitted DemandModel with price covariate."""
        from nachfrage.models import DemandModel

        df = tiny_data.copy()
        df["price"] = np.where(df["product"] == "Cake A", 5.0, 10.0)

        dm = DemandModel(model_config)
        dm.build(df)
        dm.fit(draws=5, tune=5, chains=1, random_seed=42, progressbar=False)
        return dm

    def test_returns_xarray_dataarray(self, fitted_price_model):
        """Returns xr.DataArray."""
        ppd = fitted_price_model.predict_demand_at_price(price=7.0)

        import xarray as xr

        assert isinstance(ppd, xr.DataArray)

    def test_correct_dims(self, fitted_price_model):
        """PPD has dims (sample, product)."""
        ppd = fitted_price_model.predict_demand_at_price(price=7.0)

        assert set(ppd.dims) == {"sample", "product"}
        assert ppd.sizes["product"] == 2

    def test_higher_price_lower_demand(self, fitted_price_model):
        """Higher price produces lower or equal mean demand."""
        ppd_low = fitted_price_model.predict_demand_at_price(price=5.0, random_seed=42)
        ppd_high = fitted_price_model.predict_demand_at_price(price=15.0, random_seed=43)

        mean_low = ppd_low.mean(dim="sample").values
        mean_high = ppd_high.mean(dim="sample").values

        assert np.all(mean_low >= mean_high)

    def test_raises_without_idata(self, model_config):
        """Raises RuntimeError if called before fit."""
        from nachfrage.models import DemandModel

        dm = DemandModel(model_config)
        with pytest.raises(RuntimeError):
            dm.predict_demand_at_price(price=7.0)

    def test_no_price_model_falls_back(self, tiny_data, model_config):
        """Without price covariate, predict_demand_at_price is same as PPD."""
        from nachfrage.models import DemandModel

        dm = DemandModel(model_config)
        dm.build(tiny_data)
        dm.fit(draws=5, tune=5, chains=1, random_seed=42, progressbar=False)

        ppd_pred = dm.predict_demand_at_price(price=99.0, random_seed=42)
        ppd_std = dm.sample_posterior_predictive(random_seed=42)

        np.testing.assert_array_equal(ppd_pred.values, ppd_std.values)


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

        ppd_orig = fitted_model.sample_posterior_predictive(random_seed=42)

        path = tmp_path / "test_posterior.nc"
        fitted_model.to_netcdf(path)
        assert path.exists()

        loaded = DemandModel.from_netcdf(path)
        assert loaded.idata is not None
        assert loaded.product_names == fitted_model.product_names

        # PPD with same seed should be identical
        ppd_loaded = loaded.sample_posterior_predictive(random_seed=42)
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


class TestDemandModelNetCDFPrice:
    """Tests for to_netcdf() / from_netcdf() with price metadata."""

    @pytest.fixture
    def fitted_price_model(self, tiny_data, model_config):
        """A fitted DemandModel with price covariate."""
        from nachfrage.models import DemandModel

        df = tiny_data.copy()
        df["price"] = np.where(df["product"] == "Cake A", 5.0, 10.0)

        dm = DemandModel(model_config)
        dm.build(df)
        dm.fit(draws=5, tune=5, chains=1, random_seed=42, progressbar=False)
        return dm

    def test_roundtrip_preserves_price_metadata(self, fitted_price_model, tmp_path):
        """_has_price and _mean_log_price survive roundtrip."""
        from nachfrage.models import DemandModel

        path = tmp_path / "test_price_posterior.nc"
        fitted_price_model.to_netcdf(path)

        loaded = DemandModel.from_netcdf(path)
        assert loaded._has_price is True
        assert loaded._mean_log_price is not None
        assert loaded._mean_log_price.keys() == fitted_price_model._mean_log_price.keys()
