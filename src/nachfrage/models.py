"""Bayesian demand model with censored NegativeBinomial likelihood.

The DemandModel class encapsulates the full lifecycle:
    build → fit → sample_posterior_predictive → save/load

The model is a hierarchical NegativeBinomial with product-level random effects,
right-censored at the prepared quantity (for sellout observations).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from pymc_extras.prior import Censored, Prior

DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "likelihood": Censored(
        Prior("NegativeBinomial", alpha=Prior("HalfNormal", sigma=5.0)),
    ),
    "mu_global": Prior("Normal", mu=np.log(12), sigma=0.5),
    "sigma_product": Prior("HalfNormal", sigma=0.5),
    "mu_product_raw": Prior("Normal", sigma=1.0, dims="product"),
}


class DemandModel:
    """Bayesian demand model with product-level hierarchical random effects.

    The model assumes demand follows a NegativeBinomial distribution with
    censored observations (when sellout occurs, we only know demand >= prepared).
    Product-level effects are modeled via a non-centered parameterization.

    Lifecycle:
        >>> model = DemandModel(model_config={...})
        >>> model.build(data=df)  # df has columns: sold, prepared, product
        >>> model.fit(draws=1000, tune=1000, chains=4)
        >>> ppd = model.sample_posterior_predictive(n_samples=10000)
        >>> model.to_netcdf("posterior.nc")
        >>> loaded = DemandModel.from_netcdf("posterior.nc")

    Args:
        model_config: Dict of model configuration. Keys:
            likelihood: Censored Prior for the likelihood.
            mu_global: Prior for the global mean (log-scale).
            sigma_product: Prior for product-level scale.
            mu_product_raw: Prior for raw product offsets (non-centered).
    """

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        self.model_config = {
            **DEFAULT_MODEL_CONFIG,
            **(model_config or {}),
        }
        self.model: pm.Model | None = None
        self.idata: xr.DataTree | None = None
        self.product_names: list[str] | None = None

    def build(self, data: pd.DataFrame) -> DemandModel:
        """Build the PyMC model graph from a DataFrame.

        Args:
            data: DataFrame with columns:
                - sold: Observed sales per market day (capped at prepared).
                - prepared: Units prepared per market day.
                - product: Product name for each observation.

        Returns:
            self (for method chaining).

        Raises:
            ValueError: If required columns are missing.
        """
        required = {"sold", "prepared", "product"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"DataFrame must have columns: {', '.join(sorted(required))}. "
                f"Missing: {', '.join(sorted(missing))}"
            )

        sold_arr = data["sold"].values.astype(float)
        prepared_arr = data["prepared"].values.astype(float)

        codes, unique_names = pd.factorize(data["product"])
        product_names = unique_names.tolist()
        product_id_arr = codes.astype(int)

        n_obs = len(sold_arr)

        coords = {
            "product": product_names,
            "obs": np.arange(n_obs),
        }

        with pm.Model(coords=coords) as model:
            mu_global = self.model_config["mu_global"].create_variable("mu_global")
            sigma_product = self.model_config["sigma_product"].create_variable(
                "sigma_product"
            )
            mu_product_raw = self.model_config["mu_product_raw"].create_variable(
                "mu_product_raw",
            )

            mu_product = pm.Deterministic(
                "mu_product",
                pm.math.exp(mu_global + mu_product_raw * sigma_product),
                dims="product",
            )

            mu_obs = mu_product[product_id_arr]

            upper = prepared_arr
            self.model_config["likelihood"].upper = upper

            self.model_config["likelihood"].create_likelihood_variable(
                "demand",
                mu=mu_obs,
                observed=sold_arr,
            )

        self.model = model
        self.product_names = product_names
        return self

    def fit(
        self,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 4,
        nuts_sampler: str = "nutpie",
        random_seed: int = 42,
        progressbar: bool = False,
        **kwargs: Any,
    ) -> xr.DataTree:
        """Sample the posterior distribution.

        Args:
            draws: Number of posterior draws per chain.
            tune: Number of tuning (warm-up) steps per chain.
            chains: Number of Markov chains.
            nuts_sampler: NUTS implementation ("nutpie" or "pymc").
            random_seed: Random seed for reproducibility.
            progressbar: Whether to show a progress bar.
            **kwargs: Additional arguments passed to pm.sample().

        Returns:
            xarray DataTree with posterior samples.

        Raises:
            RuntimeError: If build() has not been called.
        """
        if self.model is None:
            raise RuntimeError("Call build() before fit()")

        self.idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            nuts_sampler=nuts_sampler,
            random_seed=random_seed,
            progressbar=progressbar,
            model=self.model,
            **kwargs,
        )

        return self.idata

    def sample_posterior_predictive(
        self,
        n_samples: int = 10000,
        seed: int = 42,
    ) -> xr.DataArray:
        """Draw posterior predictive demand samples.

        Samples from NegativeBinomial(mu, alpha) using posterior draws
        of mu (per product) and alpha (global overdispersion).

        Uses xarray's stack to flatten chain/draw dimensions, then
        NumPy for the random number generation over the stacked samples.

        Args:
            n_samples: Number of posterior predictive draws.
            seed: Random seed for reproducibility.

        Returns:
            xr.DataArray with dims (sample, product) and product coordinate.

        Raises:
            RuntimeError: If idata is not available (call fit() or
                from_netcdf() first).
        """
        if self.idata is None:
            raise RuntimeError(
                "No posterior samples available. Call fit() or from_netcdf() first."
            )

        mu = self.idata.posterior["mu_product"]
        alpha = self.idata.posterior["demand_alpha"]

        # Stack chain + draw into a single sample dimension
        mu_stacked = mu.stack(sample=("chain", "draw")).transpose("sample", "product")
        alpha_stacked = alpha.stack(sample=("chain", "draw"))

        n_total = mu_stacked.sizes["sample"]
        n_products = mu_stacked.sizes["product"]

        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n_total, size=n_samples)

        mu_s = mu_stacked.values[idx]  # (n_samples, n_products)
        alpha_s = alpha_stacked.values[idx]  # (n_samples,)
        p = alpha_s[:, None] / (alpha_s[:, None] + mu_s)
        ppd_vals = rng.negative_binomial(alpha_s[:, None], p)

        return xr.DataArray(
            ppd_vals,
            dims=("sample", "product"),
            coords={
                "sample": np.arange(n_samples),
                "product": mu.coords["product"],
            },
            name="demand_ppd",
        )

    def to_netcdf(self, path: str | Path, engine: str | None = None) -> None:
        """Save posterior inference data to a netCDF file.

        Product names are stored in the DataTree attrs so they survive
        the roundtrip through from_netcdf().

        Args:
            path: File path for the netCDF output.
            engine: NetCDF backend. One of "netcdf4", "h5netcdf", "scipy",
                or None for auto-detect.

        Raises:
            RuntimeError: If idata is not available.
        """
        if self.idata is None:
            raise RuntimeError("No posterior to save. Call fit() first.")

        idata = self.idata.copy()
        if self.product_names:
            idata.attrs["product_names"] = json.dumps(self.product_names)
        idata.to_netcdf(str(path), engine=engine)

    @classmethod
    def from_idata(
        cls,
        idata: xr.DataTree,
        model_config: dict[str, Any] | None = None,
    ) -> DemandModel:
        """Create a DemandModel from an existing xarray DataTree.

        Returns a lightweight instance without a PyMC model graph — suitable
        for sample_posterior_predictive() and prediction, but not for fit()
        (call build() and fit() again to continue sampling).

        Args:
            idata: xarray DataTree with posterior samples and optional
                "product_names" attr.
            model_config: Optional model config override.

        Returns:
            DemandModel with idata and product_names loaded.
        """
        product_names_str = idata.attrs.pop("product_names", "[]")
        try:
            product_names = json.loads(product_names_str)
        except (json.JSONDecodeError, TypeError):
            product_names = None

        inst = cls(model_config=model_config)
        inst.idata = idata
        inst.product_names = product_names or None
        return inst

    @classmethod
    def from_netcdf(
        cls,
        path: str | Path,
        model_config: dict[str, Any] | None = None,
    ) -> DemandModel:
        """Load a previously saved model from a netCDF file.

        Delegates to from_idata() after reading the file.

        Args:
            path: Path to the netCDF file written by to_netcdf().
            model_config: Optional model config override.

        Returns:
            DemandModel with idata and product_names loaded.
        """
        idata = az.from_netcdf(str(path))
        return cls.from_idata(idata, model_config=model_config)
