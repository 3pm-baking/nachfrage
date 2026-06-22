"""Bayesian demand model with censored NegativeBinomial likelihood.

The DemandModel class encapsulates the full lifecycle:
    build → fit → compute_ppd → save/load

The model is a hierarchical NegativeBinomial with product-level random effects,
right-censored at the prepared quantity (for sellout observations).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pymc_extras.prior import Censored
from pymc_extras.prior import Prior


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
        >>> model.build(sold=sold, prepared=prepared, censored=censored,
        ...             product_ids=product_ids, product_names=product_names)
        >>> model.fit(draws=1000, tune=1000, chains=4)
        >>> ppd = model.compute_ppd(n_samples=10000)
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
        self.idata: az.InferenceData | None = None
        self.product_names: list[str] | None = None

    def build(
        self,
        sold: np.ndarray,
        prepared: np.ndarray,
        censored: np.ndarray,
        product_ids: np.ndarray,
        product_names: list[str],
    ) -> DemandModel:
        """Build the PyMC model graph.

        Args:
            sold: Observed sales per market day (n_obs,).
            prepared: Units prepared per market day (n_obs,).
            censored: Boolean array — True if sellout (demand >= prepared).
            product_ids: Integer index mapping each observation to a product.
            product_names: Unique product names (used as coordinate labels).

        Returns:
            self (for method chaining).

        Raises:
            ValueError: If array lengths don't match.
        """
        if not (len(sold) == len(prepared) == len(censored) == len(product_ids)):
            raise ValueError(
                "sold, prepared, censored, and product_ids must have the same length"
            )

        n_obs = len(sold)
        sold_arr = np.asarray(sold, dtype=float)
        prepared_arr = np.asarray(prepared, dtype=float)
        censored_arr = np.asarray(censored, dtype=bool)
        product_id_arr = np.asarray(product_ids, dtype=int)

        coords = {
            "product": list(product_names),
            "obs": np.arange(n_obs),
        }

        with pm.Model(coords=coords) as model:
            μ_global = self.model_config["mu_global"].create_variable("μ_global")
            σ_product = self.model_config["sigma_product"].create_variable("σ_product")
            μ_product_raw = self.model_config["mu_product_raw"].create_variable(
                "μ_product_raw",
            )

            μ_product = pm.Deterministic(
                "μ_product",
                pm.math.exp(μ_global + μ_product_raw * σ_product),
                dims="product",
            )

            μ_obs = μ_product[product_id_arr]

            # Set the data-dependent upper bound on the Censored config
            upper = pt.switch(censored_arr, prepared_arr, np.inf)
            self.model_config["likelihood"].upper = upper

            self.model_config["likelihood"].create_likelihood_variable(
                "demand", mu=μ_obs, observed=sold_arr,
            )

        self.model = model
        self.product_names = list(product_names)
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
    ) -> az.InferenceData:
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
            ArviZ InferenceData with posterior samples.

        Raises:
            RuntimeError: If build() has not been called.
        """
        if self.model is None:
            raise RuntimeError("Call build() before fit()")

        with self.model:
            self.idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                nuts_sampler=nuts_sampler,
                random_seed=random_seed,
                progressbar=progressbar,
                **kwargs,
            )

        return self.idata

    def compute_ppd(
        self,
        n_samples: int = 10000,
        seed: int = 42,
    ) -> xr.DataArray:
        """Compute posterior predictive distribution.

        Samples from NegativeBinomial(mu, alpha) using the posterior draws
        of mu (per product) and alpha (global overdispersion).

        Args:
            n_samples: Number of PPD draws.
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

        mu = self.idata.posterior["μ_product"]
        alpha = self.idata.posterior["demand_alpha"]

        # mu: (chain, draw, product), alpha: (chain, draw)
        n_chains = mu.sizes["chain"]
        n_draws = mu.sizes["draw"]
        n_products = mu.sizes["product"]

        rng = np.random.default_rng(seed)
        mu_flat = mu.values.reshape(n_chains * n_draws, n_products)
        alpha_flat = alpha.values.reshape(n_chains * n_draws)

        idx = rng.integers(0, len(alpha_flat), size=n_samples)
        alpha_s = alpha_flat[idx, None]
        p = alpha_s / (alpha_s + mu_flat[idx])
        ppd_vals = rng.negative_binomial(alpha_s, p)

        return xr.DataArray(
            ppd_vals,
            dims=("sample", "product"),
            coords={
                "sample": np.arange(n_samples),
                "product": mu.coords["product"],
            },
            name="demand_ppd",
        )

    def to_netcdf(self, path: str | Path) -> None:
        """Save posterior InferenceData to a netCDF file.

        Product names are stored in the InferenceData attrs so they survive
        the roundtrip through from_netcdf().

        Args:
            path: File path for the netCDF output.

        Raises:
            RuntimeError: If idata is not available.
        """
        if self.idata is None:
            raise RuntimeError(
                "No posterior to save. Call fit() first."
            )

        idata = self.idata.copy()
        if self.product_names:
            idata.attrs["product_names"] = json.dumps(self.product_names)
        idata.to_netcdf(str(path), engine="h5netcdf")

    @classmethod
    def from_netcdf(
        cls,
        path: str | Path,
        model_config: dict[str, Any] | None = None,
    ) -> DemandModel:
        """Load a previously saved model from a netCDF file.

        Returns a lightweight instance without a PyMC model graph — suitable
        for compute_ppd() and prediction, but not for fit() (call build() and
        fit() again to continue sampling).

        Args:
            path: Path to the netCDF file written by to_netcdf().
            model_config: Optional model config override. If None, the
                default config is used.

        Returns:
            DemandModel with idata and product_names loaded.
        """
        idata = az.from_netcdf(str(path))

        product_names_str = idata.attrs.pop("product_names", "[]")
        try:
            product_names = json.loads(product_names_str)
        except (json.JSONDecodeError, TypeError):
            product_names = None

        inst = cls(model_config=model_config)
        inst.idata = idata
        inst.product_names = product_names or None
        return inst
