"""Bayesian demand model with censored NegativeBinomial likelihood.

The linear predictor is constructed from additive ``ModelTerm`` instances,
each of which registers its own data and creates its own variables::

    from pymc_extras.prior import Censored, Prior
    from nachfrage.terms import InterceptTerm, GroupContribution, LinearCovariate
    from nachfrage.models import DemandModel

    model = DemandModel(
        likelihood=Censored(Prior("NegativeBinomial",
                            alpha=Prior("HalfNormal", sigma=5.0))),
        terms=[
            InterceptTerm(prior=Prior("Normal", mu=np.log(12), sigma=0.5)),
            GroupContribution(
                data_source="product",
                prior=Prior("Normal", mu=0,
                            sigma=Prior("HalfNormal", sigma=0.5),
                            dims="product"),
            ),
            LinearCovariate(
                data_source="log_price",
                prior=Prior("Normal", mu=-0.7, sigma=0.5),
            ),
        ],
    )
    model.build(ds)
    model.fit()
    ppd = model.sample_posterior_predictive()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from pymc_extras.prior import Censored, Prior

from nachfrage.terms import LinearCovariate, ModelTerm

DEFAULT_LIKELIHOOD = Censored(
    Prior("NegativeBinomial", alpha=Prior("HalfNormal", sigma=5.0)),
)


def _dataframe_to_dataset(df: pd.DataFrame) -> xr.Dataset:
    data_vars: dict[str, Any] = {}
    for col in df.columns:
        data_vars[col] = ("obs", df[col].values)
    return xr.Dataset(data_vars, coords={"obs": np.arange(len(df))})


class DemandModel:
    """Bayesian demand model with a compositional linear predictor.

    The model assumes demand follows a NegativeBinomial distribution with
    censored observations (when sellout occurs, we only know demand >= prepared).

    The linear predictor is built from additive ``ModelTerm`` instances
    passed to ``terms``.

    Lifecycle::

        model = DemandModel(terms=[...], likelihood=...)
        model.build(ds)            # xr.Dataset or pd.DataFrame
        model.fit(draws=1000, tune=1000, chains=4)
        ppd = model.sample_posterior_predictive()
        model.to_netcdf("posterior.nc")
        loaded = DemandModel.from_netcdf("posterior.nc")

    Args:
        terms: Additive ``ModelTerm`` instances for the linear predictor.
        likelihood: Censored Prior for the demand likelihood.
    """

    def __init__(
        self,
        *,
        terms: list[ModelTerm],
        likelihood: Prior = DEFAULT_LIKELIHOOD,
    ) -> None:
        self.terms = list(terms)
        self.likelihood = likelihood
        self.model: pm.Model | None = None
        self.idata: xr.DataTree | None = None

    def build(
        self,
        data: pd.DataFrame | xr.Dataset,
    ) -> DemandModel:
        """Build the PyMC model graph from data.

        Args:
            data: DataFrame or Dataset. Must contain ``sold`` and ``prepared``
                variables indexed by ``obs``. Additional variables are read
                by the configured terms.

        Returns:
            self (for method chaining).
        """
        if isinstance(data, pd.DataFrame):
            ds = _dataframe_to_dataset(data)
        else:
            ds = data

        required = {"sold", "prepared"}
        missing = required - set(ds.data_vars)
        if missing:
            raise ValueError(
                f"Dataset must have variables: {', '.join(sorted(required))}. "
                f"Missing: {', '.join(sorted(missing))}"
            )

        sold_arr = ds["sold"].values.astype(float)
        prepared_arr = ds["prepared"].values.astype(float)
        n_obs = ds.sizes["obs"]
        coords: dict[str, Any] = {"obs": np.arange(n_obs)}

        with pm.Model(coords=coords) as model:
            mu = 0
            for term in self.terms:
                term.setup(model, ds)
                mu = mu + term.get_contribution(model)

            upper = prepared_arr
            self.likelihood.upper = upper
            self.likelihood.create_likelihood_variable(
                "demand",
                mu=mu,
                observed=sold_arr,
            )

        self.model = model
        self._training_ds = ds
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
            random_seed: Random seed.
            progressbar: Whether to show a progress bar.
            **kwargs: Extra args passed to ``pm.sample()``.

        Returns:
            xarray DataTree with posterior samples.
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

    def rebuild_prediction_model(self) -> pm.Model:
        """Create a new PyMC model for posterior predictive sampling.

        Rebuilds the model structure identically to ``build()`` using the
        training Dataset, so ``pm.sample_posterior_predictive`` can fill
        variables from the posterior trace.
        """
        ds = getattr(self, "_training_ds", None)
        if ds is None:
            raise RuntimeError("No training data. Call build() first.")

        coords: dict[str, Any] = {"obs": np.arange(ds.sizes["obs"])}

        pred_model = pm.Model(coords=coords)
        with pred_model:
            mu = 0
            for term in self.terms:
                term.setup(pred_model, ds)
                mu = mu + term.get_contribution(pred_model)

            inner = getattr(self.likelihood, "distribution", self.likelihood)
            nan_obs = xr.DataArray(np.full(ds.sizes["obs"], np.nan), dims="obs")
            inner.create_likelihood_variable("demand", mu=mu, observed=nan_obs)
        return pred_model

    def sample_posterior_predictive(
        self,
        random_seed: int = 42,
    ) -> xr.DataArray:
        """Draw posterior predictive demand.

        Returns:
            xr.DataArray with dims ``(sample, obs)``.
        """
        if self.idata is None:
            raise RuntimeError(
                "No posterior samples available. Call fit() or from_netcdf() first."
            )

        pred_model = self.rebuild_prediction_model()
        ppd_idata = pm.sample_posterior_predictive(
            self.idata,
            model=pred_model,
            var_names=["demand"],
            random_seed=random_seed,
        )
        da = ppd_idata.posterior_predictive["demand"]
        # The obs dimension name is the first non-chain/draw dim
        obs_dim = [d for d in da.dims if d not in ("chain", "draw")][0]
        return da.stack(sample=("chain", "draw")).transpose("sample", obs_dim)

    def predict_demand_at_price(
        self,
        price: float,
        random_seed: int = 42,
    ) -> xr.DataArray:
        """Draw posterior predictive demand at a counterfactual uniform price.

        Falls back to ``sample_posterior_predictive()`` if no price term
        is found.
        """
        if self.idata is None:
            raise RuntimeError(
                "No posterior samples available. Call fit() or from_netcdf() first."
            )

        has_price_term = any(
            isinstance(t, LinearCovariate) and t.data_source == "log_price"
            for t in self.terms
        )
        if not has_price_term:
            return self.sample_posterior_predictive(random_seed=random_seed)

        ds = getattr(self, "_training_ds", None)
        if ds is None:
            raise RuntimeError("No training data. Call build() first.")

        log_price_val = np.log(price)
        coords: dict[str, Any] = {"obs": np.arange(ds.sizes["obs"])}

        pred_model = pm.Model(coords=coords)
        with pred_model:
            mu = 0
            for term in self.terms:
                if isinstance(term, LinearCovariate) and term.data_source == "log_price":
                    pm.Data(
                        "covariate_log_price",
                        np.full(ds.sizes["obs"], log_price_val),
                    )
                    mu = mu + term.get_contribution(pred_model)
                else:
                    term.setup(pred_model, ds)
                    mu = mu + term.get_contribution(pred_model)

            inner = getattr(self.likelihood, "distribution", self.likelihood)
            nan_obs = xr.DataArray(np.full(ds.sizes["obs"], np.nan), dims="obs")
            inner.create_likelihood_variable("demand", mu=mu, observed=nan_obs)

        ppd_idata = pm.sample_posterior_predictive(
            self.idata,
            model=pred_model,
            var_names=["demand"],
            random_seed=random_seed,
        )
        da = ppd_idata.posterior_predictive["demand"]
        obs_dim = [d for d in da.dims if d not in ("chain", "draw")][0]
        return da.stack(sample=("chain", "draw")).transpose("sample", obs_dim)

    def to_netcdf(self, path: str | Path, engine: str | None = None) -> None:
        """Save posterior inference data to a netCDF file."""
        if self.idata is None:
            raise RuntimeError("No posterior to save. Call fit() first.")
        self.idata.copy().to_netcdf(str(path), engine=engine)

    @classmethod
    def from_idata(
        cls,
        idata: xr.DataTree,
        *,
        terms: list[ModelTerm],
        likelihood: Prior = DEFAULT_LIKELIHOOD,
    ) -> DemandModel:
        """Create a DemandModel from an xarray DataTree (no model graph).

        Suitable for ``sample_posterior_predictive()``, not ``fit()``.
        """
        inst = cls(terms=terms, likelihood=likelihood)
        inst.idata = idata
        return inst

    @classmethod
    def from_netcdf(
        cls,
        path: str | Path,
        *,
        terms: list[ModelTerm],
        likelihood: Prior = DEFAULT_LIKELIHOOD,
    ) -> DemandModel:
        """Load a saved model from a netCDF file."""
        idata = az.from_netcdf(str(path))
        return cls.from_idata(idata, terms=terms, likelihood=likelihood)
