"""Bayesian demand model with censored NegativeBinomial likelihood.

The linear predictor is a composition of :mod:`pymc_marketing.terms` pieces::

    from pymc_extras.prior import Prior
    from pymc_marketing.terms import Dot, Intercept
    from nachfrage.terms import GroupContribution
    from nachfrage.models import DemandModel

    model = DemandModel(
        terms=[
            Intercept(prior=Prior("Normal", mu=np.log(12), sigma=0.5)),
            GroupContribution(
                data_source="product",
                prior=Prior("Normal", mu=0,
                            sigma=Prior("HalfNormal", sigma=0.5),
                            dims="product"),
            ),
        ],
    )
    model.build(ds)
    model.fit()

Two predictive resolutions are available, because they answer different
questions:

- ``sample_posterior_predictive()`` draws demand per *observation*
  (dims ``(sample, obs)``) with sellout censoring retained. This is the
  fitted model's own resolution.
- ``sample_product_predictive()`` draws demand per *group level*
  (dims ``(sample, product)``) uncensored, which is what the newsvendor
  decision math consumes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.xtensor.math as ptx_math
import xarray as xr
from pymc_extras.prior import Censored, Prior
from pymc_marketing.terms import (
    Dot,
    Sum,
    Transform,
    build_param,
    collect_coords,
    collect_terms,
    register_data,
)
from pymc_marketing.terms import serialization as _serialization

from nachfrage.terms import unique_labels

DEFAULT_LIKELIHOOD = Censored(
    Prior("NegativeBinomial", alpha=Prior("HalfNormal", sigma=5.0)),
)

LOG_PRICE_VAR = "log_price"
COVARIATE_DIM = "feature"
#: Columns reshaped to ``(obs, feature)`` design matrices for ``Dot`` terms.
COVARIATE_COLUMNS = (LOG_PRICE_VAR,)
#: DataTree group holding the predictor columns needed to rebuild a graph.
DESIGN_GROUP = "design"
#: Likelihood-only columns, kept out of the saved design so counts are not
#: written into a model artifact that prediction does not need.
LIKELIHOOD_COLUMNS = ("sold", "prepared")
#: Trace attribute holding the JSON-serialized term list, so a reloaded model
#: rebuilds the same graph without the caller re-supplying terms.
TERMS_ATTR = "nachfrage_terms"


def _dataframe_to_dataset(df: pd.DataFrame) -> xr.Dataset:
    """Build a model dataset, promoting covariate columns to design matrices.

    ``Dot`` terms compute ``data @ beta``, so a continuous column must carry
    both an ``obs`` and a ``feature`` dimension rather than a flat vector.
    """
    data_vars: dict[str, Any] = {}
    covariates: list[str] = []
    for col in df.columns:
        if col in COVARIATE_COLUMNS:
            values = np.asarray(df[col].values, dtype=float).reshape(-1, 1)
            data_vars[col] = (("obs", COVARIATE_DIM), values)
            covariates.append(col)
        else:
            data_vars[col] = ("obs", df[col].values)

    coords: dict[str, Any] = {"obs": np.arange(len(df))}
    if covariates:
        coords[COVARIATE_DIM] = [LOG_PRICE_VAR]
    return xr.Dataset(data_vars, coords=coords)


def _at_price(ds: xr.Dataset, price: float) -> xr.Dataset:
    """Return a copy of ``ds`` with the log-price design column set to ``price``."""
    out = ds.copy()
    if LOG_PRICE_VAR in ds:
        n = ds.sizes["obs"]
        out[LOG_PRICE_VAR] = (
            ("obs", COVARIATE_DIM),
            np.full((n, 1), np.log(price), dtype=float),
        )
    return out


class DemandModel:
    """Bayesian demand model with a compositional linear predictor.

    Lifecycle::

        model = DemandModel(terms=[...])
        model.build(ds)                      # xr.Dataset or pd.DataFrame
        model.fit(draws=1000, tune=1000, chains=4)
        model.sample_posterior_predictive()  # (sample, obs)
        model.sample_product_predictive()    # (sample, product)
        model.to_netcdf("posterior.nc")      # saves posterior + design + terms
        DemandModel.from_netcdf("posterior.nc")  # terms come back automatically

    Args:
        terms: Additive terms for the linear predictor.
        likelihood: Censored prior for the demand likelihood.
        group: Name of the grouping variable identifying product levels.
    """

    def __init__(
        self,
        *,
        terms: list[Any],
        likelihood: Prior = DEFAULT_LIKELIHOOD,
        group: str = "product",
    ) -> None:
        self.terms = list(terms)
        self.likelihood = likelihood
        self.group = group
        self.model: pm.Model | None = None
        self.idata: xr.DataTree | None = None
        self._training_ds: xr.Dataset | None = None

    # ---------------------------------------------------------------- build

    def build(self, data: pd.DataFrame | xr.Dataset) -> DemandModel:
        """Build the PyMC model graph from data.

        Args:
            data: DataFrame or Dataset. Must contain ``sold`` and ``prepared``
                indexed by ``obs``, plus whatever the configured terms read.

        Returns:
            self (for method chaining).
        """
        ds = _dataframe_to_dataset(data) if isinstance(data, pd.DataFrame) else data

        missing = {"sold", "prepared"} - set(ds.data_vars)
        if missing:
            raise ValueError(
                f"Dataset must have variables: {', '.join(sorted(missing))}. "
                f"Missing: {', '.join(sorted(missing))}"
            )

        self.model = self._build_graph(ds, observed=True)
        self._training_ds = ds
        return self

    def _build_graph(
        self,
        ds: xr.Dataset,
        *,
        observed: bool,
    ) -> pm.Model:
        """Assemble a PyMC model over ``ds``'s rows.

        Every predictive path funnels through here so the variable names
        match the fitted posterior, which is what lets
        ``pm.sample_posterior_predictive`` refill them.

        When ``observed`` is false the likelihood is left unobserved rather
        than filled with NaN: ``pymc.dims`` derives the response dtype from
        the distribution, so an integer distribution such as NegativeBinomial
        rejects a float NaN placeholder.
        """
        n_rows = ds.sizes["obs"]
        coords = collect_coords(*self.terms, ds=ds)
        coords["obs"] = np.arange(n_rows)
        if COVARIATE_DIM in ds.coords:
            coords[COVARIATE_DIM] = list(ds[COVARIATE_DIM].values)

        with pm.Model(coords=coords) as model:
            mu = self._build_mu(ds)
            if observed:
                self.likelihood.upper = ds["prepared"]
                self.likelihood.create_likelihood_variable(
                    "demand", mu=mu, observed=ds["sold"], xdist=True
                )
            else:
                self._create_unobserved_likelihood(mu)
        return model

    def _build_mu(self, ds: xr.Dataset) -> pm.TensorVariable:
        """Register the design on the composed terms and return the rate ``mu``.

        The terms sum to a linear predictor and a ``Transform`` wrapper maps it
        onto the demand scale. The link is a wrapper around the terms rather
        than part of any one of them, which is what lets the same terms be
        composed and re-used without each knowing about it.
        """
        linear_predictor = Sum(list(self.terms)) if self.terms else 0
        register_data(linear_predictor, ds=ds)
        return build_param(Transform(linear_predictor, ptx_math.exp))

    def _create_unobserved_likelihood(self, mu: pm.TensorVariable) -> None:
        """Create an unobserved ``demand`` variable, to be sampled rather than fit.

        ``create_likelihood_variable`` insists on observations, so the
        uncensored distribution is copied and its ``mu`` injected directly,
        mirroring how ``create_likelihood_variable`` builds its own.
        """
        dist = self._inner_likelihood().deepcopy()
        dist.parameters.pop("mu", None)
        dist.parameters["mu"] = mu
        dist.create_variable("demand", xdist=True)

    def _inner_likelihood(self) -> Prior:
        """The uncensored distribution behind a possibly-wrapped likelihood."""
        return getattr(self.likelihood, "distribution", self.likelihood)

    # ------------------------------------------------------------------ fit

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
            draws: Posterior draws per chain.
            tune: Warm-up steps per chain.
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
        self._stamp_attrs(self.idata)
        return self.idata

    # ----------------------------------------------------------- prediction

    def sample_posterior_predictive(
        self,
        price: float | None = None,
        random_seed: int = 42,
    ) -> xr.DataArray:
        """Draw posterior predictive demand per observation.

        Args:
            price: Counterfactual uniform price. When given, the log-price
                design column is replaced before sampling.
            random_seed: Random seed.

        Returns:
            xr.DataArray with dims ``(sample, obs)``.
        """
        ds = self._require_training_ds()
        design = ds if price is None else _at_price(ds, price)
        model = self._build_graph(design, observed=False)
        return self._draw(model, random_seed)

    def sample_product_predictive(
        self,
        price: float | None = None,
        random_seed: int = 42,
    ) -> xr.DataArray:
        """Draw posterior predictive demand per product level.

        Builds the same term composition over a dataset whose rows *are* the
        product levels, so the group gather becomes the identity and the
        result carries one column per product — the resolution the newsvendor
        decision math needs. Censoring is not applied here, since a product
        level has no single prepared quantity.

        Args:
            price: Counterfactual uniform price. When given, the log-price
                design column is replaced before sampling.
            random_seed: Random seed.

        Returns:
            xr.DataArray with dims ``(sample, product)``.
        """
        ds = self._require_training_ds()
        design = self._product_dataset(ds, price=price)
        model = self._build_product_graph(design)
        return self._draw(model, random_seed)

    def _draw(self, model: pm.Model, random_seed: int) -> xr.DataArray:
        """Sample ``demand`` from ``model`` using the fitted posterior."""
        if self.idata is None:
            raise RuntimeError(
                "No posterior samples available. Call fit() or from_netcdf() first."
            )

        ppd_idata = pm.sample_posterior_predictive(
            self.idata,
            model=model,
            var_names=["demand"],
            random_seed=random_seed,
        )
        da = ppd_idata.posterior_predictive["demand"]
        skip = ("chain", "draw")
        target = [d for d in da.dims if d not in skip][0]
        return da.stack(sample=skip).transpose("sample", target)

    def _product_dataset(
        self,
        ds: xr.Dataset,
        *,
        price: float | None = None,
    ) -> xr.Dataset:
        """Collapse ``ds`` to one row per product level.

        Args:
            ds: Training dataset.
            price: Counterfactual uniform price. When ``None``, each product
                keeps its own mean log price.
        """
        labels = unique_labels(ds[self.group])
        n = len(labels)

        data_vars: dict[str, Any] = {self.group: (self.group, labels)}
        if LOG_PRICE_VAR in ds:
            if price is None:
                values = self._mean_log_price_by_group(ds, labels)
            else:
                values = np.full((n, 1), np.log(price), dtype=float)
            data_vars[LOG_PRICE_VAR] = ((self.group, COVARIATE_DIM), values)

        out = xr.Dataset(data_vars, coords={COVARIATE_DIM: [LOG_PRICE_VAR]})
        # The group column must stay a data variable for the terms to read,
        # so the level labels are attached to its dim afterwards.
        return out.assign_coords({self.group: labels})

    def _mean_log_price_by_group(
        self,
        ds: xr.Dataset,
        labels: list[Any],
    ) -> np.ndarray:
        """Average each product's log price over the observations it appears in."""
        values = np.asarray(ds[LOG_PRICE_VAR].values, dtype=float).reshape(-1)
        assignments = np.asarray(ds[self.group].values).tolist()
        out = np.empty(len(labels), dtype=float)
        for i, label in enumerate(labels):
            rows = [v for v, a in zip(values, assignments, strict=True) if a == label]
            out[i] = float(np.mean(rows)) if rows else 0.0
        return out.reshape(-1, 1)

    def _build_product_graph(self, ds: xr.Dataset) -> pm.Model:
        """Assemble a PyMC model whose rows are product levels.

        ``build_param`` is not idempotent, so this runs exactly once per model.
        """
        coords = collect_coords(*self.terms, ds=ds)
        coords[self.group] = list(ds[self.group].values)
        if COVARIATE_DIM in ds.coords:
            coords[COVARIATE_DIM] = list(ds[COVARIATE_DIM].values)

        with pm.Model(coords=coords) as model:
            mu = self._build_mu(ds)
            self._create_unobserved_likelihood(mu)
        return model

    def predict_demand_at_price(
        self,
        price: float,
        random_seed: int = 42,
    ) -> xr.DataArray:
        """Draw posterior predictive demand per product at a uniform price.

        Falls back to the product-level predictive when no log-price term is
        configured.
        """
        if not self._has_price_term():
            return self.sample_product_predictive(random_seed=random_seed)
        return self.sample_product_predictive(price=price, random_seed=random_seed)

    def _require_training_ds(self) -> xr.Dataset:
        if self._training_ds is None:
            raise RuntimeError("No training data. Call build() first.")
        return self._training_ds

    def _has_price_term(self) -> bool:
        """Whether any configured term contributes a log-price design column."""
        return any(
            isinstance(t, Dot) and t.var_name == LOG_PRICE_VAR
            for t in collect_terms(self.terms)
        )

    # ------------------------------------------------------------- storage

    @property
    def product_names(self) -> list[str] | None:
        """Product labels, available after ``build()`` or a load from netCDF."""
        if self.idata is not None:
            raw = self.idata.attrs.get("product_names")
            if raw:
                return json.loads(raw)
        if self._training_ds is not None:
            return unique_labels(self._training_ds[self.group])
        return None

    def _stamp_attrs(self, idata: xr.DataTree) -> None:
        """Record product labels on the trace so they survive a save/load."""
        if self._training_ds is not None:
            labels = unique_labels(self._training_ds[self.group])
            idata.attrs["product_names"] = json.dumps(labels)

    def _design_for_storage(self) -> xr.Dataset | None:
        """The predictor columns needed to rebuild a graph after a reload.

        The likelihood columns are dropped: prediction never reads them, and a
        saved model should not carry the raw sellout counts. Label columns are
        cast to plain strings because netCDF cannot encode pandas' nullable
        string dtype.
        """
        if self._training_ds is None:
            return None
        design = self._training_ds.drop_vars(
            [c for c in LIKELIHOOD_COLUMNS if c in self._training_ds],
        )
        for name, array in design.variables.items():
            if not pd.api.types.is_numeric_dtype(array.dtype):
                design[name] = array.astype(str)
        return design

    def to_netcdf(self, path: str | Path, engine: str | None = None) -> None:
        """Save the posterior, its design matrix, and its terms to a netCDF file.

        The design travels with the trace so a reloaded model can still build
        predictive graphs, and the term list is serialized into the trace
        attributes so the graph is rebuilt without the caller re-supplying it.
        """
        if self.idata is None:
            raise RuntimeError("No posterior to save. Call fit() first.")
        tree = self.idata.copy()
        self._stamp_terms(tree)
        design = self._design_for_storage()
        if design is not None:
            tree[DESIGN_GROUP] = xr.DataTree(design)
        tree.to_netcdf(str(path), engine=engine)

    def _stamp_terms(self, idata: xr.DataTree) -> None:
        """Serialize the term list into the trace attributes as JSON."""
        idata.attrs[TERMS_ATTR] = json.dumps(
            [_serialization.serialize(t) for t in self.terms],
        )

    @classmethod
    def _resolve_terms(
        cls,
        terms: list[Any] | None,
        idata: xr.DataTree,
    ) -> list[Any]:
        """Return the caller's terms, or rebuild them from the saved trace."""
        if terms is not None:
            return list(terms)
        raw = idata.attrs.get(TERMS_ATTR)
        if not raw:
            raise ValueError(
                "No terms given and none found on the trace. Pass terms=... "
                "explicitly when loading a model that was not saved by this "
                "class, or refit it so the terms are recorded."
            )
        return [_serialization.deserialize(blob) for blob in json.loads(raw)]

    @classmethod
    def from_idata(
        cls,
        idata: xr.DataTree,
        *,
        terms: list[Any] | None = None,
        likelihood: Prior = DEFAULT_LIKELIHOOD,
        group: str = "product",
    ) -> DemandModel:
        """Create a DemandModel from an xarray DataTree (no model graph).

        Suitable for the predictive methods, not ``fit()``. ``terms`` defaults
        to the term list serialized into the trace by ``to_netcdf``, so a
        round-tripped model needs no arguments. Any saved design group is picked
        up to keep prediction working.
        """
        resolved = cls._resolve_terms(terms, idata)
        inst = cls(terms=resolved, likelihood=likelihood, group=group)
        inst.idata = idata
        if DESIGN_GROUP in getattr(idata, "children", {}):
            inst._training_ds = idata[DESIGN_GROUP].ds
        return inst

    @classmethod
    def from_netcdf(
        cls,
        path: str | Path,
        *,
        terms: list[Any] | None = None,
        likelihood: Prior = DEFAULT_LIKELIHOOD,
        group: str = "product",
    ) -> DemandModel:
        """Load a saved model from a netCDF file.

        ``terms`` defaults to the ones stored in the file by ``to_netcdf``.
        """
        idata = az.from_netcdf(str(path))
        return cls.from_idata(idata, terms=terms, likelihood=likelihood, group=group)
