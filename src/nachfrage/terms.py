"""Additive model terms for constructing linear predictors.

Each term is a building block that registers its own data nodes and creates
its own PyMC variables for the linear predictor of a demand model::

    terms = [
        InterceptTerm(prior=Prior("Normal", mu=np.log(12), sigma=0.5)),
        GroupContribution(
            data_source="product",
            prior=Prior("Normal", mu=0, sigma=Prior("HalfNormal", sigma=0.5),
                        dims="product"),
        ),
        LinearCovariate(
            data_source="log_price",
            prior=Prior("Normal", mu=-0.7, sigma=0.5),
        ),
    ]
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr
from pymc_extras.prior import Prior

IndexArray = np.ndarray | None


class ModelTerm(ABC):
    """An additive term in a demand model's linear predictor.

    Each term follows a two-phase lifecycle:

    1. **setup** — register coordinates and ``pm.Data`` nodes into the model.
       Called for all terms first so coords and data exist before variables
       are created.
    2. **get_contribution** — create prior variables and return a tensor that
       is added to the linear predictor.
    """

    @abstractmethod
    def setup(self, model: pm.Model, ds: xr.Dataset) -> None:
        """Register coords and data nodes into the model."""

    @abstractmethod
    def get_contribution(self, model: pm.Model) -> pm.TensorVariable:
        """Create prior variables and return the linear predictor contribution."""


class InterceptTerm(ModelTerm):
    """A scalar intercept (global baseline).

    Creates a single ``pm.Normal`` variable named ``"intercept"``.
    """

    def __init__(self, prior: Prior) -> None:
        self.prior = prior

    def setup(self, model: pm.Model, ds: xr.Dataset) -> None:
        pass

    def get_contribution(self, model: pm.Model) -> pm.TensorVariable:
        return self.prior.create_variable("intercept")


class GroupContribution(ModelTerm):
    """Hierarchical group-level offsets.

    Reads per-observation group assignments from ``ds[data_source]``,
    adds a dimension to the model coords, and creates a non-centered
    hierarchical intercept for the group.

    The ``prior`` should be a hierarchical prior where the scale parameter
    governs between-group variability::

        GroupContribution(
            data_source="category",
            prior=Prior("Normal", mu=0, sigma=Prior("HalfNormal", sigma=0.5),
                        dims="category"),
        )

    The scale prior (the ``sigma`` parameter of the Normal) is extracted
    and used for the non-centered parameterization. The term creates::

        {data_source}_raw      -- Normal(0, 1) raw effects (non-centered)
        {data_source}_sigma    -- scale (from prior's sigma parameter)
        {data_source}_offset   -- Deterministic: raw * sigma
    """

    def __init__(self, data_source: str, prior: Prior) -> None:
        self.data_source = data_source
        self.prior = prior
        self._index: IndexArray = None

    def setup(self, model: pm.Model, ds: xr.Dataset) -> None:
        assignments = ds[self.data_source].values
        unique = pd.unique(assignments)
        self._n_groups = len(unique)
        label_to_idx = {label: i for i, label in enumerate(unique)}
        self._index = np.array([label_to_idx[v] for v in assignments], dtype=int)
        if self.data_source not in model.coords:
            model.add_coord(self.data_source, list(unique))

    def get_contribution(self, model: pm.Model) -> pm.TensorVariable:
        raw = Prior("Normal", sigma=1.0, dims=self.data_source).create_variable(
            f"{self.data_source}_raw"
        )
        scale_prior = self.prior.parameters.get("sigma")
        if scale_prior is not None and isinstance(scale_prior, Prior):
            sigma = scale_prior.create_variable(f"{self.data_source}_sigma")
        else:
            sigma = self.prior.create_variable(f"{self.data_source}_sigma")
        offset = pm.Deterministic(
            f"{self.data_source}_offset",
            raw * sigma,
            dims=self.data_source,
        )
        return offset[self._index]


class LinearCovariate(ModelTerm):
    """Linear effect of a continuous covariate.

    Creates a scalar coefficient and multiplies it by the covariate values::

        contribution = beta * covariate

    The covariate values are registered as a ``pm.Data`` node named
    ``covariate_{data_source}``.
    """

    def __init__(self, data_source: str, prior: Prior) -> None:
        self.data_source = data_source
        self.prior = prior
        self._dim_labels: tuple[str, ...] = ("obs",)

    def setup(self, model: pm.Model, ds: xr.Dataset) -> None:
        values = ds[self.data_source].values
        self._dim_labels = self._infer_dim_labels(ds)
        pm.Data(f"covariate_{self.data_source}", values, dims=self._dim_labels)

    def get_contribution(self, model: pm.Model) -> pm.TensorVariable:
        beta = self.prior.create_variable(f"beta_{self.data_source}")
        data = model[f"covariate_{self.data_source}"]
        return beta * data

    def _infer_dim_labels(self, ds: xr.Dataset) -> tuple[str, ...]:
        if self.data_source in ds.dims:
            return (self.data_source,)
        if self.data_source in ds.data_vars:
            return ds[self.data_source].dims or ("obs",)
        return ("obs",)
