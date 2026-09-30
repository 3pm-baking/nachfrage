"""Additive model terms for constructing linear predictors.

Built on :mod:`pymc_marketing.terms`, which supplies the pieces this package
already needs: ``Intercept``, ``Parameter``, ``Dot``, ``Transform``, the
``+`` / ``*`` / ``-`` composition operators, and the ``collect_coords`` /
``register_data`` / ``build_param`` / ``set_data`` drivers.

Upstream deliberately does not ship a hierarchical group term, so that is all
this module adds — a non-centered gather of per-group offsets onto
observations::

    import numpy as np
    from pymc_extras.prior import Prior
    from pymc_marketing.terms import Dot, Intercept
    from nachfrage.terms import GroupContribution

    terms = [
        Intercept(prior=Prior("Normal", mu=np.log(12), sigma=0.5)),
        GroupContribution(
            data_source="product",
            prior=Prior("Normal", mu=0, sigma=Prior("HalfNormal", sigma=0.5),
                        dims="product"),
        ),
        Dot(
            var_name="log_price",
            name="beta_log_price",
            prior=Prior("Normal", mu=-0.7, sigma=0.5, dims="feature"),
        ),
    ]

The ``prior`` is a hierarchical prior whose ``sigma`` parameter governs
between-group variability. Following the non-centered parameterization the
term creates:

    {data_source}_raw      -- Normal(0, 1) raw effects, dims={data_source}
    {data_source}_sigma    -- between-group scale (from prior's sigma)
    {data_source}_offset   -- Deterministic: raw * sigma, dims={data_source}
    {data_source}_idx      -- pmd.Data gather index, dims=obs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pymc as pm
import pymc.dims as pmd
import xarray as xr
from pymc_extras.prior import Prior
from pymc_marketing.terms import ModelTerm
from pymc_marketing.terms import pymc_extras_deserialize as _deserialize_prior
from pymc_marketing.terms import serialization as _serialization

__all__ = ["GroupContribution", "default_terms"]


def unique_labels(values: xr.DataArray) -> list[Any]:
    """Return the distinct values of a data column, in first-seen order."""
    return list(dict.fromkeys(np.asarray(values.values).tolist()))


def default_terms(
    with_category: bool = False,
    with_price: bool = True,
) -> list[Any]:
    """The standard demand term list: intercept, group effect, log price.

    The intercept prior sits on the log scale because ``DemandModel`` sums the
    terms and applies a single ``exp`` link to the result.

    Args:
        with_category: Add a second ``GroupContribution`` over ``category``.
        with_price: Add a ``Dot`` term for the ``log_price`` design column.
    """
    from pymc_marketing.terms import Dot, Intercept

    terms: list[Any] = [
        Intercept(prior=Prior("Normal", mu=np.log(12), sigma=0.5)),
    ]

    if with_category:
        terms.append(
            GroupContribution(
                data_source="category",
                prior=Prior(
                    "Normal",
                    mu=0,
                    sigma=Prior("HalfNormal", sigma=0.5),
                    dims="category",
                ),
            ),
        )

    terms.append(
        GroupContribution(
            data_source="product",
            prior=Prior(
                "Normal",
                mu=0,
                sigma=Prior("HalfNormal", sigma=0.5),
                dims="product",
            ),
        ),
    )

    if with_price:
        terms.append(
            Dot(
                var_name="log_price",
                name="beta_log_price",
                prior=Prior("Normal", mu=-0.7, sigma=0.5, dims="feature"),
            ),
        )

    return terms


@dataclass
class GroupContribution(ModelTerm):
    """Non-centered hierarchical group offsets, gathered onto observations.

    Reads per-observation group assignments from ``ds[data_source]``, adds a
    dimension of the distinct group labels to the model coords, and creates a
    non-centered hierarchical intercept for the group.

    Args:
        data_source: Name of the dataset column holding group assignments.
        prior: Hierarchical prior. Its ``sigma`` parameter is the
            between-group scale; ``mu`` and ``dims`` are otherwise unused
            because the term supplies its own non-centered construction.
    """

    data_source: str
    prior: Prior

    @property
    def index_var(self) -> str:
        """Name of the registered gather index."""
        return f"{self.data_source}_idx"

    def get_coords(self, ds: xr.Dataset) -> dict[str, Any]:
        """Return the group's distinct labels as a model coordinate."""
        return {self.data_source: unique_labels(ds[self.data_source])}

    def register_data(self, ds: xr.Dataset) -> None:
        """Register the gather index as a ``pmd.Data`` shared variable."""
        model = pm.modelcontext(None)
        if self.index_var in model:
            return

        labels = unique_labels(ds[self.data_source])
        lookup = {label: i for i, label in enumerate(labels)}
        assignments = np.asarray(ds[self.data_source].values)
        index = np.array([lookup[v] for v in assignments.tolist()], dtype=int)

        dims = ds[self.data_source].dims
        coords = {d: ds.coords[d] for d in dims if d in ds.coords}
        pmd.Data(self.index_var, xr.DataArray(index, dims=dims, coords=coords))

    def create_variable(self) -> pm.TensorVariable:
        """Create the raw/sigma/offset variables and return the gathered offset."""
        model = pm.modelcontext(None)

        raw = pmd.Normal(f"{self.data_source}_raw", sigma=1.0, dims=self.data_source)
        sigma = self._create_sigma()
        offset = pmd.Deterministic(
            f"{self.data_source}_offset",
            raw * sigma,
            dims=self.data_source,
        )
        return offset[model[self.index_var]]

    def set_data(self, ds: xr.Dataset, model: pm.Model | None = None) -> None:
        """Update the gather index for out-of-sample prediction."""
        model = pm.modelcontext(None) if model is None else model
        if model is None or self.index_var not in model:
            return

        index = self._index_array(ds)
        coords = {d: ds.coords[d].values for d in ds.dims if d in ds.coords}
        pm.set_data({self.index_var: index}, model=model, coords=coords)

    def _create_sigma(self) -> pm.TensorVariable:
        """Create the between-group scale from the prior's ``sigma`` parameter."""
        scale_prior = self.prior.parameters.get("sigma")
        if isinstance(scale_prior, Prior):
            return scale_prior.create_variable(f"{self.data_source}_sigma", xdist=True)
        return self.prior.create_variable(f"{self.data_source}_sigma", xdist=True)

    def _index_array(self, ds: xr.Dataset) -> np.ndarray:
        """Map this dataset's group assignments onto the registered label order."""
        model = pm.modelcontext(None)
        labels = (
            list(model.coords[self.data_source].values)
            if self.data_source in model.coords
            else unique_labels(ds[self.data_source])
        )
        lookup = {label: i for i, label in enumerate(labels)}
        assignments = np.asarray(ds[self.data_source].values)
        return np.array([lookup[v] for v in assignments.tolist()], dtype=int)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the term so a saved model can rebuild it on load."""
        return {"data_source": self.data_source, "prior": self.prior.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroupContribution:
        """Reconstruct a term from its serialized form."""
        return cls(
            data_source=data["data_source"],
            prior=_deserialize_prior(data["prior"]),
        )


_serialization.register(GroupContribution)
