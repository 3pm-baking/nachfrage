"""Tidy posterior summaries over group-level grids.

One-call summary path for the most common question after fitting: *how much
demand does each group level see, with how much uncertainty, and how much of
it did we actually observe?* Replaces the hand-rolled "rebuild mu, call
compute_ppd per cell, reshape, join counts back" pipeline that consumers
otherwise need.

Demand statistics come from the same uncensored posterior predictive as
:meth:`nachfrage.models.DemandModel.sample_group_predictive` — per reference
window, per group level. Observation and censoring counts come from the
training data, so thin cells are visible as cells rather than buried in the
posterior.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from nachfrage.models import DemandModel

DEFAULT_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def summarize_demand(
    model: DemandModel,
    conditions: Mapping[str, object] | None = None,
    *,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Summarize uncensored posterior demand per group-level combination.

    Args:
        model: A fitted :class:`nachfrage.models.DemandModel`. Its group
            terms define the grid (e.g. product x market for a model with
            those two group terms).
        conditions: Optional per-group conditioning, passed through to
            ``sample_group_predictive``; restricts the summary to one level
            of that group while sweeping the rest.
        quantiles: Posterior predictive quantiles to report, plus the mean.

    Returns:
        DataFrame with one row per group-level combination and columns

        - one column per group dimension (e.g. ``product``, ``market``), in
          term order
        - ``demand_mean`` and ``q<quantile>`` columns (as ``q05``, ``q95``, …)
          from the posterior predictive draws, per reference window
        - ``n_obs``: training observations at this grid cell
        - ``n_censored``: training observations where ``sold >= prepared``
          (sellouts — demand at or above what was made); ``None`` when the
          model's training data predates a netCDF round-trip, which drops
          likelihood columns by design
    """
    ppd = model.sample_group_predictive(conditions=conditions)
    group_dims = [d for d in ppd.dims if d != "sample"]
    values = np.asarray(ppd.transpose("sample", *group_dims).values)

    flat = values.reshape(values.shape[0], -1)
    out: dict[str, object] = {}
    for i, dim in enumerate(group_dims):
        coords = ppd.coords[dim].values
        # C-order cell addressing: this dim repeats every level over the
        # trailing dims; leading dims cycle it whole-block per level.
        tail = int(np.prod([ppd.sizes[d] for d in group_dims[i + 1 :]]))
        head = int(np.prod([ppd.sizes[d] for d in group_dims[:i]]))
        out[dim] = np.tile(np.repeat(coords, tail), head)

    out["demand_mean"] = flat.mean(axis=0)
    for q in quantiles:
        out[f"q{int(q * 100):02d}"] = np.quantile(flat, q, axis=0)

    df = pd.DataFrame(out)

    counts = _observed_counts(model, group_dims)
    keys = _cell_keys(df, group_dims)
    if counts is not None and keys is not None:
        df["n_obs"] = [counts.get(k, {}).get("n", 0) for k in keys]
        df["n_censored"] = pd.array(
            [counts.get(k, {}).get("censored", 0) for k in keys], dtype="Int64"
        )
    else:
        df["n_obs"] = 0
        df["n_censored"] = pd.array([None] * len(df), dtype="Int64")

    return df


def _cell_keys(df: pd.DataFrame, group_dims: list[str]) -> list[tuple] | None:
    keys = list(zip(*(df[dim].tolist() for dim in group_dims), strict=True))
    return keys or None


def _observed_counts(
    model: DemandModel, group_dims: list[str]
) -> dict[tuple, dict[str, int | None]] | None:
    """Count observations and censored sellouts per grid key from training data.

    ``n_censored`` is ``None`` when the training dataset does not carry the
    likelihood columns (``to_netcdf`` drops them by design), rather than 0 —
    an unknown count must not look like "never sold out".
    """
    ds = model._training_ds  # noqa: SLF001 - summary is a model companion API
    if ds is None:
        return None
    censored_known = "sold" in ds and "prepared" in ds

    keys: list[tuple] = [() for _ in range(ds.sizes.get("obs", 0))]
    for dim in group_dims:
        values = np.asarray(ds[dim].values).tolist()
        keys = [k + (v,) for k, v in zip(keys, values, strict=True)]

    if censored_known:
        censored = np.asarray(ds["sold"].values) >= np.asarray(ds["prepared"].values)
    else:
        censored = np.full(len(keys), None, dtype=object)

    counts: dict[tuple, dict[str, int | None]] = {}
    for key, is_censored in zip(keys, censored, strict=True):
        entry = counts.setdefault(key, {"n": 0, "censored": None})
        entry["n"] += 1
        if is_censored is not None:
            entry["censored"] = (entry["censored"] or 0) + int(is_censored)
    return counts
