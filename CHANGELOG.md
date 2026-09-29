# Changelog

All notable changes to `nachfrage` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-09-29

### Changed

- **Migrated the linear predictor to `pymc_marketing.terms`.** The bespoke
  `ModelTerm` ABC and two of its three subclasses are gone: `InterceptTerm` is
  replaced by upstream `Intercept`, and `LinearCovariate` by upstream `Dot`.
  Only `GroupContribution` — the one term with no upstream equivalent — remains
  local, in `nachfrage.terms`.
- **Restored the `exp` link on the demand rate.** The earlier `terms.py` refactor
  dropped it silently, so `mu` was a linear predictor rather than a rate. Terms
  are now summed and wrapped once in `Transform(..., ptx_math.exp)`, giving
  `mu = exp(intercept + product_offset + beta_log_price * log_price)`.
- `DemandModel` now takes `terms` as a positional/required keyword and an
  optional `group` (default `"product"`).
- `sample_posterior_predictive()` returns draws per **observation**
  (dims `(sample, obs)`) with sellout censoring retained. Per-product draws
  moved to the new `sample_product_predictive()` (dims `(sample, product)`,
  uncensored).
- `from_idata()` / `from_netcdf()` accept a `group` and pick up a saved
  `design` group so a reloaded model can still build predictive graphs.
- `predict_demand_at_price()` now delegates to the product-level predictive.

### Added

- `sample_product_predictive()` — posterior predictive demand per group level,
  the resolution the newsvendor decision math consumes.
- `nachfrage.terms.default_terms(with_category=False, with_price=True)` — the
  standard term list, so consumers share one definition.
- Prediction at a counterfactual uniform price via
  `sample_product_predictive(price=...)`.
- Design persistence: `to_netcdf()` stores the predictor columns (excluding the
  likelihood counts) alongside the posterior, so a reloaded model is
  self-contained for prediction.

### Fixed

- Unobserved predictive graph: the likelihood is created unobserved rather than
  filled with a float-NaN placeholder, which integer distributions such as
  NegativeBinomial reject.
- Saved models could no longer predict after a load — the training design is now
  persisted.
- Product label columns are coerced to plain strings for netCDF, which cannot
  encode pandas' nullable string dtype.

### Breaking

- `InterceptTerm` and `LinearCovariate` are removed (use `Intercept` / `Dot`).
- `DemandModel()` requires `terms`.
- `sample_posterior_predictive()` return shape changed from `(sample, product)`
  to `(sample, obs)`; use `sample_product_predictive()` for the old shape.

[0.2.0]: https://github.com/3pm-baking/nachfrage/releases/tag/v0.2.0
