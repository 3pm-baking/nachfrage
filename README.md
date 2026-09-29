# nachfrage

Bayesian demand modeling and newsvendor inventory optimization for small-batch producers.

`nachfrage` (German for "demand") pairs with [`wright`](https://github.com/3pm-baking/wright) (supply/costing) to cover both sides of the production planning equation.

## Install

```bash
pip install nachfrage
# or with plotting:
pip install nachfrage[plot]
```

## Quick example

```python
import numpy as np
import pandas as pd
from nachfrage import DemandModel, optimal_quantity
from nachfrage.terms import default_terms

# --- Fake demand data: 3 products, 10 market days each ---
rng = np.random.default_rng(42)
n_products = 3
n_per = 10
true_mu = np.array([12.0, 8.0, 5.0])
alpha = 5.0
product_names = ["Cheese Cake (slice)", "Apple Strudel (piece)", "Bienenstich (slice)"]

demand = rng.negative_binomial(alpha, alpha / (alpha + true_mu),
                               size=(n_per, n_products))
prepared = np.ceil(demand * 1.2).astype(int)
sold = demand.T.ravel()
prepared = prepared.T.ravel()
censored = (sold >= prepared).astype(bool)
sold[censored] = prepared[censored]

df = pd.DataFrame({
    "sold": sold,
    "prepared": prepared,
    "product": np.repeat(product_names, n_per),
})
print(df.head())
#    sold  prepared                product
# 0  12.0      15.0  Cheese Cake (slice)
# 1   9.0      12.0  Cheese Cake (slice)
# 2  13.0      16.0  Cheese Cake (slice)
# 3   9.0      16.0  Cheese Cake (slice)
# 4   7.0      12.0  Cheese Cake (slice)

# --- Build and fit the model ---
model = DemandModel(terms=default_terms(with_price=False))
model.build(df)
model.fit(draws=1000, tune=1000, chains=4, random_seed=42)

# --- Posterior predictive, one column per product ---
ppd = model.sample_product_predictive()
# ppd is an xr.DataArray with dims (sample, product)
print(f"Shape: {ppd.sizes}")
print(ppd.coords["product"].values)

# --- Newsvendor optimization ---
pid = 0  # Cheese Cake
r = optimal_quantity(
    ppd.values[:, pid], price=5.0, unit_cost=2.0, batch_size=1,
)
print(f"Optimal prep: {r.best_q}, expected profit: ${r.profit:.2f}")

# --- Save and reload ---
model.to_netcdf("posterior.nc")
loaded = DemandModel.from_netcdf(
    "posterior.nc", terms=default_terms(with_price=False)
)
ppd_reloaded = loaded.sample_product_predictive()

# --- Plotting (delegate to arviz_plots) ---
import arviz_plots as azp
import matplotlib.pyplot as plt
import xarray as xr

dt = xr.DataTree()
dt["demand"] = xr.DataTree(ppd.to_dataset(name="demand"))
azp.plot_forest(dt, group="demand", var_names=["demand"], sample_dims=["sample"])
plt.savefig("forest.png", dpi=150, bbox_inches="tight")
```

## Model

The default model is a **hierarchical NegativeBinomial** with right-censoring.
The linear predictor is a sum of `pymc_marketing.terms` pieces, wrapped in a
single `exp` link:

```
demand ~ Censored(NegativeBinomial(mu=mu, alpha), upper=prepared)
mu     = exp(intercept + product_offset + beta_log_price @ log_price)
```

- **Censoring**: when a product sells out (`sold >= prepared`), we only know demand ≥ prepared
- **Hierarchy**: product-level random effects via non-centered parameterization
- **Overdispersion**: NegativeBinomial handles variance > mean

### Custom priors

Terms are ordinary `pymc_marketing.terms` objects, so any of them can be
swapped or extended. `nachfrage.terms.GroupContribution` adds the one piece
upstream does not ship: a non-centered gather of per-group offsets.

```python
import numpy as np
from pymc_extras.prior import Prior
from pymc_marketing.terms import Dot, Intercept
from nachfrage.terms import GroupContribution

terms = [
    Intercept(prior=Prior("Normal", mu=np.log(15), sigma=0.3)),
    GroupContribution(
        data_source="product",
        prior=Prior("Normal", mu=0, sigma=Prior("HalfNormal", sigma=0.3),
                    dims="product"),
    ),
    Dot(
        var_name="log_price",
        name="beta_log_price",
        prior=Prior("Normal", mu=-0.7, sigma=0.5, dims="feature"),
    ),
]
model = DemandModel(terms=terms)
```

## API

| Module | Key exports | Purpose |
|--------|------------|---------|
| `nachfrage.models` | `DemandModel` | Build, fit, predict, save/load |
| `nachfrage.terms` | `GroupContribution`, `default_terms` | Term composition on top of `pymc_marketing.terms` |
| `nachfrage.decision` | `optimal_quantity`, `profit_profile`, `waste_sensitivity` | Newsvendor optimization |
| `nachfrage.posterior` | `compute_ppd` | Standalone PPD computation |
| `nachfrage.analysis` | `format_scenarios`, `format_results_table` | Text tables |
| `nachfrage.plot` | `plot_sellout_curves`, `plot_calibration`, `plot_profit_curves`, ... | Application-specific matplotlib figures |

### DemandModel lifecycle

```python
model = DemandModel(terms=default_terms())
model.build(df)  # df has columns: sold, prepared, product
model.fit(draws=1000, tune=1000, chains=4)
ppd = model.sample_product_predictive()   # dims (sample, product)
obs = model.sample_posterior_predictive() # dims (sample, obs), censoring kept
model.to_netcdf("posterior.nc")           # save
loaded = DemandModel.from_netcdf("posterior.nc", terms=default_terms())  # load
loaded = DemandModel.from_idata(idata, terms=default_terms())            # in-memory
```

## License

MIT
