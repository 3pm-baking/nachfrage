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
model = DemandModel()
model.build(df)
model.fit(draws=1000, tune=1000, chains=4, random_seed=42)

# --- Posterior predictive ---
ppd = model.sample_posterior_predictive(n_samples=10000)
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
loaded = DemandModel.from_netcdf("posterior.nc")
ppd_reloaded = loaded.sample_posterior_predictive(n_samples=10000)

# --- Plotting (delegate to arviz_plots) ---
import arviz_plots as azp
import matplotlib.pyplot as plt
import xarray as xr

dt = xr.DataTree({"demand": ppd})
azp.plot_forest(dt, var_names=["demand"])
plt.savefig("forest.png", dpi=150, bbox_inches="tight")
```

## Model

The default model is a **hierarchical NegativeBinomial** with right-censoring:

```
demand ~ Censored(NegativeBinomial(mu=mu_product, alpha), upper=prepared)
mu_product = exp(mu_global + mu_product_raw * sigma_product)
```

- **Censoring**: when a product sells out (`sold >= prepared`), we only know demand ≥ prepared
- **Hierarchy**: product-level random effects via non-centered parameterization
- **Overdispersion**: NegativeBinomial handles variance > mean

### Custom priors

```python
from pymc_extras.prior import Prior, Censored

model = DemandModel(model_config={
    "likelihood": Censored(Prior("NegativeBinomial",
        alpha=Prior("HalfNormal", sigma=3.0))),
    "mu_global": Prior("Normal", mu=np.log(15), sigma=0.3),
    "sigma_product": Prior("HalfNormal", sigma=0.3),
    "mu_product_raw": Prior("Normal", sigma=1.0, dims="product"),
})
```

## API

| Module | Key exports | Purpose |
|--------|------------|---------|
| `nachfrage.models` | `DemandModel` | Build, fit, predict, save/load |
| `nachfrage.decision` | `optimal_quantity`, `profit_profile`, `waste_sensitivity` | Newsvendor optimization |
| `nachfrage.posterior` | `compute_ppd` | Standalone PPD computation |
| `nachfrage.analysis` | `format_scenarios`, `format_results_table` | Text tables |
| `nachfrage.plot` | `plot_sellout_curves`, `plot_calibration`, `plot_profit_curves`, ... | Application-specific matplotlib figures |

### DemandModel lifecycle

```python
model = DemandModel(model_config={...})
model.build(df)  # df has columns: sold, prepared, product
model.fit(draws=1000, tune=1000, chains=4)
ppd = model.sample_posterior_predictive(n_samples=10000)
model.to_netcdf("posterior.nc")      # save
loaded = DemandModel.from_netcdf("posterior.nc")  # load
loaded = DemandModel.from_idata(idata)            # from in-memory DataTree
```

## License

MIT
