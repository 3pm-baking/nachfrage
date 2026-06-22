"""nachfrage — Bayesian demand modeling and newsvendor inventory optimization."""

from nachfrage.analysis import format_results_table, format_scenarios
from nachfrage.decision import optimal_quantity, profit_profile, waste_sensitivity
from nachfrage.models import DEFAULT_MODEL_CONFIG, DemandModel
from nachfrage.posterior import compute_ppd

__version__ = "0.1.0"

__all__ = [
    "DemandModel",
    "DEFAULT_MODEL_CONFIG",
    "compute_ppd",
    "format_results_table",
    "format_scenarios",
    "optimal_quantity",
    "profit_profile",
    "waste_sensitivity",
]
