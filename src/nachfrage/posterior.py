"""Posterior predictive computation and calibration.

Pure NumPy functions — no file I/O, no PyMC imports.
"""

from __future__ import annotations

import numpy as np


def compute_ppd(
    mu: np.ndarray,
    alpha: np.ndarray,
    n_samples: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Compute posterior predictive samples from mu and alpha draws.

    Samples from NegativeBinomial(mu, alpha) using the standard
    (n, p) parameterization, where:
        n = alpha
        p = alpha / (alpha + mu)

    Args:
        mu: Array of shape (n_draws, n_products) of posterior mu samples.
        alpha: Array of shape (n_draws,) of posterior alpha samples.
        n_samples: Number of PPD samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        PPD array of shape (n_samples, n_products) with integer values.
    """
    rng = np.random.default_rng(seed)
    n_products = mu.shape[1]
    mu_flat = np.asarray(mu).reshape(-1, n_products)
    alpha_flat = np.asarray(alpha).reshape(-1)

    idx = rng.integers(0, len(alpha_flat), size=n_samples)
    alpha_s = alpha_flat[idx, None]
    p = alpha_s / (alpha_s + mu_flat[idx])
    ppd = rng.negative_binomial(alpha_s, p)

    return ppd
