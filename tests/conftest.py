"""Shared test fixtures for nachfrage."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded random number generator."""
    return np.random.default_rng(42)


@pytest.fixture
def deterministic_ppd(rng):
    """PPD where all 1000 draws equal fixed values for 3 products."""
    # 3 products: product 0 = 10, product 1 = 5, product 2 = 0
    ppd = np.zeros((1000, 3), dtype=float)
    ppd[:, 0] = 10.0
    ppd[:, 1] = 5.0
    ppd[:, 2] = 0.0
    return ppd


@pytest.fixture
def random_ppd(rng):
    """Random PPD: 10000 draws × 4 products from NegativeBinomial(20, ...)."""
    return rng.negative_binomial(5, 0.2, size=(10000, 4))


@pytest.fixture
def sample_model_config():
    """Model config with default priors. Avoids PyMC import at module level."""
    return None  # populated in test_models.py when PyMC is available
