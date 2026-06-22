"""Tests for nachfrage.posterior — PPD computation and calibration."""

import numpy as np
import pytest


class TestComputePPD:
    """Tests for compute_ppd()."""

    @pytest.fixture
    def mu_samples(self, rng):
        """100 draws × 3 products, log-scale means."""
        return rng.lognormal(mean=np.log(10), sigma=0.3, size=(100, 3))

    @pytest.fixture
    def alpha_samples(self, rng):
        """100 draws of overdispersion parameter."""
        return rng.lognormal(mean=np.log(5), sigma=0.3, size=100)

    def test_correct_shape(self, mu_samples, alpha_samples):
        """PPD has shape (n_samples, n_products)."""
        from nachfrage.posterior import compute_ppd

        ppd = compute_ppd(mu_samples, alpha_samples, n_samples=1000)

        assert ppd.shape == (1000, 3)

    def test_reproducible_with_seed(self, mu_samples, alpha_samples):
        """Same seed produces identical results."""
        from nachfrage.posterior import compute_ppd

        ppd1 = compute_ppd(mu_samples, alpha_samples, n_samples=500, seed=42)
        ppd2 = compute_ppd(mu_samples, alpha_samples, n_samples=500, seed=42)

        np.testing.assert_array_equal(ppd1, ppd2)

    def test_different_seeds_produce_different_results(self, mu_samples, alpha_samples):
        """Different seeds produce different draws."""
        from nachfrage.posterior import compute_ppd

        ppd1 = compute_ppd(mu_samples, alpha_samples, n_samples=500, seed=0)
        ppd2 = compute_ppd(mu_samples, alpha_samples, n_samples=500, seed=1)

        assert not np.array_equal(ppd1, ppd2)

    def test_non_negative_values(self, mu_samples, alpha_samples):
        """NegativeBinomial produces only non-negative values."""
        from nachfrage.posterior import compute_ppd

        ppd = compute_ppd(mu_samples, alpha_samples, n_samples=1000)

        assert np.all(ppd >= 0)
        # Should be integers or castable to int
        assert np.allclose(ppd, np.round(ppd))

    def test_default_seed_is_42(self, mu_samples, alpha_samples):
        """Default seed is 42 (matches two explicit calls)."""
        from nachfrage.posterior import compute_ppd

        with_default = compute_ppd(mu_samples, alpha_samples, n_samples=500)
        with_explicit = compute_ppd(mu_samples, alpha_samples, n_samples=500, seed=42)

        np.testing.assert_array_equal(with_default, with_explicit)

    def test_single_product(self, mu_samples, alpha_samples):
        """Works with a single product (1D mu)."""
        from nachfrage.posterior import compute_ppd

        ppd = compute_ppd(mu_samples[:, :1], alpha_samples, n_samples=100)

        assert ppd.shape == (100, 1)

    def test_mean_roughly_correct(self, mu_samples, alpha_samples):
        """PPD mean is close to posterior mu mean for moderate alpha."""
        from nachfrage.posterior import compute_ppd

        ppd = compute_ppd(mu_samples, alpha_samples, n_samples=20000)

        mu_mean = mu_samples.mean(axis=0)
        ppd_mean = ppd.mean(axis=0)

        # NegativeBinomial mean = mu, should be close
        np.testing.assert_allclose(ppd_mean, mu_mean, rtol=0.15)
