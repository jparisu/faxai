"""Tests for sampling distribution classes."""

import numpy as np
import pytest

from faxai.mathing.distribution.sampling_distributions import (
    DeltaDistribution,
    HistogramDistribution,
    WeightedDistribution,
)
from faxai.mathing.RandomGenerator import RandomGenerator


# --- DeltaDistribution Tests ---


def test_delta_distribution_basic_stats():
    """Test basic statistical properties of DeltaDistribution."""
    samples = np.array([1.0, 2.0, 3.0, 2.0, 2.0])
    dist = DeltaDistribution(samples)

    # Mean should be average of all samples
    assert dist.mean() == pytest.approx(2.0, abs=1e-10)

    # Mode should be the most frequent value (2.0)
    assert dist.moded() == pytest.approx(2.0, abs=1e-10)

    # Median
    assert dist.median() == pytest.approx(2.0, abs=1e-10)

    # Std
    expected_std = np.std(samples, ddof=0)
    assert dist.std(ddof=0) == pytest.approx(expected_std, abs=1e-10)


def test_delta_distribution_pdf():
    """Test PDF computation for DeltaDistribution."""
    samples = np.array([1.0, 2.0, 3.0, 2.0])
    dist = DeltaDistribution(samples)

    # PDF at sample points
    x = np.array([1.0, 2.0, 3.0, 4.0])
    pdf = dist.pdf(x)

    # At 1.0: 1/4 probability
    assert pdf[0] == pytest.approx(0.25, abs=1e-10)
    # At 2.0: 2/4 probability
    assert pdf[1] == pytest.approx(0.5, abs=1e-10)
    # At 3.0: 1/4 probability
    assert pdf[2] == pytest.approx(0.25, abs=1e-10)
    # At 4.0: 0 probability (not in samples)
    assert pdf[3] == pytest.approx(0.0, abs=1e-10)


def test_delta_distribution_cdf():
    """Test CDF computation for DeltaDistribution."""
    samples = np.array([1.0, 2.0, 3.0, 2.0])
    dist = DeltaDistribution(samples)

    x = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    cdf = dist.cdf(x)

    # CDF should be step function
    assert cdf[0] == pytest.approx(0.0, abs=1e-10)  # x < 1.0
    assert cdf[1] == pytest.approx(0.25, abs=1e-10)  # x = 1.0
    assert cdf[2] == pytest.approx(0.25, abs=1e-10)  # 1.0 < x < 2.0
    assert cdf[3] == pytest.approx(0.75, abs=1e-10)  # x = 2.0
    assert cdf[4] == pytest.approx(0.75, abs=1e-10)  # 2.0 < x < 3.0
    assert cdf[5] == pytest.approx(1.0, abs=1e-10)  # x = 3.0
    assert cdf[6] == pytest.approx(1.0, abs=1e-10)  # x > 3.0


def test_delta_distribution_maximum_pdf():
    """Test maximum PDF value."""
    samples = np.array([1.0, 2.0, 2.0, 2.0])
    dist = DeltaDistribution(samples)

    # Maximum should be at the most frequent value (2.0 appears 3/4 times)
    assert dist.maximum_pdf() == pytest.approx(0.75, abs=1e-10)


def test_delta_distribution_random_sample():
    """Test random sampling from DeltaDistribution."""
    samples = np.array([1.0, 2.0, 3.0])
    dist = DeltaDistribution(samples)

    rng = RandomGenerator(42)
    n = 1000
    random_samples = dist.random_sample(n=n, rng=rng)

    # All samples should be from the original set
    assert len(random_samples) == n
    for s in random_samples:
        assert s in samples


def test_delta_distribution_empty_samples():
    """Test that empty samples raise ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        DeltaDistribution(np.array([]))


# --- HistogramDistribution Tests ---


def test_histogram_distribution_basic_stats():
    """Test basic statistical properties of HistogramDistribution."""
    rng = RandomGenerator(42)
    samples = np.array(rng.gauss(mean=5.0, std=2.0, n=1000))
    dist = HistogramDistribution(samples, bins=20)

    # Mean and std should be close to sample statistics
    assert dist.mean() == pytest.approx(np.mean(samples), abs=1e-10)
    assert dist.std(ddof=0) == pytest.approx(np.std(samples, ddof=0), abs=1e-10)

    # Median should be close to sample median
    assert dist.median() == pytest.approx(np.median(samples), abs=1e-10)


def test_histogram_distribution_pdf_positive():
    """Test that PDF values are non-negative."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dist = HistogramDistribution(samples, bins=5)

    x = np.linspace(0, 6, 100)
    pdf = dist.pdf(x)

    assert np.all(pdf >= 0)


def test_histogram_distribution_cdf_bounds():
    """Test that CDF is bounded between 0 and 1."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dist = HistogramDistribution(samples, bins=5)

    x = np.linspace(-10, 10, 100)
    cdf = dist.cdf(x)

    # CDF should be 0 before samples and 1 after
    assert cdf[0] == pytest.approx(0.0, abs=1e-10)
    assert cdf[-1] == pytest.approx(1.0, abs=1e-10)
    assert np.all((cdf >= 0) & (cdf <= 1))


def test_histogram_distribution_random_sample():
    """Test random sampling from HistogramDistribution."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dist = HistogramDistribution(samples, bins=5)

    rng = RandomGenerator(123)
    random_samples = dist.random_sample(n=100, rng=rng)

    assert len(random_samples) == 100
    # Samples should be within the range of original samples
    assert np.all((random_samples >= np.min(samples)) & (random_samples <= np.max(samples)))


def test_histogram_distribution_custom_bins_range():
    """Test HistogramDistribution with custom bins range."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dist = HistogramDistribution(samples, bins=10, bins_range=(0.0, 6.0))

    # Should handle custom range without errors
    assert dist.mean() == pytest.approx(np.mean(samples), abs=1e-10)


def test_histogram_distribution_empty_samples():
    """Test that empty samples raise ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        HistogramDistribution(np.array([]))


def test_histogram_distribution_invalid_bins():
    """Test that invalid bins raise ValueError."""
    samples = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="must be positive"):
        HistogramDistribution(samples, bins=-5)


# --- WeightedDistribution Tests ---


def test_weighted_distribution_basic_stats():
    """Test basic statistical properties of WeightedDistribution."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    dist = WeightedDistribution(samples, weights, bins=5)

    # Weighted mean: (1*1 + 2*2 + 3*3 + 4*2 + 5*1) / (1+2+3+2+1) = 27/9 = 3.0
    expected_mean = np.sum(samples * weights) / np.sum(weights)
    assert dist.mean() == pytest.approx(expected_mean, abs=1e-10)


def test_weighted_distribution_pdf_positive():
    """Test that PDF values are non-negative."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    dist = WeightedDistribution(samples, weights, bins=5)

    x = np.linspace(0, 6, 100)
    pdf = dist.pdf(x)

    assert np.all(pdf >= 0)


def test_weighted_distribution_cdf_bounds():
    """Test that CDF is bounded between 0 and 1."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([1.0, 2.0, 1.0, 1.0, 1.0])
    dist = WeightedDistribution(samples, weights, bins=5)

    x = np.linspace(-10, 10, 100)
    cdf = dist.cdf(x)

    assert cdf[0] == pytest.approx(0.0, abs=1e-10)
    assert cdf[-1] == pytest.approx(1.0, abs=1e-10)
    assert np.all((cdf >= 0) & (cdf <= 1))


def test_weighted_distribution_random_sample():
    """Test random sampling from WeightedDistribution."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([10.0, 1.0, 1.0, 1.0, 1.0])  # Heavy weight on 1.0
    dist = WeightedDistribution(samples, weights, bins=5)

    rng = RandomGenerator(456)
    n = 1000
    random_samples = dist.random_sample(n=n, rng=rng)

    assert len(random_samples) == n
    # With heavy weight on 1.0, we should see more samples near 1.0
    count_near_1 = np.sum(random_samples < 2.0)
    # At least 50% should be from the first value (which has 10/14 ≈ 0.71 weight)
    assert count_near_1 > n * 0.5


def test_weighted_distribution_empty_samples():
    """Test that empty samples raise ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        WeightedDistribution(np.array([]), np.array([]))


def test_weighted_distribution_mismatched_lengths():
    """Test that mismatched sample and weight lengths raise ValueError."""
    samples = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="same length"):
        WeightedDistribution(samples, weights)


def test_weighted_distribution_negative_weights():
    """Test that negative weights raise ValueError."""
    samples = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, -2.0, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        WeightedDistribution(samples, weights)


def test_weighted_distribution_zero_sum_weights():
    """Test that zero sum weights raise ValueError."""
    samples = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="must be positive"):
        WeightedDistribution(samples, weights)


def test_weighted_distribution_weighted_median():
    """Test weighted median calculation."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Heavy weight on 3.0
    weights = np.array([0.1, 0.1, 0.6, 0.1, 0.1])
    dist = WeightedDistribution(samples, weights, bins=5)

    # Weighted median should be close to 3.0
    assert dist.median() == pytest.approx(3.0, abs=0.5)
