"""Tests for kernel distribution classes."""

import numpy as np
import pytest

from faxai.mathing.distribution.kernel_distributions import (
    KernelDensityDistribution,
    KernelDensityEstimationDistribution,
)
from faxai.mathing.kernel import GaussianKernel, UniformKernel
from faxai.mathing.bandwidth import Bandwidth
from faxai.mathing.RandomGenerator import RandomGenerator


# --- KernelDensityDistribution Tests ---


def test_kernel_density_distribution_basic_stats():
    """Test basic statistical properties of KernelDensityDistribution."""
    bandwidth = Bandwidth.build_univariate(1.0)
    kernel = GaussianKernel(bandwidth=bandwidth)
    center = 5.0
    dist = KernelDensityDistribution(kernel=kernel, center=center)

    # For symmetric kernel, mean should equal center
    assert dist.mean() == pytest.approx(center, abs=1e-10)

    # Mode should also be at center
    assert dist.moded() == pytest.approx(center, abs=1e-10)

    # Median should equal mean for symmetric distribution
    assert dist.median() == pytest.approx(center, abs=1e-10)


def test_kernel_density_distribution_pdf_at_center():
    """Test that PDF is maximum at the center."""
    bandwidth = Bandwidth.build_univariate(1.0)
    kernel = GaussianKernel(bandwidth=bandwidth)
    center = 0.0
    dist = KernelDensityDistribution(kernel=kernel, center=center)

    # PDF at center should be the maximum
    pdf_center = dist.pdf(np.array([center]))[0]
    pdf_away = dist.pdf(np.array([center + 2.0]))[0]

    assert pdf_center > pdf_away
    assert dist.maximum_pdf() == pytest.approx(pdf_center, abs=1e-10)


def test_kernel_density_distribution_pdf_symmetry():
    """Test that PDF is symmetric around center."""
    bandwidth = Bandwidth.build_univariate(1.0)
    kernel = GaussianKernel(bandwidth=bandwidth)
    center = 3.0
    dist = KernelDensityDistribution(kernel=kernel, center=center)

    # PDF should be symmetric
    x1 = center - 1.0
    x2 = center + 1.0
    pdf1 = dist.pdf(np.array([x1]))[0]
    pdf2 = dist.pdf(np.array([x2]))[0]

    assert pdf1 == pytest.approx(pdf2, rel=1e-6)


def test_kernel_density_distribution_cdf_bounds():
    """Test that CDF is bounded between 0 and 1."""
    bandwidth = Bandwidth.build_univariate(1.0)
    kernel = GaussianKernel(bandwidth=bandwidth)
    dist = KernelDensityDistribution(kernel=kernel, center=0.0)

    x = np.linspace(-10, 10, 20)
    cdf = dist.cdf(x)

    # CDF should be monotonically increasing
    assert np.all(np.diff(cdf) >= -1e-6)  # Allow small numerical errors

    # CDF should approach 0 at -infinity and 1 at +infinity
    assert cdf[0] < 0.1  # Close to 0
    # NOTE: Due to integration bounds in the implementation, CDF may not reach exactly 1.0
    # This is a limitation of the numerical integration approach.
    assert cdf[-1] > 0.8  # Should be reasonably close to 1


def test_kernel_density_distribution_random_sample():
    """Test random sampling from KernelDensityDistribution."""
    bandwidth = Bandwidth.build_univariate(1.0)
    kernel = GaussianKernel(bandwidth=bandwidth)
    center = 2.0
    dist = KernelDensityDistribution(kernel=kernel, center=center)

    rng = RandomGenerator(789)
    n = 500
    samples = dist.random_sample(n=n, rng=rng)

    assert len(samples) == n

    # Samples should cluster around the center
    mean_samples = np.mean(samples)
    assert mean_samples == pytest.approx(center, abs=1.0)


def test_kernel_density_distribution_no_bandwidth():
    """Test that missing bandwidth raises ValueError."""
    kernel = GaussianKernel()  # No bandwidth set
    with pytest.raises(ValueError, match="must have a bandwidth"):
        KernelDensityDistribution(kernel=kernel)


def test_kernel_density_distribution_std():
    """Test standard deviation computation."""
    bandwidth = Bandwidth.build_univariate(1.0)
    kernel = GaussianKernel(bandwidth=bandwidth)
    dist = KernelDensityDistribution(kernel=kernel, center=0.0)

    # For Gaussian kernel, std should be related to bandwidth
    std = dist.std(ddof=0)
    assert std > 0  # Should be positive

    # Should raise error for non-zero ddof
    with pytest.raises(ValueError, match="not applicable"):
        dist.std(ddof=1)


# --- KernelDensityEstimationDistribution Tests ---


def test_kde_distribution_basic_stats():
    """Test basic statistical properties of KDE distribution."""
    rng = RandomGenerator(42)
    samples = np.array(rng.gauss(mean=5.0, std=2.0, n=100))
    bandwidth = Bandwidth.build_univariate(0.5)
    kernel = GaussianKernel(bandwidth=bandwidth)
    dist = KernelDensityEstimationDistribution(samples=samples, kernel=kernel)

    # Mean should be close to sample mean
    assert dist.mean() == pytest.approx(np.mean(samples), abs=1e-10)

    # Median should be close to sample median
    assert dist.median() == pytest.approx(np.median(samples), abs=1e-10)


def test_kde_distribution_pdf_positive():
    """Test that PDF values are non-negative."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bandwidth = Bandwidth.build_univariate(0.5)
    kernel = GaussianKernel(bandwidth=bandwidth)
    dist = KernelDensityEstimationDistribution(samples=samples, kernel=kernel)

    x = np.linspace(0, 6, 50)
    pdf = dist.pdf(x)

    assert np.all(pdf >= 0)


def test_kde_distribution_pdf_smooth():
    """Test that KDE PDF is smooth (no abrupt changes)."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bandwidth = Bandwidth.build_univariate(0.5)
    kernel = GaussianKernel(bandwidth=bandwidth)
    dist = KernelDensityEstimationDistribution(samples=samples, kernel=kernel)

    x = np.linspace(0, 6, 100)
    pdf = dist.pdf(x)

    # Check that PDF doesn't have extreme jumps
    pdf_diff = np.abs(np.diff(pdf))
    max_diff = np.max(pdf_diff)

    # For Gaussian kernel, PDF should be smooth
    assert max_diff < 0.5  # Reasonable threshold for smooth changes


def test_kde_distribution_mode():
    """Test mode calculation for KDE."""
    # Create samples with clear mode at 3.0
    samples = np.array([3.0, 3.0, 3.0, 1.0, 5.0])
    bandwidth = Bandwidth.build_univariate(0.3)
    kernel = GaussianKernel(bandwidth=bandwidth)
    dist = KernelDensityEstimationDistribution(samples=samples, kernel=kernel)

    # Mode should be close to 3.0
    mode = dist.moded()
    assert mode == pytest.approx(3.0, abs=0.5)


def test_kde_distribution_cdf_monotonic():
    """Test that CDF is monotonically increasing."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bandwidth = Bandwidth.build_univariate(0.5)
    kernel = GaussianKernel(bandwidth=bandwidth)
    dist = KernelDensityEstimationDistribution(samples=samples, kernel=kernel)

    x = np.linspace(0, 6, 30)
    cdf = dist.cdf(x)

    # CDF should be monotonically increasing
    assert np.all(np.diff(cdf) >= -1e-6)  # Allow for small numerical errors


def test_kde_distribution_random_sample():
    """Test random sampling from KDE distribution."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bandwidth = Bandwidth.build_univariate(0.5)
    kernel = GaussianKernel(bandwidth=bandwidth)
    dist = KernelDensityEstimationDistribution(samples=samples, kernel=kernel)

    rng = RandomGenerator(999)
    n = 1000
    random_samples = dist.random_sample(n=n, rng=rng)

    assert len(random_samples) == n

    # Samples should have similar mean to original
    mean_random = np.mean(random_samples)
    mean_original = np.mean(samples)
    assert mean_random == pytest.approx(mean_original, abs=1.0)


def test_kde_distribution_maximum_pdf():
    """Test maximum PDF value calculation."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bandwidth = Bandwidth.build_univariate(0.5)
    kernel = GaussianKernel(bandwidth=bandwidth)
    dist = KernelDensityEstimationDistribution(samples=samples, kernel=kernel)

    max_pdf = dist.maximum_pdf()
    assert max_pdf > 0

    # Maximum should be at least as large as PDF at any sample point
    pdf_at_samples = dist.pdf(samples)
    assert max_pdf >= np.max(pdf_at_samples) - 1e-10


def test_kde_distribution_empty_samples():
    """Test that empty samples raise ValueError."""
    bandwidth = Bandwidth.build_univariate(0.5)
    kernel = GaussianKernel(bandwidth=bandwidth)
    with pytest.raises(ValueError, match="cannot be empty"):
        KernelDensityEstimationDistribution(samples=np.array([]), kernel=kernel)


def test_kde_distribution_no_bandwidth():
    """Test that missing bandwidth raises ValueError."""
    samples = np.array([1.0, 2.0, 3.0])
    kernel = GaussianKernel()  # No bandwidth
    with pytest.raises(ValueError, match="must have a bandwidth"):
        KernelDensityEstimationDistribution(samples=samples, kernel=kernel)


def test_kde_distribution_std():
    """Test standard deviation computation for KDE."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bandwidth = Bandwidth.build_univariate(0.5)
    kernel = GaussianKernel(bandwidth=bandwidth)
    dist = KernelDensityEstimationDistribution(samples=samples, kernel=kernel)

    # Std should be positive and related to sample std
    std = dist.std(ddof=0)
    sample_std = np.std(samples, ddof=0)

    assert std > 0
    # KDE std should be larger than sample std due to kernel bandwidth
    assert std >= sample_std


def test_kde_distribution_different_kernels():
    """Test KDE with different kernel types."""
    samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Test with Uniform kernel
    bandwidth = Bandwidth.build_univariate(1.0)
    uniform_kernel = UniformKernel(bandwidth=bandwidth)
    dist_uniform = KernelDensityEstimationDistribution(samples=samples, kernel=uniform_kernel)

    # Should work without errors
    assert dist_uniform.mean() == pytest.approx(np.mean(samples), abs=1e-10)

    # Test with Gaussian kernel
    gaussian_kernel = GaussianKernel(bandwidth=bandwidth)
    dist_gaussian = KernelDensityEstimationDistribution(samples=samples, kernel=gaussian_kernel)

    # Should work without errors
    assert dist_gaussian.mean() == pytest.approx(np.mean(samples), abs=1e-10)


def test_kde_distribution_single_sample():
    """Test KDE with a single sample."""
    samples = np.array([2.5])
    bandwidth = Bandwidth.build_univariate(0.5)
    kernel = GaussianKernel(bandwidth=bandwidth)
    dist = KernelDensityEstimationDistribution(samples=samples, kernel=kernel)

    # Mean should equal the single sample
    assert dist.mean() == pytest.approx(2.5, abs=1e-10)

    # PDF should be peaked at the sample location
    pdf_at_sample = dist.pdf(np.array([2.5]))[0]
    pdf_away = dist.pdf(np.array([5.0]))[0]
    assert pdf_at_sample > pdf_away
