from __future__ import annotations

import numpy as np
from scipy import integrate  # type: ignore[import-untyped]

from faxai.mathing.distribution.Distribution import Distribution
from faxai.mathing.kernel import Kernel
from faxai.mathing.RandomGenerator import RandomGenerator
from faxai.utils.decorators import cache_method


class KernelDensityDistribution(Distribution):
    """
    Represents a distribution given by a kernel function centered at a single point.

    This is essentially a kernel function treated as a probability distribution.
    The kernel must be properly normalized to integrate to 1.

    Parameters
    ----------
    kernel : Kernel
        The kernel function to use. Must have bandwidth set.

    center : float, optional
        The center point of the kernel. Default is 0.0.

    Notes
    -----
    - This assumes the kernel is univariate (1-dimensional).
    - The kernel itself acts as the PDF.
    - Statistical properties are computed numerically or analytically where possible.

    WARNING: This implementation assumes univariate kernels. Multi-dimensional support
    would require additional complexity.
    """

    def __init__(self, kernel: Kernel, center: float = 0.0):
        if kernel.bandwidth() is None:
            raise ValueError("Kernel must have a bandwidth set.")

        if kernel.dimension() != 1:
            raise ValueError("Only univariate kernels are currently supported.")
            # NOTE: Multi-dimensional kernels would require vector operations
            # and more complex integration methods.

        self._kernel = kernel
        self._center = float(center)

    def mean(self) -> float:
        """
        Calculate the mean of the distribution.

        For a symmetric kernel centered at 'center', the mean is the center point.
        """
        # NOTE: This assumes the kernel is symmetric, which is true for all standard kernels.
        # For asymmetric kernels, we would need to compute the integral of x * pdf(x).
        return self._center

    def std(self, ddof: int = 0) -> float:
        """
        Calculate the standard deviation of the distribution.

        Computed numerically by integrating (x - mean)^2 * pdf(x).
        """
        if ddof != 0:
            raise ValueError("ddof parameter is not applicable for kernel density distributions.")

        # Compute variance by numerical integration
        # Variance = integral of (x - mean)^2 * pdf(x) dx
        mean = self.mean()

        def integrand(x: float) -> float:
            return (x - mean) ** 2 * self.pdf(np.array([x]))[0]

        # NOTE: Integration bounds are set heuristically. For kernels with bounded support
        # (uniform, triangular, Epanechnikov), we could use tighter bounds.
        # For unbounded kernels (Gaussian), we use wide bounds.
        # This may not be accurate for extreme cases or very small/large bandwidths.
        # ISSUE: The bounds of ±10√(bandwidth) may be insufficient for accurate variance
        # calculation. For Gaussian kernels, 99.99% of mass is within ±4σ, but the
        # relationship between bandwidth and σ depends on the kernel implementation.
        # Consider using kernel-specific bounds or adaptive integration.
        bandwidth = self._kernel.bandwidth()
        if bandwidth is None:
            raise ValueError("Kernel bandwidth is not set")
        bandwidth_value = bandwidth.matrix()[0, 0]
        lower_bound = self._center - 10 * np.sqrt(bandwidth_value)
        upper_bound = self._center + 10 * np.sqrt(bandwidth_value)

        variance, _ = integrate.quad(integrand, lower_bound, upper_bound)
        return float(np.sqrt(max(0, variance)))

    def moded(self) -> float:
        """
        Calculate the mode of the distribution.

        For symmetric kernels, the mode is at the center.
        """
        # NOTE: For symmetric kernels centered at 'center', the mode is at the center.
        # For asymmetric kernels, we would need to find the maximum of the PDF.
        return self._center

    def median(self) -> float:
        """
        Calculate the median of the distribution.

        For symmetric distributions, the median equals the mean.
        """
        # NOTE: For symmetric distributions, median = mean = mode.
        return self.mean()

    @cache_method
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the probability density function.

        Evaluates the kernel at each point.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        center_array = np.array([self._center])
        for i, xi in enumerate(x):
            xi_array = np.array([xi])
            result[i] = self._kernel.apply(xi_array, center_array)

        return result

    @cache_method
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the cumulative distribution function.

        Computed by numerical integration of the PDF.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        # NOTE: CDF is computed by integrating the PDF from -inf to x.
        # This is computationally expensive and may have numerical errors.
        # For better performance, consider using analytical CDF formulas for specific kernels.

        bandwidth = self._kernel.bandwidth()
        if bandwidth is None:
            raise ValueError("Kernel bandwidth is not set")
        bandwidth_value = bandwidth.matrix()[0, 0]
        lower_bound = self._center - 10 * np.sqrt(bandwidth_value)

        for i, xi in enumerate(x):

            def integrand(t: float) -> float:
                return self.pdf(np.array([t]))[0]

            cdf_val, _ = integrate.quad(integrand, lower_bound, float(xi))
            result[i] = np.clip(cdf_val, 0.0, 1.0)

        return result

    @cache_method
    def maximum_pdf(self) -> float:
        """
        Calculate the maximum value of the PDF.

        For symmetric kernels, the maximum is at the center.
        """
        return float(self.pdf(np.array([self._center]))[0])

    def random_sample(self, n: int = 1, rng: RandomGenerator | None = None) -> np.ndarray:
        """
        Generate random samples from the distribution.

        Uses rejection sampling.

        NOTE: This implementation uses rejection sampling which may be inefficient
        for kernels with low maximum PDF values or complex shapes.
        For Gaussian kernels, a direct sampling method would be more efficient.
        """
        if rng is None:
            rng = RandomGenerator()

        samples: list[float] = []
        max_pdf = self.maximum_pdf()

        # Determine sampling bounds
        bandwidth = self._kernel.bandwidth()
        if bandwidth is None:
            raise ValueError("Kernel bandwidth is not set")
        bandwidth_value = bandwidth.matrix()[0, 0]
        lower = self._center - 5 * np.sqrt(bandwidth_value)
        upper = self._center + 5 * np.sqrt(bandwidth_value)

        # Rejection sampling
        # NOTE: This can be slow if max_pdf is small or if the kernel has a complex shape.
        # A more sophisticated approach would use inverse transform sampling or
        # mixture sampling for specific kernel types.
        # ISSUE: The sampling method is kernel-agnostic, which may not be optimal for
        # all kernel types. For specific kernels, consider implementing kernel-specific
        # sampling methods.
        max_iterations = n * 1000  # Prevent infinite loops
        iterations = 0

        while len(samples) < n and iterations < max_iterations:
            iterations += 1
            # Sample uniformly in the range
            x = rng.uniform(lower, upper, 1)[0]
            u = rng.rand()

            pdf_val = self.pdf(np.array([x]))[0]
            if u * max_pdf <= pdf_val:
                samples.append(x)

        if len(samples) < n:
            # Fallback: if rejection sampling fails, sample near the center
            # NOTE: This is a fallback to prevent complete failure, but it
            # may not accurately represent the distribution.
            # ISSUE: This fallback assumes Gaussian noise, which is incorrect for
            # non-Gaussian kernels (e.g., uniform, triangular). For production use,
            # implement kernel-specific fallback sampling or use a more robust
            # primary sampling method.
            remaining = n - len(samples)
            fallback_samples = rng.gauss(mean=self._center, std=np.sqrt(bandwidth_value), n=remaining)
            samples.extend(fallback_samples)

        return np.array(samples[:n])


class KernelDensityEstimationDistribution(Distribution):
    """
    Represents a distribution estimated from a sample using Kernel Density Estimation (KDE).

    KDE is a non-parametric way to estimate the probability density function of a random variable.
    The distribution is modeled as a sum of kernel functions centered at each sample point.

    Parameters
    ----------
    samples : np.ndarray
        Array of sample points from which to estimate the distribution.
    kernel : Kernel
        The kernel function to use for density estimation. Must have bandwidth set.

    Notes
    -----
    - The PDF is the average of kernels centered at each sample point.
    - This implementation assumes univariate (1D) data.
    - Statistical properties are computed from the samples and adjusted based on the kernel.

    WARNING: Multi-dimensional support would require significant modifications.
    """

    def __init__(self, samples: np.ndarray, kernel: Kernel):
        if len(samples) == 0:
            raise ValueError("Samples array cannot be empty.")

        if kernel.bandwidth() is None:
            raise ValueError("Kernel must have a bandwidth set.")

        if kernel.dimension() != 1:
            raise ValueError("Only univariate kernels are currently supported.")
            # NOTE: Multi-dimensional KDE would require handling of multivariate samples
            # and kernel evaluations in higher dimensions.

        self._samples = np.asarray(samples, dtype=float)
        self._kernel = kernel
        self._n_samples = len(self._samples)

    def mean(self) -> float:
        """
        Calculate the mean of the distribution.

        For KDE, the mean is approximately the mean of the samples.
        """
        # NOTE: The true mean of the KDE is the integral of x * pdf(x) dx.
        # However, for symmetric kernels, this is approximately the sample mean.
        return float(np.mean(self._samples))

    def std(self, ddof: int = 0) -> float:
        """
        Calculate the standard deviation of the distribution.

        For KDE, this is approximated by the sample standard deviation.
        """
        # NOTE: The true variance of KDE includes both the sample variance
        # and the kernel bandwidth contribution. This is an approximation.
        # A more accurate estimate would be: var(samples) + var(kernel)
        sample_var = float(np.var(self._samples, ddof=ddof))
        bandwidth = self._kernel.bandwidth()
        if bandwidth is None:
            raise ValueError("Kernel bandwidth is not set")
        bandwidth_value = bandwidth.matrix()[0, 0]

        # Total variance is approximately sample variance + kernel variance
        # For most kernels, the kernel variance is proportional to bandwidth
        total_var = sample_var + bandwidth_value

        return float(np.sqrt(total_var))

    def moded(self) -> float:
        """
        Calculate the mode of the distribution.

        Estimated by finding the point with maximum PDF value.
        """
        # NOTE: Finding the true mode of a KDE requires optimization.
        # We approximate by evaluating PDF at sample points and finding the maximum.
        # This may not be the true mode if it lies between sample points.

        pdf_values = self.pdf(self._samples)
        max_idx = np.argmax(pdf_values)
        return float(self._samples[max_idx])

    def median(self) -> float:
        """
        Calculate the median of the distribution.

        Approximated by the sample median.
        """
        # NOTE: The true median of KDE would require solving CDF(x) = 0.5.
        # We approximate with the sample median, which is reasonable for symmetric kernels.
        return float(np.median(self._samples))

    @cache_method
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the probability density function.

        KDE PDF is the average of kernel functions centered at each sample.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        # KDE: pdf(x) = (1/n) * sum_{i=1}^{n} K((x - x_i) / h)
        # where K is the kernel and h is the bandwidth
        for sample in self._samples:
            sample_array = np.array([sample])
            for i, xi in enumerate(x):
                xi_array = np.array([xi])
                result[i] += self._kernel.apply(xi_array, sample_array)

        result /= self._n_samples
        return result

    @cache_method
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the cumulative distribution function.

        Computed by numerical integration of the KDE PDF.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        # NOTE: Computing CDF for KDE is expensive as it requires integrating the PDF.
        # For each query point, we integrate from -inf to x.
        # This could be optimized by caching or using analytical formulas for specific kernels.

        bandwidth = self._kernel.bandwidth()
        if bandwidth is None:
            raise ValueError("Kernel bandwidth is not set")
        bandwidth_value = bandwidth.matrix()[0, 0]
        lower_bound = np.min(self._samples) - 10 * np.sqrt(bandwidth_value)

        for i, xi in enumerate(x):

            def integrand(t: float) -> float:
                return self.pdf(np.array([t]))[0]

            cdf_val, _ = integrate.quad(integrand, lower_bound, float(xi))
            result[i] = np.clip(cdf_val, 0.0, 1.0)

        return result

    @cache_method
    def maximum_pdf(self) -> float:
        """
        Calculate the maximum value of the PDF.

        Estimated by evaluating PDF at sample points and finding the maximum.
        """
        # NOTE: The true maximum might be between sample points.
        # A more accurate approach would use optimization.
        pdf_at_samples = self.pdf(self._samples)
        return float(np.max(pdf_at_samples))

    def random_sample(self, n: int = 1, rng: RandomGenerator | None = None) -> np.ndarray:
        """
        Generate random samples from the KDE distribution.

        Uses a two-step process:
        1. Randomly select one of the original samples
        2. Add noise from the kernel distribution

        NOTE: This assumes Gaussian kernel behavior for the noise generation.
        For non-Gaussian kernels, the noise distribution should match the kernel shape.
        """
        if rng is None:
            rng = RandomGenerator()

        # KDE sampling algorithm:
        # 1. Choose a random sample point x_i with uniform probability
        # 2. Generate a random value from the kernel centered at x_i

        result: list[float] = []
        bandwidth = self._kernel.bandwidth()
        if bandwidth is None:
            raise ValueError("Kernel bandwidth is not set")
        bandwidth_value = bandwidth.matrix()[0, 0]

        for _ in range(n):
            # Step 1: Choose a random sample
            sample_idx = rng.choice(range(self._n_samples))
            base_sample = self._samples[sample_idx]

            # Step 2: Add kernel noise
            # NOTE: This assumes a Gaussian-like kernel. For other kernel types
            # (uniform, triangular, etc.), the noise distribution should be different.
            # For example, uniform kernel would use uniform noise in [-h, h].
            # ISSUE: This implementation always uses Gaussian noise regardless of kernel type.
            # For non-Gaussian kernels, this produces incorrect sample distributions.
            # A proper implementation should:
            # - For Gaussian kernels: use Gaussian noise (current implementation)
            # - For Uniform kernels: use uniform noise in [-sqrt(bandwidth), +sqrt(bandwidth)]
            # - For other kernels: use appropriate noise matching the kernel shape
            # Consider adding a kernel-specific random_sample method to the Kernel class.
            noise = rng.gauss(mean=0.0, std=np.sqrt(bandwidth_value), n=1)[0]
            result.append(base_sample + noise)

        return np.array(result)
