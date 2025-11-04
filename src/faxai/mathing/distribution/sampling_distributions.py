from __future__ import annotations

import numpy as np

from faxai.mathing.distribution.Distribution import Distribution
from faxai.mathing.RandomGenerator import RandomGenerator
from faxai.utils.decorators import cache_method


class DeltaDistribution(Distribution):
    """
    Represents a degenerate distribution concentrated only on those points that are given as samples.

    This distribution assigns equal probability to each distinct sample value, with zero probability
    elsewhere. It is effectively a discrete uniform distribution over the provided sample points.

    Parameters
    ----------
    samples : np.ndarray
        Array of sample points where the distribution has non-zero probability.

    Notes
    -----
    - The mean is the average of all samples (with repetitions counted).
    - The mode is the most frequently occurring value.
    - The median is computed from the sorted samples.
    - PDF returns probability mass at exact sample points, zero elsewhere.
    - CDF is a step function.
    """

    def __init__(self, samples: np.ndarray):
        if len(samples) == 0:
            raise ValueError("Samples array cannot be empty.")

        self._samples = np.asarray(samples, dtype=float)
        # Store unique values and their counts for efficient computation
        self._unique_values, self._counts = np.unique(self._samples, return_counts=True)
        self._probabilities = self._counts / len(self._samples)

    def mean(self) -> float:
        """Calculate the mean of the distribution."""
        return float(np.mean(self._samples))

    def std(self, ddof: int = 0) -> float:
        """Calculate the standard deviation of the distribution."""
        return float(np.std(self._samples, ddof=ddof))

    def moded(self) -> float:
        """Calculate the mode of the distribution (most frequent value)."""
        max_count_idx = np.argmax(self._counts)
        return float(self._unique_values[max_count_idx])

    def median(self) -> float:
        """Calculate the median of the distribution."""
        return float(np.median(self._samples))

    @cache_method
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the probability mass function.

        Returns the probability mass at points that match samples, zero elsewhere.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        for i, val in enumerate(self._unique_values):
            # NOTE: Using exact equality for discrete distribution.
            # This may have precision issues with floating point comparisons.
            mask = np.isclose(x, val, rtol=1e-15, atol=1e-15)
            result[mask] = self._probabilities[i]

        return result

    @cache_method
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the cumulative distribution function.

        Returns the probability that a random sample is less than or equal to x.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        for val, prob in zip(self._unique_values, self._probabilities):
            result += np.where(x >= val, prob, 0.0)

        return result

    @cache_method
    def maximum_pdf(self) -> float:
        """Calculate the maximum value of the PDF (highest probability mass)."""
        return float(np.max(self._probabilities))

    def random_sample(self, n: int = 1, rng: RandomGenerator | None = None) -> np.ndarray:
        """
        Generate random samples from the distribution.

        Samples are drawn from the original sample set with replacement.
        """
        if rng is None:
            rng = RandomGenerator()

        # Sample indices based on probabilities
        indices = [rng.choice(range(len(self._samples))) for _ in range(n)]
        return self._samples[indices]


class HistogramDistribution(Distribution):
    """
    Represents a discrete distribution based on a given set of samples.
    Probability functions are set using histogram estimation.

    This distribution creates bins from the sample data and estimates probabilities
    based on the number of samples falling into each bin.

    Parameters
    ----------
    samples : np.ndarray
        Array of sample points to build the histogram from.
    bins : int, optional
        Number of bins for the histogram. Default is 10.
    bins_range : tuple of float, optional
        The lower and upper range of the bins as (min, max). If None, uses the range of samples.

    Notes
    -----
    - PDF is piecewise constant within each bin.
    - CDF is a step function based on bin boundaries.
    - Sampling uses inverse transform sampling.
    """

    def __init__(self, samples: np.ndarray, bins: int | None = 10, bins_range: tuple[float, float] | None = None):
        if len(samples) == 0:
            raise ValueError("Samples array cannot be empty.")
        if bins is None:
            bins = 10
        if bins <= 0:
            raise ValueError("Number of bins must be positive.")

        self._samples = np.asarray(samples, dtype=float)
        self._bins = bins

        # Determine bin range
        if bins_range is None:
            self._bins_range = (float(np.min(self._samples)), float(np.max(self._samples)))
        else:
            self._bins_range = bins_range

        # Create histogram
        self._hist, self._bin_edges = np.histogram(
            self._samples, bins=self._bins, range=self._bins_range, density=False
        )

        # Normalize to get probabilities
        self._probabilities = self._hist / len(self._samples)

        # Bin widths for PDF calculation
        self._bin_widths = np.diff(self._bin_edges)

        # PDF values (probability density, not mass)
        # NOTE: For bins with zero width, we set density to 0 to avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            self._pdf_values = self._probabilities / self._bin_widths
            self._pdf_values = np.nan_to_num(self._pdf_values, nan=0.0, posinf=0.0)

    def mean(self) -> float:
        """Calculate the mean of the distribution."""
        return float(np.mean(self._samples))

    def std(self, ddof: int = 0) -> float:
        """Calculate the standard deviation of the distribution."""
        return float(np.std(self._samples, ddof=ddof))

    def moded(self) -> float:
        """
        Calculate the mode of the distribution.

        Returns the center of the bin with the highest probability.
        """
        max_bin_idx = np.argmax(self._probabilities)
        return float((self._bin_edges[max_bin_idx] + self._bin_edges[max_bin_idx + 1]) / 2)

    def median(self) -> float:
        """Calculate the median of the distribution."""
        return float(np.median(self._samples))

    @cache_method
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the probability density function.

        Returns the density for each bin (constant within each bin).
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        for i in range(len(self._bin_edges) - 1):
            mask = (x >= self._bin_edges[i]) & (x < self._bin_edges[i + 1])
            result[mask] = self._pdf_values[i]

        # Handle the rightmost edge (inclusive)
        if len(self._bin_edges) > 0:
            mask = x == self._bin_edges[-1]
            result[mask] = self._pdf_values[-1]

        return result

    @cache_method
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the cumulative distribution function.

        Returns the cumulative probability up to x.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        cumsum = np.cumsum(self._probabilities)

        # For values before the first bin edge
        result[x < self._bin_edges[0]] = 0.0

        # For values in each bin, return cumulative probability up to and including this bin
        for i in range(len(self._bin_edges) - 1):
            mask = (x >= self._bin_edges[i]) & (x < self._bin_edges[i + 1])
            result[mask] = cumsum[i]

        # For values at or beyond the last edge
        mask = x >= self._bin_edges[-1]
        result[mask] = 1.0

        return result

    @cache_method
    def maximum_pdf(self) -> float:
        """Calculate the maximum value of the PDF."""
        return float(np.max(self._pdf_values))

    def random_sample(self, n: int = 1, rng: RandomGenerator | None = None) -> np.ndarray:
        """
        Generate random samples from the distribution.

        Samples are drawn from the original sample set with replacement.
        """
        if rng is None:
            rng = RandomGenerator()

        # Simple approach: sample from the original samples with replacement
        indices = [rng.choice(range(len(self._samples))) for _ in range(n)]
        return self._samples[indices]


class WeightedDistribution(Distribution):
    """
    Represents a distribution based on a given set of samples with an associated weight.
    Probability functions are set using histogram estimation.

    Similar to HistogramDistribution, but samples are weighted according to provided weights.

    Parameters
    ----------
    samples : np.ndarray
        Array of sample points.
    weights : np.ndarray
        Array of weights corresponding to each sample. Must have same length as samples.
    bins : int, optional
        Number of bins for the histogram. Default is 10.
    bins_range : tuple of float, optional
        The lower and upper range of the bins as (min, max). If None, uses the range of samples.

    Notes
    -----
    - Weights are normalized to sum to 1.
    - PDF and CDF calculations account for sample weights.
    """

    def __init__(
        self, samples: np.ndarray, weights: np.ndarray, bins: int | None = 10,
        bins_range: tuple[float, float] | None = None
    ):
        if len(samples) == 0:
            raise ValueError("Samples array cannot be empty.")

        self._samples = np.asarray(samples, dtype=float)
        self._weights = np.asarray(weights, dtype=float)

        if len(self._samples) != len(self._weights):
            raise ValueError("Samples and weights must have the same length.")

        if np.any(self._weights < 0):
            raise ValueError("Weights must be non-negative.")

        if np.sum(self._weights) == 0:
            raise ValueError("Sum of weights must be positive.")

        # Normalize weights
        self._weights = self._weights / np.sum(self._weights)

        if bins is None:
            bins = 10
        if bins <= 0:
            raise ValueError("Number of bins must be positive.")

        self._bins = bins

        # Determine bin range
        if bins_range is None:
            self._bins_range = (float(np.min(self._samples)), float(np.max(self._samples)))
        else:
            self._bins_range = bins_range

        # Create weighted histogram
        self._hist, self._bin_edges = np.histogram(
            self._samples, bins=self._bins, range=self._bins_range, weights=self._weights, density=False
        )

        # Probabilities (already normalized since weights sum to 1)
        self._probabilities = self._hist

        # Bin widths for PDF calculation
        self._bin_widths = np.diff(self._bin_edges)

        # PDF values (probability density)
        with np.errstate(divide='ignore', invalid='ignore'):
            self._pdf_values = self._probabilities / self._bin_widths
            self._pdf_values = np.nan_to_num(self._pdf_values, nan=0.0, posinf=0.0)

    def mean(self) -> float:
        """Calculate the weighted mean of the distribution."""
        return float(np.sum(self._samples * self._weights))

    def std(self, ddof: int = 0) -> float:
        """
        Calculate the weighted standard deviation of the distribution.

        NOTE: ddof parameter affects the denominator in the variance calculation.
        For weighted samples, this uses the reliability weights formulation.
        """
        mean = self.mean()
        variance: float = np.sum(self._weights * (self._samples - mean) ** 2)

        if ddof != 0:
            # Adjust for degrees of freedom
            # For weighted data: effective_n = (sum(w))^2 / sum(w^2)
            sum_weights: float = np.sum(self._weights)
            sum_weights_sq: float = np.sum(self._weights**2)
            effective_n = sum_weights**2 / sum_weights_sq if sum_weights_sq > 0 else 1

            denominator = sum_weights * (effective_n - ddof) / effective_n
            if denominator <= 0:
                # NOTE: Edge case where ddof is too large for the effective sample size
                return 0.0
            variance = variance / denominator * sum_weights

        return float(np.sqrt(variance))

    def moded(self) -> float:
        """
        Calculate the mode of the distribution.

        Returns the center of the bin with the highest weighted probability.
        """
        max_bin_idx = np.argmax(self._probabilities)
        return float((self._bin_edges[max_bin_idx] + self._bin_edges[max_bin_idx + 1]) / 2)

    def median(self) -> float:
        """
        Calculate the weighted median of the distribution.

        The weighted median is the value where the cumulative weight reaches 0.5.
        """
        sorted_indices = np.argsort(self._samples)
        sorted_samples = self._samples[sorted_indices]
        sorted_weights = self._weights[sorted_indices]

        cumulative_weights = np.cumsum(sorted_weights)
        median_idx: int = int(np.searchsorted(cumulative_weights, 0.5))

        if median_idx >= len(sorted_samples):
            median_idx = len(sorted_samples) - 1

        return float(sorted_samples[median_idx])

    @cache_method
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the probability density function.

        Returns the density for each bin (constant within each bin).
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        for i in range(len(self._bin_edges) - 1):
            mask = (x >= self._bin_edges[i]) & (x < self._bin_edges[i + 1])
            result[mask] = self._pdf_values[i]

        # Handle the rightmost edge (inclusive)
        if len(self._bin_edges) > 0:
            mask = x == self._bin_edges[-1]
            result[mask] = self._pdf_values[-1]

        return result

    @cache_method
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the cumulative distribution function.

        Returns the cumulative probability up to x.
        """
        x = np.asarray(x)
        result = np.zeros_like(x, dtype=float)

        cumsum = np.cumsum(self._probabilities)

        # For values before the first bin edge
        result[x < self._bin_edges[0]] = 0.0

        # For values in each bin, return cumulative probability up to and including this bin
        for i in range(len(self._bin_edges) - 1):
            mask = (x >= self._bin_edges[i]) & (x < self._bin_edges[i + 1])
            result[mask] = cumsum[i]

        # For values at or beyond the last edge
        mask = x >= self._bin_edges[-1]
        result[mask] = 1.0

        return result

    @cache_method
    def maximum_pdf(self) -> float:
        """Calculate the maximum value of the PDF."""
        return float(np.max(self._pdf_values))

    def random_sample(self, n: int = 1, rng: RandomGenerator | None = None) -> np.ndarray:
        """
        Generate random samples from the distribution using weighted sampling.

        Samples are drawn from the original sample set according to their weights.
        """
        if rng is None:
            rng = RandomGenerator()

        # Use weighted random sampling
        # Convert weights to cumulative probabilities for inverse transform
        cumulative_weights = np.cumsum(self._weights)

        result = []
        for _ in range(n):
            r = rng.rand()
            idx: int = int(np.searchsorted(cumulative_weights, r))
            if idx >= len(self._samples):
                idx = len(self._samples) - 1
            result.append(self._samples[idx])

        return np.array(result)
