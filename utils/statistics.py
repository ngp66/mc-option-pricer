import numpy as np
from scipy.stats import norm


def mean_confidence_interval(data, confidence=0.95):
    data = np.asarray(data)

    n = len(data)
    mean = np.mean(data)
    stderr = np.std(data, ddof=1) / np.sqrt(n)

    z = norm.ppf(0.5 + confidence / 2)
    margin = z * stderr

    return mean, mean - margin, mean + margin


def monte_carlo_error(estimates):
    estimates = np.asarray(estimates)
    return np.std(estimates, ddof=1) / np.sqrt(len(estimates))


def relative_error(mc_value, true_value):
    return abs(mc_value - true_value) / abs(true_value)


def running_mean(data):
    data = np.asarray(data)
    return np.cumsum(data) / np.arange(1, len(data) + 1)


def summary_stats(estimates, benchmark=None, confidence=0.95):
    estimates = np.asarray(estimates)

    mean, low, high = mean_confidence_interval(estimates, confidence)

    result = {
        "mean": mean,
        "stderr": monte_carlo_error(estimates),
        "ci_lower": low,
        "ci_upper": high,
    }

    if benchmark is not None:
        result["abs_error"] = abs(mean - benchmark)
        result["rel_error"] = relative_error(mean, benchmark)

    return result


def batch_summary(stat_matrix, axis=0):
    stat_matrix = np.asarray(stat_matrix)

    means = np.mean(stat_matrix, axis=axis)
    stderrs = np.std(stat_matrix, axis=axis, ddof=1) / np.sqrt(stat_matrix.shape[axis])

    return {
        "means": means,
        "stderrs": stderrs,
    }