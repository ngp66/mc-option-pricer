import numpy as np


def antithetic_variates(Z: np.ndarray) -> np.ndarray:
    return np.concatenate([Z, -Z], axis=0)


def antithetic_paths(paths: np.ndarray) -> np.ndarray:
    return np.vstack([paths, paths[::-1]])


def control_variate(payoff, control, control_exact):
    payoff = np.asarray(payoff)
    control = np.asarray(control)

    cov = np.cov(payoff, control, ddof=1)[0, 1]
    var_c = np.var(control, ddof=1)

    if var_c == 0:
        return payoff

    c = -cov / var_c
    return payoff + c * (control - control_exact)


def importance_sampling_log_likelihood(S, S0, r, mu_shift, sigma, T):
    S = np.asarray(S)

    drift_diff = (r - mu_shift)
    log_term = np.log(S / S0)

    exponent = (drift_diff / sigma**2) * (
        log_term - (mu_shift - 0.5 * sigma**2) * T
    )

    return np.exp(exponent)


def shift_drift_paths(S0, r, sigma, T, n_paths, n_steps, mu_shift):
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    mu = mu_shift

    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_paths)

        paths[:, t] = paths[:, t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt +
            sigma * np.sqrt(dt) * Z
        )

    return paths


def apply_control_variate(payoff, control, control_exact):
    payoff = np.asarray(payoff)
    control = np.asarray(control)

    cov = np.cov(payoff, control, ddof=1)[0, 1]
    var_c = np.var(control, ddof=1)

    if var_c == 0:
        return payoff

    c = -cov / var_c
    return payoff + c * (control - control_exact)