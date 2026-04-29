import numpy as np
import matplotlib.pyplot as plt
import os

from models.monte_carlo import MonteCarloEngine
from models.black_scholes import black_scholes
from models.heston import HestonEngine
from options.european import EuropeanOption
from options.asian import AsianOption
from models.variance_reduction import apply_control_variate
from utils.statistics import summary_stats

np.random.seed(42)


def convergence_study_european(engine, option, paths=None, n_max=20000, step=500):
    n_total = paths.shape[0]
    sizes = np.arange(step, n_total + 1, step, dtype=int)

    S_T = paths[:, -1]
    payoff = option.payoff(S_T)
    discounted = np.exp(-engine.r * engine.T) * payoff

    cumulative_means = np.cumsum(discounted) / np.arange(1, len(discounted) + 1)
    estimates = cumulative_means[sizes - 1]

    return sizes, estimates, discounted


def convergence_study_asian(engine, option, paths=None, n_max=20000, step=500):
    n_total = paths.shape[0]
    sizes = np.arange(step, n_total + 1, step, dtype=int)

    payoff = option.payoff(paths)
    discounted = np.exp(-engine.r * engine.T) * payoff

    cumulative_means = np.cumsum(discounted) / np.arange(1, len(discounted) + 1)
    estimates = cumulative_means[sizes - 1]

    return sizes, estimates


def main():
    os.makedirs("figures", exist_ok=True)

    # parameters
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2

    n_paths = 100000
    n_steps = 100

    # models
    gbm_engine = MonteCarloEngine(S0, T, r, sigma, n_paths, n_steps)

    heston_engine = HestonEngine(
        S0=S0,
        v0=0.04,
        T=T,
        r=r,
        kappa=2.0,
        theta=0.04,
        xi=0.5,
        rho=-0.7,
        n_paths=n_paths,
        n_steps=n_steps
    )

    european = EuropeanOption(K, "call")
    asian = AsianOption(K, "call")

    # paths
    gbm_paths = gbm_engine.generate_paths()
    heston_paths, _ = heston_engine.generate_paths()

    gbm_S_T = gbm_paths[:, -1]
    heston_S_T = heston_paths[:, -1]

    # benchmark
    bs_price = black_scholes(S0, K, T, r, sigma, "call")

    # payoffs
    gbm_euro = np.exp(-r * T) * european.payoff(gbm_S_T)
    heston_euro = np.exp(-r * T) * european.payoff(heston_S_T)

    gbm_asian = np.exp(-r * T) * asian.payoff(gbm_paths)
    heston_asian = np.exp(-r * T) * asian.payoff(heston_paths)

    # prices
    gbm_euro_price = np.mean(gbm_euro)
    heston_euro_price = np.mean(heston_euro)

    gbm_asian_price = np.mean(gbm_asian)
    heston_asian_price = np.mean(heston_asian)

    print("\n--- MODEL COMPARISON ---")
    print("GBM European:", gbm_euro_price)
    print("Heston European:", heston_euro_price)
    print("GBM Asian:", gbm_asian_price)
    print("Heston Asian:", heston_asian_price)
    print("Black-Scholes:", bs_price)

    # European convergence
    sizes_e, euro_conv, euro_discounted = convergence_study_european(
        gbm_engine, european, paths=gbm_paths
    )

    bs_line = np.full_like(sizes_e, bs_price, dtype=float)

    sigma_euro = np.std(euro_discounted)
    stderr_e = sigma_euro / np.sqrt(sizes_e)
    upper_e = euro_conv + 1.96 * stderr_e
    lower_e = euro_conv - 1.96 * stderr_e

    plt.figure(figsize=(14, 9))
    plt.plot(sizes_e, euro_conv, label="MC Estimate", color='blue', linewidth=3)
    plt.plot(sizes_e, bs_line, "--", label="Black-Scholes", color='black', alpha=0.8)
    plt.fill_between(sizes_e, lower_e, upper_e, color='blue', alpha=0.2,
                     label="95% Confidence Interval")

    plt.title("European Option: Monte Carlo Convergence Analysis", fontsize=24, fontweight='bold')
    plt.xlabel("Number of Paths (N)", fontsize=20)
    plt.ylabel("Option Price", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("figures/european_convergence.png", dpi=300)

    # Asian option + control variate
    sizes_a, asian_conv = convergence_study_asian(
        gbm_engine, asian, paths=gbm_paths
    )

    asian_cv = apply_control_variate(
        payoff=gbm_asian,
        control=gbm_euro,
        control_exact=bs_price
    )

    # variance reduction
    mc_var = np.var(gbm_asian, ddof=1)
    cv_var = np.var(asian_cv, ddof=1)

    vr = 1 - cv_var / mc_var

    print("\n--- VARIANCE REDUCTION ---")
    print("MC Variance:", mc_var)
    print("CV Variance:", cv_var)
    print("Variance Reduction:", vr * 100)

    # Asian convergence
    asian_cv_conv = np.cumsum(asian_cv) / np.arange(1, n_paths + 1)
    asian_cv_estimates = asian_cv_conv[sizes_a - 1]

    final_asian_price = np.mean(asian_cv)
    asian_line = np.full_like(sizes_a, final_asian_price, dtype=float)

    sigma_asian_cv = np.std(asian_cv)
    stderr_a = sigma_asian_cv / np.sqrt(sizes_a)
    upper_a = asian_cv_estimates + 1.96 * stderr_a
    lower_a = asian_cv_estimates - 1.96 * stderr_a

    plt.figure(figsize=(14, 9))
    plt.plot(sizes_a, asian_conv, label="Standard MC", color='red', linestyle='--', linewidth=2)
    plt.plot(sizes_a, asian_cv_estimates, label="Control Variate MC", color='green', linewidth=3)
    plt.plot(sizes_a, asian_line, "--", label="Converged Price", color='black', alpha=0.7)
    plt.fill_between(sizes_a, lower_a, upper_a, color='green', alpha=0.2,
                     label="CV 95% Confidence Interval")

    plt.title("Asian Option: Variance Reduction Performance", fontsize=24, fontweight='bold')
    plt.xlabel("Number of Paths (N)", fontsize=20)
    plt.ylabel("Option Price", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("figures/asian_convergence.png", dpi=300)

    # Heston vs GBM
    plt.figure(figsize=(12, 8))

    plt.hist(gbm_S_T, bins=120, density=True, alpha=0.5,
             label="GBM $S_T$", color="blue")
    plt.hist(heston_S_T, bins=120, density=True, alpha=0.5,
             label="Heston $S_T$", color="orange")

    plt.title("Terminal Asset Distribution: GBM vs Heston", fontsize=22, fontweight='bold')
    plt.xlabel("$S_T$", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/heston_vs_gbm_distribution.png", dpi=300)


if __name__ == "__main__":
    main()    cumulative_means = np.cumsum(discounted) / np.arange(1, len(discounted) + 1)
    estimates = cumulative_means[sizes - 1]

    return sizes, estimates


def main():
    os.makedirs("figures", exist_ok=True)

    # parameters
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2

    n_paths = 100000
    n_steps = 100

    # models
    gbm_engine = MonteCarloEngine(S0, T, r, sigma, n_paths, n_steps)

    heston_engine = HestonEngine(
        S0=S0,
        v0=0.04,
        T=T,
        r=r,
        kappa=2.0,
        theta=0.04,
        xi=0.5,
        rho=-0.7,
        n_paths=n_paths,
        n_steps=n_steps
    )

    european = EuropeanOption(K, "call")
    asian = AsianOption(K, "call")

    # paths
    gbm_paths = gbm_engine.generate_paths()
    heston_paths, _ = heston_engine.generate_paths()

    gbm_S_T = gbm_paths[:, -1]
    heston_S_T = heston_paths[:, -1]

    # benchmark
    bs_price = black_scholes(S0, K, T, r, sigma, "call")

    # payoffs
    gbm_euro = np.exp(-r * T) * european.payoff(gbm_S_T)
    heston_euro = np.exp(-r * T) * european.payoff(heston_S_T)

    gbm_asian = np.exp(-r * T) * asian.payoff(gbm_paths)
    heston_asian = np.exp(-r * T) * asian.payoff(heston_paths)

    # prices
    gbm_euro_price = np.mean(gbm_euro)
    heston_euro_price = np.mean(heston_euro)

    gbm_asian_price = np.mean(gbm_asian)
    heston_asian_price = np.mean(heston_asian)

    print("\n--- MODEL COMPARISON ---")
    print("GBM European:", gbm_euro_price)
    print("Heston European:", heston_euro_price)
    print("GBM Asian:", gbm_asian_price)
    print("Heston Asian:", heston_asian_price)
    print("Black-Scholes:", bs_price)

    # European convergence
    sizes_e, euro_conv, euro_discounted = convergence_study_european(
        gbm_engine, european, paths=gbm_paths
    )

    bs_line = np.full_like(sizes_e, bs_price, dtype=float)

    sigma_euro = np.std(euro_discounted)
    stderr_e = sigma_euro / np.sqrt(sizes_e)
    upper_e = euro_conv + 1.96 * stderr_e
    lower_e = euro_conv - 1.96 * stderr_e

    plt.figure(figsize=(14, 9))
    plt.plot(sizes_e, euro_conv, label="MC Estimate", color='blue', linewidth=3)
    plt.plot(sizes_e, bs_line, "--", label="Black-Scholes", color='black', alpha=0.8)
    plt.fill_between(sizes_e, lower_e, upper_e, color='blue', alpha=0.2,
                     label="95% Confidence Interval")

    plt.title("European Option: Monte Carlo Convergence Analysis", fontsize=24, fontweight='bold')
    plt.xlabel("Number of Paths (N)", fontsize=20)
    plt.ylabel("Option Price", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("figures/european_convergence.png", dpi=300)

    # Asian convergence
    sizes_a, asian_conv = convergence_study_asian(
        gbm_engine, asian, paths=gbm_paths
    )

    asian_cv = apply_control_variate(
        payoff=gbm_asian,
        control=gbm_euro,
        control_exact=bs_price
    )

    asian_cv_conv = np.cumsum(asian_cv) / np.arange(1, n_paths + 1)
    asian_cv_estimates = asian_cv_conv[sizes_a - 1]

    final_asian_price = np.mean(asian_cv)
    asian_line = np.full_like(sizes_a, final_asian_price, dtype=float)

    sigma_asian_cv = np.std(asian_cv)
    stderr_a = sigma_asian_cv / np.sqrt(sizes_a)
    upper_a = asian_cv_estimates + 1.96 * stderr_a
    lower_a = asian_cv_estimates - 1.96 * stderr_a

    plt.figure(figsize=(14, 9))
    plt.plot(sizes_a, asian_conv, label="Standard MC", color='red', linestyle='--', linewidth=2)
    plt.plot(sizes_a, asian_cv_estimates, label="Control Variate MC", color='green', linewidth=3)
    plt.plot(sizes_a, asian_line, "--", label="Converged Price", color='black', alpha=0.7)
    plt.fill_between(sizes_a, lower_a, upper_a, color='green', alpha=0.2,
                     label="CV 95% Confidence Interval")

    plt.title("Asian Option: Variance Reduction Performance", fontsize=24, fontweight='bold')
    plt.xlabel("Number of Paths (N)", fontsize=20)
    plt.ylabel("Option Price", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("figures/asian_convergence.png", dpi=300)

    # Heston vs GBM
    plt.figure(figsize=(12, 8))

    plt.hist(gbm_S_T, bins=120, density=True, alpha=0.5,
             label="GBM $S_T$", color="blue")
    plt.hist(heston_S_T, bins=120, density=True, alpha=0.5,
             label="Heston $S_T$", color="orange")

    plt.title("Terminal Asset Distribution: GBM vs Heston", fontsize=22, fontweight='bold')
    plt.xlabel("$S_T$", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/heston_vs_gbm_distribution.png", dpi=300)


if __name__ == "__main__":
    main()
