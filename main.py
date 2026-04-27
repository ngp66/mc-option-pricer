import numpy as np
import matplotlib.pyplot as plt
import os

from models.monte_carlo import MonteCarloEngine
from models.black_scholes import black_scholes
from options.european import EuropeanOption
from options.asian import AsianOption
from models.variance_reduction import apply_control_variate
from utils.statistics import summary_stats

np.random.seed(42)

def convergence_study_european(engine, option, paths=None, n_max=20000, step=500):
    if paths is None:
        paths = engine.generate_paths()
    
    n_total = paths.shape[0]
    sizes = np.arange(step, n_total + 1, step, dtype=int)

    S_T = paths[:, -1]
    payoff = option.payoff(S_T)
    discounted = np.exp(-engine.r * engine.T) * payoff

    cumulative_means = np.cumsum(discounted) / np.arange(1, len(discounted) + 1)
    estimates = cumulative_means[sizes - 1]

    return sizes, estimates


def convergence_study_asian(engine, option, paths=None, n_max=20000, step=500):
    if paths is None:
        paths = engine.generate_paths()
    
    n_total = paths.shape[0]
    sizes = np.arange(step, n_total + 1, step, dtype=int)

    payoff = option.payoff(paths)
    discounted = np.exp(-engine.r * engine.T) * payoff

    cumulative_means = np.cumsum(discounted) / np.arange(1, len(discounted) + 1)
    estimates = cumulative_means[sizes - 1]

    return sizes, estimates


def main():
    os.makedirs("figures", exist_ok=True)

    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2

    n_paths = 100000
    n_steps = 100

    engine = MonteCarloEngine(S0, T, r, sigma, n_paths, n_steps)

    european = EuropeanOption(K, "call")
    asian = AsianOption(K, "call")

    paths = engine.generate_paths()
    S_T = paths[:, -1]

    bs_price = black_scholes(S0, K, T, r, sigma, "call")

    euro_payoff = european.payoff(S_T)
    euro_discounted = np.exp(-r * T) * euro_payoff

    asian_payoff = asian.payoff(paths)
    asian_discounted = np.exp(-r * T) * asian_payoff

    euro_price = np.mean(euro_discounted)
    asian_price = np.mean(asian_discounted)

    asian_cv_discounted = apply_control_variate(
        payoff=asian_discounted,
        control=euro_discounted,
        control_exact=bs_price
    )

    euro_stats = summary_stats(euro_discounted, benchmark=bs_price)
    asian_stats = summary_stats(asian_discounted)
    asian_cv_stats = summary_stats(asian_cv_discounted)

    print("\n--- European Option ---")
    print("MC Price:", euro_price)
    print("Black-Scholes:", bs_price)
    print("Stats:", euro_stats)

    print("\n--- Asian Option ---")
    print("MC Price:", asian_price)
    print("Stats:", asian_stats)

    print("\n--- Asian (Control Variate) ---")
    print("MC Price:", np.mean(asian_cv_discounted))
    print("Stats:", asian_cv_stats)

    # European Plot
    sizes_e, euro_conv = convergence_study_european(engine, european, paths=paths, n_max=n_paths)
    bs_line = np.full_like(sizes_e, bs_price, dtype=float)
    
    sigma_euro = np.std(euro_discounted)
    stderr_e = sigma_euro / np.sqrt(sizes_e)
    upper_e = euro_conv + 1.96 * stderr_e
    lower_e = euro_conv - 1.96 * stderr_e

    plt.figure(figsize=(12, 8))
    plt.plot(sizes_e, euro_conv, label="MC Estimate", color='tab:blue', linewidth=2.5, zorder=3)
    plt.plot(sizes_e, bs_line, "--", label="Black-Scholes", color='black', alpha=0.8, zorder=2)
    plt.fill_between(sizes_e, lower_e, upper_e, color='tab:blue', alpha=0.2, label="95% Confidence Interval", zorder=1)
    
    errors_e = np.abs(euro_conv - bs_price)
    valid_e = (errors_e > 1e-9) & (sizes_e > 1000)
    rate_e = np.polyfit(np.log(sizes_e[valid_e]), np.log(errors_e[valid_e]), 1)
    
    plt.text(0.95, 0.05, f"$\mathrm{{Error}} \propto N^{{{rate_e[0]:.2f}}}$", 
             transform=plt.gca().transAxes, fontsize=18, verticalalignment='bottom', 
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.title("European Option: Monte Carlo Convergence Analysis", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Number of Paths ($N$)", fontsize=18)
    plt.ylabel("Option Price", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper right', fontsize=16, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("figures/european_convergence.png", dpi=300)

    # Asian Plot
    sizes_a, asian_conv = convergence_study_asian(engine, asian, paths=paths, n_max=n_paths)
    asian_cv_conv = np.cumsum(asian_cv_discounted) / np.arange(1, n_paths + 1)
    asian_cv_estimates = asian_cv_conv[sizes_a - 1]
    
    final_asian_price = np.mean(asian_cv_discounted)
    asian_line = np.full_like(sizes_a, final_asian_price, dtype=float)

    sigma_asian_cv = np.std(asian_cv_discounted)
    stderr_a = sigma_asian_cv / np.sqrt(sizes_a)
    upper_a = asian_cv_estimates + 1.96 * stderr_a
    lower_a = asian_cv_estimates - 1.96 * stderr_a

    plt.figure(figsize=(12, 8))
    # Redder color and dashed line for high contrast against the green CV line
    plt.plot(sizes_a, asian_conv, label="Standard MC", color='#d62728', alpha=0.8, linewidth=2, linestyle='--', zorder=1)
    plt.plot(sizes_a, asian_cv_estimates, label="Control Variate MC", color='#2ca02c', linewidth=3, zorder=3)
    plt.plot(sizes_a, asian_line, "--", label="Converged Price", color='black', alpha=0.7, zorder=2)
    plt.fill_between(sizes_a, lower_a, upper_a, color='#2ca02c', alpha=0.2, label="CV 95% Confidence Interval", zorder=2)

    errors_a = np.abs(asian_cv_estimates - final_asian_price)
    valid_a = (errors_a > 1e-9) & (sizes_a > 1000)
    rate_a = np.polyfit(np.log(sizes_a[valid_a]), np.log(errors_a[valid_a]), 1)
    
    plt.text(0.95, 0.05, f"$\mathrm{{CV\ Error}} \propto N^{{{rate_a[0]:.2f}}}$", 
             transform=plt.gca().transAxes, fontsize=18, verticalalignment='bottom', 
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.title("Asian Option: Variance Reduction Performance", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Number of Paths ($N$)", fontsize=18)
    plt.ylabel("Option Price", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='upper right', fontsize=16, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("figures/asian_convergence.png", dpi=300)


if __name__ == "__main__":
    main()