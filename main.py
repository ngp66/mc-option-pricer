import numpy as np
import matplotlib.pyplot as plt
import os

from models.monte_carlo import MonteCarloEngine
from models.black_scholes import black_scholes
from options.european import EuropeanOption
from options.asian import AsianOption

np.random.seed(42)


def convergence_study_european(engine, option, paths):
    n_total = paths.shape[0]
    step = 500
    sizes = np.arange(step, n_total + 1, step, dtype=int)

    S_T = paths[:, -1]
    payoff = option.payoff(S_T)
    discounted = np.exp(-engine.r * engine.T) * payoff

    cumulative_means = np.cumsum(discounted) / np.arange(1, len(discounted) + 1)
    estimates = cumulative_means[sizes - 1]

    return sizes, estimates


def convergence_study_asian(engine, option, paths):
    n_total = paths.shape[0]
    step = 500
    sizes = np.arange(step, n_total + 1, step, dtype=int)

    payoff = option.payoff(paths)
    discounted = np.exp(-engine.r * engine.T) * payoff

    cumulative_means = np.cumsum(discounted) / np.arange(1, len(discounted) + 1)
    estimates = cumulative_means[sizes - 1]

    return sizes, estimates


def run_engine(engine):
    return engine.generate_paths()


def main():
    os.makedirs("figures", exist_ok=True)

    S0 = 100
    K = 100
    T = 1.0
    r = 0.05

    n_paths = 100000
    n_steps = 100

    gbm_engine = MonteCarloEngine(
        S0, T, r,
        sigma=0.2,
        n_paths=n_paths,
        n_steps=n_steps
    )

    heston_engine = MonteCarloEngine(
        S0, T, r,
        sigma=0.2,
        n_paths=n_paths,
        n_steps=n_steps,
        model="heston",
        model_params={
            "v0": 0.04,
            "kappa": 2.0,
            "theta": 0.04,
            "xi": 0.5,
            "rho": -0.7
        }
    )

    european = EuropeanOption(K, "call")
    asian = AsianOption(K, "call")

    gbm_paths = run_engine(gbm_engine)
    heston_paths = run_engine(heston_engine)

    gbm_S_T = gbm_paths[:, -1]
    heston_S_T = heston_paths[:, -1]

    bs_price = black_scholes(S0, K, T, r, gbm_engine.sigma, "call")

    gbm_euro = np.exp(-r * T) * european.payoff(gbm_S_T)
    heston_euro = np.exp(-r * T) * european.payoff(heston_S_T)

    gbm_asian = np.exp(-r * T) * asian.payoff(gbm_paths)
    heston_asian = np.exp(-r * T) * asian.payoff(heston_paths)

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

    sizes, gbm_conv = convergence_study_european(gbm_engine, european, gbm_paths)
    _, heston_conv = convergence_study_european(heston_engine, european, heston_paths)

    bs_line = np.full_like(sizes, bs_price, dtype=float)

    plt.figure(figsize=(12, 8))
    plt.plot(sizes, gbm_conv, label="GBM MC", linewidth=2.5)
    plt.plot(sizes, heston_conv, label="Heston MC", linewidth=2.5)
    plt.plot(sizes, bs_line, "--", label="Black-Scholes", color="black", alpha=0.8)

    plt.title("European Option: GBM vs Heston Convergence")
    plt.xlabel("Number of Paths")
    plt.ylabel("Option Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/european_gbm_vs_heston.png", dpi=300)

    sizes_a, gbm_asian_conv = convergence_study_asian(gbm_engine, asian, gbm_paths)
    _, heston_asian_conv = convergence_study_asian(heston_engine, asian, heston_paths)

    plt.figure(figsize=(12, 8))
    plt.plot(sizes_a, gbm_asian_conv, label="GBM MC", linewidth=2)
    plt.plot(sizes_a, heston_asian_conv, label="Heston MC", linewidth=2)

    plt.title("Asian Option: GBM vs Heston Convergence")
    plt.xlabel("Number of Paths")
    plt.ylabel("Option Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/asian_gbm_vs_heston.png", dpi=300)


if __name__ == "__main__":
    main()