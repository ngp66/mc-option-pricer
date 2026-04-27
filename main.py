import numpy as np
import matplotlib.pyplot as plt

from models.monte_carlo import MonteCarloEngine
from models.black_scholes import black_scholes
from options.european import EuropeanOption
from options.asian import AsianOption
from models.variance_reduction import apply_control_variate
from utils.statistics import summary_stats


def convergence_study_european(engine, option, n_max=20000, step=500):
    sizes = list(range(step, n_max + 1, step))
    estimates = []

    for n in sizes:
        engine.n_paths = n
        paths = engine.generate_paths()

        S_T = paths[:, -1]

        payoff = option.payoff(S_T)
        discounted = np.exp(-engine.r * engine.T) * payoff

        estimates.append(np.mean(discounted))

    return sizes, np.array(estimates)

def convergence_study_asian(engine, option, n_max=20000, step=500):
    sizes = list(range(step, n_max + 1, step))
    estimates = []

    for n in sizes:
        engine.n_paths = n
        paths = engine.generate_paths()

        payoff = option.payoff(paths)
        discounted = np.exp(-engine.r * engine.T) * payoff

        estimates.append(np.mean(discounted))

    return sizes, np.array(estimates)


def main():
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2

    n_paths = 40000
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

    control_var = euro_payoff
    asian_adjusted = apply_control_variate(
        payoff=asian_payoff,
        control=control_var,
        control_exact=bs_price
    )

    asian_cv_discounted = np.exp(-r * T) * asian_adjusted

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

    # Convergence study
    sizes_e, euro_conv = convergence_study_european(engine, european, n_max = n_paths)

    plt.figure()
    plt.plot(sizes_e, euro_conv)
    plt.axhline(bs_price, linestyle="--")
    plt.title("European Option Convergence")
    plt.xlabel("Number of Paths")
    plt.ylabel("Price")

    plt.savefig("figures/european_convergence.png")

    sizes_a, asian_conv = convergence_study_asian(engine, asian, n_max = n_paths)

    plt.figure()
    plt.plot(sizes_a, asian_conv)
    plt.title("Asian Option Convergence")
    plt.xlabel("Number of Paths")
    plt.ylabel("Price")

    plt.savefig("figures/asian_convergence.png")


if __name__ == "__main__":
    main()