import numpy as np


class MonteCarloEngine:
    def __init__(self, S0, T, r, sigma, n_paths=10000, n_steps=100):
        self.S0 = S0
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = T / n_steps

    def generate_paths(self):
        paths = np.zeros((self.n_paths, self.n_steps + 1))
        paths[:, 0] = self.S0

        for t in range(1, self.n_steps + 1):
            Z = np.random.standard_normal(self.n_paths)

            paths[:, t] = paths[:, t - 1] * np.exp(
                (self.r - 0.5 * self.sigma**2) * self.dt +
                self.sigma * np.sqrt(self.dt) * Z
            )

        return paths

    def price(self, payoff_fn):
        paths = self.generate_paths()

        payoff = payoff_fn(paths)
        payoff = np.asarray(payoff)

        if payoff.ndim != 1:
            raise ValueError("Payoff function must return 1D array of shape (n_paths,)")

        discounted = np.exp(-self.r * self.T) * payoff

        price = np.mean(discounted)
        stderr = np.std(discounted, ddof=1) / np.sqrt(len(discounted))

        return price, stderr