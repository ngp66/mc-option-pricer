import numpy as np


class MonteCarloEngine:
    def __init__(self, S0, T, r, sigma, n_paths=10000, n_steps=100, model="gbm", model_params=None):
        self.S0 = S0
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.model = model
        self.model_params = model_params or {}

    def generate_paths(self):
        if self.model == "gbm":
            paths = np.zeros((self.n_paths, self.n_steps + 1))
            paths[:, 0] = self.S0

            Z = np.random.standard_normal((self.n_paths, self.n_steps))

            drift = (self.r - 0.5 * self.sigma**2) * self.dt
            diffusion = self.sigma * np.sqrt(self.dt) * Z

            log_returns = drift + diffusion
            paths[:, 1:] = self.S0 * np.exp(np.cumsum(log_returns, axis=1))

            return paths

        elif self.model == "heston":
            S0 = self.S0
            r = self.r
            dt = self.dt
            n_paths = self.n_paths
            n_steps = self.n_steps

            v0 = self.model_params["v0"]
            kappa = self.model_params["kappa"]
            theta = self.model_params["theta"]
            xi = self.model_params["xi"]
            rho = self.model_params["rho"]

            S = np.zeros((n_paths, n_steps + 1))
            v = np.zeros((n_paths, n_steps + 1))

            S[:, 0] = S0
            v[:, 0] = v0

            Z1 = np.random.standard_normal((n_paths, n_steps))
            Z2 = np.random.standard_normal((n_paths, n_steps))

            W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

            for t in range(n_steps):
                v[:, t] = np.maximum(v[:, t], 0)

                S[:, t + 1] = S[:, t] * np.exp(
                    (r - 0.5 * v[:, t]) * dt + np.sqrt(v[:, t] * dt) * Z1[:, t]
                )

                v[:, t + 1] = np.abs(
                    v[:, t]
                    + kappa * (theta - v[:, t]) * dt
                    + xi * np.sqrt(v[:, t] * dt) * W2[:, t]
                )

            return S

        else:
            raise ValueError("Unknown model type")

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