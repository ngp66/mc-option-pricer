import numpy as np


class HestonEngine:
    def __init__(self, S0, v0, T, r, kappa, theta, xi, rho, n_paths=10000, n_steps=100):
        self.S0 = S0
        self.v0 = v0
        self.T = T
        self.r = r

        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = T / n_steps

    def generate_paths(self):
        S = np.zeros((self.n_paths, self.n_steps + 1))
        v = np.zeros((self.n_paths, self.n_steps + 1))

        S[:, 0] = self.S0
        v[:, 0] = self.v0

        for t in range(self.n_steps):
            Z1 = np.random.normal(size=self.n_paths)
            Z2 = np.random.normal(size=self.n_paths)

            Z_v = Z1
            Z_s = self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * Z2

            v_t = np.maximum(v[:, t], 0)

            v[:, t + 1] = (
                v[:, t]
                + self.kappa * (self.theta - v[:, t]) * self.dt
                + self.xi * np.sqrt(v_t * self.dt) * Z_v
            )
            v[:, t + 1] = np.maximum(v[:, t + 1], 0)

            S[:, t + 1] = S[:, t] * np.exp(
                (self.r - 0.5 * v_t) * self.dt
                + np.sqrt(v_t * self.dt) * Z_s
            )

        return S, v