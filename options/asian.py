import numpy as np


class AsianOption:
    def __init__(self, strike: float, option_type: str = "call", averaging: str = "arithmetic"):
        self.K = strike
        self.option_type = option_type
        self.averaging = averaging

        if option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")

        if averaging not in ["arithmetic", "geometric"]:
            raise ValueError("averaging must be 'arithmetic' or 'geometric'")

    def _average(self, paths: np.ndarray) -> np.ndarray:
        if self.averaging == "arithmetic":
            return np.mean(paths, axis=1)
        return np.exp(np.mean(np.log(paths), axis=1))

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        avg = self._average(paths)

        if self.option_type == "call":
            return np.maximum(avg - self.K, 0.0)
        return np.maximum(self.K - avg, 0.0)

    def discounted_payoff(self, paths: np.ndarray, r: float, T: float) -> np.ndarray:
        return np.exp(-r * T) * self.payoff(paths)