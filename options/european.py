import numpy as np

class EuropeanOption:
    def __init__(self, strike: float, option_type: str = "call"):
        self.K = strike

        if option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")

        self.option_type = option_type

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        S_T = np.asarray(S_T)

        if self.option_type == "call":
            return np.maximum(S_T - self.K, 0.0)
        return np.maximum(self.K - S_T, 0.0)

    def discounted_payoff(self, S_T: np.ndarray, r: float, T: float) -> np.ndarray:
        return np.exp(-r * T) * self.payoff(S_T)