import numpy as np

def heston_paths(S0, v0, r, kappa, theta, xi, rho, T, N, M):
    dt = T / N

    S = np.zeros((M, N+1))
    v = np.zeros((M, N+1))

    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(N):
        Z1 = np.random.normal(size=M)
        Z2 = np.random.normal(size=M)

        Z_v = Z1
        Z_s = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        v[:, t+1] = v[:, t] + kappa*(theta - v[:, t])*dt + xi*np.sqrt(np.maximum(v[:, t], 0))*np.sqrt(dt)*Z_v
        v[:, t+1] = np.maximum(v[:, t+1], 0)

        S[:, t+1] = S[:, t] * np.exp(
            (r - 0.5*v[:, t])*dt + np.sqrt(np.maximum(v[:, t], 0))*np.sqrt(dt)*Z_s
        )

    return S, v