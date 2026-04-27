# Monte Carlo Option Pricing Framework

A Python Monte Carlo simulation framework for pricing European and path-dependent options, with variance reduction techniques and statistical validation tools.

The project is designed to demonstrate numerical methods used in quantitative finance, including stochastic simulation, derivative pricing, and Monte Carlo convergence analysis.

---

## 📌 Features

### 🧮 Option Pricing
- European call/put options (Black–Scholes benchmark available)
- Asian options (arithmetic and geometric averaging)

### 📊 Monte Carlo Simulation
- Geometric Brownian Motion (GBM) path generation
- Flexible multi-path, multi-step simulation engine
- Risk-neutral pricing framework

### ⚙️ Variance Reduction
- Control variates (European ↔ Asian correlation)
- Extensible design for additional techniques

### 📉 Statistical Analysis
- Standard error estimation
- Confidence intervals
- Benchmark error evaluation

### 📈 Convergence Analysis
- Monte Carlo convergence visualization
- Empirical verification of \( O(1/\sqrt{N}) \) behavior

---

## 🧠 Mathematical Model

Asset dynamics under risk-neutral measure:

\[
dS_t = r S_t dt + \sigma S_t dW_t
\]

Discretization:

\[
S_{t+\Delta t} = S_t \exp\left((r - \frac{1}{2}\sigma^2)\Delta t + \sigma \sqrt{\Delta t} Z\right)
\]

Option pricing:

\[
V = e^{-rT} \mathbb{E}[\text{payoff}]
\]

---

## 📁 Project Structure
mc_option_pricer/
│
├── models/
│   ├── monte_carlo.py        # GBM simulation engine
│   ├── black_scholes.py      # Analytical Black–Scholes formula
│   ├── variance_reduction.py # Control variates & variance reduction tools
│
├── options/
│   ├── european.py           # European call/put payoff functions
│   ├── asian.py              # Asian (path-dependent) payoff functions
│
├── utils/
│   ├── statistics.py         # Confidence intervals, stderr, error metrics
│
├── main.py                   # Full experiment runner + convergence study
└── README.md

---

## 🚀 Installation

### Requirements

```bash
pip install numpy scipy matplotlib