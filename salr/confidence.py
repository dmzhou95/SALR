
import numpy as np


def solve_confidence(B, E, gamma):
    """E: (N, V) alignment errors -> S (N, V), eta (N,)."""
    y = -E / (2.0 * gamma)
    u = B.sort_desc(y, 1)
    css = B.cumsum(u, 1)
    V = E.shape[1]
    j = B.asarray(np.arange(1, V + 1))[None, :]
    cond = (u - (css - 1.0) / j) > 0
    rho = B.count_true(cond, 1)                       # >= 1 always
    tau = (B.gather(css, (rho - 1)[:, None])[:, 0] - 1.0) / rho
    S = B.maximum(y - tau[:, None], 0.0)
    eta = -2.0 * gamma * tau
    return S, eta


def solve_bisection(E, gamma, iters=100):
    """Reference solver (numpy): bisection for eta_i, as in the proof of Theorem 2."""
    E = np.asarray(E, dtype=np.float64)
    lo = E.min(1)
    hi = lo + 2.0 * gamma
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        tot = np.maximum(0.0, (mid[:, None] - E) / (2.0 * gamma)).sum(1)
        lo = np.where(tot < 1.0, mid, lo)
        hi = np.where(tot >= 1.0, mid, hi)
    eta = 0.5 * (lo + hi)
    return np.maximum(0.0, (eta[:, None] - E) / (2.0 * gamma)), eta


def average_weights(B, n, V):
    """Average Weight ablation (#5): s_i^(v) = 1/V."""
    return B.full((n, V), 1.0 / V)


def binary_weights(B, mask):
    """Binary Weight Matrix ablation (#6): weight 1 on observed views, 0 on imputed ones."""
    m = B.asarray(mask)
    return m / m.sum(1)[:, None]
