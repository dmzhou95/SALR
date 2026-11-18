"""Unit tests for the SALR implementation.

Run:  python tests/test_core.py        (or: python -m pytest tests -q)
The numpy backend is always tested; if PyTorch is installed, the torch backend is
additionally checked against numpy on the same inputs (CPU, float64).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from salr.backend import NumpyBackend                                   # noqa: E402
from salr import graph as G                                             # noqa: E402
from salr.confidence import solve_confidence, solve_bisection           # noqa: E402
from salr.impute import impute_from_z                                   # noqa: E402
from salr.masks import mcar_mask, view_imbalanced_mask, class_dependent_mask   # noqa: E402
from salr.metrics import cluster_acc                                    # noqa: E402
from salr.spectral import random_init                                  # noqa: E402

B = NumpyBackend()
RNG = np.random.default_rng(0)


def _dense_hyper(g, n):
    H = np.zeros((n, n))
    for e in range(n):
        H[g.idx[e], e] = 1.0
    W = np.diag(g.w)
    De = np.diag(H.sum(0))
    Dv = np.diag((H * g.w[None, :]).sum(1))
    return H, W, De, Dv


def test_hypergraph_theorem1():
    n, k, d = 60, 5, 4
    F = RNG.normal(size=(n, 7))
    g = G.build_hypergraph(B, F, k)
    H, W, De, Dv = _dense_hyper(g, n)
    assert np.allclose(np.diag(De), k + 1)                       # every hyperedge has k+1 vertices
    assert np.allclose(np.diag(Dv), g.deg)
    L = Dv - H @ W @ np.linalg.inv(De) @ H.T                      # Eq. (2)
    Z = RNG.normal(size=(n, d))
    assert np.allclose(G.laplacian_matvec(B, g, Z), L @ Z)
    brute = 0.0                                                   # Eq. (11): weighted pairwise distances
    for e in range(n):
        mem = g.idx[e]
        for i in mem:
            for j in mem:
                brute += 0.5 * g.w[e] / (k + 1) * np.sum((Z[i] - Z[j]) ** 2)
    assert np.isclose(np.trace(Z.T @ L @ Z), brute)
    assert np.isclose(G.hc_loss(B, [g], Z), brute)
    assert brute >= 0
    assert np.allclose(L @ np.ones(n), 0)                         # L 1 = 0
    ev = np.linalg.eigvalsh(0.5 * (L + L.T))
    assert ev.min() > -1e-10 and G.lambda_bound(B, [g]) >= ev.max() - 1e-10
    assert np.allclose(G.to_scipy_laplacian(g).toarray(), L)
    assert ev.max() - 1e-9 <= G.lambda_power(B, [g], iters=60) <= G.lambda_bound(B, [g]) + 1e-9


def test_pairwise_graph():
    n, k = 50, 4
    F = RNG.normal(size=(n, 5))
    g = G.build_pairwise(B, F, k)
    A0 = np.zeros((n, n))
    for i in range(n):
        A0[i, g.idx[i]] = g.w[i]
    A = 0.5 * (A0 + A0.T)
    L = np.diag(A.sum(1)) - A
    Z = RNG.normal(size=(n, 3))
    assert np.allclose(G.laplacian_matvec(B, g, Z), L @ Z)
    assert np.allclose(G.to_scipy_laplacian(g).toarray(), L)


def test_knn_exact():
    F = RNG.normal(size=(80, 6))
    d2, nb = G.knn(B, F, F, 5, exclude_self=True, chunk=17)
    D = ((F[:, None] - F[None]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    assert np.array_equal(np.sort(nb, 1), np.sort(np.argsort(D, 1)[:, :5], 1))
    assert np.allclose(d2, np.sort(D, 1)[:, :5])


def test_confidence_theorem2():
    from scipy.optimize import minimize
    for V in (2, 3, 6):
        E = RNG.gamma(2.0, 1.0, size=(200, V)) * RNG.uniform(0.01, 1, size=(200, 1))
        for gamma in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            S, eta = solve_confidence(B, E, gamma)
            Sb, etab = solve_bisection(E, gamma)
            assert np.allclose(S.sum(1), 1) and (S >= 0).all()
            assert np.allclose(S, Sb, atol=1e-8)
            assert np.allclose(eta, etab, atol=1e-7)
            assert ((E >= eta[:, None] - 1e-12) <= (S <= 1e-12)).all()   # e >= eta  ->  s = 0
            assert np.allclose(S, np.maximum(0, (eta[:, None] - E) / (2 * gamma)))  # Eq. (19)
            # larger error -> smaller (or equal) confidence
            o = np.argsort(E, 1)
            assert (np.diff(np.take_along_axis(S, o, 1), axis=1) <= 1e-12).all()
        # global optimum vs a generic constrained solver on a few rows
        for i in range(5):
            e, gamma = E[i], 0.05
            f = lambda s: s @ e + gamma * s @ s
            res = minimize(f, np.full(V, 1 / V), method="SLSQP", bounds=[(0, None)] * V,
                           constraints=[{"type": "eq", "fun": lambda s: s.sum() - 1}], tol=1e-12)
            s_star = solve_confidence(B, e[None], gamma)[0][0]
            assert f(s_star) <= res.fun + 1e-9
    S, _ = solve_confidence(B, np.ones((4, 5)), 0.3)                       # equal errors -> 1/V
    assert np.allclose(S, 0.2)


def test_gpi_monotone():
    n, k, d, alpha = 120, 6, 8, 0.7
    gs = [G.build_hypergraph(B, RNG.normal(size=(n, 9)), k) for _ in range(3)]
    lam = G.lambda_bound(B, gs)
    M = RNG.normal(size=(n, d)) / np.sqrt(n)
    Z = random_init(n, d, 1)
    lap = lambda Z: sum(G.laplacian_matvec(B, g, Z) for g in gs)
    obj = lambda Z: np.sum(Z * lap(Z)) - 2 * alpha * np.sum(Z * M)
    prev = obj(Z)
    for _ in range(30):
        P, _, Qt = np.linalg.svd(lam * Z - lap(Z) + alpha * M, full_matrices=False)   # Eq. (21)
        Z = P @ Qt
        cur = obj(Z)
        assert cur <= prev + 1e-10
        assert np.allclose(Z.T @ Z, np.eye(d), atol=1e-10)
        prev = cur


def test_imputation_eq22():
    n, k = 40, 3
    Z = RNG.normal(size=(n, 4))
    X = RNG.normal(size=(n, 6))
    m = np.ones(n, dtype=bool)
    m[[1, 5, 9]] = False
    oi, mi = np.flatnonzero(m), np.flatnonzero(~m)
    Xb = impute_from_z(B, Z, [X], [m.astype(float)], k, [oi], [mi])[0]
    for i in mi:
        d = ((Z[oi] - Z[i]) ** 2).sum(1)
        assert np.allclose(Xb[i], X[oi[np.argsort(d)[:k]]].mean(0))
    assert np.allclose(Xb[oi], X[oi])


def test_masks():
    rng = np.random.default_rng(3)
    for V in (2, 3, 6):
        m = mcar_mask(1000, V, 0.6, rng)
        assert (m.sum(1) >= 1).all()
        assert abs((m.sum(1) < V).mean() - 0.6) < 1e-9
    m = view_imbalanced_mask(5000, 3, rng)
    assert (m.sum(1) >= 1).all() and abs(1 - m[:, 0].mean() - 0.7) < 0.03
    y = rng.integers(0, 10, 4000)
    m = class_dependent_mask(y, 3, rng)
    rates = sorted((m[y == c].sum(1) < 3).mean() for c in range(10))
    assert np.allclose(rates[:5], 0.1, atol=0.02) and np.allclose(rates[5:], 0.7, atol=0.02)


def test_acc():
    y = np.repeat(np.arange(4), 25)
    perm = np.array([2, 0, 3, 1])
    assert cluster_acc(y, perm[y]) == 1.0


def test_end_to_end_and_no_leakage():
    from salr import SALR, SALRConfig, make_toy, standardize_views, clustering_metrics
    views, y = make_toy(n=600, K=4, dims=(20, 30, 15), seed=1)
    mask = mcar_mask(y.size, 3, 0.4, np.random.default_rng(1))
    X = standardize_views(views, mask)
    cfg = SALRConfig(backend="numpy", d=16, k=7, hidden=(64, 64), max_iter=20, min_iter=20,
                     warmup_epochs=10, theta_epochs=2, batch_size=128, alpha=1.0, gamma=1e-3,
                     kmeans_n_init=5, verbose=False, seed=0)
    out = SALR(cfg).fit(X, mask, n_clusters=4)
    m = clustering_metrics(y, out["labels"])
    assert m["ACC"] > 85, m
    assert np.allclose(out["Z"].T @ out["Z"], np.eye(16), atol=1e-8)
    assert np.allclose(out["S"].sum(1), 1)
    # garbage in the missing rows must not change anything
    X2 = [x.copy() for x in X]
    for v in range(3):
        X2[v][mask[:, v] == 0] = 1e3 * np.random.default_rng(v).normal(size=X2[v][mask[:, v] == 0].shape)
    out2 = SALR(cfg).fit(X2, mask, n_clusters=4)
    assert np.allclose(out["Z"], out2["Z"]) and np.array_equal(out["labels"], out2["labels"])


def test_torch_matches_numpy():
    try:
        import torch
    except ImportError:
        print("  (torch not installed: torch-vs-numpy test skipped)")
        return
    from salr.backend import TorchBackend
    T = TorchBackend(device="cpu", dtype=torch.float64)
    n, k = 300, 6
    F = RNG.normal(size=(n, 10))
    Z = random_init(n, 5, 2)
    for build in (G.build_hypergraph, G.build_pairwise):
        gn, gt = build(B, F, k), build(T, T.asarray(F), k)
        assert np.array_equal(gn.idx, T.to_numpy(gt.idx))
        assert np.allclose(gn.deg, T.to_numpy(gt.deg))
        assert np.allclose(G.laplacian_matvec(B, gn, Z), T.to_numpy(G.laplacian_matvec(T, gt, T.asarray(Z))))
    E = RNG.gamma(2.0, 1.0, size=(100, 4))
    assert np.allclose(solve_confidence(B, E, 0.1)[0], T.to_numpy(solve_confidence(T, T.asarray(E), 0.1)[0]))
    X = RNG.normal(size=(n, 7))
    m = np.ones(n); m[::7] = 0
    oi, mi = np.flatnonzero(m > 0), np.flatnonzero(m == 0)
    a = impute_from_z(B, Z, [X], [m], 4, [oi], [mi])[0]
    b = impute_from_z(T, T.asarray(Z), [T.asarray(X)], [T.asarray(m)], 4, [T.asindex(oi)], [T.asindex(mi)])[0]
    assert np.allclose(a, T.to_numpy(b))


def test_torch_end_to_end():
    try:
        import torch
    except ImportError:
        print("  (torch not installed: torch end-to-end test skipped)")
        return
    from salr import SALR, SALRConfig, make_toy, standardize_views, clustering_metrics
    views, y = make_toy(n=600, K=4, dims=(20, 30, 15), seed=1)
    mask = mcar_mask(y.size, 3, 0.4, np.random.default_rng(1))
    X = standardize_views(views, mask)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for variant in ("full", "pwg", "bwm"):
        cfg = SALRConfig(backend="torch", device=dev, d=16, k=7, hidden=(64, 64), max_iter=20, min_iter=20,
                         warmup_epochs=10, theta_epochs=2, batch_size=128, gamma=1e-3, kmeans_n_init=5,
                         verbose=False, variant=variant)
        out = SALR(cfg).fit(X, mask, n_clusters=4)
        m = clustering_metrics(y, out["labels"])
        assert np.allclose(out["Z"].T @ out["Z"], np.eye(16), atol=1e-4)
        assert m["ACC"] > 80, (variant, m)
        print(f"  torch ({dev}) {variant}: ACC={m['ACC']:.1f}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
