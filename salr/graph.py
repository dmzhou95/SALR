
from dataclasses import dataclass
import numpy as np


@dataclass
class Graph:
    kind: str            # "hyper" (SALR) or "pairwise" (PWG ablation, variant #4)
    idx: object          # hyper: members (N, k+1) incl. the centre; pairwise: neighbours (N, k)
    w: object            # hyper: hyperedge weights (N,); pairwise: edge weights (N, k)
    deg: object          # vertex degrees D_v(i, i) (N,)


def knn(B, query, ref, k, exclude_self=False, chunk=4096):
    """k nearest neighbours (squared Euclidean) of each query row among ref rows."""
    nq = query.shape[0]
    sq_r = (ref * ref).sum(1)
    out_d, out_i = [], []
    for s in range(0, nq, chunk):
        q = query[s:s + chunk]
        d = (q * q).sum(1)[:, None] + sq_r[None, :] - 2.0 * (q @ ref.T)
        d = B.clamp_min(d, 0.0)
        if exclude_self:
            r = B.arange(q.shape[0])
            d[r, r + s] = float("inf")
        v, i = B.topk_smallest(d, k)
        out_d.append(v)
        out_i.append(i)
    return _cat(B, out_d), _cat(B, out_i)


def _cat(B, xs):
    if B.name == "numpy":
        return np.concatenate(xs, 0)
    return B.torch.cat(xs, 0)


def build_hypergraph(B, F, k):
    n = F.shape[0]
    d2, nbr = knn(B, F, F, k, exclude_self=True)
    sigma2 = B.item(d2.mean()) + 1e-12
    w = B.exp(-d2 / sigma2).mean(1)                                   # W_v(e_i)
    centre = B.arange(n)[:, None]
    members = _cat_cols(B, centre, nbr)                               # (N, k+1)
    deg = B.zeros((n,))
    for c in range(members.shape[1]):                                 # D_v(i) = sum_e W(e) H(i, e)
        deg = B.index_add(deg, members[:, c], w)
    return Graph("hyper", members, w, deg)


def build_pairwise(B, F, k):
    """Point-wise k-NN graph used by the PWG ablation (#4): A = (A0 + A0^T) / 2."""
    n = F.shape[0]
    d2, nbr = knn(B, F, F, k, exclude_self=True)
    sigma2 = B.item(d2.mean()) + 1e-12
    a = B.exp(-d2 / sigma2)                                           # (N, k)
    deg = 0.5 * a.sum(1)
    for c in range(k):
        deg = B.index_add(deg, nbr[:, c], 0.5 * a[:, c])
    return Graph("pairwise", nbr, a, deg)


def _cat_cols(B, a, b):
    if B.name == "numpy":
        return np.concatenate([a, b], 1)
    return B.torch.cat([a, b], 1)


def adjacency_matvec(B, g, Z, chunk=16384):
    """Theta_v Z (hyper) or A Z (pairwise), without forming the N x N matrix."""
    out = B.zeros_like(Z)
    n = Z.shape[0]
    if g.kind == "hyper":
        k1 = g.idx.shape[1]
        for s in range(0, n, chunk):
            mem = g.idx[s:s + chunk]
            e = Z[mem].sum(1) * (g.w[s:s + chunk] / k1)[:, None]      # W D_e^{-1} H^T Z  (per hyperedge)
            for c in range(k1):
                out = B.index_add(out, mem[:, c], e)                  # H ( . )
        return out
    k = g.idx.shape[1]
    for s in range(0, n, chunk):
        nb, a = g.idx[s:s + chunk], g.w[s:s + chunk]
        zc = Z[s:s + chunk]
        out[s:s + chunk] = out[s:s + chunk] + 0.5 * (a[:, :, None] * Z[nb]).sum(1)     # A0 Z / 2
        for c in range(k):
            out = B.index_add(out, nb[:, c], 0.5 * a[:, c:c + 1] * zc)                # A0^T Z / 2
    return out


def laplacian_matvec(B, g, Z):
    return g.deg[:, None] * Z - adjacency_matvec(B, g, Z)


def hc_loss(B, graphs, Z):
    """L_hc(Z) = sum_v Tr(Z^T L_v Z)   (Eq. 7)."""
    return sum(B.item((Z * laplacian_matvec(B, g, Z)).sum()) for g in graphs)


def lambda_bound(B, graphs):
    """lambda = 2 max_i sum_v D_v(i, i) >= lambda_max(sum_v L_v)  (used in Eq. 21)."""
    tot = graphs[0].deg
    for g in graphs[1:]:
        tot = tot + g.deg
    return 2.0 * B.max(tot)


def lambda_power(B, graphs, iters=30, margin=1.05, seed=0):
    """Tighter alternative to `lambda_bound`: power-iteration estimate of lambda_max(sum_v L_v),
    inflated by `margin` (power iteration approaches lambda_max from below)."""
    n = graphs[0].deg.shape[0]
    x = B.randn((n, 1), np.random.default_rng(seed))
    lam = 0.0
    for _ in range(iters):
        y = laplacian_matvec(B, graphs[0], x)
        for g in graphs[1:]:
            y = y + laplacian_matvec(B, g, x)
        lam = B.item(B.sqrt((y * y).sum())) / max(B.item(B.sqrt((x * x).sum())), 1e-300)
        x = y / max(B.item(B.sqrt((y * y).sum())), 1e-300)
    return min(margin * lam, lambda_bound(B, graphs))


def graph_to_numpy(B, g):
    return Graph(g.kind, B.to_numpy(g.idx).astype(np.int64), B.to_numpy(g.w).astype(np.float64),
                 B.to_numpy(g.deg).astype(np.float64))


def to_scipy_laplacian(g):
    """Explicit sparse L_v (numpy graph), used only for the spectral initialisation."""
    import scipy.sparse as sp
    n = g.deg.shape[0]
    if g.kind == "hyper":
        k1 = g.idx.shape[1]
        rows = g.idx.reshape(-1)
        cols = np.repeat(np.arange(n), k1)
        H = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
        theta = H @ sp.diags(g.w / k1) @ H.T
    else:
        k = g.idx.shape[1]
        A0 = sp.csr_matrix((g.w.reshape(-1), (np.repeat(np.arange(n), k), g.idx.reshape(-1))), shape=(n, n))
        theta = 0.5 * (A0 + A0.T)
    return (sp.diags(g.deg) - theta).tocsr()
