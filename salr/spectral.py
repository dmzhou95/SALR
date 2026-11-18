
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

from .graph import to_scipy_laplacian


def spectral_init(np_graphs, d, seed=0, tol=1e-4, maxiter=None):
    L = to_scipy_laplacian(np_graphs[0])
    for g in np_graphs[1:]:
        L = L + to_scipy_laplacian(g)
    n = L.shape[0]
    lam = 2.0 * float(sum(g.deg for g in np_graphs).max())
    A = (sp.identity(n, format="csr") * lam - L).tocsr()
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n)
    try:
        vals, vecs = eigsh(A, k=d, which="LA", tol=tol, v0=v0, maxiter=maxiter or 20 * n)
    except ArpackNoConvergence as err:
        warnings.warn(f"ARPACK did not converge ({err.eigenvectors.shape[1]} of {d} vectors); "
                      "padding with random directions.")
        vecs = err.eigenvectors
        if vecs.shape[1] < d:
            pad = rng.standard_normal((n, d - vecs.shape[1]))
            vecs = np.concatenate([vecs, pad], 1)
    Q, _ = np.linalg.qr(vecs)
    return Q[:, :d]


def random_init(n, d, seed=0):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, d)))
    return Q
