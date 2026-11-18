
import os
import numpy as np

VIEW_KEYS = ("X", "data", "fea", "Xs", "x")
LABEL_KEYS = ("Y", "y", "gt", "truth", "label", "labels", "gnd")


def _to_dense(a):
    if hasattr(a, "toarray"):
        a = a.toarray()
    return np.asarray(a, dtype=np.float64)


def load_mat(path):
    try:
        from scipy.io import loadmat
        raw = loadmat(path)
        get = raw.get
    except NotImplementedError:                       # MATLAB v7.3 (HDF5)
        import h5py
        f = h5py.File(path, "r")
        raw = {k: f[k] for k in f.keys()}

        def get(k, default=None):
            if k not in raw:
                return default
            obj = raw[k]
            if obj.dtype == object:                    # cell array of references
                return np.array([[f[r][()] for r in row] for row in obj[()]], dtype=object)
            return obj[()]
    y = next((get(k) for k in LABEL_KEYS if get(k) is not None), None)
    if y is None:
        raise KeyError(f"no label key {LABEL_KEYS} in {path}")
    y = np.asarray(y).reshape(-1)
    n = y.size
    cell = next((get(k) for k in VIEW_KEYS if get(k) is not None), None)
    if cell is None:
        raise KeyError(f"no view key {VIEW_KEYS} in {path}")
    views = []
    for item in np.asarray(cell, dtype=object).reshape(-1):
        a = _to_dense(item)
        if a.shape[0] != n and a.shape[1] == n:
            a = a.T
        if a.shape[0] != n:
            raise ValueError(f"view of shape {a.shape} does not match {n} labels")
        views.append(a)
    return views, y


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    keys = sorted([k for k in z.files if k.startswith("X") or k.startswith("view")],
                  key=lambda s: int("".join(ch for ch in s if ch.isdigit()) or 0))
    return [_to_dense(z[k]) for k in keys], np.asarray(z["y"]).reshape(-1)


def load_dataset(path):
    ext = os.path.splitext(path)[1].lower()
    views, y = load_mat(path) if ext == ".mat" else load_npz(path)
    _, y = np.unique(y, return_inverse=True)           # labels -> 0..K-1
    return views, y.astype(np.int64)


def standardize_views(views, mask):
    """z-score every view with statistics of its OBSERVED rows only; missing rows are
    zeroed so that no information from an unobserved view can be used."""
    out = []
    for v, X in enumerate(views):
        obs = mask[:, v] > 0
        mu = X[obs].mean(0)
        sd = X[obs].std(0)
        sd[sd < 1e-8] = 1.0
        Xs = (X - mu) / sd
        Xs[~obs] = 0.0
        out.append(Xs)
    return out


def make_toy(n=1500, K=5, dims=(40, 60, 30), noise=0.6, seed=0):
    """Small synthetic multi-view data set for smoke tests (K well-separated clusters
    observed through view-specific random non-linear maps)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, size=n)
    centres = rng.normal(0, 1.0, (K, 8)) * 2.5
    latent = centres[y] + rng.normal(0, 1.0, (n, 8))
    views = []
    for dv in dims:
        W1 = rng.normal(0, 1 / np.sqrt(8), (8, 64))
        W2 = rng.normal(0, 1 / np.sqrt(64), (64, dv))
        X = np.tanh(latent @ W1) @ W2
        views.append(X + noise * X.std() * rng.normal(size=X.shape))
    return views, y
