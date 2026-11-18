
import numpy as np


def mcar_mask(n, V, rho, rng):
    mask = np.ones((n, V), dtype=np.int64)
    if V < 2 or rho <= 0:
        return mask
    inc = rng.choice(n, size=int(round(rho * n)), replace=False)
    for i in inc:
        n_miss = rng.integers(1, V)                 # 1 .. V-1 views missing
        mask[i, rng.choice(V, size=n_miss, replace=False)] = 0
    return mask


def view_imbalanced_mask(n, V, rng, first=0.7, others=0.1):
    p = np.full(V, others)
    p[0] = first
    mask = (rng.random((n, V)) >= p[None, :]).astype(np.int64)
    empty = np.flatnonzero(mask.sum(1) == 0)
    mask[empty, rng.integers(0, V, size=empty.size)] = 1
    return mask


def class_dependent_mask(labels, V, rng, high=0.7, low=0.1):
    labels = np.asarray(labels)
    classes = np.unique(labels)
    hi = set(rng.permutation(classes)[: len(classes) // 2].tolist())
    mask = np.ones((labels.size, V), dtype=np.int64)
    for c in classes:
        idx = np.flatnonzero(labels == c)
        sub = mcar_mask(idx.size, V, high if c in hi else low, rng)
        mask[idx] = sub
    return mask


def make_mask(kind, n, V, rho, rng, labels=None):
    if kind == "mcar":
        return mcar_mask(n, V, rho, rng)
    if kind == "vi":
        return view_imbalanced_mask(n, V, rng)
    if kind == "cd":
        return class_dependent_mask(labels, V, rng)
    raise ValueError(kind)
