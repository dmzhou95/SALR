
from .graph import knn


def mean_impute(B, X_list, mask_list):
    """Initial completion (Alg. 1): missing rows <- mean of the observed rows."""
    out = []
    for X, m in zip(X_list, mask_list):
        Xb = X.clone() if B.name == "torch" else X.copy()
        obs = m > 0
        mu = X[obs].mean(0)
        Xb[~obs] = mu
        out.append(Xb)
    return out


def impute_from_z(B, Z, X_list, mask_list, k, obs_idx, mis_idx):
    """Eq. (22).  obs_idx / mis_idx: per view, index arrays of observed / missing rows."""
    out = []
    for X, oi, mi in zip(X_list, obs_idx, mis_idx):
        Xb = X.clone() if B.name == "torch" else X.copy()
        if mi.shape[0] > 0:
            kk = min(k, oi.shape[0])
            _, nb = knn(B, Z[mi], Z[oi], kk)                 # neighbours among observing samples
            Xo = X[oi]
            Xb[mi] = Xo[nb].mean(1)
        out.append(Xb)
    return out
