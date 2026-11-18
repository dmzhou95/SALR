
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def cluster_acc(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    _, yt = np.unique(y_true, return_inverse=True)
    _, yp = np.unique(y_pred, return_inverse=True)
    C = np.zeros((yp.max() + 1, yt.max() + 1), dtype=np.int64)
    np.add.at(C, (yp, yt), 1)
    r, c = linear_sum_assignment(-C)
    return C[r, c].sum() / y_true.size


def clustering_metrics(y_true, y_pred):
    return {"ACC": 100 * cluster_acc(y_true, y_pred),
            "NMI": 100 * normalized_mutual_info_score(y_true, y_pred),
            "ARI": 100 * adjusted_rand_score(y_true, y_pred)}
