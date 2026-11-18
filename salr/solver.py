"""SALR optimisation (Algorithm 1) by block-coordinate descent on Eq. (20):

    J_total = L_hc(Z) + alpha * L_ca(Z, S),
    L_hc(Z)    = sum_v Tr(Z^T L_v Z)                                        (Eq. 7)
    L_ca(Z, S) = sum_{i,v} s_i^(v) ||z_i - F_v(xbar_i^(v))||^2 + gamma ||S||_F^2   (Eq. 13)
    s.t. Z^T Z = I_d,  S 1 = 1,  S >= 0.

One outer iteration:
  1. encode the completed views, rebuild the k-NN hypergraphs (every `rebuild_every` iterations);
  2. Z-step: generalized power iteration  Z <- P Q^T,  P S Q^T = svd((lambda I - L) Z + alpha M)
     with M_i = sum_v s_i^(v) f_i^(v)  (Eq. 21);
  3. theta-step: Adam epochs on the weighted alignment term;
  4. S-step: closed form of Theorem 2 (Eq. 19);
  5. re-impute missing entries from the k nearest neighbours in Z (Eq. 22);
  6. evaluate J_total; stop when its relative change < tol.
Finally k-means on Z gives the cluster labels.
"""
from dataclasses import dataclass, asdict
import time
import numpy as np

from . import graph as G
from .backend import get_backend
from .confidence import solve_confidence, average_weights, binary_weights
from .impute import mean_impute, impute_from_z
from .spectral import spectral_init, random_init

# variant -> (use L_hc, graph kind, use L_ca in the Z-step, weighting)
VARIANTS = {
    "full":     (True,  "hyper",    True,  "adaptive"),   # #7 SALR
    "wo_hc":    (False, "hyper",    True,  "adaptive"),   # #1 w/o L_hc
    "wo_ca":    (True,  "hyper",    False, "average"),    # #2 w/o L_ca
    "rec_only": (False, "hyper",    True,  "average"),    # #3 w/ J_rec only (Eq. 4)
    "pwg":      (True,  "pairwise", True,  "adaptive"),   # #4 point-wise graph
    "aw":       (True,  "hyper",    True,  "average"),    # #5 average weight
    "bwm":      (True,  "hyper",    True,  "binary"),     # #6 binary weight matrix
}


@dataclass
class SALRConfig:
    d: int = 128                 # latent dimension
    k: int = 7                   # neighbours per hyperedge (also used in Eq. 22)
    alpha: float = 1.0           # weight of L_ca
    gamma: float = 1e-2          # confidence smoothness
    gamma_mode: str = "absolute" # "absolute" (as in Eq. 13) or "relative" (gamma * mean error)
    max_iter: int = 100          # T
    min_iter: int = 10
    tol: float = 1e-4            # relative change of J_total
    gpi_steps: int = 5           # power-iteration steps per Z-step
    lambda_mode: str = "paper"   # "paper": 2 max_i sum_v D_v(i,i) (Eq. 21); "power": 1.05 x estimated lambda_max
    kmeans_normalize: bool = True  # row-normalise Z before k-means
    theta_epochs: int = 5        # encoder epochs per outer iteration
    warmup_epochs: int = 20      # encoder epochs fitted to the initial Z
    rebuild_every: int = 1       # r: rebuild the hypergraphs every r iterations
    impute: bool = True          # Eq. 22
    init: str = "spectral"       # "spectral" or "random"
    variant: str = "full"
    hidden: tuple = (512, 512)
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 1024
    batchnorm: bool = True
    kmeans_n_init: int = 10
    eval_every: int = 0          # >0: k-means + metrics every n iterations (needs labels)
    seed: int = 0
    backend: str = "torch"
    device: str = "cuda"
    verbose: bool = True


class SALR:
    def __init__(self, cfg: SALRConfig):
        self.cfg = cfg
        self.B = get_backend(cfg.backend, cfg.device)
        if cfg.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {list(VARIANTS)}")

    # ------------------------------------------------------------------ helpers
    def _encoders(self, dims, n):
        c = self.cfg
        if self.B.name == "torch":
            from .encoders import TorchEncoders
            return TorchEncoders(dims, c.d, n, c.hidden, c.lr, c.weight_decay, c.batchnorm,
                                 c.batch_size, c.device, c.seed)
        from .encoders import NumpyEncoders
        return NumpyEncoders(dims, c.d, n, c.hidden, c.lr, c.weight_decay, c.batch_size, c.seed)

    def _build(self, F):
        kind = VARIANTS[self.cfg.variant][1]
        fn = G.build_hypergraph if kind == "hyper" else G.build_pairwise
        return [fn(self.B, f, self.cfg.k) for f in F]

    def _lap(self, graphs, Z):
        out = G.laplacian_matvec(self.B, graphs[0], Z)
        for g in graphs[1:]:
            out = out + G.laplacian_matvec(self.B, g, Z)
        return out

    def _errors(self, Z, F):
        E = self.B.zeros((Z.shape[0], len(F)))
        for v, f in enumerate(F):
            r = Z - f
            E[:, v] = (r * r).sum(1)
        return E

    def _weights(self, E, mask_b, gamma):
        kind = VARIANTS[self.cfg.variant][3]
        n, V = E.shape
        if kind == "adaptive":
            return solve_confidence(self.B, E, gamma)[0]
        if kind == "average":
            return average_weights(self.B, n, V)
        return binary_weights(self.B, mask_b)

    def _kmeans(self, Z, n_clusters):
        from sklearn.cluster import KMeans
        if self.cfg.kmeans_normalize:
            Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        km = KMeans(n_clusters=n_clusters, n_init=self.cfg.kmeans_n_init, random_state=self.cfg.seed)
        return km.fit_predict(Z)

    # ------------------------------------------------------------------ main
    def fit(self, X_list, mask, n_clusters, labels=None):
        """X_list: list of V arrays (N, d_v), already standardised, missing rows arbitrary
        (they are never read); mask: (N, V) 0/1 with at least one observed view per row."""
        c, B = self.cfg, self.B
        use_hc, _, use_ca, _ = VARIANTS[c.variant]
        mask = np.asarray(mask).astype(np.float64)
        n, V = mask.shape
        rng = np.random.default_rng(c.seed)

        X = [B.asarray(x) for x in X_list]
        mask_cols = [B.asarray(mask[:, v]) for v in range(V)]
        mask_b = B.asarray(mask)
        obs_idx = [B.asindex(np.flatnonzero(mask[:, v] > 0)) for v in range(V)]
        mis_idx = [B.asindex(np.flatnonzero(mask[:, v] == 0)) for v in range(V)]
        Xbar = mean_impute(B, X, mask_cols)

        t0 = time.perf_counter()
        if c.init == "spectral":
            g0 = self._build(Xbar)
            Z = B.asarray(spectral_init([G.graph_to_numpy(B, g) for g in g0], c.d, c.seed))
        else:
            Z = B.asarray(random_init(n, c.d, c.seed))
        S = average_weights(B, n, V)
        enc = self._encoders([x.shape[1] for x in X], n)
        enc.train_steps(Xbar, Z, S, c.warmup_epochs)
        if c.verbose:
            print(f"[SALR] init done in {time.perf_counter() - t0:.1f}s (N={n}, V={V}, d={c.d}, k={c.k}, "
                  f"variant={c.variant}, backend={B.name})")

        history, J_prev, graphs, lam = [], None, None, None
        for t in range(c.max_iter):
            B.sync(); t_it = time.perf_counter()
            F = enc.encode(Xbar)
            t_graph = 0.0
            if use_hc and (graphs is None or t % c.rebuild_every == 0):
                tg = time.perf_counter()
                graphs = self._build(F)
                lam = G.lambda_bound(B, graphs) if c.lambda_mode == "paper" else G.lambda_power(B, graphs)
                B.sync(); t_graph = time.perf_counter() - tg

            # ---- Z-step (Eq. 21)
            if use_ca:
                M = S[:, 0:1] * F[0]
                for v in range(1, V):
                    M = M + S[:, v:v + 1] * F[v]
            else:
                M = B.zeros_like(Z)
            steps = c.gpi_steps if use_hc else 1
            for _ in range(steps):
                Bm = c.alpha * M
                if use_hc:
                    Bm = Bm + lam * Z - self._lap(graphs, Z)
                P, _, Qt = B.svd_thin(Bm)
                Z = P @ Qt

            # ---- theta-step
            enc_loss = enc.train_steps(Xbar, Z, S, c.theta_epochs)

            # ---- S-step (Eq. 19)
            F = enc.encode(Xbar)
            E = self._errors(Z, F)
            gamma = c.gamma * (B.item(E.mean()) if c.gamma_mode == "relative" else 1.0)
            S = self._weights(E, mask_b, gamma)

            # ---- objective (Eq. 20), before re-imputation
            l_hc = G.hc_loss(B, graphs, Z) if use_hc else 0.0
            l_ca = B.item((S * E).sum()) + gamma * B.item((S * S).sum())
            J = l_hc + c.alpha * l_ca

            # ---- imputation (Eq. 22)
            if c.impute:
                Xbar = impute_from_z(B, Z, X, mask_cols, c.k, obs_idx, mis_idx)
            B.sync(); dt = time.perf_counter() - t_it

            Sn = B.to_numpy(S) * V
            rec = {"iter": t + 1, "J": J, "L_hc": l_hc, "L_ca": l_ca, "enc_loss": enc_loss,
                   "conf_observed": float(Sn[mask > 0].mean()),
                   "conf_imputed": float(Sn[mask == 0].mean()) if (mask == 0).any() else float("nan"),
                   "zero_conf_pct": float(100 * (Sn == 0).mean()),
                   "time_iter": dt, "time_graph": t_graph}
            if c.eval_every and labels is not None and (t + 1) % c.eval_every == 0:
                from .metrics import clustering_metrics
                rec.update(clustering_metrics(labels, self._kmeans(B.to_numpy(Z), n_clusters)))
            history.append(rec)
            if c.verbose:
                extra = "".join(f" {m}={rec[m]:.4f}" for m in ("ACC", "NMI", "ARI") if m in rec)
                print(f"[SALR] it {t + 1:3d}  J={J:.6e}  L_hc={l_hc:.4e}  L_ca={l_ca:.4e}  "
                      f"conf(obs/imp)={rec['conf_observed']:.2f}/{rec['conf_imputed']:.2f}  {dt:.2f}s{extra}")
            if J_prev is not None and t + 1 >= c.min_iter and abs(J_prev - J) <= c.tol * max(abs(J_prev), 1e-12):
                break
            J_prev = J

        Zn = B.to_numpy(Z)
        pred = self._kmeans(Zn, n_clusters)
        return {"Z": Zn, "S": B.to_numpy(S), "labels": pred, "history": history,
                "n_iter": len(history), "config": asdict(c)}
