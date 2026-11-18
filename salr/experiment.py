
import json
import time
from dataclasses import replace
import numpy as np

from .data import standardize_views
from .masks import make_mask
from .metrics import clustering_metrics
from .solver import SALR


def run_once(views, y, cfg, rho, seed, mechanism="mcar"):
    rng = np.random.default_rng(10_000 + seed)
    mask = make_mask(mechanism, y.size, len(views), rho, rng, labels=y)
    X = standardize_views(views, mask)
    c = replace(cfg, seed=seed)
    t0 = time.perf_counter()
    out = SALR(c).fit(X, mask, n_clusters=int(y.max()) + 1, labels=y)
    m = clustering_metrics(y, out["labels"])
    m.update({"seed": seed, "rho": rho, "n_iter": out["n_iter"], "time_s": time.perf_counter() - t0})
    return m, out


def run_protocol(views, y, cfg, rhos, runs, mechanism="mcar", log_path=None):
    summary = {}
    for rho in rhos:
        rows = []
        for r in range(runs):
            m, _ = run_once(views, y, cfg, rho, r, mechanism)
            rows.append(m)
            print(f"rho={rho} run {r + 1}/{runs}: ACC={m['ACC']:.2f} NMI={m['NMI']:.2f} ARI={m['ARI']:.2f} "
                  f"({m['n_iter']} it, {m['time_s']:.1f}s)", flush=True)
            if log_path:
                with open(log_path, "a") as f:
                    f.write(json.dumps(m) + "\n")
        agg = {}
        for k in ("ACC", "NMI", "ARI"):
            a = np.array([row[k] for row in rows])
            agg[k] = (float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0)
        summary[rho] = {"mean_std": agg, "runs": rows}
    return summary


def format_summary(summary):
    lines = ["rho   ACC              NMI              ARI"]
    for rho, s in summary.items():
        ms = s["mean_std"]
        lines.append(f"{rho:<5} " + "  ".join(f"{ms[k][0]:6.2f}+-{ms[k][1]:5.2f}   " for k in ("ACC", "NMI", "ARI")))
    return "\n".join(lines)
