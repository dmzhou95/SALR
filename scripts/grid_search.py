
import argparse
import csv
import itertools
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from salr import SALRConfig, load_dataset                               # noqa: E402
from salr.experiment import run_protocol                               # noqa: E402

SPACE = {"alpha": [1e-3, 1e-2, 1e-1, 1.0, 10.0], "gamma": [1e-3, 1e-2, 1e-1, 1.0, 10.0],
         "d": [32, 64, 128, 256, 512], "k": [3, 5, 7, 9, 11, 15]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--rho", type=float, default=0.2)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--grid", nargs="+", default=["alpha", "gamma"], choices=list(SPACE))
    p.add_argument("--backend", default="torch")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/grid.csv")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="extra fixed SALRConfig fields, e.g. --set max_iter=50 theta_epochs=3")
    a = p.parse_args()
    views, y = load_dataset(a.data)
    base = SALRConfig(backend=a.backend, device=a.device, verbose=False)
    for kv in a.set:
        key, val = kv.split("=", 1)
        old = getattr(base, key)
        val = tuple(int(x) for x in val.split(",")) if isinstance(old, tuple) else \
            (val.lower() in ("1", "true", "yes")) if isinstance(old, bool) else type(old)(val)
        base = replace(base, **{key: val})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(a.grid + ["ACC", "ACC_std", "NMI", "NMI_std", "ARI", "ARI_std"])
        for combo in itertools.product(*[SPACE[g] for g in a.grid]):
            cfg = replace(base, **dict(zip(a.grid, combo)))
            s = run_protocol(views, y, cfg, [a.rho], a.runs)[a.rho]["mean_std"]
            w.writerow(list(combo) + [round(x, 2) for k in ("ACC", "NMI", "ARI") for x in s[k]])
            f.flush()
            print(dict(zip(a.grid, combo)), {k: round(s[k][0], 2) for k in s})
    print("saved", a.out)


if __name__ == "__main__":
    main()
