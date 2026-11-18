
import argparse
import json
import os
import sys
from dataclasses import fields, replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from salr import SALRConfig, VARIANTS, load_dataset                     # noqa: E402
from salr.experiment import run_protocol, format_summary               # noqa: E402


def build_config(args):
    cfg = SALRConfig()
    if args.dataset:                                                    # per-dataset defaults
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "datasets.json")
        table = json.load(open(path))
        cfg = replace(cfg, **{k: (tuple(v) if k == "hidden" else v)
                              for k, v in table.get(args.dataset, {}).items() if not k.startswith("_")})
    over = {}
    for f in fields(SALRConfig):
        val = getattr(args, f.name, None)
        if val is not None:
            over[f.name] = tuple(val) if f.name == "hidden" else val
    return replace(cfg, **over)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help=".mat or .npz file")
    p.add_argument("--dataset", default=None, help="key in configs/datasets.json")
    p.add_argument("--rhos", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8])
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--mechanism", choices=["mcar", "vi", "cd"], default="mcar")
    p.add_argument("--out", default=None, help="JSON summary path")
    p.add_argument("--variant", choices=list(VARIANTS), default=None)
    for name, typ in [("d", int), ("k", int), ("alpha", float), ("gamma", float), ("gamma_mode", str),
                      ("max_iter", int), ("min_iter", int), ("tol", float), ("gpi_steps", int),
                      ("theta_epochs", int), ("warmup_epochs", int), ("rebuild_every", int),
                      ("init", str), ("lambda_mode", str), ("lr", float), ("weight_decay", float), ("batch_size", int),
                      ("kmeans_n_init", int), ("eval_every", int), ("backend", str), ("device", str)]:
        p.add_argument("--" + name, type=typ, default=None)
    p.add_argument("--hidden", type=int, nargs="+", default=None)
    p.add_argument("--no_impute", dest="impute", action="store_const", const=False, default=None)
    p.add_argument("--no_batchnorm", dest="batchnorm", action="store_const", const=False, default=None)
    p.add_argument("--no_kmeans_normalize", dest="kmeans_normalize", action="store_const", const=False, default=None)
    p.add_argument("--quiet", dest="verbose", action="store_const", const=False, default=None)
    args = p.parse_args()

    cfg = build_config(args)
    views, y = load_dataset(args.data)
    print(f"{args.data}: N={y.size}, V={len(views)}, K={y.max() + 1}, dims={[v.shape[1] for v in views]}")
    print(cfg)
    os.makedirs("results", exist_ok=True)
    name = args.dataset or os.path.splitext(os.path.basename(args.data))[0]
    rhos = "-".join(f"{r:g}" for r in args.rhos)
    tag = f"{name}_{cfg.variant}_{args.mechanism}_rho{rhos}"
    out = args.out or os.path.join("results", tag + ".json")
    summary = run_protocol(views, y, cfg, args.rhos, args.runs, args.mechanism,
                           log_path=os.path.splitext(out)[0] + "_runs.jsonl")
    print(format_summary(summary))
    json.dump({"config": cfg.__dict__, "summary": {str(k): v["mean_std"] for k, v in summary.items()}},
              open(out, "w"), indent=1, default=list)
    print("saved", out)


if __name__ == "__main__":
    main()
