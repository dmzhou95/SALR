#!/usr/bin/env bash
# Full protocol of Section 4: 8 datasets x rho in {0.2,0.4,0.6,0.8} x 30 runs (RQ1),
# ablations at rho = 0.2 on MNIST / YouTubeFace (RQ2), non-uniform missingness (RQ4).
# Put the data files in data/<Name>.mat first (see README).  Results go to results/.
set -e
cd "$(dirname "$0")/.."
DATASETS="MNIST YouTubeFace Hdigit Caltech7 Fashion MBSP BDGP CIFAR10"

# RQ1: main results
for ds in $DATASETS; do
  python scripts/run_salr.py --data data/$ds.mat --dataset $ds --rhos 0.2 0.4 0.6 0.8 --runs 30 --quiet
done

# RQ2: ablation variants (#1-#6) at rho = 0.2
for ds in MNIST YouTubeFace; do
  for v in wo_hc wo_ca rec_only pwg aw bwm; do
    python scripts/run_salr.py --data data/$ds.mat --dataset $ds --variant $v --rhos 0.2 --runs 30 --quiet
  done
done

# RQ4: non-uniform missing mechanisms and the extreme ratio rho = 0.9
for ds in MNIST YouTubeFace Caltech7 CIFAR10; do
  python scripts/run_salr.py --data data/$ds.mat --dataset $ds --mechanism vi --rhos 0 --runs 30 --quiet
  python scripts/run_salr.py --data data/$ds.mat --dataset $ds --mechanism cd --rhos 0 --runs 30 --quiet
  python scripts/run_salr.py --data data/$ds.mat --dataset $ds --rhos 0.9 --runs 30 --quiet
done
