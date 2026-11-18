#!/usr/bin/env bash
# RQ1 on an 8-GPU machine: one dataset per GPU, all in parallel.
# Logs: logs/<dataset>.log   Results: results/<dataset>_full_mcar_rho0.2-0.4-0.6-0.8.json
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
DATASETS=(MNIST YouTubeFace Hdigit Caltech7 Fashion MBSP BDGP CIFAR10)
for i in "${!DATASETS[@]}"; do
  ds=${DATASETS[$i]}
  CUDA_VISIBLE_DEVICES=$i nohup python scripts/run_salr.py --data data/$ds.mat --dataset $ds \
      --rhos 0.2 0.4 0.6 0.8 --runs 30 --quiet > logs/$ds.log 2>&1 &
  echo "started $ds on GPU $i"
done
wait
echo "all done"
