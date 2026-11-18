# SALR: Structure-Aware Latent Representations for Incomplete Multi-view Clustering

This is an implementation of the method presented in *Structure-Aware Latent Representations for Incomplete Multi-view Clustering* (SALR)

---

## 1. Directory Structure

```
SALR-code/
├── salr/
│   ├── backend.py     
│   ├── graph.py       
│   ├── confidence.py   
│   ├── impute.py       
│   ├── encoders.py     
│   ├── spectral.py     
│   ├── solver.py       
│   ├── data.py        
│   ├── masks.py        
│   ├── metrics.py      
│   └── experiment.py   
├── scripts/
│   ├── run_salr.py            
│   ├── grid_search.py            
│   ├── run_paper_protocol.sh     
│   ├── run_parallel_8gpu.sh      
│   └── make_toy_data.py          
├── configs/datasets.json  
├── tests/test_core.py     
└── requirements.txt
```

## 2. Installation

```bash
pip install -r requirements.txt     # numpy, scipy, scikit-learn, torch>=1.12
```

## 3. Quick Start

```bash
python tests/test_core.py                       # run the tests first
python scripts/make_toy_data.py                 # generate data/toy.npz
python scripts/run_salr.py --data data/toy.npz --rhos 0.4 --runs 3 \
       --d 16 --hidden 64 64 --max_iter 20 --gamma 0.001            # GPU (default)
python scripts/run_salr.py --data data/toy.npz --rhos 0.4 --runs 3 \
       --d 16 --hidden 64 64 --max_iter 20 --gamma 0.001 --backend numpy   # CPU only
```

```python
from salr import SALR, SALRConfig, load_dataset, standardize_views, make_mask, clustering_metrics
import numpy as np
views, y = load_dataset("data/MNIST.mat")
mask = make_mask("mcar", y.size, len(views), 0.4, np.random.default_rng(0))
X = standardize_views(views, mask)               # compute statistics from observed rows only; set missing rows to zero
out = SALR(SALRConfig(alpha=1.0, gamma=1e-2, d=128, k=7)).fit(X, mask, n_clusters=y.max() + 1, labels=y)
print(clustering_metrics(y, out["labels"]))      # out also contains Z, S, and the history for each iteration
```





