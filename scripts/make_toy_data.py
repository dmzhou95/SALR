"""Write a small synthetic 3-view data set (data/toy.npz) for smoke tests."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from salr.data import make_toy                                          # noqa: E402

if __name__ == "__main__":
    views, y = make_toy(n=1500, K=5, dims=(40, 60, 30), seed=0)
    os.makedirs("data", exist_ok=True)
    np.savez("data/toy.npz", **{f"X{v}": X for v, X in enumerate(views)}, y=y)
    print("saved data/toy.npz", [X.shape for X in views], "K =", y.max() + 1)
