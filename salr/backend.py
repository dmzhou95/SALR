
import numpy as np


class NumpyBackend:
    name = "numpy"

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    # ---- creation / conversion
    def asarray(self, x):
        return np.asarray(x, dtype=self.dtype)

    def asindex(self, x):
        return np.asarray(x, dtype=np.int64)

    def sync(self):
        pass

    def to_numpy(self, x):
        return np.asarray(x)

    def zeros(self, shape):
        return np.zeros(shape, dtype=self.dtype)

    def zeros_like(self, x):
        return np.zeros_like(x)

    def full(self, shape, value):
        return np.full(shape, value, dtype=self.dtype)

    def arange(self, n):
        return np.arange(n, dtype=np.int64)

    def randn(self, shape, rng):
        return rng.standard_normal(shape).astype(self.dtype)

    # ---- elementwise / reductions
    def exp(self, x):
        return np.exp(x)

    def sqrt(self, x):
        return np.sqrt(x)

    def clamp_min(self, x, v):
        return np.maximum(x, v)

    def maximum(self, x, y):
        return np.maximum(x, y)

    def max(self, x):
        return float(np.max(x))

    def item(self, x):
        return float(x)

    def cumsum(self, x, axis):
        return np.cumsum(x, axis=axis)

    def sort_desc(self, x, axis):
        return -np.sort(-x, axis=axis)

    def gather(self, x, idx):
        """x[i, idx[i, j]] along axis 1."""
        return np.take_along_axis(x, idx, axis=1)

    def count_true(self, cond, axis):
        return cond.sum(axis=axis).astype(np.int64)

    # ---- linear algebra
    def svd_thin(self, a):
        return np.linalg.svd(a, full_matrices=False)

    def topk_smallest(self, d, k):
        """k smallest entries per row (sorted ascending): values, indices."""
        idx = np.argpartition(d, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(d, idx, axis=1)
        order = np.argsort(vals, axis=1)
        return np.take_along_axis(vals, order, axis=1), np.take_along_axis(idx, order, axis=1)

    def index_add(self, out, idx, src):
        """out[idx[j]] += src[j] (rows); idx 1-D, src 2-D or 1-D."""
        n = out.shape[0]
        if src.ndim == 1:
            out += np.bincount(idx, weights=src, minlength=n)
            return out
        for c in range(src.shape[1]):
            out[:, c] += np.bincount(idx, weights=src[:, c], minlength=n)
        return out


class TorchBackend:
    name = "torch"

    def __init__(self, device="cuda", dtype=None):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = dtype or torch.float32

    def asarray(self, x):
        t = self.torch
        if isinstance(x, t.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return t.as_tensor(np.asarray(x), dtype=self.dtype, device=self.device)

    def asindex(self, x):
        t = self.torch
        if isinstance(x, t.Tensor):
            return x.to(device=self.device, dtype=t.long)
        return t.as_tensor(np.asarray(x), dtype=t.long, device=self.device)

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def zeros(self, shape):
        return self.torch.zeros(shape, dtype=self.dtype, device=self.device)

    def zeros_like(self, x):
        return self.torch.zeros_like(x)

    def full(self, shape, value):
        return self.torch.full(shape, value, dtype=self.dtype, device=self.device)

    def arange(self, n):
        return self.torch.arange(n, device=self.device, dtype=self.torch.long)

    def randn(self, shape, rng):
        return self.asarray(rng.standard_normal(shape))

    def exp(self, x):
        return self.torch.exp(x)

    def sqrt(self, x):
        return self.torch.sqrt(x)

    def clamp_min(self, x, v):
        return self.torch.clamp(x, min=v)

    def maximum(self, x, y):
        if not isinstance(y, self.torch.Tensor):
            return self.torch.clamp(x, min=y)
        return self.torch.maximum(x, y)

    def max(self, x):
        return float(x.max().item())

    def item(self, x):
        return float(x.item()) if hasattr(x, "item") else float(x)

    def cumsum(self, x, axis):
        return self.torch.cumsum(x, dim=axis)

    def sort_desc(self, x, axis):
        return self.torch.sort(x, dim=axis, descending=True).values

    def gather(self, x, idx):
        return self.torch.gather(x, 1, idx)

    def count_true(self, cond, axis):
        return cond.sum(dim=axis).long()

    def svd_thin(self, a):
        return self.torch.linalg.svd(a, full_matrices=False)

    def topk_smallest(self, d, k):
        vals, idx = self.torch.topk(d, k, dim=1, largest=False, sorted=True)
        return vals, idx

    def index_add(self, out, idx, src):
        return out.index_add_(0, idx.contiguous(), src.contiguous())


def get_backend(name="torch", device="cuda"):
    if name == "numpy":
        return NumpyBackend()
    if name == "torch":
        return TorchBackend(device=device)
    raise ValueError(f"unknown backend {name}")
