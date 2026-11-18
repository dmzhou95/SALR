
import numpy as np


class TorchEncoders:
    def __init__(self, dims, d, n, hidden=(512, 512), lr=1e-3, weight_decay=0.0,
                 batchnorm=True, batch_size=1024, device="cuda", seed=0):
        import torch
        import torch.nn as nn
        self.torch = torch
        torch.manual_seed(seed)
        self.scale = 1.0 / np.sqrt(n)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.nets = nn.ModuleList()
        for dv in dims:
            layers, prev = [], dv
            for h in hidden:
                layers.append(nn.Linear(prev, h))
                if batchnorm:
                    layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU(inplace=True))
                prev = h
            layers.append(nn.Linear(prev, d))
            self.nets.append(nn.Sequential(*layers))
        self.nets.to(self.device)
        self.opt = torch.optim.Adam(self.nets.parameters(), lr=lr, weight_decay=weight_decay)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def encode(self, X_list, chunk=8192):
        torch = self.torch
        self.nets.eval()
        out = []
        with torch.no_grad():
            for net, X in zip(self.nets, X_list):
                parts = [net(X[s:s + chunk]) for s in range(0, X.shape[0], chunk)]
                out.append(torch.cat(parts, 0) * self.scale)
        return out

    def train_steps(self, X_list, Z, S, epochs):
        """Minimise sum_{i,v} s_iv ||z_i - F_v(x_iv)||^2 (Z, S fixed); returns mean loss."""
        torch = self.torch
        self.nets.train()
        n = Z.shape[0]
        target = (Z / self.scale).detach()          # sqrt(N) z_i, O(1) entries
        S = S.detach()
        bs = min(self.batch_size, n)
        tot, cnt = 0.0, 0
        for _ in range(epochs):
            perm = torch.randperm(n, generator=self.gen).to(self.device)
            for s in range(0, n - bs + 1 if n >= bs else 1, bs):   # drop the last partial batch (BatchNorm)
                b = perm[s:s + bs]
                loss = 0.0
                for v, net in enumerate(self.nets):
                    r = target[b] - net(X_list[v][b])
                    loss = loss + (S[b, v] * (r * r).sum(1)).mean()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                tot += float(loss.item()); cnt += 1
        return tot / max(cnt, 1) * self.scale ** 2


class NumpyEncoders:
    """Same model without normalisation layers: Linear-ReLU-...-Linear, Adam, manual backprop."""

    def __init__(self, dims, d, n, hidden=(256, 256), lr=1e-3, weight_decay=0.0,
                 batch_size=512, seed=0, **_):
        self.rng = np.random.default_rng(seed)
        self.scale = 1.0 / np.sqrt(n)
        self.lr, self.wd, self.batch_size = lr, weight_decay, batch_size
        self.params = []
        for dv in dims:
            layers, prev = [], dv
            for h in list(hidden) + [d]:
                bound = 1.0 / np.sqrt(prev)                  # PyTorch default init range
                W = self.rng.uniform(-bound, bound, (prev, h))
                b = self.rng.uniform(-bound, bound, h)
                layers.append([W, b])
                prev = h
            self.params.append(layers)
        self.m = [[[np.zeros_like(W), np.zeros_like(b)] for W, b in L] for L in self.params]
        self.v = [[[np.zeros_like(W), np.zeros_like(b)] for W, b in L] for L in self.params]
        self.t = 0

    @staticmethod
    def _forward(layers, X):
        acts = [X]
        h = X
        for li, (W, b) in enumerate(layers):
            h = h @ W + b
            if li < len(layers) - 1:
                h = np.maximum(h, 0.0)
            acts.append(h)
        return acts

    def encode(self, X_list):
        return [self._forward(L, X)[-1] * self.scale for L, X in zip(self.params, X_list)]

    def train_steps(self, X_list, Z, S, epochs):
        n = Z.shape[0]
        target = Z / self.scale
        bs = min(self.batch_size, n)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tot, cnt = 0.0, 0
        for _ in range(epochs):
            perm = self.rng.permutation(n)
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                self.t += 1
                loss = 0.0
                for v, layers in enumerate(self.params):
                    acts = self._forward(layers, X_list[v][idx])
                    r = target[idx] - acts[-1]
                    w = S[idx, v][:, None]
                    loss += float((w[:, 0] * (r * r).sum(1)).mean())
                    g = -2.0 * w * r / len(idx)                     # d loss / d output
                    for li in range(len(layers) - 1, -1, -1):
                        W, b = layers[li]
                        gW = acts[li].T @ g + self.wd * W
                        gb = g.sum(0)
                        if li > 0:
                            g = (g @ W.T) * (acts[li] > 0)
                        for p, gp, slot in ((W, gW, 0), (b, gb, 1)):
                            m = self.m[v][li][slot]; vv = self.v[v][li][slot]
                            m *= b1; m += (1 - b1) * gp
                            vv *= b2; vv += (1 - b2) * gp * gp
                            mh = m / (1 - b1 ** self.t); vh = vv / (1 - b2 ** self.t)
                            p -= self.lr * mh / (np.sqrt(vh) + eps)
                tot += loss; cnt += 1
        return tot / max(cnt, 1) * self.scale ** 2
