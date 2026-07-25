import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataprep import load_fold

torch.manual_seed(0)


class Net(nn.Module):
    def __init__(self, ns, d=64):
        super().__init__()

        self.hist = nn.EmbeddingBag(ns, d, mode='sum')
        self.item = nn.Embedding(ns, d)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))

        nn.init.normal_(self.hist.weight, std=.05)
        nn.init.normal_(self.item.weight, std=.05)

    def user(self, X):
        idx = torch.from_numpy(X.indices.astype(np.int64))
        off = torch.from_numpy(X.indptr[:-1].astype(np.int64))

        w = torch.from_numpy(X.data.astype(np.float32))
        cnt = torch.from_numpy(np.diff(X.indptr).clip(1).astype(np.float32))

        h = self.hist(idx, off, per_sample_weights=w) / cnt[:, None]
        return F.normalize(self.mlp(h), dim=-1)

    def items(self, ids=None):
        if ids is None:
            return F.normalize(self.item.weight, dim=-1)
        return F.normalize(self.item(ids), dim=-1)


class TwoTower:
    def __init__(self, views, d=64, weeks=6, epochs=3, tau=0.05, lr=3e-3, batch=1024):
        self.views = views
        self.ns = views.n_series

        self.d, self.weeks, self.epochs = d, weeks, epochs
        self.tau, self.lr, self.batch = tau, lr, batch

        self.name = f'two_tower_d{d}'

    def fit(self, f):
        c = f.t
        H, U = [], []

        for w in range(max(2, c - self.weeks), c):
            fw = load_fold(self.views, w)

            # Yn = fw.Y.multiply(fw.H == 0).tocoo()
            Yn = (fw.Y - fw.Y.multiply(fw.H)).tocsr()
            Yn.eliminate_zeros()
            Yn = Yn.tocoo()

            H.append(fw.Xtr)
            U.append((len(H) - 1, Yn.row, Yn.col, Yn.data))

        u_week = np.concatenate([np.full(len(r), k) for k, r, _, _ in U])
        u_row = np.concatenate([r for _, r, _, _ in U])
        pos = np.concatenate([s for _, _, s, _ in U])
        wts = np.concatenate([d for _, _, _, d in U]).astype(np.float32)

        logq = np.log(np.bincount(pos, minlength=self.ns) + 1) - np.log(len(pos))
        logq = torch.from_numpy(logq).float()

        net = Net(self.ns, self.d)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)

        rng = np.random.default_rng(0)
        n = len(pos)

        for ep in range(self.epochs):
            perm = rng.permutation(n)
            tot = 0

            for b0 in range(0, n, self.batch):
                b = perm[b0:b0 + self.batch]
                ks = np.unique(u_week[b])

                uvec = torch.cat([net.user(H[k][u_row[b[u_week[b] == k]]]) for k in ks])
                b = np.concatenate([b[u_week[b] == k] for k in ks])

                p = torch.from_numpy(pos[b])
                it = net.items(p)

                # logits = uvec @ it.T / self.tau
                logits = uvec @ it.T / self.tau - logq[p][None, :]

                loss = F.cross_entropy(logits, torch.arange(len(b)), reduction='none')
                loss = (loss * torch.from_numpy(wts[b])).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

                tot += loss.item() * len(b)

            print(f'two-tower cutoff {c} epoch {ep}: loss {tot / n:.3f} on {n} positives')

        net.eval()
        self.net = net

        with torch.no_grad():
            self.I = net.items().numpy()

        return self

    def score(self, f, ub):
        with torch.no_grad():
            u = self.net.user(f.Xtr[ub]).numpy()
        return (u @ self.I.T).astype(np.float32)
