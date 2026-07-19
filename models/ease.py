import numpy as np


class EASE:
    def __init__(self, lam=200.0, n_items=8000, binary=True):
        self.lam = lam
        self.n = n_items
        self.binary = binary
        self.name = f'ease_l{lam}_n{n_items}'

    def fit(self, f):
        X = f.H if self.binary else f.Xtr

        pop = np.asarray((X > 0).sum(0)).ravel()
        self.items = np.sort(np.argsort(-pop)[:self.n])
        # self.items = np.arange(f.ns)

        Xs = X[:, self.items]
        G = (Xs.T @ Xs).toarray().astype(np.float64)
        G[np.diag_indices_from(G)] += self.lam

        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        np.fill_diagonal(B, 0)

        self.B = B.astype(np.float32)
        self.Xsrc = X
        return self

    def score(self, f, ub):
        S = np.zeros((len(ub), f.ns), np.float32)
        S[:, self.items] = self.Xsrc[ub][:, self.items] @ self.B
        return S
