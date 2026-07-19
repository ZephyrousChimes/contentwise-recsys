import numpy as np
import scipy.sparse as sp


def row_norm(X):
    d = np.asarray(X.sum(1)).ravel()
    d[d == 0] = 1
    return sp.diags(1 / d) @ X


def blockwise_topk(left, right, k, post=None, block=2048):
    ns = right.shape[1]
    rows, cols, vals = [], [], []

    for b0 in range(0, ns, block):
        cb = np.arange(b0, min(b0 + block, ns))

        B = left @ right[:, cb]
        if sp.issparse(B):
            B = B.toarray()
        B = np.asarray(B, dtype=np.float32)

        if post is not None:
            B = post(B, cb)
        B[cb, np.arange(len(cb))] = 0

        kk = min(k, B.shape[0] - 1)
        idx = np.argpartition(-B, kk, 0)[:kk]
        v = np.take_along_axis(B, idx, 0)

        rows.append(idx.ravel(order='F'))
        cols.append(np.repeat(cb, kk))
        vals.append(v.ravel(order='F'))

    S = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ns, ns))
    S.eliminate_zeros()
    return S


class ItemKNN:
    def __init__(self, k=100, shrink=10, binary=False):
        self.k = k
        self.shrink = shrink
        self.binary = binary
        self.name = f'itemknn_k{k}_sh{shrink}' + ('_bin' if binary else '')

    def X(self, f):
        if self.binary:
            return f.H
        return f.Xtr

    def fit(self, f):
        X = self.X(f).tocsc()
        n = np.sqrt(np.asarray(X.multiply(X).sum(0)).ravel())

        post = lambda B, cb: B / (n[:, None] * n[cb][None, :] + self.shrink + 1e-9)

        self.S = blockwise_topk(X.T.tocsr(), X, self.k, post)
        return self

    def score(self, f, ub):
        return (self.X(f)[ub] @ self.S).toarray()


class RP3bColumn:
    def __init__(self, k=100, alpha=1.0, beta=0.5):
        self.k, self.a, self.b = k, alpha, beta
        self.name = f'rp3b_col_k{k}_a{alpha}_b{beta}'

    def fit(self, f):
        X = f.H
        Pui = row_norm(X).power(self.a).tocsr()
        Piu = row_norm(X.T.tocsr()).power(self.a).tocsr()

        pop = np.asarray(X.sum(0)).ravel()
        pop[pop == 0] = 1

        post = lambda B, cb: B / (pop[cb][None, :] ** self.b)
        self.S = blockwise_topk(Piu, Pui.tocsc(), self.k, post)
        return self

    def score(self, f, ub):
        return (f.H[ub] @ self.S).toarray()


class RP3b:
    def __init__(self, k=100, alpha=1.0, beta=0.5):
        self.k, self.a, self.b = k, alpha, beta
        self.name = f'rp3b_k{k}_a{alpha}_b{beta}'

    def fit(self, f):
        X = f.H
        pop = np.asarray(X.sum(0)).ravel()
        pop[pop == 0] = 1

        Pui = row_norm(X).power(self.a).tocsc()
        Piu = row_norm(X.T.tocsr()).power(self.a).tocsr()

        rows, cols, vals = [], [], []
        for b0 in range(0, f.ns, 2048):
            rb = np.arange(b0, min(b0 + 2048, f.ns))

            B = (Piu[rb] @ Pui).toarray().astype(np.float32) / (pop[None, :] ** self.b)
            B[np.arange(len(rb)), rb] = 0

            idx = np.argpartition(-B, self.k, 1)[:, :self.k]

            rows.append(np.repeat(rb, self.k))
            cols.append(idx.ravel())
            vals.append(np.take_along_axis(B, idx, 1).ravel())

        self.S = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(f.ns, f.ns))
        return self

    def score(self, f, ub):
        return (f.H[ub] @ self.S).toarray()
