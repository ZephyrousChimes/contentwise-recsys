import numpy as np


class Random:
    name = 'random'

    def fit(self, f):
        self.rng = np.random.default_rng(0)
        return self

    def score(self, f, ub):
        return self.rng.random((len(ub), f.ns), dtype=np.float32)


class Pop:
    def __init__(self, which='lastweek'):
        self.which = which
        self.name = 'pop_' + which

    def fit(self, f):
        if self.which == 'lastweek':
            p = f.pop_lastweek
        else:
            p = f.pop_all

        self.p = (p / (p.max() + 1e-9)).astype(np.float32)
        return self

    def score(self, f, ub):
        return np.broadcast_to(self.p, (len(ub), f.ns)).copy()


class ContinueRecency:
    name = 'cw_recency'

    def fit(self, f):
        self.pop = Pop().fit(f)
        return self

    def score(self, f, ub):
        R = f.last_ts[ub].toarray()
        R = np.where(R > 0, R / R.max(), 0)

        return R + 1e-3 * self.pop.score(f, ub) * (R == 0)


class ContinueDecayed:
    def __init__(self, tau=7):
        self.tau = tau
        self.name = f'cw_decay{tau}'

    def fit(self, f):
        self.pop = Pop().fit(f)
        return self

    def score(self, f, ub):
        R = f.last_ts[ub].toarray()
        X = f.Xtr[ub].toarray()

        now = f.t * 7 * 86400
        age = np.where(R > 0, (now - R) / 86400, np.inf)

        S = X * np.exp(-age / self.tau)
        return S + 1e-3 * self.pop.score(f, ub) * (S == 0)
