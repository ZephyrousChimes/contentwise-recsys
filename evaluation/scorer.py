import numpy as np


class NextWeekScorer:
    def __init__(self, fold, theta):
        self.f = fold
        self.theta = np.asarray(theta, dtype=np.float32)

    def score(self, users, lists):
        users = np.asarray(users)
        lists = np.asarray(lists)
        n, K = lists.shape

        th = self.theta[:K]
        if K > len(self.theta):
            th = np.r_[self.theta, np.full(K - len(self.theta), self.theta[-1])]

        Y = self.f.Y[users].toarray()
        H = self.f.H[users].toarray()

        ok = lists >= 0
        li = np.maximum(lists, 0)

        W = np.where(ok, np.take_along_axis(Y, li, 1), 0.0)
        new = np.where(ok, np.take_along_axis(H, li, 1) == 0, False)

        value = (th * W).sum(1)
        best = (th * -np.sort(-Y, 1)[:, :K]).sum(1)

        with np.errstate(invalid='ignore', divide='ignore'):
            out = dict(value=value,
                       norm_value=np.where(best > 0, value / best, np.nan),
                       disc=np.where(value > 0, (th * W * new).sum(1) / value, np.nan),
                       new_slots=new.sum(1) / ok.sum(1).clip(1))
        return out

    def summary(self, users, lists, batch=1024):
        users = np.asarray(users)
        lists = np.asarray(lists)

        # s = self.score(users, lists)
        parts = [self.score(users[i:i + batch], lists[i:i + batch]) for i in range(0, len(users), batch)]
        s = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}

        return {'value': float(s['value'].mean()),
                'norm_value': float(np.nanmean(s['norm_value'])),
                'disc': float(np.nanmean(s['disc'])),
                'new_slots': float(s['new_slots'].mean())}
