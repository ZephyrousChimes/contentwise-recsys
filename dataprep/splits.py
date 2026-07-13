import numpy as np
from sklearn.utils import Bunch

from .loaders import T0, WEEK
from .folds import load_fold, to_csr


N_HIST = 30
N_TOTAL = 100

FEATURES = ['in_hist', 'days_since', 'u_w', 'u_eps', 'pop_1w', 'pop_2w', 'trend', 'pop_all', 'n_users',
            'age_w', 'is_film', 'u_nhist', 'u_lastwk']


def top_k(S, k):
    idx = np.argpartition(-S, k, 1)[:, :k]
    o = np.argsort(-np.take_along_axis(S, idx, 1), 1)
    return np.take_along_axis(idx, o, 1)


def check_no_lookahead(v, c):
    cut = T0 + c * WEEK

    # assert f.last_ts.data.max() + T0 < cut
    assert v.ts[v.week < c].max() < cut
    assert v.ts[v.week == c].min() >= cut


def load_rank_split(views, c, is_film, explore, extra=None, fold=None, max_users=None, seed=0):
    v = views.frame
    nu, ns = views.n_users, views.n_series

    check_no_lookahead(v, c)

    f = fold if fold is not None else load_fold(views, c)
    extra = extra or {}

    users = f.users
    if max_users and len(users) > max_users:
        users = np.sort(np.random.default_rng(seed).choice(users, max_users, replace=False))

    tr = v[v.week < c]

    p1 = np.bincount(tr.s[tr.week == c - 1], minlength=ns).astype(np.float32)
    p2 = np.bincount(tr.s[tr.week == c - 2], minlength=ns).astype(np.float32)

    first = np.full(ns, np.nan, np.float32)
    fw = tr.groupby('s').week.min()
    first[fw.index] = fw.values

    g = {}
    g['pop_1w'] = np.log1p(p1)
    g['pop_2w'] = np.log1p(p2)
    g['trend'] = np.log1p(p1) - np.log1p(p2)
    g['pop_all'] = np.log1p(np.bincount(tr.s, minlength=ns).astype(np.float32))
    g['n_users'] = np.log1p(np.asarray(f.H.sum(0)).ravel())
    g['age_w'] = c - first
    g['is_film'] = is_film

    eps = tr.groupby(['u', 's']).item_id.nunique()
    Eps = to_csr(eps.astype('float32'), nu, ns)

    u_nhist = np.diff(f.H.indptr).astype(np.float32)
    u_lastwk = np.bincount(tr.u[tr.week == c - 1], minlength=nu).astype(np.float32)

    feats = FEATURES + list(extra)
    Cs, Xs, Ys = [], [], []

    # for b0 in range(0, len(users), 2048):
    for b0 in range(0, len(users), 1024):
        ub = users[b0:b0 + 1024]

        Hd = f.H[ub].toarray()
        R = f.last_ts[ub].toarray()

        # hist = np.argsort(-np.where(Hd > 0, R, -np.inf), 1)[:, :N_HIST]
        hist = top_k(np.where(Hd > 0, R, -np.inf), N_HIST)
        nh = np.minimum((Hd > 0).sum(1), N_HIST)

        # tops = [np.argsort(-np.where(Hd > 0, -np.inf, m.score(f, ub)), 1)[:, :N_TOTAL] for m in explore]
        tops = [top_k(np.where(Hd > 0, -np.inf, m.score(f, ub)), N_TOTAL) for m in explore]

        C = np.full((len(ub), N_TOTAL), -1, np.int64)
        for i in range(len(ub)):
            row = list(hist[i, :nh[i]])
            seen = set(row)

            for cand in zip(*[t[i] for t in tops]):
                for s in cand:
                    if s not in seen:
                        row.append(s)
                        seen.add(s)
                if len(row) >= N_TOTAL:
                    break

            C[i, :min(len(row), N_TOTAL)] = row[:N_TOTAL]

        ok = C >= 0
        Ci = np.maximum(C, 0)

        last = np.take_along_axis(R, Ci, 1)

        cols = {}
        cols['in_hist'] = np.take_along_axis(Hd, Ci, 1)
        cols['days_since'] = np.where(last > 0, (c * WEEK - last) / 86400, 999)
        cols['u_w'] = np.take_along_axis(f.Xtr[ub].toarray(), Ci, 1)
        cols['u_eps'] = np.take_along_axis(Eps[ub].toarray(), Ci, 1)
        cols['u_nhist'] = np.broadcast_to(u_nhist[ub, None], C.shape)
        cols['u_lastwk'] = np.broadcast_to(u_lastwk[ub, None], C.shape)

        for k, a in g.items():
            cols[k] = a[Ci]

        for k, m in extra.items():
            cols[k] = np.take_along_axis(m.score(f, ub), Ci, 1)

        X = np.stack([np.where(ok, cols[k], np.nan) for k in feats], -1).astype(np.float32)
        y = np.where(ok, np.take_along_axis(f.Y[ub].toarray(), Ci, 1), 0).astype(np.float32)

        Cs.append(C)
        Xs.append(X)
        Ys.append(y)

    C = np.concatenate(Cs)
    X = np.concatenate(Xs)
    y = np.concatenate(Ys)

    tot = np.asarray(f.Y[users].sum(1)).ravel()
    cand_recall = float((y.sum(1)[tot > 0] / tot[tot > 0]).mean())

    return Bunch(X=X, y=y, C=C, users=users, feature_names=feats, fold=f, week=c, cand_recall=cand_recall)


def flatten(split):
    ok = split.C >= 0
    return split.X[ok], split.y[ok], ok.sum(1)
