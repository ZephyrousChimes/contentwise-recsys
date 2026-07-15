import numpy as np

from .scorer import NextWeekScorer
from .metrics import build_feed, topk


def lists_from_scores(split, S, K=20, rho=None):
    S = np.where(split.C >= 0, S, -np.inf)

    order = np.argsort(-S, 1)
    ranked = np.take_along_axis(split.C, order, 1)

    if rho is None:
        return ranked[:, :K]

    j = split.feature_names.index('in_hist')
    inh = np.take_along_axis(split.X[:, :, j], order, 1) > 0

    n_ex = int(np.ceil(rho * K))
    n_cw = K - n_ex

    out = np.full((len(S), K), -1)
    for i in range(len(S)):
        h = ranked[i][inh[i] & (ranked[i] >= 0)][:n_cw]
        e = ranked[i][~inh[i] & (ranked[i] >= 0)][:K - len(h)]

        r = np.concatenate([h, e])[:K]
        out[i, :len(r)] = r

    return out


def evaluate_ranker(split, S, name, theta, K=20):
    sc = NextWeekScorer(split.fold, theta)
    res = {'model': name, 'week': split.week}

    for tag, rho in [('top', None), ('feed.5', 0.5)]:
        lists = lists_from_scores(split, S, K, rho)
        for k, x in sc.summary(split.users, lists).items():
            res[f'{tag}_{k}'] = round(x, 4)

    return res


def baselines(split, theta, cw, ease, K=20):
    f = split.fold
    u = split.users
    sc = NextWeekScorer(f, theta)

    # C = cw.score(f, u); E = ease.score(f, u); Hd = f.H[u].toarray()
    top, feed = [], []
    for i in range(0, len(u), 1024):
        ub = u[i:i + 1024]

        C = cw.score(f, ub)
        E = ease.score(f, ub)
        Hd = f.H[ub].toarray()

        top.append(topk(C, K))
        feed.append(build_feed(C, E, Hd, K, .5))

    top = np.concatenate(top)
    feed = np.concatenate(feed)

    r = {'model': 'cw_recency', 'week': split.week}
    for k, x in sc.summary(u, top).items():
        r['top_' + k] = round(x, 4)
    for k, x in sc.summary(u, feed).items():
        r['feed.5_' + k] = round(x, 4)

    return [r]
