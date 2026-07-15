import numpy as np


def topk(S, K):
    idx = np.argpartition(-S, K, 1)[:, :K]
    o = np.argsort(-np.take_along_axis(S, idx, 1), 1)
    return np.take_along_axis(idx, o, 1)


def ndcg_rows(rec, Yd, K):
    g = np.take_along_axis(Yd, rec[:, :K], 1)
    disc = 1 / np.log2(np.arange(2, K + 2))

    dcg = (g * disc).sum(1)
    ideal = -np.sort(-Yd, 1)[:, :K]
    idcg = (ideal * disc).sum(1)

    with np.errstate(invalid='ignore', divide='ignore'):
        nd = np.where(idcg > 0, dcg / idcg, np.nan)
    return nd, g


def build_feed(cw_s, ex_s, Hd, K, rho):
    n_ex = int(np.ceil(rho * K))
    n_cw = K - n_ex

    cw = np.where(Hd > 0, cw_s, -np.inf)
    ex = np.where(Hd > 0, -np.inf, ex_s)

    if n_cw:
        cw_top = topk(cw, n_cw)
    else:
        cw_top = np.zeros((len(Hd), 0), int)
    ex_top = topk(ex, K)

    nh = (Hd > 0).sum(1)
    out = np.empty((len(Hd), K), dtype=np.int64)

    for i in range(len(Hd)):
        c = cw_top[i, :min(n_cw, nh[i])]
        out[i] = np.concatenate([c, ex_top[i, :K - len(c)]])

    return out


def evaluate(fold, model, K=20, rhos=(0, .25, .5, .75, 1.0), cw_model=None, batch=2048, users=None):
    if users is None:
        users = fold.users

    acc = {'all': [], 'new': [], 'rep': []}
    feed = {r: ([], [], []) for r in rhos}

    for b0 in range(0, len(users), batch):
        ub = users[b0:b0 + batch]

        S = model.score(fold, ub).astype(np.float32)
        Hd = fold.H[ub].toarray()
        Yd = fold.Y[ub].toarray()

        Ynew = np.where(Hd > 0, 0, Yd)
        Yrep = np.where(Hd > 0, Yd, 0)

        acc['all'].append(ndcg_rows(topk(S, K), Yd, K)[0])
        acc['new'].append(ndcg_rows(topk(np.where(Hd > 0, -np.inf, S), K), Ynew, K)[0])
        acc['rep'].append(ndcg_rows(topk(np.where(Hd > 0, S, -np.inf), K), Yrep, K)[0])

        if cw_model is None:
            continue

        C = cw_model.score(fold, ub)
        for r in rhos:
            rec = build_feed(C, S, Hd, K, r)
            n, g = ndcg_rows(rec, Yd, K)

            isnew = np.take_along_axis(Hd, rec, 1) == 0
            feed[r][0].append(n)
            feed[r][1].append((g * isnew).sum(1))
            feed[r][2].append(g.sum(1))

    res = {}
    for k, a in acc.items():
        res['ndcg_' + k] = float(np.nanmean(np.concatenate(a)))

    for r, (n, gn, ga) in feed.items():
        if not n:
            continue

        gn = np.concatenate(gn)
        ga = np.concatenate(ga)
        m = ga > 0

        res[f'feed{r:.2f}_ndcg'] = float(np.nanmean(np.concatenate(n)))
        res[f'feed{r:.2f}_disc'] = float((gn[m] / ga[m]).mean())

    return res


def true_discovery(fold):
    Yd = fold.Y[fold.users]
    H = fold.H[fold.users]

    tot = np.asarray(Yd.sum(1)).ravel()
    rep = np.asarray(Yd.multiply(H).sum(1)).ravel()

    return float(((tot - rep) / tot).mean())
