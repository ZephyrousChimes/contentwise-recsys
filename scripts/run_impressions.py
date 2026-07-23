import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from dataprep import load_views, load_impressions, load_fold
from models import Pop, ContinueRecency, ItemKNN, RP3b, EASE
from evaluation import ndcg_rows


views = load_views()
v = views.frame
sids = views.series_ids

imp = load_impressions().frame

lv = v[v.recommendation_id >= 0]
clk = lv.groupby(['recommendation_id', 'series_id']).agg(W=('vision_factor', 'max'), u=('u', 'first'), week=('week', 'min')).reset_index()
print('click pairs', len(clk), 'lists', clk.recommendation_id.nunique())

clk = clk.merge(imp[['recommendation_id', 'recommended_series_list']], on='recommendation_id')
clk['pos'] = [list(l).index(s) if s in set(l) else -1 for s, l in zip(clk.series_id, clk.recommended_series_list)]

print('found in list', (clk.pos >= 0).mean())
print(clk[clk.pos >= 0].pos.value_counts(normalize=True).sort_index().head(12).round(3))
print(clk[clk.pos >= 0].groupby('pos').W.mean().head(8).round(3))

ex = imp.explode('recommended_series_list')
ex['slot'] = ex.groupby(level=0).cumcount()
var = ex.groupby('recommended_series_list').slot.agg(['nunique', 'size'])
print('varied slots', (var[var['size'] >= 20]['nunique'] >= 5).mean())

rows = []
for t in [10, 11, 12, 13]:
    f = load_fold(views, t)

    c = clk[(clk.week == t) & (clk.pos >= 0)]
    lists = c.groupby('recommendation_id').agg(u=('u', 'first')).join(imp.set_index('recommendation_id').recommended_series_list)
    lists = lists[lists.recommended_series_list.map(len) >= 2]

    gains = {rid: dict(zip(d.series_id, d.W)) for rid, d in c.groupby('recommendation_id')}

    n = len(lists)
    Kmax = int(lists.recommended_series_list.map(len).max())

    I = np.full((n, Kmax), -1)
    G = np.zeros((n, Kmax), np.float32)

    for i, (rid, l) in enumerate(zip(lists.index, lists.recommended_series_list)):
        l = np.asarray(l)
        g = gains[rid]

        idx = np.searchsorted(sids, l)
        idx[~np.isin(l, sids)] = -1

        I[i, :len(l)] = idx
        G[i, :len(l)] = [g.get(s, 0.0) for s in l]

    pad = np.arange(Kmax)[None, :] >= lists.recommended_series_list.map(len).values[:, None]

    def run(name, S):
        S = np.where(pad, -np.inf, S)
        order = np.argsort(-S, 1, kind='stable')

        nd, _ = ndcg_rows(order, G, 10)
        hit1 = (np.take_along_axis(G, order[:, :1], 1) > 0).mean()

        rows.append(dict(week=t, model=name, lists=n, ndcg10=float(np.nanmean(nd)), hit1=float(hit1)))
        print(t, name, round(rows[-1]['ndcg10'], 4), round(hit1, 4))

    run('logged_order', -np.arange(Kmax, dtype=float)[None, :].repeat(n, 0))
    run('random', np.random.default_rng(0).random((n, Kmax)))

    for m in [Pop('lastweek'), ContinueRecency(), ItemKNN(200, 10), RP3b(200, 1.0, 0.3), EASE(500, 8000)]:
        m.fit(f)

        us = lists.u.values
        S = np.zeros((n, Kmax), np.float32)

        for b0 in range(0, n, 2048):
            full = m.score(f, us[b0:b0 + 2048])
            Ib = I[b0:b0 + 2048]
            S[b0:b0 + 2048] = np.where(Ib >= 0, np.take_along_axis(full, np.maximum(Ib, 0), 1), -1e9)

        run(m.name, S)

df = pd.DataFrame(rows)
df.to_csv('results/impressions.csv', index=False)
print(df.groupby('model')[['ndcg10', 'hit1', 'lists']].mean().round(4).sort_values('ndcg10'))
