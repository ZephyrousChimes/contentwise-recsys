import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import gc
import json
import numpy as np
import pandas as pd

from dataprep import load_views, load_fold, load_rank_split, load_theta, series_is_film
from models import EASE, ContinueRecency
from models.gbdt import fit_gbdt, predict
from evaluation import evaluate_ranker, baselines


views = load_views()
is_film = series_is_film(views)
theta = load_theta(views)

weeks = [12, 13]
if len(sys.argv) > 1:
    weeks = [int(w) for w in sys.argv[1].split(',')]


def make_split(c, n):
    f = load_fold(views, c)
    ease = EASE(500, 8000).fit(f)

    sp = load_rank_split(views, c, is_film, explore=[ease], extra={'ease': ease}, fold=f, max_users=n)
    return sp, ease


rows = []
for t in weeks:
    tr, _ = make_split(t - 2, 8000)
    va, _ = make_split(t - 1, 4000)
    te, ease = make_split(t, 12000)
    gc.collect()

    print('week', t, 'cand recall', round(tr.cand_recall, 3), round(va.cand_recall, 3), round(te.cand_recall, 3))

    rows += baselines(te, theta, ContinueRecency().fit(te.fold), ease)

    order = -np.arange(te.C.shape[1])[None, :].repeat(len(te.users), 0)
    rows.append(evaluate_ranker(te, order, 'candidate order', theta))

    pop_feats = ['pop_1w', 'pop_2w', 'trend', 'pop_all', 'n_users', 'age_w', 'is_film']

    for name, feats in [('gbdt_pop_only', pop_feats), ('gbdt_all', None)]:
        m, idx = fit_gbdt(tr, va, feats)

        r = evaluate_ranker(te, predict(m, idx, te), name, theta)
        r['trees'] = m.best_iteration_

        rows.append(r)
        print(json.dumps(r))

    imp = pd.Series(m.booster_.feature_importance('gain'), index=[te.feature_names[i] for i in idx])
    print((imp / imp.sum()).sort_values(ascending=False).round(3))

    del tr, va, te, ease
    gc.collect()

df = pd.DataFrame(rows)
df.to_csv('results/model1.csv', index=False)
print(df.groupby('model')[[c for c in df.columns if c.startswith(('top_', 'feed'))]].mean().round(4))
