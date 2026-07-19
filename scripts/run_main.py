import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

from dataprep import load_views, load_fold
from models import Pop, ContinueRecency, ItemKNN, RP3b, EASE
from evaluation import evaluate, true_discovery


views = load_views()

rule = 'max'
if len(sys.argv) > 1:
    rule = sys.argv[1]

rows = []
for t in [10, 11, 12, 13]:
    f = load_fold(views, t, rule=rule)
    cw = ContinueRecency().fit(f)

    print('week', t, 'true disc', round(true_discovery(f), 3))

    for m in [Pop('lastweek'), ItemKNN(200, 10), RP3b(200, 1.0, 0.3), EASE(500, 8000), ContinueRecency()]:
        m.fit(f)

        r = evaluate(f, m, cw_model=cw, rhos=(0, .25, .5, .75))
        r.update(model=m.name, week=t, rule=rule)

        rows.append(r)
        print(m.name, round(r['ndcg_all'], 4), round(r['ndcg_new'], 4), round(r['ndcg_rep'], 4))

df = pd.DataFrame(rows)
df.to_csv(f'results/main_{rule}.csv', index=False)
print(df.drop(columns=['week', 'rule']).groupby('model').mean().round(4).T)
