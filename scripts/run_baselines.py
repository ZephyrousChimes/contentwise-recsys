import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import pandas as pd

from dataprep import load_views, load_fold
from models import Random, Pop, ContinueRecency, ContinueDecayed
from evaluation import evaluate, true_discovery


views = load_views()
print('users', views.n_users, 'series', views.n_series)

weeks = [10, 11, 12, 13]
if len(sys.argv) > 1:
    weeks = [int(w) for w in sys.argv[1].split(',')]

rows = []
for t in weeks:
    f = load_fold(views, t)
    print('week', t, 'users', len(f.users), 'cold', len(f.cold_users), 'true disc', round(true_discovery(f), 3))

    cw = ContinueRecency().fit(f)

    for m in [Random(), Pop('lastweek'), Pop('all'), ContinueRecency(), ContinueDecayed(3), ContinueDecayed(14)]:
        m.fit(f)

        r = evaluate(f, m, cw_model=cw)
        r['model'] = m.name
        r['week'] = t

        rows.append(r)
        print(json.dumps(r))

df = pd.DataFrame(rows)
df.to_csv('results/baselines.csv', index=False)
print(df.drop(columns=['week']).groupby('model').mean().round(4).T)
