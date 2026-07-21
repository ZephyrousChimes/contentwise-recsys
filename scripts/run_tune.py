import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np

from dataprep import load_views, load_fold
from models import Pop, ItemKNN, RP3b, RP3bColumn, EASE
from evaluation import evaluate


views = load_views()
f = load_fold(views, 9)

sub = np.random.default_rng(0).choice(f.users, 8000, replace=False)

grid = [Pop('lastweek'),
        ItemKNN(50, 10), ItemKNN(200, 10), ItemKNN(200, 100), ItemKNN(200, 10, binary=True),
        # RP3bColumn(100, 1.0, 0.0), RP3bColumn(200, 0.8, 0.5),
        RP3b(100, 1.0, 0.0), RP3b(100, 1.0, 0.5), RP3b(200, 0.8, 0.5), RP3b(200, 1.0, 0.3),
        EASE(50, 8000), EASE(500, 8000), EASE(2000, 8000)]

for m in grid:
    m.fit(f)
    r = evaluate(f, m, users=sub)
    print(m.name, json.dumps({k: round(x, 4) for k, x in r.items()}))
