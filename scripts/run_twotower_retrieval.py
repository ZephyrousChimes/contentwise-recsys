import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np

from dataprep import load_views, load_fold
from models import Pop, EASE
from models.twotower import TwoTower
from evaluation import evaluate


views = load_views()
f = load_fold(views, 13)

sub = np.random.default_rng(0).choice(f.users, 8000, replace=False)

# for m in [Pop('lastweek'), EASE(500, 8000), TwoTower(views, d=64), TwoTower(views, d=128, epochs=5)]:
for m in [TwoTower(views, d=64), TwoTower(views, d=128, epochs=5)]:
    m.fit(f)
    r = evaluate(f, m, users=sub)
    print(m.name, json.dumps({k: round(x, 4) for k, x in r.items()}))
