import numpy as np
import scipy.sparse as sp
from sklearn.utils import Bunch

from .loaders import T0


def series_W(df, rule='max'):
    it = df.groupby(['u', 's', 'item_id'], sort=False).vision_factor.max()
    g = it.groupby(level=[0, 1], sort=False)

    if rule == 'max':
        w = g.max()
    elif rule == 'mean':
        w = g.mean()
    elif rule == 'bin':
        w = g.max() * 0 + 1.0
    elif rule == 'sat':
        w = 1 - np.exp(-g.sum())

    return w.astype('float32')


def to_csr(w, nu, ns):
    u = w.index.get_level_values(0).values
    s = w.index.get_level_values(1).values

    return sp.csr_matrix((w.values, (u, s)), shape=(nu, ns), dtype=np.float32)


def load_fold(views, t, rule='max'):
    v = views.frame
    nu, ns = views.n_users, views.n_series

    tr = v[v.week < t]
    te = v[v.week == t]

    Xtr = to_csr(series_W(tr, rule), nu, ns)
    H = (Xtr > 0).astype(np.float32).tocsr()
    Y = to_csr(series_W(te, rule), nu, ns)

    last = tr.groupby(['u', 's']).ts.max()
    last_ts = to_csr(last.astype('float32') - float(T0), nu, ns)

    lw = tr[tr.week == t - 1]
    pop_lastweek = np.asarray(to_csr(series_W(lw, rule), nu, ns).sum(0)).ravel()
    pop_all = np.asarray(Xtr.sum(0)).ravel()

    has_hist = np.diff(H.indptr) > 0
    has_test = np.diff(Y.indptr) > 0

    users = np.where(has_hist & has_test)[0]
    cold_users = np.where(~has_hist & has_test)[0]

    return Bunch(t=t, nu=nu, ns=ns, Xtr=Xtr, H=H, Y=Y, last_ts=last_ts,
                 pop_lastweek=pop_lastweek, pop_all=pop_all, users=users, cold_users=cold_users)
