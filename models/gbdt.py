import numpy as np
import lightgbm as lgb

from dataprep import flatten


def label(y):
    # return (y > 0).astype(int)
    return np.ceil(4 * y).astype(int)


def fit_gbdt(tr, va, feats=None):
    Xt, yt, gt = flatten(tr)
    Xv, yv, gv = flatten(va)

    if feats is None:
        feats = tr.feature_names
    idx = [tr.feature_names.index(f) for f in feats]

    m = lgb.LGBMRanker(objective='lambdarank', n_estimators=500, learning_rate=0.05, num_leaves=31,
                       min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

    m.fit(Xt[:, idx], label(yt), group=gt,
          eval_set=[(Xv[:, idx], label(yv))], eval_group=[gv], eval_at=[20],
          callbacks=[lgb.early_stopping(30, verbose=False)])

    return m, idx


def predict(m, idx, split):
    n, K, F = split.X.shape
    s = m.predict(split.X.reshape(-1, F)[:, idx])
    return s.reshape(n, K)
