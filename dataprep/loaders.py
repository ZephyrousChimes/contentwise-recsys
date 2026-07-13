import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
from sklearn.utils import Bunch


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RAW = os.path.join(ROOT, 'data', 'ContentWiseImpressions', 'data', 'ContentWiseImpressions', 'CW10M')
CACHE = os.path.join(ROOT, 'cache')

WEEK = 7 * 86400
T0 = int(pd.Timestamp('2019-01-07 09:00:00').timestamp())


def build_cache():
    os.makedirs(CACHE, exist_ok=True)

    cols = ['user_id', 'item_id', 'series_id', 'item_type', 'recommendation_id',
            'interaction_type', 'vision_factor', 'utc_ts_milliseconds']

    # df = pd.read_parquet(RAW + '/interactions')
    # imp = pd.read_parquet(RAW + '/impressions-non-direct-link')
    t = pq.read_table(os.path.join(RAW, 'interactions'), columns=cols)
    t = t.filter(pc.equal(t['interaction_type'], 0))

    # v = t.to_pandas()
    v = t.to_pandas().reset_index()
    del t

    v['ts'] = (v.utc_ts_milliseconds // 1000).astype('int32')
    v = v.drop(columns=['interaction_type', 'utc_ts_milliseconds'])

    v['vision_factor'] = v.vision_factor.astype('float32')
    v = v.sort_values('ts').reset_index(drop=True)

    v.to_parquet(os.path.join(CACHE, 'views.parquet'))
    print('views', len(v))

    imp = pq.read_table(os.path.join(RAW, 'impressions-direct-link')).to_pandas().reset_index()
    imp.to_parquet(os.path.join(CACHE, 'imp_direct.parquet'))
    print('impressions', len(imp))


def load_views():
    path = os.path.join(CACHE, 'views.parquet')
    if not os.path.exists(path):
        build_cache()

    v = pd.read_parquet(path)
    v['week'] = ((v.ts - T0) // WEEK).astype('int8')

    sids = np.sort(v.series_id.unique())
    uids = np.sort(v.user_id.unique())

    v['s'] = np.searchsorted(sids, v.series_id).astype('int32')
    v['u'] = np.searchsorted(uids, v.user_id).astype('int32')

    return Bunch(frame=v, n_users=len(uids), n_series=len(sids), series_ids=sids, user_ids=uids)


def load_impressions():
    path = os.path.join(CACHE, 'imp_direct.parquet')
    if not os.path.exists(path):
        build_cache()

    imp = pd.read_parquet(path)
    return Bunch(frame=imp)


def series_is_film(views):
    v = views.frame

    it = v.groupby('series_id').item_type.agg(lambda x: x.mode().iat[0])
    it = it.reindex(views.series_ids).values

    return (it == 0).astype(np.float32)


def load_theta(views, K=20):
    path = os.path.join(CACHE, 'theta.npy')
    if os.path.exists(path):
        return np.load(path)

    v = views.frame
    imp = load_impressions().frame

    lv = v[v.recommendation_id >= 0][['recommendation_id', 'series_id']].drop_duplicates()
    lv = lv.merge(imp[['recommendation_id', 'recommended_series_list']], on='recommendation_id')

    pos = np.array([list(l).index(s) for s, l in zip(lv.series_id, lv.recommended_series_list)])
    L = imp[imp.recommendation_id.isin(lv.recommendation_id)].recommendation_list_length.values

    clicks = np.bincount(pos, minlength=50)[:K]
    exposed = np.array([(L > k).sum() for k in range(K)])

    # theta = clicks / clicks[0]
    rate = clicks / exposed
    theta = (rate / rate[0]).astype(np.float32)

    np.save(path, theta)
    return theta
