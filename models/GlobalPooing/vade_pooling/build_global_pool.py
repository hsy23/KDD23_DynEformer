import pickle

from global_pool import global_pool
from dataloader import get_pworkload
import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt

# aic_l = pickle.load(open('seasonal_aic_l.pkl', 'rb'))
# bic_l = pickle.load(open('seasonal_bic_l.pkl', 'rb'))
# plt.plot(aic_l, label='aic')
# plt.plot(bic_l, label='bic')
# plt.xlabel('n_clusters')
# plt.ylabel('aic/bic')
# plt.xticks(range(len(aic_l)), labels=range(1, 600, 50))
# plt.title('seasonal clusters valuation')
# plt.legend()
# plt.show()

# trend_cluster_res = pkl.load(open("trend_cluster_res", 'rb'))


def build_global_pool(best_n, args):
    seasonal_cluster_res = pkl.load(open('ns_cluster_res_n{}_s{}_s{}'.format(best_n, args.series_len, args.step), 'rb'))

    # trends = pkl.load(open('../../../raw_data/X_0801_0830_trend.pkl', 'rb'))
    seasonals = pkl.load(open('../../../../raw_data/X_0801_0830_seasonal.pkl', 'rb'))[:, :24*25]

    # S_trend, _ = get_pworkload_all(trends, trends, std='minmax')
    # S_trend = np.asarray(S_trend)

    _, _, S_sea = get_pworkload(seasonals, seasonals, std='minmax', series_len=args.series_len, step=args.step)
    S_sea = np.asarray(S_sea)

    # pickle.dump(S_sea, open('../../draw_pics/S_sea.pkl', 'wb'))

    cluster_num = best_n
    global_pool_ = global_pool(cluster_num, cluster_num)
    # global_pool_.build_pool_trend(S_trend, trend_cluster_res)
    global_pool_.build_pool_seasonal(S_sea, seasonal_cluster_res)

    pkl.dump(global_pool_, open('global_pool_c{}_s{}_s{}.pkl'.format(cluster_num, args.series_len, args.step), 'wb'))


# with open(r'global_pool_c551_s48_s12.pkl', 'rb') as f:
#     timeFeature_pool = pickle.load(f)
#     pass

