import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt



def draw_series(s):
    plt.plot(s, label='s')
    plt.xlabel('t')
    plt.ylabel('workload_minmax')
    # plt.title('trend clusters valuation')
    plt.show()


class global_pool():
    def __init__(self, trend_class_num, seasonal_class_num):
        self.trend_pool = []
        self.seasonal_pool = []

        self.trend_class_num = trend_class_num
        self.seasonal_class_num = seasonal_class_num

    def build_pool_trend(self, series, classes):
        for i in range(self.trend_class_num):
            cluster_index = np.argwhere(classes == i).squeeze(1)
            if len(cluster_index) != 0:
                self.trend_pool.append(np.average(series[cluster_index], axis=0))
            else:
                self.trend_pool.append(np.zeros(series.shape[1]))
        self.trend_pool = np.asarray(self.trend_pool)

    def build_pool_seasonal(self, series, classes):
        for i in range(self.seasonal_class_num):
            cluster_index = np.argwhere(classes == i).squeeze(1)
            if len(cluster_index) != 0:
                self.seasonal_pool.append(np.average(series[cluster_index], axis=0))
            else:
                # self.seasonal_pool.append(np.zeros(series.shape[1]))
                pass
        self.seasonal_pool = np.asarray(self.seasonal_pool)


# trend_cluster_res = pkl.load(open("trend_cluster_res", 'rb'))
# seasonal_cluster_res = pkl.load(open("seasonal_cluster_res", 'rb'))
#
# trends = pkl.load(open('../../../raw_data/X_0801_0830_trend.pkl', 'rb'))
# seasonals = pkl.load(open('../../../raw_data/X_0801_0830_seasonal.pkl', 'rb'))
#
# S_trend, _ = get_pworkload_all(trends, trends, std='minmax')
# S_trend = np.asarray(S_trend)
#
# S_sea, _ = get_pworkload_all(seasonals, seasonals, std='minmax')
# S_sea = np.asarray(S_sea)
#
# cluster_num = 150
# global_pool_ = global_pool(cluster_num, cluster_num)
# global_pool_.build_pool_trend(S_trend, trend_cluster_res)
# global_pool_.build_pool_seasonal(S_sea, seasonal_cluster_res)
#
# pkl.dump(global_pool_, open('global_pool_trend{}_sea{}_series_minmax.pkl'.format(cluster_num, cluster_num), 'wb'))
# gp = pkl.load(open('global_pool_trend{}_sea{}_series_minmax.pkl'.format(150, 150), 'rb'))
# pass

