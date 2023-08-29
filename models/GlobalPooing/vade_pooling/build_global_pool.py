import numpy as np
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


def build_global_pool(decomp_X, seasonal_cluster_res, best_n):
    cluster_num = best_n
    global_pool_ = global_pool(cluster_num, cluster_num)
    global_pool_.build_pool_seasonal(decomp_X, seasonal_cluster_res)

    return global_pool_