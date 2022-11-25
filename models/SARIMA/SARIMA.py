import itertools
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# load data
def load_data(file_name, train_days):
    bd = pd.read_pickle(file_name)
    bd = bd[(bd['name'] == 'ecdnd')].sort_values(by='time_id')
    bd['bw_upload'] = bd['bw_upload']/1024/1024/1024  # GB
    bd_y = pd.Series(bd['bw_upload'])
    bd_y.index = pd.Index(range(len(bd_y)))
    bd_y.plot()
    plt.show()

    bd_y_train = bd_y.iloc[:288*train_days]
    bd_y_test = bd_y.iloc[288*train_days:]
    return bd_y, bd_y_train, bd_y_test


def tune_model():
    # 2.下面我们先对非平稳时间序列进行时间序列的差分，找出适合的差分次数d的值：
    # fig = plt.figure(figsize=(12, 8))
    # ax1 = fig.add_subplot(111)
    # diff1 = bd_y.diff(1)
    # diff1.plot(ax=ax1)
    # # 这里是做了1阶差分，可以看出时间序列的均值和方差基本平稳，不过还是可以比较一下二阶差分的效果：
    #
    # # 这里进行二阶差分
    # fig = plt.figure(figsize=(12, 8))
    # ax2 = fig.add_subplot(111)
    # diff2 = bd_y.diff(2)
    # diff2.plot(ax=ax2)
    # # 由下图可以看出来一阶跟二阶的差分差别不是很大，所以可以把差分次数d设置为1，上面的一阶和二阶程序我们注释掉
    #
    # # 这里我们使用一阶差分的时间序列
    # # 3.接下来我们要找到ARIMA模型中合适的p和q值：
    # data1 = bd_y.diff(1)
    # data1.dropna(inplace=True)
    # # 加上这一步，不然后面画出的acf和pacf图会是一条直线
    #
    # # 第一步：先检查平稳序列的自相关图和偏自相关图
    # fig = plt.figure(figsize=(12, 8))
    # ax1 = fig.add_subplot(211)
    # fig = sm.graphics.tsa.plot_acf(data1, lags=40, ax=ax1)
    # #lags 表示滞后的阶数
    # #第二步：下面分别得到acf 图和pacf 图
    # ax2 = fig.add_subplot(212)
    # fig = sm.graphics.tsa.plot_pacf(data1, lags=40, ax=ax2)

    # plt.show()

    # 首先定义 p、d、q 的参数值范围，这里取 0 - 2.
    p = d = q = range(0, 2)

    # 然后用itertools函数生成不同的参数组合
    pdq = list(itertools.product(p, d, q))

    # 同理处理季节周期性参数，也生成相应的多个组合
    seasonal_pdq = [(x[0], x[1], x[2], 288) for x in list(itertools.product(p, d, q))]

    smallest = float('inf')
    sr = {}

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = sm.tsa.statespace.SARIMAX(bd_y.astype(float).values, order=param, seasonal_order=param_seasonal,
                                                  enforce_stationarity=False, enforce_invertibility=False)
                results = model.fit(disp=0)

                if smallest > results.aic:
                    smallest = results
                    sr = {'results': results, 'param': param, 'param_seasonal': param_seasonal}

                print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))

            except:
                continue


def test(y, y_train, y_test, pre_len, param, param_seasonal):
    param = (0, 1, 0)
    param_seasonal = (0, 1, 0, 288)
    # model = sm.tsa.statespace.SARIMAX(y_train.astype(float).values, order=param, seasonal_order=param_seasonal,
    #                                               enforce_stationarity=False, enforce_invertibility=False)
    #
    # results = model.fit(disp=0)
    # print("aic:{}".format(results.aic))
    #
    # results.save("sarima2.pkl")

    results = pickle.load(open("sarima2.pkl", 'rb'))

    pred = results.get_prediction(start=len(y_train), end=len(y_train)+pre_len-1, dynamic=False)  # 预测值
    pred_ci = pred.conf_int()  # 置信区间

    # 画出预测值和真实值的plot图

    ax = y.plot(label='observed')
    pre_draw = pd.Series(pred.predicted_mean)
    pre_draw.index = pd.Index(range(len(y_train), len(y_train)+pre_len))
    pre_draw.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

    # ax.fill_between(pred_ci.index,
    #                 pred_ci.iloc[:, 0],
    #                 pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Time')
    ax.set_ylabel('Workloads')
    plt.legend()

    plt.savefig("normal test2.jpg")

    y_forecasted = pred.predicted_mean
    y_truth = y_test.values

    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('The Mean Squared Error of forecasts is {}'.format(mse))
    print('The Root Mean Squared Error of forecasts is {}'.format(np.sqrt(sum((y_forecasted-y_truth)**2)/len(y_forecasted))))
    print('The Mean Absolute Percentage Error is {}'.format(sum(abs(y_forecasted-y_truth)/y_truth)/len(y_forecasted)))


if __name__ == '__main__':
    data_path = "../../raw_data/f96b_20220825-20220902_bd.pkl"
    bd_y, bd_y_train, bd_y_test = load_data(data_path, 8)

    test(bd_y, bd_y_train, bd_y_test, len(bd_y_test), (), ())
