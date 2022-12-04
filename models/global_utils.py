import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import datetime
import pickle
import random

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

MinMax = preprocessing.MinMaxScaler()
StdScaler = preprocessing.StandardScaler()


Task_ID_Dict = pickle.load(open(r"C:\Users\Admin\Desktop\WWW_master\raw_data\Task_ID_Dict.pkl", 'rb'))


def train_test_split(X, y, train_ratio=0.7):
    num_ts, num_periods, num_features = X.shape
    # train_periods = int(num_periods * train_ratio)
    train_periods = int(24 * 25)
    random.seed(2)
    Xtr = X[:, :train_periods, :]
    ytr = y[:, :train_periods]
    Xte = X[:, train_periods:, :]
    yte = y[:, train_periods:]
    return Xtr, ytr, Xte, yte


class StandardScaler:

    def fit_transform(self, y):
        self.mean = np.mean(y)
        self.std = np.std(y) + 1e-4
        return (y - self.mean) / self.std

    def inverse_transform(self, y):
        return y * self.std + self.mean

    def transform(self, y):
        return (y - self.mean) / self.std


class MaxScaler:

    def fit_transform(self, y):
        self.max = np.max(y)
        return y / self.max

    def inverse_transform(self, y):
        return y * self.max

    def transform(self, y):
        return y / self.max


class MinMaxScaler:

    def fit_transform(self, y):
        self.max = np.max(y)
        self.min = np.min(y)
        return (y - self.min) / (self.max - self.min)

    def inverse_transform(self, y):
        return y * (self.max - self.min) + self.min

    def transform(self, y):
        return (y - self.min) / (self.max - self.min)


class MeanScaler:

    def fit_transform(self, y):
        self.mean = np.mean(y)
        return y / self.mean

    def inverse_transform(self, y):
        return y * self.mean

    def transform(self, y):
        return y / self.mean


class LogScaler:

    def fit_transform(self, y):
        return np.log1p(y)

    def inverse_transform(self, y):
        return np.expm1(y)

    def transform(self, y):
        return np.log1p(y)


def get_use_rate(data_base, data_95):  # 这里有一点逻辑问题，一台机器上的混跑业务利用率相同
    merge_use = pd.merge(data_base, data_95, how='left', left_on=['device_uuid', 'dt'], right_on=['machine_id', 'dt'])
    merge_use['use_rate'] = merge_use['bw_real_upload2_95']/merge_use['upbandwidth_base']

    # 删除无用数据
    data_base = pd.merge(data_base, merge_use[['device_uuid', 'use_rate', 'dt']], how='left', on=['device_uuid', 'dt'])
    return data_base


def isin_req(data_base):
    group_cols = ['province', 'isp', 'bandwidth_type', 'upbandwidth', 'dt']
    tmp = data_base[group_cols].copy()
    tmp['upbandwidth'] = tmp['upbandwidth'].apply(lambda x: int(x/1024/1024/1000))  # GB
    tmp = tmp.groupby(['province', 'isp', 'bandwidth_type', 'dt']).sum().reset_index()
    tmp['isin_req'] = tmp['upbandwidth'] <= 100
    data_base = pd.merge(data_base, tmp[['province', 'isp', 'bandwidth_type', 'dt', 'isin_req']],
                         on=['province', 'isp', 'bandwidth_type', 'dt'])
    return data_base


def task2id(task_name):  # todo:20220405 生成字典时带上字典对应的数据日期，在编码完成后再进行存储
    if task_name not in Task_ID_Dict.keys():
        print("字典未知任务:", task_name)
        Task_ID_Dict[task_name] = max(Task_ID_Dict.values()) + 1
        pickle.dump(Task_ID_Dict, open("Task_ID_Dick.pkl", 'wb'))
    return Task_ID_Dict[task_name]


def std_data(train_df, test_df, cols, way):
    train_res = train_df.copy()
    test_res = test_df.copy()
    std_scalers = []
    if way == 'minmax':
        scaler = MinMax
    else:
        scaler = StdScaler

    for col in cols:
        train_res[col] = scaler.fit_transform(np.array(train_res[col]).reshape(-1, 1))
        test_res[col] = scaler.transform(np.array(test_res[col]).reshape(-1, 1))
        std_scalers.append(scaler)

    # 关于标准化后复原的参考：
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.
    # MinMaxScaler.html?highlight=minmaxscaler#sklearn.preprocessing.MinMaxScaler.fit
    # bandwidth = new_df.loc[:, 'upbandwidth']
    # bd_scaler = scaler.fit(np.array(bandwidth).reshape(-1, 1))
    # new_df.loc[:, 'upbandwidth'] = bd_scaler.transform(np.array(bandwidth).reshape(-1, 1))  # 带宽做归一化
    # new_df_test.loc[:, 'upbandwidth'] = bd_scaler.transform(np.array(new_df_test.loc[:, 'upbandwidth']).reshape(-1, 1))
    #
    # memory_size = new_df['memory_size']
    # ms_scaler = scaler.fit(np.array(memory_size).reshape(-1, 1))
    # new_df.loc[:, 'memory_size'] = ms_scaler.transform(np.array(memory_size).reshape(-1, 1))  # 内存做归一化
    # new_df_test.loc[:, 'memory_size'] = ms_scaler.transform(np.array(new_df_test.loc[:, 'memory_size']).reshape(-1, 1))
    #
    # disk_size = new_df['disk_size']
    # ds_scaler = scaler.fit(np.array(disk_size).reshape(-1, 1))
    # new_df.loc[:, 'disk_size'] = ds_scaler.transform(np.array(disk_size).reshape(-1, 1))  # 硬盘做归一化
    # new_df_test.loc[:, 'disk_size'] = ds_scaler.transform(np.array(new_df_test.loc[:, 'disk_size']).reshape(-1, 1))  # 硬盘做归一化
    #
    # cpu_size = new_df['cpu_num']
    # cpu_scaler = scaler.fit(np.array(cpu_size).reshape(-1, 1))
    # new_df.loc[:, 'cpu_num'] = cpu_scaler.transform(np.array(cpu_size).reshape(-1, 1))  # CPU数目做归一化
    # new_df_test.loc[:, 'cpu_num'] = cpu_scaler.transform(np.array(new_df_test.loc[:, 'cpu_num']).reshape(-1, 1))  # CPU数目做归一化

    # year = new_df['year']
    # year_scaler = scaler.fit(np.array(year).reshape(-1, 1))
    # new_df.loc[:, 'year'] = year_scaler.transform(np.array(year).reshape(-1, 1))  # CPU数目做归一化
    # new_df_test.loc[:, 'year'] = year_scaler.transform(
    #     np.array(new_df_test.loc[:, 'year']).reshape(-1, 1))  # CPU数目做归一化
    #
    # mon = new_df['mon']
    # mon_scaler = scaler.fit(np.array(mon).reshape(-1, 1))
    # new_df.loc[:, 'mon'] = mon_scaler.transform(np.array(mon).reshape(-1, 1))  # CPU数目做归一化
    # new_df_test.loc[:, 'mon'] = mon_scaler.transform(
    #     np.array(new_df_test.loc[:, 'mon']).reshape(-1, 1))  # CPU数目做归一化
    #
    # week = new_df['week']
    # week_scaler = scaler.fit(np.array(week).reshape(-1, 1))
    # new_df.loc[:, 'week'] = week_scaler.transform(np.array(week).reshape(-1, 1))  # CPU数目做归一化
    # new_df_test.loc[:, 'week'] = week_scaler.transform(
    #     np.array(new_df_test.loc[:, 'week']).reshape(-1, 1))  # CPU数目做归一化
    #
    # day = new_df['day']
    # day_scaler = scaler.fit(np.array(day).reshape(-1, 1))
    # new_df.loc[:, 'day'] = day_scaler.transform(np.array(day).reshape(-1, 1))  # CPU数目做归一化
    # new_df_test.loc[:, 'day'] = day_scaler.transform(
    #     np.array(new_df_test.loc[:, 'day']).reshape(-1, 1))  # CPU数目做归一化

    # profit = new_df['profit']
    # pf_scaler = StdScaler.fit(np.array(profit).reshape(-1, 1))
    # new_df.loc[:, 'profit'] = pf_scaler.transform(np.array(profit).reshape(-1, 1))  # 收益做归一化
    #
    # bd_pro = new_df['biz_bandwidth_expenses']
    # bd_pro_scaler = StdScaler.fit(np.array(bd_pro).reshape(-1, 1))
    # new_df.loc[:, 'biz_bandwidth_expenses'] = bd_pro_scaler.transform(np.array(bd_pro).reshape(-1, 1))  # 运用商收益做归一化

    return train_res, test_res, std_scalers


def std_data_test(df, **scalers):  # 将离散数值编码为code, 在K-prototype算法集成了这一步
    new_df = df.copy()
    bandwidth = new_df.loc[:, 'upbandwidth']
    memory_size = new_df['memory_size']
    disk_size = new_df['disk_size']
    cpu_size = new_df['cpu_num']
    year = new_df['year']
    mon = new_df['mon']
    week = new_df['week']
    day = new_df['day']

    new_df.loc[:, 'upbandwidth'] = scalers['bd_scaler'].transform(np.array(bandwidth).reshape(-1, 1))  # 带宽做归一化
    new_df.loc[:, 'memory_size'] = scalers['ms_scaler'].transform(np.array(memory_size).reshape(-1, 1))  # 内存做归一化
    new_df.loc[:, 'disk_size'] = scalers['ds_scaler'].transform(np.array(disk_size).reshape(-1, 1))  # 硬盘做归一化
    new_df.loc[:, 'cpu_num'] = scalers['cpu_scaler'].transform(np.array(cpu_size).reshape(-1, 1))  # CPU数目做归一化
    new_df.loc[:, 'year'] = scalers['year'].transform(np.array(year).reshape(-1, 1))  # CPU数目做归一化
    new_df.loc[:, 'mon'] = scalers['mon'].transform(np.array(mon).reshape(-1, 1))  # CPU数目做归一化
    new_df.loc[:, 'week'] = scalers['week'].transform(np.array(week).reshape(-1, 1))  # CPU数目做归一化
    new_df.loc[:, 'day'] = scalers['day'].transform(np.array(day).reshape(-1, 1))  # CPU数目做归一化
    return new_df


def discrete_feats_encoder(train_x, test_x, train_y, cols, std=None):
    train_res = train_x.copy()
    test_res = test_x.copy()
    encode_dicts = []
    std_scalers = []

    if std == 'minmax':
        scaler = MinMax
    else:
        scaler = StdScaler

    y_mean = train_y.mean()
    smoothing = 0.8
    for col in cols:
        print(col)
        train_x_col = train_res[col].copy()
        test_x_col = test_res[col].copy()
        target_xy = pd.concat((train_x_col, train_y), axis=1)

        # sorted_x = sorted(set(train_x_col))
        # encode_dict = dict(zip(sorted_x, range(1, len(sorted_x) + 1)))
        # encode_dict = train_x_col.value_counts().to_dict()

        count_dict = train_x_col.value_counts().to_dict()
        mean_dict = target_xy.groupby(col).mean().to_dict()['use_rate']

        count_encoding = train_x_col.replace(count_dict)
        mean_encoding = train_x_col.replace(mean_dict)

        weight = smoothing / (1 + np.exp(-(count_encoding - 1)))
        target_encoding = mean_encoding * weight + float(y_mean) * (1 - weight)
        train_encoding = target_encoding

        # test dataset
        count_encoding = test_x_col.replace(count_dict)
        mean_encoding = test_x_col.replace(mean_dict)

        # 用训练集均值替代测试集中新出现的特征编码
        count_encoding = count_encoding.apply(pd.to_numeric, errors='coerce')
        count_encoding = count_encoding.fillna(count_encoding.mean())

        mean_encoding = mean_encoding.apply(pd.to_numeric, errors='coerce')
        mean_encoding = mean_encoding.fillna(mean_encoding.mean())

        weight = smoothing / (1 + np.exp(-(count_encoding - 1)))  # 测试集存在训练集不存在的离散特征
        target_encoding = mean_encoding * weight + float(y_mean) * (1 - weight)
        test_encoding = target_encoding

        train_res[col] = train_encoding
        test_res[col] = test_encoding
        encode_dicts.append([count_dict, mean_dict])

        if std:
            train_res[col] = scaler.fit_transform(np.array(train_encoding).reshape(-1, 1))
            test_res[col] = scaler.transform(np.array(test_encoding).reshape(-1, 1))
            std_scalers.append(scaler)

    return train_res, test_res, encode_dicts, std_scalers


def judge_Outier(q1, q3, iqr, v, k):
    if v > q3 + (k * iqr) or v < q1 - (k * iqr):  # 正常为1.5 效果会变差 但删除后会有一些极端值被保留，因此测试5倍
        return False
    else:
        return True


def judge_Outier2(avg, sigma, v):
    if v > avg + 3 * sigma or v < avg - 3 * sigma:
        return False
    else:
        return True


def delete_Outier(df, num_index_feas, style):
    k = 30
    for i in num_index_feas:
        series = df.loc[:, i]
        a = np.array(series)
        q1 = np.percentile(a, 25)
        q3 = np.percentile(a, 75)
        iqr = q3 - q1

        avg = np.average(series)
        sigma = np.std(series)
        if style == 1:
            df = df[df.loc[:, i].apply(lambda x: judge_Outier(q1, q3, iqr, x, k))]
        else:
            df = df[df.loc[:, i].apply(lambda x: judge_Outier2(avg, sigma, x))]
    return df


def draw_displot(data, discrete, name):  # 为数据绘制分布图
    data = data.values
    if discrete:
        sns.displot(data)
    else:
        sns.kdeplot(data)
    plt.title(name)
    # plt.savefig("feature pics/XGBoosting20-20/w20_{}".format(name))
    # plt.clf()
    plt.show()


def get_corr_draw(df):  # 绘制相关性热力图
    corrmat = df.corr()
    f, ax = plt.subplots(figsize=(20, 9))  # 绘制画布
    # sns.heatmap(corrmat, vmax=0.8, square=True)  # 绘制相关矩阵的热力图
    # sns.set(font_scale=1.25)
    hm = sns.heatmap(corrmat, cbar=False, annot=True, square=True, fmt='.2f', annot_kws={'size': 10})
    plt.show()
    # plt.savefig("feature pics/V2_result/num_correlation")


def descibe_feats(df, feats1, feat2, des_fun):  # 描述离散特征与连续特征之间的关系
    plt.subplots(figsize=(20, 10))  # 绘制画布
    # sns.set(font_scale=1.25)
    for i, feat1 in enumerate(feats1):
        df_p = df.pivot_table(index=feat1,
                              values=feat2,
                              aggfunc=des_fun)  # 聚合函数
        plt.subplot(1, len(feats1), i+1)  # 绘制画布
        hm = sns.heatmap(df_p, cbar=True, annot=True, square=False, fmt='.4f', annot_kws={'size': 10})
    plt.tight_layout()
    plt.show()
    # plt.savefig("feature pics/MLP特征分析/dis_cor_{}".format(df.columns[feat1]))


def disk_info_explode(data_base):  # 将data_base的disk_info_v2数据进行解析
    data_base_dict = data_base.to_dict(orient='records')
    for i, v in enumerate(data_base_dict):
        disk_dict = eval(v['disk_info_v2'])
        for j in disk_dict.keys():
            disk_type, disk_size = disk_dict[j]['type'], disk_dict[j]['size']
            if disk_type in v.keys():
                v[disk_type] += int(disk_size/1024/1024/1000)  # GB
                v[disk_type + '_num'] += 1
            else:
                v[disk_type] = int(disk_size/1024/1024/1000)
                v[disk_type+'_num'] = 1
    data_base = pd.DataFrame(data_base_dict)
    data_base.fillna(0, inplace=True)
    return data_base


def get_week_of_month(dt):
    """
    获取指定的某天是某个月中的第几周
    周一作为一周的开始
    """
    b_dt = dt[:-2] + '01'
    end = int(datetime.strptime(dt, "%Y%m%d").strftime("%W"))
    begin = int(datetime.strptime(b_dt, "%Y%m%d").strftime("%W"))
    return end - begin + 1


def dt_2_dayl(df):  # 将dt转为day label，作为时序预测的标签
    df_tmp = df.copy()
    df_tmp['day'] = df['dt'].apply(lambda x: int(datetime.strptime(x, "%Y%m%d").weekday())+1)  # 一个星期的第几天
    df_tmp['week'] = df['dt'].apply(lambda x: get_week_of_month(x))  # 一个月内的第几周
    df_tmp['mon'] = df['dt'].apply(lambda x: int(x[-4:-2]))  # 一年内的第几个月
    df_tmp['year'] = df['dt'].apply(lambda x: int(x[:4]))  # 第几年
    return df_tmp


def shuffle_and_split(df):
    index = np.arange(len(df))
    np.random.shuffle(index)
    df = df.iloc[index, :]

    borderline = int(0.8 * len(df))
    train_data = df.iloc[:borderline, :]
    test_data = df.iloc[borderline:, :]
    return train_data, test_data


def shuffle_and_split2(df, train_rate=0.8):
    borderline = int(train_rate * len(df))
    train_data = df.iloc[:borderline, :]
    test_data = df.iloc[borderline:, :]
    return train_data, test_data


def shuffle_and_split3(x_pro, y, train_rate=0.8):
    borderline = int(train_rate * len(x_pro))
    train_X = x_pro[:borderline, :]
    test_X = x_pro[borderline:, :]
    train_y = y[:borderline]
    test_y = y[borderline:]

    return train_X, test_X, train_y, test_y


def shuffle_and_split4(x_pro, y, train_rate=0.8):
    borderline = int(train_rate * len(x_pro))
    train_X = x_pro.iloc[:borderline, :]
    test_X = x_pro.iloc[borderline:, :]
    train_y = y.iloc[:borderline]
    test_y = y.iloc[borderline:]

    return train_X, test_X, train_y, test_y


def draw_true_pre_compare(predictions, test_Y, mae=0):  # 预测值与实际值进行画图对比
    x = np.arange(len(predictions))
    plt.figure(facecolor='w')  # figure 先画一个白色的画板
    plt.plot(x, test_Y, 'r-', linewidth=2, label='label')
    plt.plot(x, predictions, 'g-', linewidth=2, label='pre')
    plt.legend(loc='upper left')  # legend 该函数用于设置标签的位置，此处设置为左上角
    plt.grid(True)  # grid 该函数是设置是否需要网格线
    plt.show()

    # x = np.arange(len(mae))
    # plt.figure(facecolor='w')  # figure 先画一个白色的画板
    # plt.plot(x, mae, 'r-', linewidth=2, label='label')
    # plt.xticks(range(20))
    # plt.grid(True)  # grid 该函数是设置是否需要网格线
    # plt.show()


def my_mae(predictions, labels):
    res_df = pd.DataFrame(columns=['index', 'pre', 'label', 'ae'])
    res_df['pre'] = predictions
    res_df['label'] = labels
    res_df['ae'] = abs(res_df['pre'] - res_df['label'])
    return res_df


def delete_abnoraml_mac(df):
    # 存在过异常（超高或超低）：
    df_tmp = df.copy()
    macs = df_tmp[(df_tmp['use_rate'] >= 2) | (df_tmp['use_rate'] < 0.1)]['device_uuid']
    df_tmp = df_tmp[df_tmp['device_uuid'].apply(lambda x: x not in macs.values)]
    return df_tmp


def delete_history_mac(df, test_dt):
    df_tmp = df.copy()
    df_tmp['t_dif'] = abs(df_tmp['dt'].apply(lambda x: float(x)-float(test_dt)))
    min_v, max_v = np.min(df_tmp['t_dif']), np.max(df_tmp['t_dif'])
    random.seed(23)
    df_tmp['random_v'] = random.choices(range(int(min_v), int(max_v)), k=len(df_tmp))
    df_tmp['t_dif'] = df_tmp.apply(lambda x: x['t_dif'] <= x['random_v'] or float(x['dt']) >= float(test_dt), axis=1)
    # for i in range(len(df_tmp)):
    #     df_tmp.iloc[i, -1] = df_tmp.iloc[i, -1] <= random_v[i] or float(df_tmp.iloc[i, -2]) >= float(min_test_dt)
    df_tmp = df_tmp[df_tmp['t_dif']==True]
    return df_tmp.drop(['t_dif', 'random_v'], axis=1)


def get_pre_mac(macs, exit_macs, **args):  # 生成一个待预测机器，可能是已有机器+未来时间 或者新机器（除时间属性外，其他任意属性存在变更）
    new_macs = macs.copy()
    # for mac_id in macs['device_uuid'].unique():
    #     if mac_id not in exit_macs:  # 已有机器的新时间
    #         print("机器{}从未在模型遇到过，预测准确率可能较低".format(mac_id))

    new_dt = args['dt']
    new_macs['dt'] = new_dt
    new_macs = dt_2_dayl(new_macs)  # 将日期转为时间数组
    index_features = ['device_uuid', 'specific_tasks', 'task_id', 'province', 'bandwidth_type', 'nat_type', 'isp', 'billing_rule', 'day',
                      'week', 'mon', 'year', 'upbandwidth', 'cpu_num', 'memory_size', 'disk_size', 'test_sat', 'loss_sat', 'dt']
    index_num = ['upbandwidth', 'cpu_num', 'memory_size', 'disk_size', 'test_sat', 'loss_sat', 'day', 'week', 'mon', 'year', 'dt']
    index_cate = ['task_id', 'province', 'bandwidth_type', 'nat_type', 'isp', 'billing_rule']

    new_macs = new_macs[index_features]
    new_macs.loc[:, index_num] = new_macs.loc[:, index_num].astype(float)
    new_macs.loc[:, index_cate] = new_macs.loc[:, index_cate].astype(str)

    return new_macs


def get_pre_mac_week(mac, exit, **args):  # 生成一个待预测机器，可能是已有机器+未来时间 或者新机器（除时间属性外，其他任意属性存在变更）
    new_mac = mac.copy()
    if not exit:  # 已有机器的新时间
        print("该机器从未在模型遇到过，预测准确率可能较低")

    new_dt = args['dt']
    new_mac['dt'] = new_dt
    index_features = ['device_uuid', 'task_id', 'province', 'bandwidth_type', 'nat_type', 'isp', 'billing_rule',
                      'upbandwidth', 'cpu_num', 'memory_size', 'disk_size', 'test_sat', 'loss_sat', 'dt']
    index_num = ['upbandwidth', 'cpu_num', 'memory_size', 'disk_size', 'test_sat', 'loss_sat']
    index_cate = ['task_id', 'province', 'bandwidth_type', 'nat_type', 'isp', 'billing_rule']

    new_mac = new_mac[index_features]
    new_mac.loc[:, index_num] = new_mac.loc[:, index_num].astype(float)
    new_mac.loc[:, index_cate] = new_mac.loc[:, index_cate].astype(str)

    return new_mac


def day_to_week(df, seq_len, index_cate, pre_dt):
    df_tmp = df.copy()
    all_dt = list(df_tmp['dt'].unique())  # 数据中所有的日期
    if int(pre_dt) not in all_dt:  # 封闭测试
        pre_dt = (datetime.datetime.strptime(pre_dt, "%Y%m%d") - datetime.timedelta(days=7)).strftime("%Y%m%d")
    pre_dt_index = all_dt.index(int(pre_dt))
    df_week = pd.DataFrame(columns=df_tmp.columns)
    for i in range(pre_dt_index%7, pre_dt_index+1, seq_len):
        tmp_dt = all_dt[i:i+seq_len]
        a = df_tmp[df_tmp['dt'].apply(lambda x: x in tmp_dt)].copy()
        a['dt'] = tmp_dt[0]  # 该周时间以第一天作为时序特征
        b = a.groupby(index_cate).mean().reset_index()  # 按照类别属性分组，对其余属性求平均
        df_week = df_week.append(b)
    return df_week


def day_to_week_series(df, seq_len, index_cate, pre_dt, way='leap'):  # 时间窗跨越/滑动
    df_tmp = df.copy()
    all_dt = list(df_tmp['dt'].unique())  # 数据中所有的日期
    if float(pre_dt) not in all_dt:  # 封闭测试 滑动取周
        last_week = (datetime.datetime.strptime(pre_dt, "%Y%m%d") - datetime.timedelta(days=14)).strftime("%Y%m%d")
    else:
        last_week = pre_dt

    keep_cols = list(df_tmp.columns)
    keep_cols.remove('use_rate')
    df_week = pd.DataFrame(columns=keep_cols+['use_rate_x', 'use_rate_y'])

    bg_index = 0
    lw_index = all_dt.index(float(last_week))
    step = 1
    if way == 'leap':
        bg_index = lw_index % 7
        step = seq_len
    for i in range(bg_index, lw_index+1, step):
        tmp_dt = all_dt[i:i+seq_len]
        next_dt = all_dt[i+seq_len:i+2*seq_len]
        a = df_tmp[df_tmp['dt'].apply(lambda x: x in tmp_dt)].copy()  # 获取该时期全部机器数据
        next_a = df_tmp[df_tmp['dt'].apply(lambda x: x in next_dt)].copy()
        a['dt'] = tmp_dt[0]  # 该周时间以第一天作为时序特征
        b = a.groupby(index_cate).mean().reset_index()  # 按照类别属性分组，对其余属性求平均
        next_b = next_a.groupby(index_cate)['use_rate'].mean().reset_index()  # 按照类别属性分组，对其余属性求平均
        b = pd.merge(b, next_b[['device_uuid', 'task_id', 'use_rate']], on=['device_uuid', 'task_id'])
        df_week = df_week.append(b)
    return df_week


def nat_type_judge(mac_nat, task_nat):
    if mac_nat == 'inner' and 'NAT' in task_nat:
        return True
    if mac_nat == 'public' and '公网' in task_nat:
        return True
    return False



if __name__ == '__main__':
    tmp = [0.06622082821091543, 0.10792547409917186, 0.10438078182687105,
           0.1138254163275628, 0.129922253574408, 0.10720586096822364, 0.10325594164854597,
           0.11234439568360048, 0.13983193254038248, 0.139804582131574, 0.14771209446249722]

    # matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    x = range(2, 13)
    # "r" 表示红色，ms用来设置*的大小
    plt.plot(x, tmp, "r", marker='*', ms=10, label="MAE")
    # plt.plot([1, 2, 3, 4], [20, 30, 80, 40], label="b")
    plt.xticks(rotation=45)
    plt.xlabel("日期")
    plt.ylabel("预测误差")
    # upper left 将图例a显示到左上角
    plt.legend(loc="upper left")
    # 在折线图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
    for x1, y1 in zip(x, tmp):
        plt.text(x1, round(y1, 3), str(round(y1, 3)), ha='center', va='bottom', fontsize=20, rotation=0)
    plt.savefig("a.jpg")
    plt.show()
