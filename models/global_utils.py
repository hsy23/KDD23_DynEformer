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


def train_test_split(X, y, train_ratio=0.6, test_ratio=0.2):
    num_ts, num_periods, num_features = X.shape
    # train_periods = int(num_periods * train_ratio)
    train_periods = int(len(num_periods) * train_ratio)
    test_periods = int(len(num_periods) * test_ratio)

    random.seed(2)
    Xtr = X[:, :train_periods, :]
    ytr = y[:, :train_periods]
    Xte = X[:, -test_periods:, :]
    yte = y[:, -test_periods:]
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
    def fit(self, y):
        self.max = np.max(y)
        self.min = np.min(y)

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


def task2id(task_name):  # todo:20220405 生成字典时带上字典对应的数据日期，在编码完成后再进行存储
    if task_name not in Task_ID_Dict.keys():
        print("unknown tasks", task_name)
        Task_ID_Dict[task_name] = max(Task_ID_Dict.values()) + 1
        pickle.dump(Task_ID_Dict, open("Task_ID_Dick.pkl", 'wb'))
    return Task_ID_Dict[task_name]


def discrete_feats_encoder(df, encode_cols, y_col):
    df = df.copy()
    std_scalers = []

    y_mean = df[y_col].mean()
    smoothing = 0.8
    for col in encode_cols:
        train_x_col = df[col].copy()

        sorted_x = sorted(set(train_x_col))
        encode_dict = dict(zip(sorted_x, range(1, len(sorted_x) + 1)))
        df[col] = df[col].replace(encode_dict)
        # encode_dict = train_x_col.value_counts().to_dict()

        # label encoder
        # count_dict = train_x_col.value_counts().to_dict()
        # mean_dict = target_xy.groupby(col).mean().to_dict()['use_rate']
        #
        # count_encoding = train_x_col.replace(count_dict)
        # mean_encoding = train_x_col.replace(mean_dict)
        #
        # weight = smoothing / (1 + np.exp(-(count_encoding - 1)))
        # target_encoding = mean_encoding * weight + float(y_mean) * (1 - weight)
        # train_encoding = target_encoding

        # test dataset
        # count_encoding = test_x_col.replace(count_dict)
        # mean_encoding = test_x_col.replace(mean_dict)

        # 用训练集均值替代测试集中新出现的特征编码
        # count_encoding = count_encoding.apply(pd.to_numeric, errors='coerce')
        # count_encoding = count_encoding.fillna(count_encoding.mean())
        #
        # mean_encoding = mean_encoding.apply(pd.to_numeric, errors='coerce')
        # mean_encoding = mean_encoding.fillna(mean_encoding.mean())
        #
        # weight = smoothing / (1 + np.exp(-(count_encoding - 1)))  # 测试集存在训练集不存在的离散特征
        # target_encoding = mean_encoding * weight + float(y_mean) * (1 - weight)
        # test_encoding = target_encoding
        #
        # train_res[col] = train_encoding
        # test_res[col] = test_encoding

    return df


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


def mac_attributes_pro(mac_attr, workload_data):  # 合并机器数据
    # 填充tcp重传率数据和tcp重传带宽
    data_base = mac_attr
    retrans_ratio_mean = data_base['dbtr_tcp_retransmission_ratio'].mean()
    data_base['dbtr_tcp_retransmission_ratio'] = data_base['dbtr_tcp_retransmission_ratio'].fillna(retrans_ratio_mean)
    data_base['avg_test_upbandwidthwith_tcp'].fillna(-1, inplace=True)
    data_base['avg_iops_per_line'].fillna(0, inplace=True)

    # 数据清理
    data_base.dropna(inplace=True)
    data_base = data_base[data_base['upbandwidth_base'] != 0]

    # 磁盘属性清洗和处理
    data_base = disk_info_explode(data_base)

    # 任务属性清洗和处理
    data_base = data_base[data_base['specific_tasks'] != '[]']
    data_base['specific_tasks'] = data_base['specific_tasks'].apply(
        lambda x: re.split(r'[,，;、\s]\s*', x.strip('[]')))  # 删除空格
    data_base = data_base[data_base['specific_tasks'].apply(lambda x: len(x) == 1)]  # 只考虑独跑
    data_base['specific_tasks'] = data_base['specific_tasks'].apply(lambda x: x[0])
    data_base['task_id'] = data_base['specific_tasks'].apply(lambda x: task2id(x))

    # 获取压测满意度和重传满意度
    data_base['test_sat'] = data_base['avg_test_upbandwidth'] / data_base['upbandwidth']
    data_base['test_sat_base'] = data_base['upbandwidth_base'] / data_base['upbandwidth']
    data_base['loss_sat'] = 1 - data_base['dbtr_tcp_retransmission_ratio'] / 100
    data_base['loss_sat2'] = data_base['avg_test_upbandwidthwith_tcp'].astype(float) / data_base['upbandwidth']
    data_base.loc[data_base['loss_sat2'] >= 0, 'loss_sat'] = data_base[data_base['loss_sat2'] >= 0]['loss_sat2']

    # 对连续数值做泛化处理
    data_base['upbandwidth'] = data_base['upbandwidth'].apply(lambda x: round(float(x / 1024 / 1024 / 1000), 2))  # GB
    data_base['memory_size'] = data_base['memory_size'].apply(lambda x: round(float(x / 1024 / 1024 / 1000), 2))  # GB
    data_base['disk_size'] = data_base['disk_size'].apply(
        lambda x: round(float(x / 1024 / 1024 / 1024 / 1024), 2))  # TB
    data_base['upbandwidth_base'] = data_base['upbandwidth_base'].apply(lambda x: round(float(x / 1024 / 1024 /1000), 2))


    data_base['dt'] = data_base['dt'].astype(str)

    index_used = ['device_uuid', 'task_id', 'province', 'bandwidth_type', 'nat_type', 'isp', 'billing_rule',
                  'upbandwidth', 'upbandwidth_base', 'cpu_num', 'memory_size', 'disk_size', 'test_sat', 'loss_sat']
    data_base = data_base[index_used]

    # all_cols = ['device_uuid', 'specific_tasks', 'task_id', 'location_', 'province', 'city',
    #             'bandwidth_type', 'nat_type',
    #             'isp', 'upbandwidth_base', 'test_sat_base', 'avg_test_upbandwidth',
    #             'upbandwidth', 'upbandwidth_perline', 'cpu_num', 'memory_size', 'disk_size',
    #             'ssd', 'ssd_num', 'hdd',
    #             'hdd_num',
    #             'nvme', 'nvme_num', 'test_sat', 'loss_sat', 'avg_iops_per_line', 'dbtr_tcp_retransmission_ratio',
    #             'use_rate', 'billing_rule',
    #             'is_filter_stand', 'is_hardware_result', 'is_net_result', 'device_disk_upband',
    #             'df_days',
    #             'dt']
    # final_res_df = data_base[all_cols]
    data_base.dropna(inplace=True)  # 删除空数据
    final_res_df = data_base
    final_res_df.drop_duplicates(inplace=True)

    index_cate = ['device_uuid', 'task_id', 'province', 'bandwidth_type', 'nat_type', 'isp', 'billing_rule']
    index_cate2 = ['province', 'bandwidth_type', 'nat_type', 'isp', 'billing_rule']
    final_res_df = final_res_df.groupby(index_cate).mean().reset_index()
    final_res_df.drop_duplicates(subset=['device_uuid'], inplace=True)

    data = pd.merge(left=workload_data, right=final_res_df, left_on='machine_id', right_on='device_uuid', how='left')
    data.dropna(inplace=True)
    data = discrete_feats_encoder(data, index_cate2, 'bw_upload')

    return data


def draw_true_pre_compare(history_Y, predictions, test_Y, p_id):  # 预测值与实际值进行画图对比
    his_x = np.arange(len(history_Y))
    pre_x = np.arange(48, len(his_x))

    plt.figure(facecolor='w')  # figure 先画一个白色的画板
    plt.plot(his_x, history_Y, 'r-', linewidth=2)
    plt.vlines(48, np.min(history_Y), np.max(history_Y), color="blue", linestyles="dashed", linewidth=2)
    plt.plot(pre_x, test_Y, 'r-', linewidth=2, label='label')
    plt.plot(pre_x, predictions, 'g-', linewidth=2, label='pre')
    plt.legend(loc='upper left')  # legend 该函数用于设置标签的位置，此处设置为左上角
    plt.grid(True)  # grid 该函数是设置是否需要网格线
    # plt.show()
    plt.savefig(r'saved_res_pics/app_switch_{}.png'.format(p_id))


def draw_true_pre_compare_normal(history_Y, predictions, test_Y, p_id):  # 预测值与实际值进行画图对比
    his_x = np.arange(len(history_Y))
    pre_x = np.arange(len(his_x), len(his_x)+len(predictions))

    plt.figure(facecolor='w')  # figure 先画一个白色的画板
    plt.plot(his_x, history_Y, 'r-', linewidth=2)
    plt.vlines(len(his_x), np.min(history_Y), np.max(history_Y), color="blue", linestyles="dashed", linewidth=2)
    plt.plot(pre_x, test_Y, 'r-', linewidth=2, label='label')
    plt.plot(pre_x, predictions, 'g-', linewidth=2, label='pre')
    plt.legend(loc='upper left')  # legend 该函数用于设置标签的位置，此处设置为左上角
    plt.grid(True)  # grid 该函数是设置是否需要网格线
    # plt.show()
    plt.savefig(r'saved_res_pics/app_switch_{}.png'.format(p_id))


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
