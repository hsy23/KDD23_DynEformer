import os.path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import datetime
import pickle
import random
from models.global_utils import *

AB_PATH = r"C:\Users\Hsy\Desktop\WWW_master\raw_data"

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

MinMax = preprocessing.MinMaxScaler()
StdScaler = preprocessing.StandardScaler()

TASK_ID_DICT = pickle.load(open(os.path.join(AB_PATH, "Task_ID_DicT.pkl"), 'rb'))


def get_data_path(file_name):
    # folder = os.path.dirname(__file__)
    # return os.path.join(folder, "raw_data")
    return os.path.join(AB_PATH, file_name)


def app_filter(df):  # 将实时带宽数据过滤出主要应用，如果是混跑则进行拆分,合并同一业务的不同docker
    # 过滤有效应用，混跑默认拆分
    mac_ids = list(df['machine_id'].unique())
    df_main = df[df['name'].apply(lambda x: x in TASK_ID_DICT.keys())]

    # 合并业务dockers
    df_group = df_main.groupby(['machine_id', 'name', 'time_id', 'dt']).sum().reset_index()

    # 抽取部分机器
    # chosen_macs = random.sample(mac_ids, 100)
    # df_part = df_group[df_group['machine_id'].apply(lambda x: x in chosen_macs)]
    # print("随机抽取mac:{}".format(chosen_macs))
    # 随机抽取mac:['c93af9e36844fd2161e2f45a1cccb3bd', '56f0545aa49ba67e1c332d62434061a7', '26a2368185d17c6d0477cb6ded81b3ab']
    return df_group


def get_week_of_month(dt):
    """
    获取指定的某天是某个月中的第几周
    周一作为一周的开始
    """
    b_dt = dt[:-2] + '01'
    end = int(datetime.datetime.strptime(dt, "%Y%m%d").strftime("%W"))
    begin = int(datetime.datetime.strptime(b_dt, "%Y%m%d").strftime("%W"))
    return end - begin + 1


def dt_2_dayl(df):  # 将dt转为day label，作为时序预测的标签。
    df_tmp = df.copy()
    df_tmp['min'] = df_tmp['time_id'].apply(lambda x: int(x[-2:]))  # 一小时的第几分钟
    df_tmp['hour'] = df_tmp['time_id'].apply(lambda x: int(x[-4:-2]))  # 一天的第几个小时
    df_tmp['day'] = df['dt'].apply(lambda x: int(datetime.datetime.strptime(x, "%Y%m%d").weekday())+1)  # 一个星期的第几天
    df_tmp['week'] = df['dt'].apply(lambda x: get_week_of_month(x))  # 一个月内的第几周
    # df_tmp['mon'] = df['dt'].apply(lambda x: int(x[-4:-2]))  # 一年内的第几个月
    # df_tmp['year'] = df['dt'].apply(lambda x: int(x[:4]))  # 第几年
    return df_tmp


def data_merge(begin_dt, end_dt, step, num_periods_tmp, num_periods_all, keep_diff_app):
    begin_date = datetime.datetime.strptime(begin_dt, "%Y%m%d")
    middle_date = begin_date + datetime.timedelta(days=step-1)
    res = pd.DataFrame(columns=['machine_id', 'name', 'bw_upload', 'bw_download', 'time_id', 'dt'])
    while middle_date.strftime("%Y%m%d") <= end_dt:
        data_path = "../raw_data/ams_{}-{}_bd.pkl".format(begin_date.strftime("%Y%m%d"), middle_date.strftime("%Y%m%d"))
        data = pickle.load(open(data_path, 'rb'))
        data = app_filter(data)  # 过滤非主要应用

        if keep_diff_app:  # Todo:区分混跑和切换
            tmp = data.groupby('machine_id').size().reset_index()
            tmp2 = tmp[tmp[0] == num_periods_tmp]
            tmp3 = pd.merge(data, tmp2, on=['machine_id']).iloc[:, :-1]
        else:
            tmp = data.groupby(['machine_id', 'name']).size().reset_index()
            tmp2 = tmp[tmp[0] == num_periods_tmp]
            tmp3 = pd.merge(data, tmp2, on=['machine_id', 'name']).iloc[:, :-1]

        res = res.append(tmp3)
        begin_date += datetime.timedelta(days=step)
        middle_date = begin_date + datetime.timedelta(days=step-1)

    if keep_diff_app:
        tmp = res.groupby('machine_id').size().reset_index()
        tmp2 = tmp[tmp[0] == num_periods_all]
        tmp3 = pd.merge(res, tmp2, on=['machine_id']).iloc[:, :-1]
    else:
        tmp = res.groupby(['machine_id', 'name']).size().reset_index()
        tmp2 = tmp[tmp[0] == num_periods_all]
        tmp3 = pd.merge(res, tmp2, on=['machine_id', 'name']).iloc[:, :-1]

    # tmp3.to_pickle("../raw_data/pro_{}-{}_bd.pkl".format(begin_dt, end_dt))
    # tmp3.to_csv("../raw_data/pro_{}-{}_bd.csv".format(begin_dt, end_dt))


def mac_merge(data_base):  # 合并机器数据
    # 填充tcp重传率数据和tcp重传带宽
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
    data_base['specific_tasks'] = data_base['specific_tasks'].apply(lambda x: re.split(r'[,，;、\s]\s*', x.strip('[]')))  # 删除空格
    data_base = data_base.explode('specific_tasks')
    # data_base['task_id'] = data_base['specific_tasks'].apply(lambda x: task2id(x))

    # 获取压测满意度和重传满意度
    data_base['test_sat'] = data_base['avg_test_upbandwidth'] / data_base['upbandwidth']
    data_base['loss_sat'] = 1 - data_base['dbtr_tcp_retransmission_ratio'] / 100
    data_base['loss_sat2'] = data_base['avg_test_upbandwidthwith_tcp'].astype(float) / data_base['upbandwidth']
    data_base.loc[data_base['loss_sat2'] >= 0, 'loss_sat'] = data_base[data_base['loss_sat2'] >= 0]['loss_sat2']

    # 对连续数值做泛化处理
    data_base['upbandwidth'] = data_base['upbandwidth'].apply(lambda x: round(float(x/1024/1024), 2))  # MB
    data_base['upbandwidth_perline'] = data_base['upbandwidth_perline'].apply(lambda x: round(float(x/1024/1024), 2))  # MB
    data_base['memory_size'] = data_base['memory_size'].apply(lambda x: round(float(x/1024/1024/1000), 2))  # GB
    data_base['disk_size'] = data_base['disk_size'].apply(lambda x: round(float(x/1024/1024/1024/1024), 2))  # TB
    data_base['test_sat'] = data_base['test_sat'].apply(lambda x: round(x, 2))
    data_base['loss_sat'] = data_base['loss_sat'].apply(lambda x: round(x, 2))
    data_base['avg_iops_per_line'] = data_base['avg_iops_per_line'].astype(float)

    # 合并收益信息
    # data_base = get_use_rate(data_base, data_95)  # 计算利用率
    data_base['dt'] = data_base['dt'].astype(str)
    all_cols = ['device_uuid', 'specific_tasks', 'location_', 'province', 'bandwidth_type', 'nat_type', 'isp',
         'upbandwidth', 'upbandwidth_perline', 'cpu_num', 'memory_size', 'disk_size', 'ssd', 'ssd_num', 'hdd', 'hdd_num',
         'nvme', 'nvme_num', 'test_sat', 'loss_sat', 'avg_iops_per_line',  'dbtr_tcp_retransmission_ratio',
                'billing_rule', 'dt']
    index_num_split_t = ['upbandwidth', 'test_sat', 'loss_sat']
    # index_num_split_t = ['upbandwidth', 'cpu_num', 'memory_size', 'disk_size', 'test_sat', 'loss_sat', 'use_rate']
    final_res_df = data_base[all_cols]
    final_res_df.dropna(inplace=True)  # 删除空数据
    # final_res_df = final_res_df.sort_values(by='dt')
    final_res_df.drop_duplicates(inplace=True)
    final_res_df = delete_Outier(final_res_df, index_num_split_t, style=1)  # style1为箱线图法
    # final_res_df = delete_abnoraml_mac(final_res_df)  # 筛选利用率异常机器
    return final_res_df


def extend_T(s):
    news = []
    for i in range(0, len(s), 12):
        news.append(np.max(s[i:i+12]))
    return news

# def cate_feats_encode(df, style):
#     if style == 'one_hot':
#
#     elif style == '':


if __name__ == "__main__":
    data_merge('20220801', '20220830', 2, 288*2, 288*30, True)
    print("test")


