import os.path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import datetime
import pickle
import random
from tqdm import tqdm

Raw_Data_Path = r'data/ams_{}-{}_bd.pkl'

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

MinMax = preprocessing.MinMaxScaler()
StdScaler = preprocessing.StandardScaler()


def app_filter(df):
    # Filters the real-time bandwidth data for the main applications; if applications are mixed, they are split and different dockers of the same service are merged.
    TASK_ID_DICT = pickle.load(open("../data/Task_ID_Dict.pkl", 'rb'))
    df_main = df[df['name'].apply(lambda x: x in TASK_ID_DICT.keys())]
    df_main = df
    # Merges different dockers of the same service
    df_group = df_main.groupby(['machine_id', 'name', 'time_id', 'dt']).sum().reset_index()
    return df_group


def get_week_of_month(dt):
    """
    Returns the week of the month for a given day
    Monday is considered the start of the week
    """
    b_dt = dt[:-2] + '01'
    end = int(datetime.datetime.strptime(dt, "%Y%m%d").strftime("%W"))
    begin = int(datetime.datetime.strptime(b_dt, "%Y%m%d").strftime("%W"))
    return end - begin + 1


def dt_2_dayl(df):  # Converts dt to day label for time series prediction.
    df_tmp = df.copy()
    df_tmp['min'] = df_tmp['time_id'].apply(lambda x: int(x[-2:]))  # The minute of the hour
    df_tmp['hour'] = df_tmp['time_id'].apply(lambda x: int(x[-4:-2]))  # The hour of the day
    df_tmp['day'] = df['dt'].apply(lambda x: int(datetime.datetime.strptime(x, "%Y%m%d").weekday())+1)  # The day of the week
    df_tmp['week'] = df['dt'].apply(lambda x: get_week_of_month(x))  # The week of the month
    # df_tmp['mon'] = df['dt'].apply(lambda x: int(x[-4:-2]))  # The month of the year
    # df_tmp['year'] = df['dt'].apply(lambda x: int(x[:4]))  # The year
    return df_tmp



def data_merge(begin_dt, end_dt, step, num_periods_tmp, num_periods_all, keep_diff_app):
    '''
    Merges bandwidth data from multiple days within a month
    '''
    begin_date = datetime.datetime.strptime(begin_dt, "%Y%m%d")
    middle_date = begin_date + datetime.timedelta(days=step-1)
    res = pd.DataFrame(columns=['machine_id', 'name', 'bw_upload', 'time_id', 'dt'])
    while middle_date.strftime("%Y%m%d") <= end_dt:
        data_path = Raw_Data_Path.format(begin_date.strftime("%Y%m%d"), middle_date.strftime("%Y%m%d"))
        data = pickle.load(open(data_path, 'rb'))
        data = app_filter(data)  # Process and filter the main applications
        if keep_diff_app:
            tmp = data.groupby('machine_id').size().reset_index()
            tmp2 = tmp[tmp[0] == num_periods_tmp]
            tmp3 = pd.merge(data, tmp2, on=['machine_id']).iloc[:, :-1]
        else:
            tmp = data.groupby(['machine_id', 'name']).size().reset_index()
            tmp2 = tmp[tmp[0] == num_periods_tmp]
            tmp3 = pd.merge(data, tmp2, on=['machine_id', 'name']).iloc[:, :-1]

        res = res.append(tmp3)
        print('{}-{} done'.format(begin_date, middle_date))
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

    data = tmp3.sort_values(by=['machine_id', 'time_id'])
    data = dt_2_dayl(data)
    data['bw_upload'] = data['bw_upload'] / 1024 / 1024 / 1024  # GB

    data.to_pickle(open("../../raw_data/merged_0901_0930_bd_t_feats.pkl", 'wb'))
    data.to_csv("../../raw_data/merged_0901_0930_bd_t_feats.csv")
    return data


def extend_T(s):  # Converts minute-level data to hour-level
    news = []
    for i in range(0, len(s), 12):
        news.append(np.max(s[i:i+12]))
    return news


pd.options.mode.chained_assignment = None  # default='warn'
def transfor_T(df, day_periods=10):  # Transforms data from minute-level to hour-level
    res_df = pd.DataFrame(columns=df.columns)
    for j in tqdm(range(0, len(df), 12)):
        s_hour = df.iloc[j:j + 12]
        workload = np.max(s_hour['bw_upload'])
        tmp_df = s_hour.iloc[0]
        tmp_df.loc['bw_upload'] = workload
        res_df = res_df.append(tmp_df)
    return res_df


def get_newEntity(begin_dt, end_dt, step, num_periods_tmp):  # 找到新加入的应用和app
    begin_date = datetime.datetime.strptime(begin_dt, "%Y%m%d")
    middle_date = begin_date + datetime.timedelta(days=step-1)
    res = pd.DataFrame(columns=['machine_id', 'name', 'bw_upload', 'time_id', 'dt'])
    while middle_date.strftime("%Y%m%d") <= end_dt:
        data_path = Raw_Data_Path.format(begin_date.strftime("%Y%m%d"), middle_date.strftime("%Y%m%d"))
        data = pickle.load(open(data_path, 'rb'))
        data = app_filter(data)  # 过滤非主要应用

        tmp = data.groupby('machine_id').size().reset_index()
        tmp2 = tmp[tmp[0] == num_periods_tmp]
        tmp3 = pd.merge(data, tmp2, on=['machine_id']).iloc[:, :-1]

        res = res.append(tmp3)
        print('{}-{} done'.format(begin_date, middle_date))
        begin_date += datetime.timedelta(days=step)
        middle_date = begin_date + datetime.timedelta(days=step-1)

    # tmp = res.groupby('machine_id').size().reset_index()
    # tmp2 = tmp[tmp[0] >= 15*288]
    # tmp3 = pd.merge(res, tmp2, on=['machine_id']).iloc[:, :-1]

    data = res.sort_values(by=['machine_id', 'time_id'])
    data = dt_2_dayl(data)
    data['bw_upload'] = data['bw_upload'] / 1024 / 1024 / 1024  # GB

    old_data = pickle.load(open("../../raw_data/merged_0801_0830_bd_t_feats_hour.pkl", 'rb'))
    old_macs = old_data['machine_id'].unique()
    old_app = old_data['name'].unique()

    data_app = data[data['name'].apply(lambda x:x not in old_app and x != 'default')]
    # data_mac = data[data['machine_id'].apply(lambda x:x not in old_macs)]

    data_app = transfor_T(data_app.sort_values(by=['machine_id', 'time_id']))
    # data_mac = transfor_T(data_mac.sort_values(by=['machine_id', 'time_id']))

    data_app.to_pickle(open("../../raw_data/0801_0830_newAdd_app.pkl", 'wb'))
    # data_mac.to_pickle(open("../../raw_data/0825_0830_newAdd_mac.pkl", 'wb'))
    return data


def task2id(task_name, Task_ID_Dict):
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


def static_features_merge(mac_attr, workload_data):  # Merging workload data with static features
    Task_ID_Dict = pickle.load(open(r"../data/Task_ID_Dict.pkl", 'rb'))

    data_base = mac_attr
    retrans_ratio_mean = data_base['dbtr_tcp_retransmission_ratio'].mean()
    data_base['dbtr_tcp_retransmission_ratio'] = data_base['dbtr_tcp_retransmission_ratio'].fillna(retrans_ratio_mean)
    data_base['avg_test_upbandwidthwith_tcp'].fillna(-1, inplace=True)
    data_base['avg_iops_per_line'].fillna(0, inplace=True)

    # Data Cleaning
    data_base.dropna(inplace=True)
    data_base = data_base[data_base['upbandwidth_base'] != 0]

    # Disk property cleaning and processing
    data_base = disk_info_explode(data_base)

    # Task attribute cleaning and processing
    data_base = data_base[data_base['specific_tasks'] != '[]']
    data_base['specific_tasks'] = data_base['specific_tasks'].apply(
        lambda x: re.split(r'[,，;、\s]\s*', x.strip('[]')))
    data_base = data_base[data_base['specific_tasks'].apply(lambda x: len(x) == 1)]
    data_base['specific_tasks'] = data_base['specific_tasks'].apply(lambda x: x[0].replace("'", "").replace('"', ''))
    data_base['task_id'] = data_base['specific_tasks'].apply(lambda x: task2id(x, Task_ID_Dict))

    # Obtain pressure test satisfaction and retransmission satisfaction
    data_base['test_sat'] = data_base['avg_test_upbandwidth'] / data_base['upbandwidth']
    data_base['test_sat_base'] = data_base['upbandwidth_base'] / data_base['upbandwidth']
    data_base['loss_sat'] = 1 - data_base['dbtr_tcp_retransmission_ratio'] / 100
    data_base['loss_sat2'] = data_base['avg_test_upbandwidthwith_tcp'].astype(float) / data_base['upbandwidth']
    data_base.loc[data_base['loss_sat2'] >= 0, 'loss_sat'] = data_base[data_base['loss_sat2'] >= 0]['loss_sat2']

    # Generalization of continuous values
    data_base['upbandwidth'] = data_base['upbandwidth'].apply(lambda x: round(float(x / 1024 / 1024 / 1000), 2))  # GB
    data_base['memory_size'] = data_base['memory_size'].apply(lambda x: round(float(x / 1024 / 1024 / 1000), 2))  # GB
    data_base['disk_size'] = data_base['disk_size'].apply(
        lambda x: round(float(x / 1024 / 1024 / 1024 / 1024), 2))  # TB
    data_base['upbandwidth_base'] = data_base['upbandwidth_base'].apply(lambda x: round(float(x / 1024 / 1024 /1000), 2))


    data_base['dt'] = data_base['dt'].astype(str)

    index_used = ['device_uuid', 'task_id', 'province', 'bandwidth_type', 'nat_type', 'isp', 'billing_rule',
                  'upbandwidth', 'upbandwidth_base', 'cpu_num', 'memory_size', 'disk_size', 'test_sat', 'loss_sat']
    data_base = data_base[index_used]

    # final_res_df = data_base[all_cols]
    data_base.dropna(inplace=True)
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


if __name__ == "__main__":
    merged_data_min = data_merge('20220901', '20220930', 1, 288*1, 288*30, True)
    merged_data_min = pickle.load(open("../_data/merged_0901_0930_bd_t_feats.pkl", 'rb'))
    merged_data_hour = transfor_T(merged_data_min)