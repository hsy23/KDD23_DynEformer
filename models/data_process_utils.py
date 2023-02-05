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

AB_PATH = r"C:\Users\Admin\Desktop\WWW_master\raw_data"
Raw_Data_Path = r'../../raw_data/ams_{}-{}_bd.pkl'

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
    # exist_tasks = df['name'].unique()
    # not_exist_tasks = set(exist_tasks) - set(TASK_ID_DICT.keys())
    # print('filtered:{}'.format(not_exist_tasks))

    mac_ids = list(df['machine_id'].unique())
    # df_main = df[df['name'].apply(lambda x: x in TASK_ID_DICT.keys())]
    df_main=df
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
    '''
        合并一个月内多天的带宽数据
    '''
    begin_date = datetime.datetime.strptime(begin_dt, "%Y%m%d")
    middle_date = begin_date + datetime.timedelta(days=step-1)
    res = pd.DataFrame(columns=['machine_id', 'name', 'bw_upload', 'time_id', 'dt'])
    while middle_date.strftime("%Y%m%d") <= end_dt:
        data_path = Raw_Data_Path.format(begin_date.strftime("%Y%m%d"), middle_date.strftime("%Y%m%d"))
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


def extend_T(s):  # 将分钟级别的数据转为小时级
    news = []
    for i in range(0, len(s), 12):
        news.append(np.max(s[i:i+12]))
    return news


pd.options.mode.chained_assignment = None  # default='warn'
def transfor_T(df, day_periods=10):  # 将数据从分钟级别转为小时级别
    res_df = pd.DataFrame(columns=df.columns)
    for j in tqdm(range(0, len(df), 12)):
        s_hour = df.iloc[j:j + 12]
        workload = np.max(s_hour['bw_upload'])
        tmp_df = s_hour.iloc[0]
        tmp_df.loc['bw_upload'] = workload
        res_df = res_df.append(tmp_df)
    # res_df.to_pickle(open("../../raw_data/merged_0901_0930_bd_t_feats_hour.pkl", 'wb'))
    # res_df.to_csv("../../raw_data/merged_0901_0930_bd_t_feats_hour.csv")
    return res_df


def get_newEntity(begin_dt, end_dt, step, num_periods_tmp):
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


if __name__ == "__main__":
    # merged_data_min = data_merge('20220901', '20220930', 1, 288*1, 288*30, True)
    # merged_data_min = pickle.load(open("../../raw_data/merged_0901_0930_bd_t_feats.pkl", 'rb'))
    # merged_data_hour = transfor_T(merged_data_min)
    # print("test")

    get_newEntity('20220801', '20220830', 2, 288*2)


