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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

MinMax = preprocessing.MinMaxScaler()
StdScaler = preprocessing.StandardScaler()

TASK_ID_DICT = pickle.load(open(os.path.join(AB_PATH, "Task_ID_DicT.pkl"), 'rb'))


def get_data_path(file_name):
    return os.path.join(AB_PATH, file_name)


def app_filter(df):
    # Filters the real-time bandwidth data for the main applications; if applications are mixed, they are split and different dockers of the same service are merged.
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
    merged_data_min = data_merge('20220901', '20220930', 1, 288*1, 288*30, True)
    merged_data_min = pickle.load(open("../../raw_data/merged_0901_0930_bd_t_feats.pkl", 'rb'))
    merged_data_hour = transfor_T(merged_data_min)