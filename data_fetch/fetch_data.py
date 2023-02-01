import pickle

from odps import ODPS
from tqdm import tqdm
from odps.df import DataFrame
from collections import defaultdict
import pandas as pd
import datetime


opt_file = 'secret keys.txt'
with open(opt_file, 'r') as f:
    acc_id, acc_key = f.readline().split(',')
# file_handle = open('ods_device.txt', mode='w')
o = ODPS(acc_id, acc_key, 'xxx', 'xxx')


def fetch(begin_date, end_date, save=True):
    dt_range = begin_date + '-' + end_date
    dts = getEveryDay(begin_date, end_date)
    print("开始爬取数据{}，总时间:{}天".format(dt_range, len(dts)))
    t_ods_device_order = get_order_from_txt("sql/sql_cluster_t_ods_device.txt")
    device_95_order = get_order_from_txt("sql/sql_cluster_95.txt")
    device_pro_order = get_order_from_txt("sql/sql_cluster_profit.txt")
    demand_area_order = get_order_from_txt("sql/sql_demand_area.txt")
    device_bd_order = get_order_from_txt("sql/sql_cluster_5bd.txt")

    t_ods_device_order = t_ods_device_order.format('(' + str(dts).strip('[]') + ')')
    device_95_order = device_95_order.format('(' + str(dts).strip('[]') + ')')
    device_pro_order = device_pro_order.format(end_date, limit='--')  # 收益数据是累计得到的
    device_bd_order = device_bd_order.format('(' + str(dts).strip('[]') + ')', mac_limit='--',
                                            mac_id="'f96be0f0a07c890bf9792a581b83caeb'", num_limit='--')

    # data_base = exe_sql(t_ods_device_order)
    # data_95 = exe_sql(device_95_order)
    # data_pro = exe_sql(device_pro_order)
    # demand_area = exe_sql(demand_area_order)
    device_bd = exe_sql(device_bd_order)

    if save:
        print("存储数据ing...")
        # pickle.dump(data_base, open("../raw_data/{}_mac_attr.pkl".format(dt_range), 'wb'), protocol=4)
        # pickle.dump(data_95, open("../raw_data/{}_mac_95.pkl".format(dt_range), 'wb'), protocol=4)
        # pickle.dump(data_pro, open("../raw_data/{}_mac_pro.pkl".format(dt_range), 'wb'), protocol=4)
        # pickle.dump(demand_area, open("../raw_data/{}_demand_area.pkl".format(end_date), 'wb'), protocol=4)
        pickle.dump(device_bd, open("../raw_data/{}_bd.pkl".format('bd_'+dt_range), 'wb'), protocol=4)

    # return data_base, data_95  # , data_pro


def get_odps_table(tb_name):
   data = DataFrame(o.get_table(tb_name, project='ppio_env_test'))
   # data['ds'] = data['ds'].astype('int')
   return data


def exe_sql(sql):
   print("执行{}......".format(sql[:10]))
   with o.execute_sql(sql).open_reader(tunnel=True) as reader:
       d = defaultdict(list)  # collection默认一个dict
       for record in tqdm(reader, total=reader.count):
           for res in record:
                d[res[0]].append(res[1])  # 解析record中的每一个元组，存储方式为(k,v)，以k作为key，存储每一列的内容；
       data = pd.DataFrame.from_dict(d, orient='index').T  # 转换为数据框，并转置，不转置的话是横条数据
   return data


def get_order_from_txt(order_path):
    sql_order = ""
    with open(order_path, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            sql_order = sql_order + line
    return sql_order


def get_last_dat(day):
    big_m = [1, 3, 5, 7, 8, 10, 12]
    if day in big_m:
        return 31
    else:
        return 30


def getEveryDay(begin_date, end_date):
    date_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y%m%d")
    end_date = datetime.datetime.strptime(end_date, "%Y%m%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y%m%d")
        date_list.append(date_str)
        begin_date += datetime.timedelta(days=1)
    return date_list


if __name__ == '__main__':
    for begin_date in range(20220901, 20220930, 2):
        end_date = begin_date + 1  # 每次获取两天数据
        begin_date, end_date = str(begin_date), str(end_date)
        fetch(begin_date, end_date, save=True)
    # fetch('20220901', '20220930', save=True)
