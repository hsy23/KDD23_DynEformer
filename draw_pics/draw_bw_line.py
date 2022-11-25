import pickle

import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

db = pickle.load(open('../raw_data/7f5b_20220822-20220825_bd.pkl', 'rb'))
db['bw_upload'] = db['bw_upload']/1024/1024/1024  # GB
db_ks_zj = db[(db['name'] == 'ks') | (db['name'] == 'zjtd')].sort_values(by='time_id')
db_ks_zj['idx'] = range(len(db_ks_zj))
db_ks = db_ks_zj[db_ks_zj['name'] == 'ks'].sort_values(by='time_id')
db_zj = db_ks_zj[db_ks_zj['name'] == 'zjtd'].sort_values(by='time_id')

ks_x = list(db_ks['idx'])
zj_x = list(db_zj['idx'])
ks_db = list(db_ks['bw_upload'])
zj_db = list(db_zj['bw_upload'])

db2 = pickle.load(open('../raw_data/7f5b_20220815-20220818_bd.pkl', 'rb'))
db2 = db2[db2['name'] == 'ks'].sort_values(by='time_id')
db2['bw_upload'] = db2['bw_upload']/1024/1024/1024  # GB
db2['idx'] = range(len(db2))
db2_x = list(db2['idx'])
db2_y = list(db2['bw_upload'])

db3 = pickle.load(open('../raw_data/53d_20220802-20220805_bd.pkl', 'rb'))
db3 = db3[db3['name'] == 'tx80'].sort_values(by='time_id')
db3['bw_upload'] = db3['bw_upload']/1024/1024/1024  # GB
db3['idx'] = range(len(db3))
db3_x = list(db3['idx'])
db3_y = list(db3['bw_upload'])

# # 密度图
# with open("gps0.pkl", 'rb') as f:
#     gps0 = pkl.load(f)
# with open("gps1.pkl", 'rb') as f:
#     gps1 = pkl.load(f)
#
# gps0 = np.sort(gps0)
# gps1 = np.sort(gps1)
#
# index0 = range(0, len(gps0), 100)
# index1 = range(0, len(gps1), 100)
#
# gps0 = gps0[index0].astype(int)
# gps1 = gps1[index1].astype(int)

# sns.set()  # 恢复默认的格式
# 设置图例并且设置图例的字体及大小
font = {'family': 'Times New Roman',
'weight' : 'normal',
'size'   : 10,
}

# Draw Plot
fig = plt.figure()
plt.rcParams['xtick.direction'] ='in'  # 刻度线内向
plt.rcParams['ytick.direction'] ='in'

ax1 = fig.add_subplot(3, 1, 1)
plt.grid(axis='y', linestyle=':', zorder=0)
l_db2 = ax1.plot(db2_x, db2_y, color="#545E75", label="KWai", alpha=.7, linewidth=1.5)
x_length = len(db2_x)
x_idx = [int(0.25*x_length), int(0.5*x_length), int(0.75*x_length), int(x_length)]
plt.xlabel('(a) Work Steadily', fontdict={'family': 'Times New Roman', 'size': 13})
plt.xticks([(x-100) for x in x_idx], ['day1', 'day2', 'day3', 'day4'])
plt.ylim(0.0, 1.0)
plt.yticks([0.0, 0.3, 0.6, 0.9])
plt.tick_params(labelsize=10)

# ax1.legend(prop=font, frameon=False, loc='upper left')

ax = fig.add_subplot(3, 1, 2)
plt.grid(axis='y', linestyle=':', zorder=0)
l_ks = ax.plot(ks_x, ks_db, color="#545E75", label="KWai", alpha=.7, linewidth=1.5)
l_cold = ax.plot(np.arange(len(ks_x), len(ks_x)+50), [0]*50, color="#E09F3E", alpha=.7, linewidth=1.5)
l_zj = ax.plot([x+50 for x in zj_x], zj_db, color="#E09F3E", label="Tik Tok", alpha=.7, linewidth=1.5)
x_length = len(ks_x)+len(zj_x)+50
x_idx = [int(0.25*x_length), int(0.5*x_length), int(0.75*x_length), int(x_length)]
plt.xticks([(x-100) for x in x_idx], ['day1', 'day2', 'day3', 'day4'])
plt.yticks([0.00, 0.40, 0.80, 1.20])
plt.tick_params(labelsize=10)
plt.xlabel('(b) Application Switching', fontdict={'family': 'Times New Roman', 'size': 13})

# ax.legend(prop=font, frameon=False, loc='upper left')
#
ax2 = fig.add_subplot(3, 1, 3)
plt.grid(axis='y', linestyle=':', zorder=0)
l_db3 = ax2.plot(db3_x, db3_y, color="#FF0000", label="new app", alpha=.7, linewidth=1.5)
x_length = len(db3_x)
x_idx = [int(0.25*x_length), int(0.5*x_length), int(0.75*x_length), int(x_length)]
plt.xticks([(x-100) for x in x_idx], ['day1', 'day2', 'day3', 'day4'])
plt.ylim(0.0, 4.7)
plt.yticks([0.0, 1.5, 3.0, 4.5])
plt.tick_params(labelsize=10)
plt.xlabel('(c) New App Appears', fontdict={'family': 'Times New Roman', 'size': 13})
# ax2.legend(prop=font, frameon=False, loc='upper left')


# sns.lineplot(x=index0, y=gps0, color="#545E75", label="D2D", alpha=.7, ax=ax)
# sns.lineplot(x=index1, y=gps1, color="#E09F3E", label="Gowalla", alpha=.7, ax=ax1)


# l2 = ax1.plot(index1, gps1, color="#545E75", label="Gowalla", alpha=.7, linewidth=5)



# ax1.set_ylabel('Gowalla', fontsize=30)
# ax1.legend(prop=font, loc='upper right')

# ax1.set_xlabel('GPS Positional Accuracy (decimal places)', fontsize=30)

# lns = l1 + l2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, prop=font, loc='upper left')

# plt.xlim((4.0, 16.5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
plt.show()

fig.savefig('pics/workloads2.svg', dpi=600, format='svg')
