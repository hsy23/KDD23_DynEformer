import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import datetime
import pickle
import random

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

MinMax = preprocessing.MinMaxScaler()
StdScaler = preprocessing.StandardScaler()


def train_test_split(X, y, train_ratio=0.8, test_ratio=0.2):
    num_ts, num_periods, num_features = X.shape
    # train_periods = int(num_periods * train_ratio)
    train_periods = int(num_periods * train_ratio)
    test_periods = int(num_periods * test_ratio)

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


def draw_data_plots(workloads):  # Visualize the workload
    x = np.arange(len(workloads))

    plt.figure(facecolor='w')  # Create a new white figure
    plt.plot(x, workloads, 'r-', linewidth=2)
    plt.grid(True)  # Display the grid lines
    plt.show()


def draw_true_pre_compare(history_Y, predictions, test_Y, p_id):  # Compare prediction and actual value by plotting
    his_x = np.arange(len(history_Y))
    pre_x = np.arange(48, len(his_x))

    plt.figure(facecolor='w')  # Create a new white figure
    plt.plot(his_x, history_Y, 'r-', linewidth=2)
    plt.vlines(48, np.min(history_Y), np.max(history_Y), color="blue", linestyles="dashed", linewidth=2)
    plt.plot(pre_x, test_Y, 'r-', linewidth=2, label='label')
    plt.plot(pre_x, predictions, 'g-', linewidth=2, label='pre')
    plt.legend(loc='upper left')  # Set the location of the legend to the upper left
    plt.grid(True)  # Display the grid lines
    # plt.show()
    plt.savefig(r'saved_res_pics/app_switch_{}.png'.format(p_id))  # Save the plot


def draw_true_pre_compare_normal(history_Y, predictions, test_Y, p_id):  # Compare prediction and actual value by plotting
    his_x = np.arange(len(history_Y))
    pre_x = np.arange(len(his_x), len(his_x)+len(predictions))

    plt.figure(facecolor='w')  # Create a new white figure
    plt.plot(his_x, history_Y, 'r-', linewidth=2)
    plt.vlines(len(his_x), np.min(history_Y), np.max(history_Y), color="blue", linestyles="dashed", linewidth=2)
    plt.plot(pre_x, test_Y, 'r-', linewidth=2, label='label')
    plt.plot(pre_x, predictions, 'g-', linewidth=2, label='pre')
    plt.legend(loc='upper left')  # Set the location of the legend to the upper left
    plt.grid(True)  # Display the grid lines
    # plt.show()
    plt.savefig(r'saved_res_pics/app_switch_{}.png'.format(p_id))  # Save the plot