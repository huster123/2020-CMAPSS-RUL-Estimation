import numpy as np
from keras import backend as K


def modelrmse(y_actual, y_predicted):
    return K.mean(K.square(y_actual - y_predicted)) ** 0.5


def rmse(y_actual, y_predicted):
    res = [(i - j) ** 2 for i, j in zip(y_actual, y_predicted)]
    return np.mean(res) ** 0.5


def accuracy(y_actual, y_predicted):  # 准确率，重视提前预测
    res = [1 if -13 <= (i - j) <= 10 else 0 for i, j in zip(y_actual, y_predicted)]
    return np.mean(res) * 100


def mape(y_actual, y_predicted):  # 平均绝对误差百分比
    res = [np.abs(i - j) / j for i, j in zip(y_actual, y_predicted)]
    return np.mean(res) * 100


def score(y_actual, y_predicted):  # 得分函数，重视提前预测
    res = [np.exp((i - j) / 13) - 1 if i > j else np.exp((j - i) / 10) - 1 for i, j in zip(y_actual, y_predicted)]
    return np.sum(res)
