import numpy as np
import pandas as pd
from tqdm import tqdm

dir_name = "C:/Users/Administrator/Desktop/2020-Aircraft-Engine-RUL-Estimation/CMAPSSData/"

column = ['unit number', 'time, in cycles', 'operational setting 1', 'operational setting 2',
          'operational setting 3', 'sensor measurement  1', 'sensor measurement  2', 'sensor measurement  3',
          'sensor measurement  4', 'sensor measurement  5', 'sensor measurement  6', 'sensor measurement  7',
          'sensor measurement  8', 'sensor measurement  9', 'sensor measurement  10', 'sensor measurement  11',
          'sensor measurement  12', 'sensor measurement  13', 'sensor measurement  14', 'sensor measurement  15',
          'sensor measurement  16', 'sensor measurement  17', 'sensor measurement  18', 'sensor measurement  19',
          'sensor measurement  20', 'sensor measurement  21']
df = pd.DataFrame(columns=['sub_dataset'] + column + ['rul', 'real_rul'])
rul = []
for i in tqdm(range(4)):
    x = np.loadtxt(dir_name + "train_FD00" + str(i + 1) + ".txt")
    y = np.loadtxt(dir_name + "test_FD00" + str(i + 1) + ".txt")
    for j in range(21):
        if x[:, -j].max() == x[:, -j].min():
            x[:, -j] = np.zeros(len(x))
        if x[:, -j].max() != x[:, -j].min():
            x[:, -j] = 2 * (x[:, -j] - x[:, -j].min()) / (x[:, -j].max() - x[:, -j].min()) - 1
        if y[:, -j].max() == y[:, -j].min():
            y[:, -j] = np.zeros(len(y))
        if y[:, -j].max() != y[:, -j].min():
            y[:, -j] = 2 * (y[:, -j] - y[:, -j].min()) / (y[:, -j].max() - y[:, -j].min()) - 1
    x = pd.concat([pd.DataFrame(['train_FD00' + str(i + 1)] * len(x), columns=['sub_dataset']),
                   pd.DataFrame(x, columns=column)], axis=1)
    y = pd.concat([pd.DataFrame(['test_FD00' + str(i + 1)] * len(y), columns=['sub_dataset']),
                   pd.DataFrame(y, columns=column)], axis=1)
    df = pd.concat([df, x], axis=0)
    df = pd.concat([df, y], axis=0)
    rul.append(np.loadtxt(dir_name + "RUL_FD00" + str(i + 1) + ".txt"))


def label(cycle, remain):
    time = (-1) * cycle + cycle.max() + remain
    max_rul = 125
    lpwrul = [max_rul if k > max_rul else k for k in time.values]
    realrul = [k for k in time.values]
    return lpwrul, realrul


train_RUL = pd.DataFrame(columns=['rul'])
train_RUL_real = pd.DataFrame(columns=['real_rul'])
for i in tqdm(range(4)):
    x = df[df.loc[:, 'sub_dataset'] == 'train_FD00' + str(i + 1)]
    for j in range(int(x.loc[:, 'unit number'].max())):
        y = x[x.loc[:, 'unit number'] == j + 1].loc[:, 'time, in cycles']
        m, n = label(y, 0)
        m = pd.DataFrame(m, columns=['rul'])
        n = pd.DataFrame(n, columns=['real_rul'])
        train_RUL = pd.concat([train_RUL, m], axis=0)
        train_RUL_real = pd.concat([train_RUL_real, n], axis=0)
test_RUL = pd.DataFrame(columns=['rul'])
test_RUL_real = pd.DataFrame(columns=['real_rul'])
for i in tqdm(range(4)):
    x = df[df.loc[:, 'sub_dataset'] == 'test_FD00' + str(i + 1)]
    for j in range(int(x.loc[:, 'unit number'].max())):
        y = x[x.loc[:, 'unit number'] == j + 1].loc[:, 'time, in cycles']
        m, n = label(y, rul[i][j])
        m = pd.DataFrame(m, columns=['rul'])
        n = pd.DataFrame(n, columns=['real_rul'])
        test_RUL = pd.concat([test_RUL, m], axis=0)
        test_RUL_real = pd.concat([test_RUL_real, n], axis=0)

RUL = pd.concat([train_RUL, test_RUL], axis=0)
RUL.reset_index(drop=True, inplace=True)
RUL_real = pd.concat([train_RUL_real, test_RUL_real], axis=0)
RUL_real.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)
df = pd.concat([df, RUL, RUL_real], axis=1)

df.to_csv("data/data.csv", index=False)
