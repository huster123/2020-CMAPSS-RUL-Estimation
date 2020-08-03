import pandas as pd
from keras.callbacks import Callback
from keras.optimizers import Adam
from tqdm import tqdm
from package.mymodels import *
from package.evaluationindex import *

df = pd.read_csv("data/data.csv")
num = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
sensor = ['sensor measurement  ' + str(i) for i in num]

#滑窗的方式生成样本
def sequence(subdataset, stepsize, overslap):
    train_input, train_output1, train_output2 = [], [], []
    df2 = df[df.loc[:, 'sub_dataset'] == 'train_FD00' + str(subdataset)]
    df2.reset_index(drop=True, inplace=True)
    p = [i + 1 for i in range(int(df2.loc[:, 'unit number'].max()))]
    for i in tqdm(p):
        arr1 = df2[df2.loc[:, 'unit number'] == i].loc[:, sensor].values
        arr2 = df2[df2.loc[:, 'unit number'] == i].rul.values
        arr3 = df2[df2.loc[:, 'unit number'] == i].real_rul.values
        j = 0
        while j + stepsize - 1 < len(arr1):
            train_input.append(arr1[j:j + stepsize, :])
            train_output1.append(arr2[j + stepsize - 1])
            train_output2.append(arr3[j + stepsize - 1])
            j += overslap

    test_input, test_output1, test_output2 = [], [], []
    df2 = df[df.loc[:, 'sub_dataset'] == 'test_FD00' + str(subdataset)]
    df2.reset_index(drop=True, inplace=True)
    p = [i + 1 for i in range(int(df2.loc[:, 'unit number'].max()))]
    for i in tqdm(p):
        arr1 = df2[df2.loc[:, 'unit number'] == i].loc[:, sensor].values
        arr2 = df2[df2.loc[:, 'unit number'] == i].rul.values
        arr3 = df2[df2.loc[:, 'unit number'] == i].real_rul.values
        j = 0
        while j + stepsize - 1 < len(arr1):
            test_input.append(arr1[j:j + stepsize, :])
            test_output1.append(arr2[j + stepsize - 1])
            test_output2.append(arr3[j + stepsize - 1])
            j += overslap
    tail_input, tail_output1, tail_output2 = [], [], []
    df2 = df[df.loc[:, 'sub_dataset'] == 'test_FD00' + str(subdataset)]
    df2.reset_index(drop=True, inplace=True)
    p = [i + 1 for i in range(int(df2.loc[:, 'unit number'].max()))]
    for i in tqdm(p):
        arr1 = df2[df2.loc[:, 'unit number'] == i].loc[:, sensor].values
        arr2 = df2[df2.loc[:, 'unit number'] == i].rul.values
        arr3 = df2[df2.loc[:, 'unit number'] == i].real_rul.values
        tail_input.append(arr1[-stepsize:, :])
        tail_output1.append(arr2[-1])
        tail_output2.append(arr3[-1])
    return np.array(train_input), np.array(train_output1), np.array(train_output2), np.array(test_input), np.array(
        test_output1), np.array(test_output2), np.array(tail_input), np.array(tail_output1), np.array(tail_output2)
    return np.array(train_input).reshape(-1, stepsize, 14, 1), np.array(train_output1), np.array(
        train_output2), np.array(test_input).reshape(-1, stepsize, 14, 1), np.array(test_output1), np.array(
        test_output2), np.array(tail_input).reshape(-1, stepsize, 14, 1), np.array(tail_output1), np.array(
        tail_output2)


class StepLR(Callback):
    '''
    第二种方式,台阶式的学习率调整,不一定是衰减,也可以是增加,
    epoch_list = [100, 200, 300]
    lr_list    = [0.5, 0.1, 0.8]
    举个例子,参数设置如上, 在epoch 0~99区域内, 学习率是 0.5, 在100~199范围内,学习率是0.1, 在200~300范围内,学习率是0.8
    注意两个列表一定要一样长, 这是我简单的实现, 不是很鲁棒.
    而且, 将 "lr"放入logs时,tensorboard是可以观察 lr 的,
    '''

    def __init__(self, lr_list, epoch_list, verbose=1):
        super().__init__()

        self.lr_list = lr_list
        self.epoch_list = epoch_list
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.epoch_list[0]:
            K.set_value(self.model.optimizer.lr, self.lr_list[0])
        elif epoch == self.epoch_list[0]:
            self.lr_list.pop(0)
            self.epoch_list.pop(0)
            K.set_value(self.model.optimizer.lr, self.lr_list[0])
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRate'
                  'is %s.' % (epoch + 1, K.get_value(self.model.optimizer.lr)))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


subdataset = 1#数据集编号
stepsize = 30#滑窗长度
overslap = 1#滑窗间隔
train_input, train_output1, train_output2, test_input, test_output1, test_output2, tail_input, tail_output1, tail_output2 \
    = sequence(subdataset, stepsize, overslap)

# model = lstm(stepsize)
model = lstmattentionv1(stepsize)
# model = lstmencoderdecoderattentionv1(stepsize)
# model = onedimcnn(stepsize)
# model = sepcnn(stepsize)
adam = Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=[modelrmse])
model.summary()

#分段定学习率训练
epoch_list = [200, 250]
lr_list = [0.001, 0.0001]
stepLR = StepLR(lr_list, epoch_list)

history = model.fit(x=train_input, y=train_output1, batch_size=512, epochs=250, shuffle=True,
                    validation_data=(test_input, test_output1),
                    callbacks=[stepLR], verbose=2)
#收敛曲线
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(history.history['modelrmse'], label='root_mean_square_error')
plt.plot(history.history['val_modelrmse'], label='val_root_mean_square_error')
plt.legend()
plt.show()

#结果表格
res = []
y_prediction = model.predict(train_input)
res.append([rmse(train_output1, y_prediction), rmse(train_output2, y_prediction), '/', '/', '/', '/', '/', '/'])

y_prediction = model.predict(test_input)
res.append([rmse(test_output1, y_prediction), rmse(test_output2, y_prediction), '/', '/', '/', '/', '/', '/'])

y_prediction = model.predict(tail_input)
res.append(
    [rmse(tail_output1, y_prediction), rmse(tail_output2, y_prediction), score(tail_output1, y_prediction),
     score(tail_output2, y_prediction), accuracy(tail_output1, y_prediction), accuracy(tail_output2, y_prediction),
     mape(tail_output1, y_prediction), mape(tail_output2, y_prediction)])
results = pd.DataFrame(res, index=['训练集', '测试集', '测试集特定点'],
                       columns=['修正RMSE', '无修正RMSE', '修正得分', '无修正得分', '修正准确率', '无修正准确率',
                                '修正MAPE', '无修正MAPE'])
pd.set_option("display.max_columns", None)
print(results)
