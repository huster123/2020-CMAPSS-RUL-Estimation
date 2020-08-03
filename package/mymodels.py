from keras.layers import Input, Dense, Dropout, LSTM, Bidirectional, Flatten, Concatenate, Reshape, RepeatVector, \
    Conv2D, SeparableConv2D
from keras.models import Model
from matplotlib import pyplot as plt
from keras import backend as K
from keras.engine.topology import Layer
from keras import regularizers
from keras.engine.topology import Layer
from keras import regularizers
from keras.layers.core import Lambda
from keras_ordered_neurons import ONLSTM

K.clear_session()

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')


class AttentionLayer(Layer):
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size

        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                 initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                 initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        self.V = K.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        score = K.softmax(K.dot(H, self.V), axis=1)
        outputs = K.sum(score * inputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


# 一维CNN
def onedimcnn(step_size):
    x = Input(shape=(step_size, 14, 1))
    y = Conv2D(10, (10, 1), activation='tanh', padding='same')(x)
    y = Conv2D(10, (10, 1), activation='tanh', padding='same')(y)
    y = Conv2D(10, (10, 1), activation='tanh', padding='same')(y)
    y = Conv2D(10, (10, 1), activation='tanh', padding='same')(y)
    y = Conv2D(1, (3, 1), activation='tanh', padding='same')(y)
    y = Flatten()(y)
    y = Dropout(0.5)(y)
    y = Dense(100, activation='tanh')(y)
    y = Dense(1)(y)
    return Model(x, y)

#使用SeparableCNN取代普通CNN
def sepcnn(step_size):
    x = Input(shape=(step_size, 14, 1))
    y = SeparableConv2D(10, (10, 1), activation='tanh', padding='same')(x)
    y = SeparableConv2D(10, (10, 1), activation='tanh', padding='same')(y)
    y = SeparableConv2D(10, (10, 1), activation='tanh', padding='same')(y)
    y = SeparableConv2D(10, (10, 1), activation='tanh', padding='same')(y)
    y = SeparableConv2D(1, (10, 1), activation='tanh', padding='same')(y)
    y = Flatten()(y)
    y = Dropout(0.5)(y)
    y = Dense(100, activation='tanh')(y)
    y = Dense(1)(y)
    return Model(x, y)

#简单的lstm模型
def lstm(step_size):
    x = Input(shape=(step_size, 14))
    y = LSTM(7, return_sequences=True, activation='tanh')(x)
    y = LSTM(7, return_sequences=True, activation='tanh')(y)
    y = LSTM(7, return_sequences=True, activation='tanh')(y)
    y = LSTM(7, return_sequences=True, activation='tanh')(y)
    y = LSTM(7, return_sequences=True, activation='tanh')(y)
    y = Flatten()(y)
    y = Dropout(0.5)(y)
    y = Dense(100, activation='tanh')(y)
    y = Dense(1)(y)
    return Model(x, y)

#双向lstm模型
def blstm(step_size):
    x = Input(shape=(step_size, 14))
    y = Bidirectional(LSTM(7, return_sequences=True, activation='tanh'))(x)
    y = Bidirectional(LSTM(7, return_sequences=True, activation='tanh'))(y)
    y = Bidirectional(LSTM(7, return_sequences=True, activation='tanh'))(y)
    y = Bidirectional(LSTM(7, return_sequences=True, activation='tanh'))(y)
    y = Bidirectional(LSTM(7, return_sequences=True, activation='tanh'))(y)
    y = Flatten()(y)
    y = Dropout(0.5)(y)
    y = Dense(100, activation='tanh')(y)
    y = Dense(1)(y)
    return Model(x, y)

#带Attention机制的LSTM-ED模型
def lstmencoderdecoderattentionv1(step_size):
    x = Input(shape=(step_size, 14))
    node = 5
    y = LSTM(node, return_sequences=True, activation='tanh')(x)
    y = LSTM(node, return_sequences=True, activation='tanh')(y)
    y = AttentionLayer(node)(y)
    y = RepeatVector(step_size)(y)
    y = LSTM(node, return_sequences=True, activation='tanh')(y)
    y = LSTM(node, return_sequences=True, activation='tanh')(y)
    y = Flatten()(y)
    y = Dropout(0.5)(y)
    y = Dense(100, activation='tanh')(y)
    y = Dense(1)(y)
    return Model(x, y)

#带Attention机制的LSTM模型
def lstmattentionv1(step_size):
    x = Input(shape=(step_size, 14))
    y = LSTM(30, return_sequences=True, activation='tanh')(x)
    y = LSTM(30, return_sequences=True, activation='tanh')(y)
    y = AttentionLayer(attention_size=30)(y)
    y = Dropout(0.5)(y)
    y = Dense(20, activation='tanh')(y)
    y = Dense(1)(y)
    return Model(x, y)

#带Attention机制的双向LSTM模型
def blstmattentionv1(step_size):
    x = Input(shape=(step_size, 14))
    y = Bidirectional(LSTM(30, return_sequences=True, activation='tanh'))(x)
    y = Bidirectional(LSTM(30, return_sequences=True, activation='tanh'))(y)
    y = AttentionLayer(attention_size=30)(y)
    y = Dropout(0.5)(y)
    y = Dense(20, activation='tanh')(y)
    y = Dense(1)(y)
    return Model(x, y)