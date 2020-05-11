import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
num_features = 3000
sequence_length = 300
embedding_dimension = 100
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_features)
x_train = pad_sequences(x_train, maxlen=sequence_length)
x_test = pad_sequences(x_test, maxlen=sequence_length)
epoch = 4

## kernel_size=5,单一尺寸
# def imdb_cnn():
#     model = keras.Sequential([
#         # embedding的三个维度  one-hot维度, embedding后的维度, 序列长度
#         layers.Embedding(input_dim=num_features, output_dim=embedding_dimension,input_length=sequence_length),
#         layers.Conv1D(filters=50, kernel_size=5, strides=1, padding='valid'),# #params = (5*100+1)*50
#         layers.MaxPool1D(2, padding='valid'),
#         layers.Flatten(),
#         layers.Dense(10, activation='relu'),
#         layers.Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer=keras.optimizers.Adam(1e-3),
#                  loss=keras.losses.BinaryCrossentropy(),
#                  metrics=['accuracy'])
#
#     return model
# model = imdb_cnn()
# model.summary()
# history = model.fit(x_train, y_train, batch_size=64, epochs=epoch, validation_split=0.1)

## kernel_size=5,三种卷积核尺寸
filter_sizes=[3,4,5]
def convolution():  #要放入keras.Sequential的自定义层,输入inn,返回model
    inn = layers.Input(shape=(sequence_length, embedding_dimension, 1))
    cnns = []
    for size in filter_sizes:
        conv = layers.Conv2D(filters=64, kernel_size=(size, embedding_dimension),
                            strides=1, padding='valid', activation='relu')(inn)
        pool = layers.MaxPool2D(pool_size=(sequence_length-size+1, 1), padding='valid')(conv)
        cnns.append(pool)
    outt = layers.concatenate(cnns)
    # 进入keras.Model,其中有基于keras.Model创建自定义model的说明, 自定义layer见https://zhuanlan.zhihu.com/p/59481536
    model = keras.Model(inputs=inn, outputs=outt)
    return model

def cnn_mulfilter():
    model = keras.Sequential([
        layers.Embedding(input_dim=num_features, output_dim=embedding_dimension,
                        input_length=sequence_length),
        layers.Reshape((sequence_length, embedding_dimension, 1)),
        convolution(),            # 自定义层放入 keras.Sequential
        layers.Flatten(),
        layers.Dense(10, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')

    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
    return model

model = cnn_mulfilter()
model.summary()
history = model.fit(x_train, y_train, batch_size=64, epochs=epoch, validation_split=0.1)
