import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def lstm_model():
    model = keras.Sequential([
        layers.Embedding(input_dim=30000, output_dim=32, input_length=maxlen),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(1, activation='sigmoid', return_sequences=False)
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
    return model

num_words = 30000
maxlen = 200
embedding_size = 32
# 返回一个np.array,其中每个元素是一个不定长的list
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)
## 将 一维np.array[list] 转化为 2维 np.array[int]
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen, padding='post')
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen, padding='post')
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)
## lstm gru 的输入为3维向量(batch_size, sequence_len, input_size),输出为2维向量(当return_sequences=False,只返回sequence中last lstm单元的output)
# 或三维向量(当return_sequences=True,返回sequence中每个input经过lstm的output)
def lstm_model():
    model = keras.Sequential([
        layers.Embedding(input_dim=num_words, output_dim=embedding_size, input_length=maxlen),
        layers.LSTM(32, return_sequences=True),
        layers.LSTM(1, activation='sigmoid', return_sequences=False)
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
    return model

model = lstm_model()
model.summary()