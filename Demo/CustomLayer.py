from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.keras.backend.clear_session()
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


# 定义网络层就是：设置网络权重和输出到输入的计算过程
# Method1 : add weights by tf.Variable() method
class MyLayer1(layers.Layer):
    def __init__(self, input_dim=32, unit=32):
        super(MyLayer1, self).__init__()

        w_init = tf.random_normal_initializer()
        self.weight = tf.Variable(initial_value=w_init(
            shape=(input_dim, unit), dtype=tf.float32), trainable=True)

        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=b_init(
            shape=(unit,), dtype=tf.float32), trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias


# Method2 : add weights by add_weight() method
class MyLayer2(layers.Layer):
    def __init__(self, input_dim=32, unit=32):
        super(MyLayer2, self).__init__()
        # trainable represents whether the variable ...
        self.weight = self.add_weight(shape=(input_dim, unit),
                                      initializer=keras.initializers.RandomNormal(),
                                      trainable=True)
        self.bias = self.add_weight(shape=(unit,),
                                    initializer=keras.initializers.Zeros(),
                                    trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias


# Method3: 当定义网络时不知道网络的维度是可以重写build(self, input_shape)函数，用获得的shape构建网络
class MyLayer3(layers.Layer):
    def __init__(self, unit=32):
        super(MyLayer3, self).__init__()
        self.unit = unit

    def build(self, input_shape):
        self.weight = self.add_weight(shape=(input_shape[-1], self.unit),
                                      initializer=keras.initializers.RandomNormal(),
                                      trainable=True)
        self.bias = self.add_weight(shape=(self.unit,),
                                    initializer=keras.initializers.Zeros(),
                                    trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias

#
# x = tf.ones((3, 5))
# # my_layer = MyLayer2(5, 4)
# my_layer = MyLayer3()
# out = my_layer(x)
# print(out)


# 使用多个子layer自定义一个Block.注意:也要继承自layers.Layer
class MyBlock(layers.Layer):
    def __init__(self):
        super(MyBlock, self).__init__()
        self.layer1 = MyLayer3(32)
        self.layer2 = MyLayer3(16)
        self.layer3 = MyLayer3(2)

    def call(self, inputs):
        h1 = self.layer1(inputs)
        h1 = tf.nn.relu(h1)
        h2 = self.layer2(h1)
        h2 = tf.nn.relu(h2)
        return self.layer3(h2)


my_block = MyBlock()
print('trainable weights:', len(my_block.trainable_weights))
y = my_block(tf.ones(shape=(3, 64)))
# 构建网络在build()里面，所以执行了才有网络
print('trainable weights:', len(my_block.trainable_weights))


#通过构建网络层的方法来收集loss
class LossLayer(layers.Layer):

    def __init__(self, rate=1e-2):
        super(LossLayer, self).__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs


class OutLayer(layers.Layer):
    def __init__(self):
        super(OutLayer, self).__init__()
        self.loss_fun = LossLayer(1e-2)

    def call(self, inputs):
        return self.loss_fun(inputs)

my_layer = OutLayer()
print(len(my_layer.losses)) # 还未call
y = my_layer(tf.zeros(1,1))
print(len(my_layer.losses)) # 执行call之后
y = my_layer(tf.zeros(1,1))
print(len(my_layer.losses)) # call之前会重新置0
# 0
# 1
# 1


# 配置只有训练时可以执行的网络层
class MyDropout(layers.Layer):
    def __init__(self, rate, **kwargs):
        super(MyDropout, self).__init__(**kwargs)
        self.rate = rate
    def call(self, inputs, training=None):
        return tf.cond(training,
                       lambda: tf.nn.dropout(inputs, rate=self.rate),
                      lambda: inputs)


# 使自己的网络层可以序列化
class Linear(layers.Layer):

    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config

layer = Linear(125)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)
# {'name': 'linear_1', 'trainable': True, 'dtype': None, 'units': 125}


