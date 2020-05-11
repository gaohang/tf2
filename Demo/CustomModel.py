import tensorflow as tf
from tensorflow.keras import layers
# print(tf.__version__)
# print(tf.keras.__version__)
import numpy as np
train_x = np.random.random((4096, 72))
train_y = np.random.random((4096, 10))
val_x = np.random.random((256, 72))
val_y = np.random.random((256, 10))
ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(batch_size=64).repeat(count=2)
val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(batch_size=64).repeat(count=1)
test_x = np.random.random((1000, 72))
test_y = np.random.random((1000, 10))
test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_data = test_data.batch(32).repeat()


class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        self.layer1 = layers.Dense(32, activation='relu')
        self.layer2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        h1 = self.layer1(inputs)
        out = self.layer2(h1)
        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


model = MyModel(num_classes=10)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=['accuracy'])

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(train_x, train_y, batch_size=64, epochs=4, callbacks=callbacks)
model.save_weights('custom_model', save_format="tf")

# model = tf.keras.models.load_model('all_model')
# estimator = tf.keras.estimator.model_to_estimator(model)
# print(estimator)
