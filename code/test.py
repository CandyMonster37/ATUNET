import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


class net(keras.Model):
    def __init__(self):
        super(net, self).__init__()
        self.conv1 = layers.Conv2D(64, 3)
        self.conv2 = layers.Conv2D(128, 3)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x


a = np.random.random((2, 10, 16, 16, 256))
print('before:', a.shape)
# atten = layers.MultiHeadAttention(num_heads=2, key_dim=3, attention_axes=(2, 3, 4))
convt = layers.Conv2D(filters=128, kernel_size=3, strides=(1, 1))
a = convt(a)
# pool = layers.MaxPool2D(pool_size=(2, 2), input_shape=a.shape[2:-1])
# ap = pool(a)
# ap = tf.reshape(ap, (-1, 10, ap.shape[-3], ap.shape[-2], ap.shape[-1]))
print('after:', a.shape)
# print(ap.shape, id(ap))
# print('a', a.shape)
# la = net()
# na = la.call(a)
# print('na', na.shape)
