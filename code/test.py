import numpy as np
import tensorflow as tf
from model import dice_coef_loss

ypred = tf.random.normal((2, 20, 480, 560))
ytrue = tf.random.normal((2, 20, 480, 560))
res = dice_coef_loss(ytrue, ypred)
print(res.shape, res, round(float(res), 4))
# print(res)
