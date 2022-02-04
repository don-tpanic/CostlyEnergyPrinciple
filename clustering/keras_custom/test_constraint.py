import os 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.python.keras.constraints import Constraint 
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend



w = tf.Variable(-0.5)
print(math_ops.greater_equal(w, 0.))   # False
w = w * math_ops.cast(math_ops.greater_equal(w, 0.), backend.floatx())
print(w)
# conclusion, casting doesn;t work like clipping
# in tf, nonneg is not doing clipping but a True/False condition.

# import numpy as np
# from constraints import SumToOne

# inputs = tf.keras.layers.Input(shape=(2,))
# x = tf.keras.layers.Dense(
#     2, 
#     name='hid', 
#     kernel_constraint=SumToOne(), 
#     kernel_initializer='ones',
#     use_bias=False)(inputs)

# outputs = tf.keras.layers.Dense(
#     2, 
#     name='out', 
#     trainable=False,
#     kernel_initializer='ones',
#     use_bias=False)(x)

# model = tf.keras.Model(
#     inputs=inputs,
#     outputs=outputs
# )

# optimizer = tf.keras.optimizers.SGD(learning_rate=100)
# loss_fn = tf.keras.losses.MSE
# np.random.seed(44)
# y_true = [[1, 0]]

# for i in range(10):
#     print('====== epoch ', i, '=======')
#     x = np.random.random((1, 2))
#     print('x = ', x)
#     with tf.GradientTape() as tape:
#         y_pred = model(x, training=True)
#         print('weights Before =', model.get_layer('hid').get_weights())
#         # print('y_pred = ', y_pred)
#         loss_value = loss_fn(y_true, y_pred)    

#         grads = tape.gradient(loss_value, model.trainable_weights)
#         # print('grads = ', grads)
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))
#         print('weights After =', model.get_layer('hid').get_weights())

# conclusion: constraint is applied after update.