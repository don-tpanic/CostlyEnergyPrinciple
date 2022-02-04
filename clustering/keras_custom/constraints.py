from tensorflow.python.keras.constraints import Constraint 
from tensorflow.python.keras import backend
from tensorflow.python.ops import math_ops
import tensorflow as tf

"""
Custom kernel constraints.
"""


class GreaterThanZero(Constraint):
    def __call__(self, w):
        w = tf.clip_by_value(w, clip_value_min=0., clip_value_max=tf.float32.max)
        return math_ops.cast(w, backend.floatx())


class GreaterEqualEpsilon(Constraint):
    """
    To avoid attention weights initialize from 0,
    we intialize them from epsilon=1e-6, hence the lower bound
    of attention weights naturally become epsilon
    """
    def __call__(self, w):
        w = tf.clip_by_value(w, clip_value_min=1.0e-6, clip_value_max=tf.float32.max)
        return math_ops.cast(w, backend.floatx())


class GreaterEqualOne(Constraint):
    def __call__(self, w):
        w = tf.clip_by_value(w, clip_value_min=1., clip_value_max=tf.float32.max)
        return math_ops.cast(w, backend.floatx())
    

class BetweenZeroAndOne(Constraint):
    def __call__(self, w):
        w = tf.clip_by_value(w, clip_value_min=0., clip_value_max=1.)
        return math_ops.cast(w, backend.floatx())


class SumToOne(Constraint):
    def __call__(self, w):
        # first apply non-neg 
        w = tf.clip_by_value(w, clip_value_min=1.0e-6, clip_value_max=tf.float32.max)
        w = math_ops.cast(w, backend.floatx())

        # then apply sum-to-one
        w = tf.clip_by_value(
            w / tf.reduce_sum(w), 
                clip_value_min=1.0e-6, clip_value_max=tf.float32.max)
        return math_ops.cast(w, backend.floatx())