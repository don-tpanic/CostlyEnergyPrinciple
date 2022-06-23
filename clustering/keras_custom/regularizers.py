from tensorflow.python.keras.regularizers import Regularizer 
from tensorflow.python.keras import backend
from tensorflow.python.ops import math_ops
import tensorflow as tf

"""
Custom kernel regularizers
"""

class EntropyMinimizer(Regularizer):
    """
    Minimize the entropy of a weight vector.
    """
    def __init__(self, strength):
        self.strength = strength

    def __call__(self, x):    
        ent = - tf.reduce_sum( x * tf.experimental.numpy.log2(x) )
        return self.strength * ent

    def get_config(self):
        return {'strength': self.strength}