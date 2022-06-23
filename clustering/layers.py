import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.keras import activations

try:
    from clustering.keras_custom import constraints, regularizers
except ModuleNotFoundError:
    from keras_custom import constraints, regularizers

"""
Custom layers used in the clustering model.
"""

class Distance(Layer):
    """
    Compute dimension-wise distance between 
    the input item and the cluster's center (SUSTAIN Eq.4)

    trainable:
    ----------
        mus: centers of each dimension
            - initializer:
                initializer doesn't matter because the center
                will always be one of the items when recruitment
                happens.
            - constraint:
                non negative

    inputs:
    -------
        An array-like item
    
    return:
    -------
        Array with the same dim as input. All values need to be [epsilon, 1].
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Distance, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mus = self.add_weight(
            name='mus',
            shape=(input_shape[1],),
            initializer=tf.keras.initializers.Constant([0.5, 0.5, 0.5]),
            constraint=tf.keras.constraints.NonNeg(),
            trainable=True)                           
        super(Distance, self).build(input_shape)

    def call(self, inputs):
        # clip to [epsilon, +inf]
        # can't allow diff to be 0 because
        # 0 unit output leads to 0 grads.
        distances = tf.clip_by_value(
            tf.abs(inputs - self.mus),
            clip_value_min=1e-06, 
            clip_value_max=tf.float32.max
        )
        return distances

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = super(Distance, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config


class DimensionWiseAttn(Layer):
    """
    Compute attn weighted dimension-wise distances (part of ALCOVE Eq.1)

    trainable:
    ----------
        ALPHAS: dimension-wise attention strength. init=epsilon
               constraint option 1: unbounded (>= epsilon)
               constraint option 2: sum to 1.
        
    hyperparam:
    -----------
        r: default to 2

    inputs:
    -------
        An array of hamming distances
    
    return:
    -------
        An array of attn weighted and r powered distances
    """
    def __init__(
            self, output_dim,
            r, 
            high_attn_constraint,
            high_attn_regularizer,
            high_attn_reg_strength,
            **kwargs):
        self.output_dim = output_dim
        self.r = r
        self.high_attn_constraint = high_attn_constraint
        self.high_attn_regularizer = high_attn_regularizer
        self.high_attn_reg_strength = high_attn_reg_strength
        super(DimensionWiseAttn, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.high_attn_constraint == 'nonneg':
            # checked: will infer based on shape.
            initializer = tf.keras.initializers.Constant([1/input_shape[1]])  
            constraint = constraints.GreaterEqualEpsilon()

        elif self.high_attn_constraint == 'sumtoone':
            initializer = tf.keras.initializers.Constant([1/input_shape[1]])
            constraint = constraints.SumToOne()
            
        if self.high_attn_regularizer == 'entropy':
            regularizer = regularizers.EntropyMinimizer(
                strength=self.high_attn_reg_strength)
        
        self.ALPHAS = self.add_weight(
            name='ALPHAS',
            shape=(input_shape[1],),
            initializer=initializer,                       
            constraint=constraint,      
            regularizer=regularizer,   
            trainable=True
        )
        super(DimensionWiseAttn, self).build(input_shape)

    def call(self, distances):
        # r powered distance for each dimension 
        r_powered_distances = tf.pow(distances, self.r)

        # alpha multiply to each powered distance
        attn_distances = tf.multiply(self.ALPHAS, r_powered_distances)
        return attn_distances

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = super(DimensionWiseAttn, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config


class ClusterActivation(Layer):
    """
    Compute cluster activation (part of ALCOVE Eq.1).

    trainable:
    ----------
        c: specificity (Default to untrainable).

    hyperparam:
    -----------
        r: default to 2
        q: default to 1
        trainable_specificity: default to False
    
    inputs:
    -------
        Array of attn weighted dimension-wise and r powered distances
    
    return:
    -------
        A scalar represents overall activation of a cluster.

    """
    def __init__(
            self, output_dim, r, q, 
            specificity, trainable_specificity, 
            **kwargs):
        self.output_dim = output_dim
        self.r = r
        self.q = q
        self.specificity = specificity
        self.trainable_specificity = trainable_specificity
        super(ClusterActivation, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.trainable_specificity:
            initializer = tf.keras.initializers.Constant([1.])
        else:
            initializer = tf.keras.initializers.Constant([self.specificity])
        self.c = self.add_weight(
            name='specificity',
            shape=(1,),
            initializer=initializer,
            constraint=constraints.GreaterThanZero(),
            trainable=self.trainable_specificity
        )
        super(ClusterActivation, self).build(input_shape)

    def call(self, attn_distances):
        # sum
        sum_of_distances = tf.reduce_sum(attn_distances, axis=1)

        # q/r powered sum and multiply specificity
        qr_powered_sum = tf.pow(sum_of_distances, self.q / self.r)
        c_sum = tf.multiply(qr_powered_sum, self.c)

        # get cluster activation.
        cluster_actv = tf.expand_dims( tf.exp(-c_sum), axis=1 )
        return cluster_actv

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = super(ClusterActivation, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config
    
    
class ClusterInhibition(Layer):
    """
    Apply lateral inhibition to all clusters (SUSTAIN Eq.6)
    
    trainable:
    ----------
        no trainable weights
        
    hyperparam:
    -----------
        beta: lateral inhibition param. Non-negative.
              if larger, means weakly inhibited. Default to ones.
    inputs:
    -------
        An array of cluster activation scalars. One scalar for
        each cluster.
    
    return:
    -------
        An array of cluster activation post inhibition.
        Note, actually we only want to keep the max. 
        Therefore, after this layer, the winner index will
        be used to set the non-max association weights to 0.
    """
    def __init__(self, output_dim, beta, **kwargs):
        self.output_dim = output_dim
        self.beta = beta
        super(ClusterInhibition, self).__init__(**kwargs)

    def call(self, H_concat):
        # H_concat are [H_1_act, H_2_act, ...]
        H_j_act2beta = tf.pow(H_concat, self.beta)
        denom = tf.reduce_sum(H_j_act2beta, axis=1)
        divided = tf.divide(H_j_act2beta, denom)
        H_j_out = tf.multiply(divided, H_concat)
        return H_j_out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = super(ClusterInhibition, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config

    
class Mask(Layer):
    """
    This Mask is used in two different locations in 
    the network:

        1. Zero out clusters that have not been recruited.
        This is to make sure both cluster inhibition and
        competition only happen within the recruited clusters.

        2. Zero out cluster activations to the decision unit
        except for the winner cluster. When there is a winner, 
        the unit corresponds to winner index will have weight flipped to 1.

    inputs:
    -------
        An array of cluster_actv from ClusterActivation

    params:
    -------
        Default weights are zeros at initialisation. 

    returns:
    --------
        Same shape as input, but masked.
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Mask, self).__init__(**kwargs)

    def build(self, input_shape):
        self.mask = self.add_weight(
            name='mask',
            shape=(input_shape[1],),
            initializer='zeros',  
            trainable=False)

    def call(self, inputs):
        masked = tf.multiply(inputs, self.mask)
        return masked

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = super(Mask, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config


class Classification(Layer):
    """
    Final decision layer.
    
    trainable:
    ----------
        w: association weights
    
    return:
    -------
        array of 2 scalars (probabilities)
    """
    def __init__(self, output_dim, activation, Phi, **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.Phi = Phi 
        super(Classification, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            name='w',
            shape=(input_shape[1], self.output_dim),
            initializer='zeros',
            trainable=True
        )
        super(Classification, self).build(input_shape)

    def call(self, H_out):
        # H_out is [None, H_out]
        C_out = tf.matmul(H_out, self.w)

        if self.activation is not None:
            output = self.activation(
                tf.multiply(C_out, self.Phi)
            )
            output._C_out = C_out
            return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def get_config(self):
        config = super(Classification, self).get_config()
        config.update({'output_dim': self.output_dim})
        return config