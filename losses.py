import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.python.ops import clip_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.keras import backend_config


def binary_crossentropy(
        reduction_method='SUM_OVER_BATCH_SIZE', 
        from_logits=False):
    """
    https://github.com/tensorflow/tensorflow/blob/61cb2f0aad70f8798bdbabf96d4103bb72828af6/tensorflow/python/keras/backend.py#L5008
    
    Element-wise BCE.
    """
    def compute_loss(
            target, 
            output, 
            reduction_method=reduction_method,
            from_logits=from_logits):

        target = tf.convert_to_tensor(target)
        output = tf.convert_to_tensor(output)

        if from_logits:
            NotImplementedError()
        else:
            epsilon = backend_config.epsilon
            epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
            output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)
            bce = target * tf.math.log(output + epsilon())
            bce += (1 - target) * tf.math.log(1 - output + epsilon())
            raw_loss = -bce
            # raw_loss is (N, D)
        return reduction(raw_loss, reduction_method=reduction_method)
    return compute_loss


def _constant_to_tensor(x, dtype):
    return constant_op.constant(x, dtype=dtype)


def _safe_mean(losses, num_present):
    total_loss = tf.reduce_sum(losses)
    return tf.math.divide_no_nan(total_loss, num_present, name='value')


def _num_elements(losses):
    with backend.name_scope('num_elements') as scope:
        return tf.cast(tf.size(losses, name=scope), dtype=losses.dtype)


def reduction(raw_loss, reduction_method='SUM_OVER_BATCH_SIZE'):
    if reduction_method is None:
        loss = raw_loss
    else:
        if reduction_method == 'SUM_OVER_BATCH_SIZE':
            loss = _safe_mean(raw_loss, _num_elements(raw_loss))
        elif reduction_method == 'SUM_OVER_DIMENSION':
            loss = tf.math.divide_no_nan(
                tf.reduce_sum(raw_loss, axis=0), raw_loss.shape[0])
    return loss