import tensorflow as tf
import tensorflow.keras.backend as K
from functools import partial, update_wrapper


def wbce(tf_y_true, tf_y_pred, pos_class_wgt):
    y_true = tf.cast(tf_y_true, dtype=tf_y_pred.dtype)
    y_pred = tf.cast(tf_y_pred, dtype=tf_y_pred.dtype)

    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())

    #logloss = -((y_true * K.log(y_pred) * class_wgts_dict[1]) + ((1 - y_true) * K.log(1 - y_pred) * class_wgts_dict[0]) )
    #tf.math.scalar_mul(
    #logloss = -( tf.math.scalar_mul(class_wgts_dict[1], (y_true* K.log(y_pred)) ) + tf.math.scalar_mul(class_wgts_dict[0], ((1 - y_true) * K.log(1 - y_pred)) ))

    pos_class = y_true * K.log(y_pred)
    neg_class = (1 - y_true) * K.log(1 - y_pred)

    logloss = -((pos_class*pos_class_wgt) + neg_class)

    return K.mean(logloss, axis=-1)


def wrapped_partial(func, *args, **kwargs):
    """
    http://louistiao.me/posts/adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/
    """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func
