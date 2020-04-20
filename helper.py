"""
This script is inspired from the work of:
        https://danijar.com/structuring-your-tensorflow-models/
        Author: Danjar Hafner

Lazy property decorator is used in the model of my project.
"""
import functools
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def cache_func(func):
    attr = '_cache_' + func.__name__

    @property
    @functools.wraps(func)
    def decorator(self):
        if not hasattr(self, attr):
            setattr(self, attr, func(self))
        return getattr(self, attr)

    return decorator


INIT_METHODS = {"xavier": tf.initializers.glorot_normal,
                "gaussian": tf.random_normal_initializer,
                "uniform": tf.uniform_unit_scaling_initializer}
