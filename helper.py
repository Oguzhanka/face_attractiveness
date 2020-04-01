"""
This script is taken from the website:
        https://danijar.com/structuring-your-tensorflow-models/
        Author: Danjar Hafner

Lazy property decorator is used throughout my project.
"""
import functools
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


INIT_METHODS = {"xavier": tf.initializers.glorot_normal,
                "gaussian": tf.random_normal_initializer,
                "uniform": tf.uniform_unit_scaling_initializer}
