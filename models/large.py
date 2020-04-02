from helper import INIT_METHODS
from models.base_model import BaseModel
import tensorflow as tf


class Large(BaseModel):
    def __init(self, data, target, model_params, data_params):
        self.num_layers = 8
        if self.params["weight_init"] == "gaussian":
            init_args = {"mean": 0.00,
                         "stddev": 0.01}
        else:
            init_args = {}

        self.weight_dict = {"W_c_1": tf.compat.v1.get_variable(shape=(7, 7, self.data_params["input_dims"], 32),
                                                               initializer=INIT_METHODS[self.params["weight_init"]](
                                                                   **init_args),
                                                               name="W_c_1"),
                            "b_c_1": tf.Variable(tf.zeros([32])),
                            "W_c_2": tf.compat.v1.get_variable(shape=(7, 7, 32, 64),
                                                               initializer=INIT_METHODS[self.params["weight_init"]](
                                                                   **init_args),
                                                               name="W_c_2"),
                            "b_c_2": tf.Variable(tf.zeros([64])),
                            "W_c_3": tf.compat.v1.get_variable(shape=(5, 5, 64, 128),
                                                               initializer=INIT_METHODS[self.params["weight_init"]](
                                                                   **init_args),
                                                               name="W_c_3"),
                            "b_c_3": tf.Variable(tf.zeros([128])),
                            "W_c_4": tf.compat.v1.get_variable(shape=(5, 5, 128, 256),
                                                               initializer=INIT_METHODS[self.params["weight_init"]](
                                                                   **init_args),
                                                               name="W_c_4"),
                            "b_c_4": tf.Variable(tf.zeros([256])),
                            "W_c_5": tf.compat.v1.get_variable(shape=(5, 5, 256, 512),
                                                               initializer=INIT_METHODS[self.params["weight_init"]](
                                                                   **init_args),
                                                               name="W_c_4"),
                            "b_c_5": tf.Variable(tf.zeros([512])),
                            "W_c_6": tf.compat.v1.get_variable(shape=(3, 3, 512, 1024),
                                                               initializer=INIT_METHODS[self.params["weight_init"]](
                                                                   **init_args),
                                                               name="W_c_4"),
                            "b_c_6": tf.Variable(tf.zeros([1024])),
                            "W_c_7": tf.compat.v1.get_variable(shape=(3, 3, 1024, 1024),
                                                               initializer=INIT_METHODS[self.params["weight_init"]](
                                                                   **init_args),
                                                               name="W_c_4"),
                            "b_c_7": tf.Variable(tf.zeros([1024])),
                            "W_1": tf.compat.v1.get_variable(shape=(32 ** 2 * 1024, 1024),
                                                             initializer=INIT_METHODS[self.params["weight_init"]](
                                                                 **init_args),
                                                             name="W_1"),
                            "b_1": tf.Variable(tf.zeros([1024])),
                            "W_2": tf.compat.v1.get_variable(shape=[1024, 256],
                                                             initializer=INIT_METHODS[self.params["weight_init"]](
                                                                 **init_args),
                                                             name="W_2"),
                            "b_2": tf.Variable(tf.zeros([256])),
                            "W_3": tf.compat.v1.get_variable(shape=(256, 1),
                                                             initializer=INIT_METHODS[self.params["weight_init"]](
                                                                 **init_args),
                                                             name="W_3"),
                            "b_3": tf.Variable(tf.zeros([1]))}

        super(Large, self).__init__(data, target, model_params, data_params)
