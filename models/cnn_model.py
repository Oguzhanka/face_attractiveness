"""
Model implementation.
"""
from helper import cache_func, INIT_METHODS
import tensorflow as tf


class CNNModel:
    """
    CNN model implementation. Covers the implementations for both the large and the
    compact network.
    """
    def __init__(self, data, target, model_params, data_params):
        self.data = data
        self.target = target
        self.params = model_params
        self.data_params = data_params

        # Weight initialization object arguments.
        if self.params["weight_init"] == "gaussian":
            init_args = {"mean": 0.0,
                         "stddev": 1}
        else:
            init_args = {}

        if model_params["model_type"] == "large":       # LargeNet Implementation.
            self.num_layers = 9                         # Number of conv. layers.
            self.num_deep = 4                           # Number of dense layers.
            self.strides = [1, 1, 1, 1, 1, 1, 1, 1]
            self.weight_dict = {"W_c_1": tf.compat.v1.get_variable(shape=(7, 7, self.data_params["input_dims"], 16),
                                                                   initializer=INIT_METHODS[self.params["weight_init"]](**init_args),
                                                                   name="W_c_1"),
                                "b_c_1": tf.Variable(tf.zeros([16])),
                                "W_c_2": tf.compat.v1.get_variable(shape=(7, 7, 16, 16),
                                                                   initializer=INIT_METHODS[self.params["weight_init"]](**init_args),
                                                                   name="W_c_2"),
                                "b_c_2": tf.Variable(tf.zeros([16])),
                                "W_c_3": tf.compat.v1.get_variable(shape=(5, 5, 16, 32),
                                                                   initializer=INIT_METHODS[self.params["weight_init"]](**init_args),
                                                                   name="W_c_3"),
                                "b_c_3": tf.Variable(tf.zeros([32])),
                                "W_c_4": tf.compat.v1.get_variable(shape=(5, 5, 32, 32),
                                                                   initializer=INIT_METHODS[self.params["weight_init"]](**init_args),
                                                                   name="W_c_4"),
                                "b_c_4": tf.Variable(tf.zeros([32])),
                                "W_c_5": tf.compat.v1.get_variable(shape=(5, 5, 32, 64),
                                                                   initializer=INIT_METHODS[self.params["weight_init"]](**init_args),
                                                                   name="W_c_5"),
                                "b_c_5": tf.Variable(tf.zeros([64])),
                                "W_c_6": tf.compat.v1.get_variable(shape=(3, 3, 64, 128),
                                                                   initializer=INIT_METHODS[self.params["weight_init"]](**init_args),
                                                                   name="W_c_6"),
                                "b_c_6": tf.Variable(tf.zeros([128])),
                                "W_c_7": tf.compat.v1.get_variable(shape=(3, 3, 128, 256),
                                                                   initializer=INIT_METHODS[self.params["weight_init"]](**init_args),
                                                                   name="W_c_7"),
                                "b_c_7": tf.Variable(tf.zeros([256])),
                                "W_c_8": tf.compat.v1.get_variable(shape=(3, 3, 256, 512),
                                                                   initializer=INIT_METHODS[self.params["weight_init"]](**init_args),
                                                                   name="W_c_8"),
                                "b_c_8": tf.Variable(tf.zeros([512])),
                                "W_1": tf.compat.v1.get_variable(shape=(34 ** 2 * 512, 512),
                                                                 initializer=INIT_METHODS[self.params["weight_init"]](**init_args),
                                                                 name="W_1"),
                                "b_1": tf.Variable(tf.zeros([512])),
                                "W_2": tf.compat.v1.get_variable(shape=[512, 256],
                                                                 initializer=INIT_METHODS[self.params["weight_init"]](**init_args),
                                                                 name="W_2"),
                                "b_2": tf.Variable(tf.zeros([256])),
                                "W_3": tf.compat.v1.get_variable(shape=(256, 1),
                                                                 initializer=INIT_METHODS[self.params["weight_init"]](**init_args),
                                                                 name="W_3"),
                                "b_3": tf.Variable(tf.zeros([1]))}

        else:                           # Implementation of the compact network.
            self.num_layers = 5
            self.num_deep = 4
            self.strides = [3, 1, 1, 1]
            self.weight_dict = {"W_c_1": tf.compat.v1.get_variable(shape=(5, 5, self.data_params["input_dims"], 16),
                                                                   initializer=INIT_METHODS[self.params["weight_init"]](
                                                                       **init_args),
                                                                   name="W_c_1"),
                                "b_c_1": tf.Variable(tf.zeros([16])),
                                "W_c_2": tf.compat.v1.get_variable(shape=(5, 5, 16, 32),
                                                                   initializer=INIT_METHODS[self.params["weight_init"]](
                                                                       **init_args),
                                                                   name="W_c_2"),
                                "b_c_2": tf.Variable(tf.zeros([32])),
                                "W_c_3": tf.compat.v1.get_variable(shape=(3, 3, 32, 64),
                                                                   initializer=INIT_METHODS[self.params["weight_init"]](
                                                                       **init_args),
                                                                   name="W_c_3"),
                                "b_c_3": tf.Variable(tf.zeros([64])),
                                "W_c_4": tf.compat.v1.get_variable(shape=(3, 3, 64, 128),
                                                                   initializer=INIT_METHODS[self.params["weight_init"]](
                                                                       **init_args),
                                                                   name="W_c_4"),
                                "b_c_4": tf.Variable(tf.zeros([128])),
                                "W_1": tf.compat.v1.get_variable(shape=(10 ** 2 * 128, 1024),
                                                                 initializer=INIT_METHODS[self.params["weight_init"]](**init_args),
                                                                 name="W_1"),
                                "b_1": tf.Variable(tf.zeros([1024])),
                                "W_2": tf.compat.v1.get_variable(shape=[1024, 256],
                                                                 initializer=INIT_METHODS[self.params["weight_init"]](**init_args),
                                                                 name="W_2"),
                                "b_2": tf.Variable(tf.zeros([256])),
                                "W_3": tf.compat.v1.get_variable(shape=(256, 1),
                                                                 initializer=INIT_METHODS[self.params["weight_init"]](**init_args),
                                                                 name="W_3"),
                                "b_3": tf.Variable(tf.zeros([1]))}

        for key, val in self.weight_dict.items():   # Rename the layers.
            self.weight_dict[key] = tf.identity(self.weight_dict[key], name=key)

        # Flag indicating whether the current session is training or prediction.
        self.training = tf.compat.v1.placeholder(tf.bool, shape=[])
        self.layer_out = None

        self.convs = {}
        self.biass = {}
        self.activations = {}
        self.pools = {}
        self.weight = {"W_c_1": None}

        self.densed = {}
        self.biased = {}
        self.activated = {}
        self.dense_weights = {}
        self.bias_weights = {}
        self._loss = None

        # Initialize the lazy properties.
        _ = self.optimize
        _ = self.eval_loss
        self.epoch_counter = 0

    @cache_func
    def predict(self):
        """
        Forward function for the models.
        :return: Output of the model.
        """
        for c in range(1, self.num_layers):
            if c == 1:
                input_ = self.data
                self.weight["W_c_1"] = tf.transpose(self.weight_dict["W_c_1"], [3, 0, 1, 2])
            else:
                input_ = pool
            conv = tf.nn.conv2d(input_, self.weight_dict[f"W_c_{c}"], self.strides[c-1], "VALID")
            self.convs[c] = tf.transpose(conv, [3, 0, 1, 2])
            bias = tf.nn.bias_add(conv, self.weight_dict[f"b_c_{c}"])
            self.biass[c] = tf.transpose(bias, [3, 0, 1, 2])
            if self.params["batch_norm"]:
                bias = tf.compat.v1.layers.batch_normalization(bias, axis=-1, training=self.training,
                                                               momentum=0.7)
            activation = tf.nn.relu(bias)
            self.activations[c] = tf.transpose(activation, [3, 0, 1, 2])
            pool = tf.nn.pool(activation, (3, 3),
                              "MAX", padding="VALID")
            self.pools[c] = tf.transpose(pool, [3, 0, 1, 2])

        flatten = tf.compat.v1.layers.flatten(pool, name=None, data_format='channels_last')

        layer_out = flatten
        for d in range(1, self.num_deep):
            densed = tf.matmul(layer_out, self.weight_dict[f"W_{d}"],
                               transpose_a=False, transpose_b=False)
            self.densed[d] = densed
            self.dense_weights[d] = self.weight_dict[f"W_{d}"]
            layer_out = densed + self.weight_dict[f"b_{d}"]
            self.biased[d] = layer_out
            self.bias_weights = self.weight_dict[f"b_{d}"]

            if self.params["batch_norm"]:
                layer_out = tf.compat.v1.layers.batch_normalization(layer_out, axis=-1, training=self.training,
                                                                    momentum=0.7)

            if d != self.num_deep - 1:
                layer_out = tf.nn.relu(layer_out)
                self.activated[d] = layer_out

                if self.params["dropout"]:
                    layer_out = tf.cond(self.training, lambda: tf.nn.dropout(layer_out, self.params["keep_rate"]),
                                        lambda: layer_out)

        return layer_out

    @cache_func
    def optimize(self):
        """
        One step optimization for the specified loss function.
        :return: optimizer.
        """
        loss = self.loss
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.params["learning_rate"],
                                                     beta1=0.8, beta2=0.8)
        train_op = optimizer.minimize(loss)
        return train_op

    @cache_func
    def loss(self):
        """
        Overall loss function that contains the data loss and the regularization losses.
        :return: Overall loss.
        """
        self._loss = self.data_loss
        if self.params["l2_loss"]:
            self._loss += self.l2_loss
        return self._loss

    @cache_func
    def data_loss(self):
        """
        Data loss from the label predictions.
        :return: data loss.
        """
        if self.params["loss_type"] == "l1":
            loss = tf.reduce_mean(tf.abs(tf.subtract(self.target, self.predict)))
        elif self.params["loss_type"] == "l2":
            loss = tf.reduce_mean(tf.pow(tf.subtract(self.target, self.predict), 2))
        else:
            loss = 0.0
        return loss

    @cache_func
    def eval_loss(self):
        """
        Evaluation loss, L1.
        :return: evaluation loss.
        """
        loss = tf.reduce_mean(tf.abs(tf.subtract(self.target, tf.math.round(self.predict))))
        return loss

    @cache_func
    def l2_loss(self):
        """
        L2 regularization loss.
        :return: Regularization loss.
        """
        l2_loss = 0.0
        for key, val in self.weight_dict.items():
            l2_loss += self.params["alpha"] * tf.nn.l2_loss(val)
        return l2_loss

    def evaluate(self):
        """

        :return:
        """
        return tf.reduce_mean(tf.abs(tf.subtract(self.target, tf.math.round(self.predict))))
