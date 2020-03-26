from helper import lazy_property, INIT_METHODS
import tensorflow as tf


class CNNModel:
    def __init__(self, data, target, model_params, data_params):
        self.data = data
        self.target = target
        self.params = model_params
        self.data_params = data_params

        self.training = tf.placeholder(tf.bool, shape=[])

        _ = self.predict
        _ = self.optimize
        _ = self.loss

    @lazy_property
    def predict(self):
        layer_out = self.data
        for c in range(1, 8):
            conv = tf.nn.conv2d(layer_out, self.weight_dict[f"W_c_{c}"], 1, "SAME")
            biased = tf.nn.bias_add(conv, self.weight_dict[f"b_c_{c}"])
            layer_out = tf.nn.relu(biased)

            if self.params["batch_norm"]:
                layer_out = tf.compat.v1.layers.batch_normalization(layer_out, axis=-1, training=self.training,
                                                                    momentum=0.8)
        flatten = tf.compat.v1.layers.flatten(layer_out, name=None, data_format='channels_last')

        layer_out = flatten
        for d in range(1, 4):
            densed = tf.matmul(layer_out, self.weight_dict[f"W_{d}"],
                               transpose_a=False, transpose_b=False)
            biased = densed + self.weight_dict[f"b_{d}"]
            layer_out = tf.nn.relu(biased)
            if self.params["batch_norm"]:
                tf.conlayer_out = tf.compat.v1.layers.batch_normalization(layer_out, axis=-1, training=self.training,
                                                                          momentum=0.8)

            layer_out = tf.cond(self.training, lambda: tf.nn.dropout(layer_out, self.params["keep_rate"]),
                                lambda: layer_out)

        return layer_out

    @lazy_property
    def optimize(self):
        loss = self.loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.params["learning_rate"])
        train_op = optimizer.minimize(loss)
        return train_op

    @lazy_property
    def loss(self):
        loss = tf.reduce_mean(tf.abs(self.target - self.predict))
        if self.params["l2_loss"]:
            loss += self.l2_loss
        return loss

    @lazy_property
    def l2_loss(self):
        l2_loss = 0.0
        for key, val in self.weight_dict.items():
            l2_loss += self.params["alpha"] * tf.nn.l2_loss(val)
        return l2_loss

    @property
    def weight_dict(self):
        weight_dict = {"W_c_1": tf.Variable(tf.random_normal((7, 7, self.data_params["input_dims"], 8))),
                       "b_c_1": tf.Variable(tf.random_normal([8])),
                       "W_c_2": tf.Variable(tf.random_normal((5, 5, 8, 8))),
                       "b_c_2": tf.Variable(tf.random_normal([8])),
                       "W_c_3": tf.Variable(tf.random_normal((5, 5, 8, 16))),
                       "b_c_3": tf.Variable(tf.random_normal([16])),
                       "W_c_4": tf.Variable(tf.random_normal((5, 5, 16, 32))),
                       "b_c_4": tf.Variable(tf.random_normal([32])),
                       "W_c_5": tf.Variable(tf.random_normal((3, 3, 32, 16))),
                       "b_c_5": tf.Variable(tf.random_normal([16])),
                       "W_c_6": tf.Variable(tf.random_normal((3, 3, 16, 8))),
                       "b_c_6": tf.Variable(tf.random_normal([8])),
                       "W_c_7": tf.Variable(tf.random_normal((3, 3, 8, 1))),
                       "b_c_7": tf.Variable(tf.random_normal([1])),

                       "W_1": tf.Variable(tf.random_normal((self.data_params["input_size"]**2, 1024))),
                       "b_1": tf.Variable(tf.random_normal([1024])),
                       "W_2": tf.Variable(tf.random_normal([1024, 256])),
                       "b_2": tf.Variable(tf.random_normal([256])),
                       "W_3": tf.Variable(tf.random_normal((256, 10))),
                       "b_3": tf.Variable(tf.random_normal([10]))}

        for key, val in weight_dict.items():
            weight_dict[key] = tf.identity(weight_dict[key], name=key)
            weight_dict[key].initializer = INIT_METHODS[self.params["weight_init"]]()
        return weight_dict
