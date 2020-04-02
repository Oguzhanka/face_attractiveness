from helper import lazy_property, INIT_METHODS
import tensorflow as tf


class BaseModel:
    def __init__(self, data, target, model_params, data_params):
        self.data = data
        self.target = target
        self.params = model_params
        self.data_params = data_params

        for key, val in self.weight_dict.items():
            self.weight_dict[key] = tf.identity(self.weight_dict[key], name=key)

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

        _ = self.optimize
        self.epoch_counter = 0

    @lazy_property
    def predict(self):
        for c in range(1, self.num_layers):
            if c == 1:
                input_ = self.data
                self.weight["W_c_1"] = tf.transpose(self.weight_dict["W_c_1"], [3, 0, 1, 2])
            else:
                input_ = pool
            conv = tf.nn.conv2d(input_, self.weight_dict[f"W_c_{c}"], 1, "VALID")
            self.convs[c] = tf.transpose(conv, [3, 0, 1, 2])
            bias = tf.nn.bias_add(conv, self.weight_dict[f"b_c_{c}"])
            self.biass[c] = tf.transpose(bias, [3, 0, 1, 2])
            if self.params["batch_norm"]:
                bias = tf.compat.v1.layers.batch_normalization(bias, axis=-1, training=self.training,
                                                               momentum=0.7)
            activation = tf.nn.relu(bias)
            self.activations[c] = tf.transpose(activation, [3, 0, 1, 2])
            pool = tf.nn.pool(activation, self.weight_dict[f"W_c_{c}"].shape[:-2],
                              "MAX", padding="VALID")
            self.pools[c] = tf.transpose(pool, [3, 0, 1, 2])

        flatten = tf.compat.v1.layers.flatten(pool, name=None, data_format='channels_last')

        layer_out = flatten
        for d in range(1, 4):
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

            if d != 3:
                layer_out = tf.nn.relu(layer_out)
                self.activated[d] = layer_out

                if self.params["dropout"]:
                    layer_out = tf.cond(self.training, lambda: tf.nn.dropout(layer_out, self.params["keep_rate"]),
                                        lambda: layer_out)

        # layer_out = tf.multiply(8.0, tf.sigmoid(layer_out))
        return layer_out

    @lazy_property
    def optimize(self):
        loss = self.loss
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.params["learning_rate"])
        train_op = optimizer.minimize(loss)
        return train_op

    @lazy_property
    def loss(self):
        loss = self.data_loss
        if self.params["l2_loss"]:
            loss += self.l2_loss
        return loss

    @lazy_property
    def data_loss(self):
        if self.params["loss_type"] == "l1":
            loss = tf.reduce_mean(tf.abs(tf.subtract(self.target, self.predict)))
        elif self.params["loss_type"] == "l2":
            loss = tf.reduce_mean(tf.pow(tf.subtract(self.target, self.predict), 2))
        else:
            loss = 0.0
        return loss

    @lazy_property
    def l2_loss(self):
        l2_loss = 0.0
        for key, val in self.weight_dict.items():
            l2_loss += self.params["alpha"] * tf.nn.l2_loss(val)
        return l2_loss

    def evaluate(self):
        return tf.reduce_mean(tf.abs(tf.subtract(self.target, self.predict)))
