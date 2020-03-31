from helper import lazy_property, INIT_METHODS
import matplotlib.pyplot as plt
import tensorflow as tf


class CNNModel:
    def __init__(self, data, target, model_params, data_params):
        self.data = data
        self.target = target
        self.params = model_params
        self.data_params = data_params

        self.training = tf.compat.v1.placeholder(tf.bool, shape=[])
        self.layer_out = None
        self.first_time = True

        _ = self.predict
        _ = self.optimize
        _ = self.loss
        _ = self.data_loss

    @lazy_property
    def predict(self):
        layer_out = self.data
        for c in range(1, 9):
            conv = tf.nn.conv2d(layer_out, self.weight_dict[f"W_c_{c}"], 1, "VALID")
            biased = tf.nn.bias_add(conv, self.weight_dict[f"b_c_{c}"])
            if self.params["batch_norm"]:
                biased = tf.compat.v1.layers.batch_normalization(biased, axis=-1, training=self.training,
                                                                 momentum=0.7)
            layer_out = tf.nn.relu(biased)

            layer_out = tf.nn.pool(layer_out, self.weight_dict[f"W_c_{c}"].shape[:-2], "AVG", padding="VALID")

        flatten = tf.compat.v1.layers.flatten(layer_out, name=None, data_format='channels_last')

        layer_out = flatten
        for d in range(1, 4):
            densed = tf.matmul(layer_out, self.weight_dict[f"W_{d}"],
                               transpose_a=False, transpose_b=False)
            biased = densed + self.weight_dict[f"b_{d}"]
            if self.params["batch_norm"]:
                biased = tf.compat.v1.layers.batch_normalization(biased, axis=-1, training=self.training,
                                                                 momentum=0.7)

            layer_out = tf.nn.relu(biased)

            layer_out = tf.cond(self.training, lambda: tf.nn.dropout(layer_out, self.params["keep_rate"]),
                                lambda: layer_out)

        layer_out = tf.multiply(8.0, tf.sigmoid(layer_out))
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
        loss = tf.reduce_mean(tf.abs(tf.subtract(self.target, self.predict)))
        return loss

    @lazy_property
    def l2_loss(self):
        l2_loss = 0.0
        for key, val in self.weight_dict.items():
            l2_loss += self.params["alpha"] * tf.nn.l2_loss(val)
        return l2_loss

    @lazy_property
    def weight_dict(self):
        weight_dict = {"W_c_1": tf.Variable(tf.random.normal((7, 7, self.data_params["input_dims"], 16))),
                       "b_c_1": tf.Variable(tf.random.normal([16])),
                       "W_c_2": tf.Variable(tf.random.normal((7, 7, 16, 16))),
                       "b_c_2": tf.Variable(tf.random.normal([16])),
                       "W_c_3": tf.Variable(tf.random.normal((5, 5, 16, 32))),
                       "b_c_3": tf.Variable(tf.random.normal([32])),
                       "W_c_4": tf.Variable(tf.random.normal((5, 5, 32, 32))),
                       "b_c_4": tf.Variable(tf.random.normal([32])),
                       "W_c_5": tf.Variable(tf.random.normal((5, 5, 32, 64))),
                       "b_c_5": tf.Variable(tf.random.normal([64])),
                       "W_c_6": tf.Variable(tf.random.normal((3, 3, 64, 128))),
                       "b_c_6": tf.Variable(tf.random.normal([128])),
                       "W_c_7": tf.Variable(tf.random.normal((3, 3, 128, 256))),
                       "b_c_7": tf.Variable(tf.random.normal([256])),
                       "W_c_8": tf.Variable(tf.random.normal((3, 3, 256, 512))),
                       "b_c_8": tf.Variable(tf.random.normal([512])),

                       "W_1": tf.Variable(tf.random.normal((20**2*512, 1024))),
                       "b_1": tf.Variable(tf.random.normal([1024])),
                       "W_2": tf.Variable(tf.random.normal([1024, 256])),
                       "b_2": tf.Variable(tf.random.normal([256])),
                       "W_3": tf.Variable(tf.random.normal((256, 1))),
                       "b_3": tf.Variable(tf.random.normal([1]))}

        if self.first_time:
            for key, val in weight_dict.items():
                weight_dict[key] = tf.identity(weight_dict[key], name=key)
                weight_dict[key].initializer = INIT_METHODS[self.params["weight_init"]]()
            self.first_time = False
        return weight_dict

    def visualize(self):

        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(((weights[0][:, :, :, i].eval() + 3) / 6 * 255).astype(int))

        plt.show()
