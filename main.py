import tensorflow as tf
import numpy as np

import config


if __name__ == "__main__":
    model_params = config.ModelParams().__dict__

    x = tf.placeholder("float", (None, 80, 80, 3))
    y = tf.placeholder("float", (None, 1))

    W = tf.Variable(tf.random_normal((5, 5, 3, 1)))
    b = tf.Variable(tf.random_normal([1]))

    W_2 = tf.Variable(tf.random_normal((1, 6400)))
    b_2 = tf.Variable(tf.random_normal([1]))

    conv = tf.nn.conv2d(x, W, 1, "SAME")
    biased = tf.nn.bias_add(conv, b)

    activation = tf.nn.relu(biased)
    flatten = tf.compat.v1.layers.flatten(activation, name=None, data_format='channels_last')

    densed = tf.matmul(W_2, flatten, transpose_a=False, transpose_b=True)
    bias = densed + b_2

    loss = tf.reduce_mean(tf.abs(y-bias))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(2000):
            sess.run(train_op, feed_dict={x: np.zeros((32, 80, 80, 3)),
                                          y: np.ones((32, 1))})

            cost = sess.run(loss, feed_dict={x: np.zeros((32, 80, 80, 3)),
                                             y: np.ones((32, 1))})

            print(cost)
