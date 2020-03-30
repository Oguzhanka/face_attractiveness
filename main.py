import tensorflow as tf
from data_processor import DataProcessor

# from models.cnn_model import CNNModel
import config

if __name__ == "__main__":
    data_params = config.DataParams().__dict__
    model_params = config.ModelParams().__dict__

    data = DataProcessor("train")
    data.init_random_batches(model_params["batch_size"])

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, reshape=False)

    x = tf.placeholder("float", (None, data_params["input_size"],
                                 data_params["input_size"], data_params["input_dims"]))
    y = tf.placeholder("float", (None, 10))

    model = CNNModel(model_params=model_params, data_params=data_params, data=x, target=y)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        batch_xs, batch_ys = data.next_train_batches()
        if not data.has_next():
            print("All data finished")
            data.init_random_batches()

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        for epoch in range(2000):
            batch_xs, batch_ys = data.next_batch(model_params["batch_size"])
            sess.run([model.optimize, extra_update_ops], feed_dict={model.data: batch_xs,
                                                                    model.target: batch_ys,
                                                                    model.training: True})

            cost = sess.run(model.loss, feed_dict={model.data: batch_xs,
                                                   model.target: batch_ys,
                                                   model.training: False})

            print(cost)
