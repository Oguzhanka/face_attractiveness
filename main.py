import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from data_processor import DataProcessor

from models.cnn_model import CNNModel
import config

if __name__ == "__main__":
    data_params = config.DataParams().__dict__
    model_params = config.ModelParams().__dict__

    data = DataProcessor("train")
    data.init_random_batches(model_params["batch_size"])

    with tf.device('/device:GPU:0'):
        x = tf.placeholder("float", (None, data_params["input_size"],
                                     data_params["input_size"], data_params["input_dims"]))
        y = tf.placeholder("float", (None, 1))

        model = CNNModel(model_params=model_params, data_params=data_params, data=x, target=y)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            file_writer = tf.summary.FileWriter('./', sess.graph)
            sess.run(init)
            if not data.has_next():
                print("All data finished")
                data.init_random_batches(model_params["batch_size"])

            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            for epoch in range(2000):
                model.visualize()
                print("\nEPOCH: " + str(epoch))
                data.init_random_batches(batch_size=model_params["batch_size"])
                mean_cost = 0.0
                len_ = 0
                while True:
                    batch_xs, batch_ys = data.next_batches()
                    if batch_xs is None:
                        break
                    sess.run([model.optimize, extra_update_ops], feed_dict={model.data: batch_xs,
                                                                            model.target: batch_ys,
                                                                            model.training: True})

                    cost = sess.run(model.data_loss, feed_dict={model.data: batch_xs,
                                                                model.target: batch_ys,
                                                                model.training: False})

                    print(f"\rTraining Loss: {cost}", end="", flush=True)
                    mean_cost += cost
                    len_ += 1

                print(f"\rTraining Loss: {mean_cost / len_}", end="", flush=True)
