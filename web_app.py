import config
import numpy as np
from flask import Flask
from flask import request
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
from models.cnn_model import CNNModel
from data_processor import DataProcessor


tf.disable_v2_behavior()
tf.disable_eager_execution()
app = Flask(__name__)
model = None
tf_saver = None
data_params = None
model_params = None


@app.route('/train', methods=['GET', 'POST'])
def train():
    global model
    global tf_saver
    global model_params, data_params
    data = DataProcessor("train")
    data.init_random_batches(model_params["batch_size"])

    val_data = DataProcessor("val")

    with tf.device('/device:GPU:0'):
        with tf.Session() as sess:
            x = tf.placeholder("float", (None, data_params["input_size"],
                                         data_params["input_size"], data_params["input_dims"]))
            y = tf.placeholder("float", (None, 1))

            model = CNNModel(model_params=model_params,
                             data_params=data_params,
                             data=x,
                             target=y)
            tf_saver.restore(sess, "./tmp/model")

            for epoch in range(model_params["num_epochs"]):
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

                print(f"\rTraining Loss: {mean_cost / len_}")
                val_x, val_y = val_data.get_data()
                val_loss = sess.run(model.data_loss, feed_dict={model.data: val_x,
                                                                model.target: val_y,
                                                                model.training: False})

                print(f"Validation Loss: {val_loss}")


@app.route("/caption", methods=['GET', 'POST'])
def rate():
    test_data = DataProcessor("test")

    pass


@app.route("/reset")
def reset():
    pass


if __name__ == "__main__":
    data_params = config.DataParams().__dict__
    model_params = config.ModelParams().__dict__

    data = DataProcessor("train")
    data.init_random_batches(model_params["batch_size"])

    val_data = DataProcessor("val")
    test_data = DataProcessor("test")
    x = tf.placeholder("float", (None, data_params["input_size"],
                                 data_params["input_size"], data_params["input_dims"]))
    y = tf.placeholder("float", (None, 1))
    model = CNNModel(model_params=model_params, data_params=data_params, data=x, target=y)
    init = tf.global_variables_initializer()
    tf_saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(init)
    file_writer = tf.summary.FileWriter('./outs', sess.graph)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    tf_saver.save(sess, "./tmp/model")

    print("Model is ready...")
    app.run(host='0.0.0.0')
