import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from data_processor import DataProcessor
from models.cnn_model import CNNModel
import config

tf.disable_v2_behavior()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == "__main__":
    data_params = config.DataParams().__dict__
    model_params = config.ModelParams().__dict__

    # Data generation for the training, validation and the test.
    data = DataProcessor("train")
    data.init_random_batches(model_params["batch_size"])
    val_data = DataProcessor("val")
    test_data = DataProcessor("test")

    # Placeholders for the model input and output.
    x = tf.placeholder("float", (None, data_params["input_size"],
                                 data_params["input_size"], data_params["input_dims"]))
    y = tf.placeholder("float", (None, 1))

    # Model class and weight initialization. Also, the step initialization for the epochs.
    model = CNNModel(model_params=model_params, data_params=data_params, data=x, target=y)
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    init = tf.global_variables_initializer()

    # Initiation of the main Session.
    with tf.Session() as sess:
        # Summary writer folder.
        file_writer = tf.summary.FileWriter('./outs', sess.graph)
        sess.run(init)
        if not data.has_next():
            print("All data finished")
            data.init_random_batches(model_params["batch_size"])

        # Enable updating through batch normalization and dropout layers.
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Namefield descriptions of the parameters that are logged through each iteration.
        with tf.name_scope("Layer-1") as scope:
            tf.summary.image(f"Filters", model.weight[f"W_c_{1}"])  # First conv. layer

        for c in range(1, model.num_layers):
            with tf.name_scope("ConvLayer-" + str(c)) as scope:

                tf.summary.histogram(f"Convolved Image", model.convs[c])        # Image after convolution.
                tf.summary.histogram(f"Filter", model.weight_dict[f"W_c_{1}"])  # Filter histogram.

                tf.summary.histogram(f"Biased Image", model.biass[c])           # Biased image histogram.
                tf.summary.histogram(f"Bias", model.weight_dict[f"b_c_{c}"])    # Bias histogram.

                tf.summary.histogram(f"Activation Image", model.activations[c]) # Image hist. after the activation.
                tf.summary.histogram(f"Pooled Image", model.pools[c])           # Image hist. after the pooling.

        for d in range(1, model.num_deep):
            with tf.name_scope("DenseLayer-" + str(d)) as scope:
                tf.summary.histogram("Dense Output", model.densed[d])           # Output hist. after the weights.
                tf.summary.histogram("Biased Output", model.biased[d])          # Output hist after the bias.
                try:
                    tf.summary.histogram("Activated Output", model.activated[d])    # Output after batch_norm layer.
                except KeyError:
                    pass

        merged_summary_op = tf.summary.merge_all()      # Merges all summaries from different logs.

        for epoch in range(model_params["num_epochs"]):
            global_step += 1
            print("\nEPOCH: " + str(epoch))
            data.init_random_batches(batch_size=model_params["batch_size"])     # Shuffle the training data.
            mean_cost = 0.0
            len_ = 0
            while True:     # For all mini-batches...
                batch_xs, batch_ys = data.next_batches()    # Get the next mini-batch.
                if batch_xs is None:
                    break
                sess.run([model.optimize, extra_update_ops], feed_dict={model.data: batch_xs,
                                                                        model.target: batch_ys,
                                                                        model.training: True})
                # Optimize the model for a single step.

                cost = sess.run(model.eval_loss, feed_dict={model.data: batch_xs,
                                                            model.target: batch_ys,
                                                            model.training: False})
                # Compute the loss function.
                # And print for every batch.
                print(f"\rTraining Loss: {cost}", end="", flush=True)
                mean_cost += cost
                len_ += 1

            # Print the average loss for all batches in the training set.
            print(f"\rTraining Loss: {mean_cost / len_}")
            val_x, val_y = val_data.get_data()
            val_loss, summ = sess.run([model.eval_loss, merged_summary_op], feed_dict={model.data: val_x,
                                                                                       model.target: val_y,
                                                                                       model.training: False})

            file_writer.add_summary(summ, global_step.eval(session=sess))
            file_writer.add_summary(summary=tf.Summary(value=[tf.Summary.Value(tag="Train-Loss",
                                                                               simple_value=mean_cost / len_),
                                                              ]), global_step=global_step.eval(session=sess))
            file_writer.add_summary(summary=tf.Summary(value=[tf.Summary.Value(tag="Reg-Loss",
                                                                               simple_value=model.l2_loss.eval()),
                                                              ]), global_step=global_step.eval(session=sess))
            file_writer.add_summary(summary=tf.Summary(value=[tf.Summary.Value(tag="Val-Loss", simple_value=val_loss),
                                                       ]), global_step=global_step.eval(session=sess))

            print(f"Validation Loss: {val_loss}")
            file_writer.flush()

        test_x, test_y = test_data.get_data()
        test_loss = sess.run([model.eval_loss], feed_dict={model.data: test_x,
                                                           model.target: test_y,
                                                           model.training: False})

        print("Test Loss: " + str(test_loss))

        random_indices = np.random.choice(list(range(test_x.shape[0])), 9)
        random_images = [test_x[idx][None, :] for idx in random_indices]
        random_preds = [sess.run(model.predict, feed_dict={model.data: image, model.training: False})
                        for image in random_images]
        random_labels = [test_y[idx] for idx in random_indices]

        for i, image, prediction, label in zip(range(9), random_images, random_preds, random_labels):
            plt.subplot(3, 3, i+1)
            plt.tight_layout()
            plt.imshow(np.mean(((image[0] - image.min()) / (image.max() - image.min()) * 255), axis=-1).astype(int))
            plt.title("True Score: {}".format(label))
            plt.xlabel("Model prediction: \n{}".format(prediction[0]))

        plt.savefig("./samples.png", figsize=(70, 70))
