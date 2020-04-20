"""
Configuration file to change the model & experiment parameters.
"""

IMAGE_SIZE = (80, 80, 3)                    # Spatial dimensions of the image.


class DataParams:
    """
    Configuration Parameters related to data preprocessing.
    """
    def __init__(self):
        self.input_size = IMAGE_SIZE[0]
        self.input_dims = IMAGE_SIZE[2]


class ModelParams:
    """
    Model parameters.
    """
    def __init__(self):
        """
        :attr model_type: Name of the model type. Choose large or compact.
        :attr batch_size: Batch size for the images.
        :attr num_epochs: Number of epochs.
        :attr weight_init: Weight initialization method. Choose xavier or gaussian.
        :attr learning_rate: Learning rate.
        :attr batch_norm: Flag toggling the batch normalization layers.
        :attr loss_type: Loss type for the training. Choos l1 or l2.
        :attr l2_loss: Flag toggling the L2 weight regularization.
        :attr alpha: L2 regularization parameter.
        :attr dropout: Flag toggling the dropout layers.
        :attr keep_rate: Dropout keep probability.
        """
        self.model_type = "compact"
        self.batch_size = 32
        self.num_epochs = 25
        self.weight_init = "xavier"
        self.learning_rate = 1e-3
        self.batch_norm = False

        self.loss_type = "l2"

        self.l2_loss = False
        self.alpha = 1e-3

        self.dropout = False
        self.keep_rate = 0.5
