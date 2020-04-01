

IMAGE_SIZE = (80, 80, 3)


class DataParams:
    def __init__(self):
        self.input_size = IMAGE_SIZE[0]
        self.input_dims = IMAGE_SIZE[2]


class ModelParams:
    def __init__(self):
        self.batch_size = 32
        self.num_epochs = 5
        self.weight_init = "gaussian"
        self.learning_rate = 1e-4
        self.batch_norm = False

        self.loss_type = "l1"

        self.l2_loss = False
        self.alpha = 1e-8

        self.dropout = False
        self.keep_rate = 0.5
