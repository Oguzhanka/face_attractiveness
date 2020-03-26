

IMAGE_SIZE = (28, 28, 1)


class DataParams:
    def __init__(self):
        self.input_size = IMAGE_SIZE[0]
        self.input_dims = IMAGE_SIZE[2]


class ModelParams:
    def __init__(self):
        self.batch_size = 64
        self.weight_init = "xavier"
        self.learning_rate = 1e-3
        self.batch_norm = True

        self.l2_loss = True
        self.alpha = 0.001

        self.dropout = True
        self.keep_rate = 0.5
