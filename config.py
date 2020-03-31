

IMAGE_SIZE = (80, 80, 3)


class DataParams:
    def __init__(self):
        self.input_size = IMAGE_SIZE[0]
        self.input_dims = IMAGE_SIZE[2]


class ModelParams:
    def __init__(self):
        self.batch_size = 128
        self.weight_init = "xavier"
        self.learning_rate = 1e-3
        self.batch_norm = True

        self.l2_loss = True
        self.alpha = 1e-8

        self.dropout = False
        self.keep_rate = 0.5
