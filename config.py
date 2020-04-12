

IMAGE_SIZE = (80, 80, 3)


class DataParams:
    def __init__(self):
        self.input_size = IMAGE_SIZE[0]
        self.input_dims = IMAGE_SIZE[2]


class ModelParams:
    def __init__(self):
        self.model_type = "compact"
        self.batch_size = 32
        self.num_epochs = 45
        self.weight_init = "xavier"
        self.learning_rate = 1e-3
        self.batch_norm = False

        self.loss_type = "l2"

        self.l2_loss = False
        self.alpha = 1e-3

        self.dropout = False
        self.keep_rate = 0.5


class ArchitectureTuning:
    def __init__(self):
        self.model_type = ["large"]
        self.batch_size = [64]
        self.num_epochs = [50]
        self.weight_init = ["gaussian", "xavier"]
        self.learning_rate = [1e-2, 1e-3]
        self.batch_norm = [False]

        self.loss_type = ["l1"]
        self.l2_loss = [True, False]
        self.alpha = [1e-8]

        self.droput = [False]
        self.keep_rate = [0.5]


class HyperParamTuning:
    def __init__(self):
        self.model_type = ["compact"]
        self.batch_size = [8, 64, 128]
        self.num_epochs = [10, 50, 100]
        self.weight_init = ["xavier"]
        self.learning_rate = [1e-2, 1e-3, 1e-5]
        self.batch_norm = [False]

        self.loss_type = ["l1"]
        self.l2_loss = [False]
        self.alpha = [1e-8]

        self.droput = [False]
        self.keep_rate = [0.5]


class LossTuning:
    def __init__(self):
        self.model_type = ["compact"]
        self.batch_size = [64]
        self.num_epochs = [50]
        self.weight_init = ["gaussian"]
        self.learning_rate = [1e-3, 1e-4]
        self.batch_norm = [False]

        self.loss_type = ["l1", "l2"]
        self.l2_loss = [False]
        self.alpha = [1e-8]

        self.dropout = [False]
        self.keep_rate = [0.5]


class InitTuning:
    def __init__(self):
        self.model_type = ["compact"]
        self.batch_size = [64]
        self.num_epochs = [50]
        self.weight_init = ["gaussian", "xavier"]
        self.learning_rate = [4e-4]
        self.batch_norm = [False]

        self.loss_type = ["l1"]
        self.l2_loss = [False]
        self.alpha = [1e-8]

        self.dropout = [False]
        self.keep_rate = [0.5]


class BatchNormTuning:
    def __init__(self):
        self.model_type = ["compact"]
        self.batch_size = [64]
        self.num_epochs = [50]
        self.weight_init = ["xavier"]
        self.learning_rate = [4e-4]
        self.batch_norm = [True, False]

        self.loss_type = ["l1"]
        self.l2_loss = [False]
        self.alpha = [1e-8]

        self.dropout = [False]
        self.keep_rate = [0.5]


class RegTuning:
    def __init__(self):
        self.model_type = ["compact"]
        self.batch_size = [64]
        self.num_epochs = [50]
        self.weight_init = ["xavier"]
        self.learning_rate = [4e-4]
        self.batch_norm = [True]

        self.loss_type = ["l1"]
        self.l2_loss = [False, True]
        self.alpha = [1e-8]

        self.dropout = [False]
        self.keep_rate = [0.5]


class DropoutTuning:
    def __init__(self):
        self.model_type = ["compact"]
        self.batch_size = [64]
        self.num_epochs = [50]
        self.weight_init = ["xavier"]
        self.learning_rate = [4e-4]
        self.batch_norm = [True]

        self.loss_type = ["l1"]
        self.l2_loss = [True]
        self.alpha = [1e-8]

        self.dropout = [True]
        self.keep_rate = [0.5]
