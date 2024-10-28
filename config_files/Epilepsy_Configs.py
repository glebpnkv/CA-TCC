class Config(object):
    def __init__(self):
        # Model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = 2
        self.dropout = 0.35
        self.features_len = 24

        # Training configs
        self.num_epoch = 40

        # Optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # Data parameters
        self.drop_last = True
        self.batch_size = 128

        self.Context_Cont = ContextContConfigs()
        self.TC = TC()
        self.augmentation = Augmentations()


class Augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.001
        self.jitter_ratio = 0.001
        self.max_seg = 5


class ContextContConfigs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 10
