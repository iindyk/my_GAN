import numpy as np
import tensorflow as tf
from utils import *


class Generator:
    a = 1.0
    layers = []
    test_data = []
    test_labels = []

    def __init__(self, layers_profile, initial_train_dataset, initial_train_labels, test_dataset, test_labels, dim):
        self.test_data = test_data
        self.test_labels = test_labels
        # add layers
        for layer_type in layers_profile:
            self.layers.append(Layer(layer_type))

    def act(self, z):
        ret = np.copy(z)
        # sequentially apply layers
        for layer in self.layers:
            ret = layer.act(ret)
        return ret

    def loss(self, discriminator, z):
        return -tf.reduce_mean(tf.log(discriminator.act(self.act(z))))
