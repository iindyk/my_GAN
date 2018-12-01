import numpy as np
import tensorflow as tf
from utils import *


class Discriminator:
    layers = []

    def __init__(self, layers_profile):
        # add layers
        for layer_type in layers_profile:
            self.layers.append(Layer(layer_type))

    def act(self, d):
        ret = np.copy(d)
        # sequentially apply layers
        for layer in self.layers:
            ret = layer.act(ret)
        return ret

    def loss(self, d_real, d_generated):
        return -tf.reduce_mean(tf.log(self.act(d_real))+tf.log(1. - self.act(d_generated)))