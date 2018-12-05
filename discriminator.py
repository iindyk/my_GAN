import numpy as np
import tensorflow as tf
from utils import *


class Discriminator:
    layers = []

    def __init__(self, layers_profile):
        # add layers
        for l_prof in layers_profile:
            self.layers.append(Layer(l_prof))

    def act(self, d):
        ret = np.copy(d)
        # sequentially apply layers
        for layer in self.layers:
            ret = layer.act(ret)
        return ret

    def loss(self, d_real, d_generated):
        return np.mean(np.log(self.act(d_real))+np.log(1. - self.act(d_generated)))

    def loss_grad(self, layer_id, d_real, d_generated):     # todo
        if self.layers[layer_id].type_ != 'linear':
            raise Exception('call of a gradient for a non-linear layer')

        # gradient of D(x):
        n_r = len(d_real)
        n_g = len(d_generated)
        d_grad_real = Layer.layers_grad(self.layers, layer_id, d_real)
        d_grad_gen = Layer.layers_grad(self.layers, layer_id, d_generated)
        return {'w': (1./self.act(d_real))*d_grad_real['w']/n_r-(1./(1.-self.act(d_generated)))*d_grad_gen['w']/n_g,
                'b': (1./self.act(d_real))*d_grad_real['b']/n_r-(1./(1.-self.act(d_generated)))*d_grad_gen['b']/n_g,
                'x': (1./self.act(d_real))*d_grad_real['x']/n_r-(1./(1.-self.act(d_generated)))*d_grad_gen['x']/n_g}
