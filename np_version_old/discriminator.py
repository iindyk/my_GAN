import numpy as np
import tensorflow as tf
from utils import *
import memory_profiler


class Discriminator:
    layers = []

    def __init__(self, layers_profile):
        # add layers
        for l_prof in layers_profile:
            self.layers.append(Layer(l_prof))

    def act(self, d):
        return Layer.layers_act(self.layers, d)

    def loss(self, d_real, d_generated):
        return np.mean(np.log(self.act(d_real))+np.log(1. - self.act(d_generated)))

    def _loss_grad(self, layer_id, d_real, d_generated):     # todo: return gradient for each layer
        if self.layers[layer_id].type_ != 'linear':
            raise Exception('call of a gradient for a non-linear layer')

        # gradient of D(x):
        n_r = len(d_real)
        n_g = len(d_generated)
        d_grad_real = Layer.layers_grad_(self.layers, layer_id, d_real)
        d_grad_gen = Layer.layers_grad_(self.layers, layer_id, d_generated)

        d_w_r = 0
        d_b_r = 0
        d_x_r = 0
        d_inv = 1./self.act(d_real)
        for i in range(n_r):
            d_w_r += d_inv[i]*d_grad_real['w'][i, :, :]
            d_b_r += d_inv[i]*d_grad_real['b'][i, :]
            d_x_r += d_inv[i]*d_grad_real['x'][i, :]

        d_w_g = 0
        d_b_g = 0
        d_x_g = 0
        g_inv = 1./(1.-self.act(d_generated))
        for j in range(n_g):
            d_w_g += g_inv[j]*d_grad_gen['w'][j, :, :]
            d_b_g += g_inv[j]*d_grad_gen['b'][j, :]
            d_x_g += g_inv[j]*d_grad_gen['x'][j, :]
        return {'w': d_w_r/n_r - d_w_g/n_g,
                'b': d_b_r/n_r - d_b_g/n_g,
                'x': d_x_r/n_r - d_x_g/n_g}

    #@memory_profiler.profile
    def loss_grad(self, d_real, d_generated):
        # gradient of D(x):
        n_r = len(d_real)
        n_g = len(d_generated)
        d_grad_real = Layer.layers_grad(self.layers, d_real)
        d_grad_gen = Layer.layers_grad(self.layers, d_generated)
        d_inv = 1. / self.act(d_real)
        g_inv = 1. / (1. - self.act(d_generated))

        n_layers = len(self.layers)
        loss_grad = {}
        for layer_id in range(n_layers):
            if self.layers[layer_id].type_ == 'linear':
                d_w_r = 0
                d_b_r = 0
                d_x_r = 0
                for i in range(n_r):
                    d_w_r += d_inv[i] * d_grad_real[layer_id]['w'][i, :, :]
                    d_b_r += d_inv[i] * d_grad_real[layer_id]['b'][i, :]
                    d_x_r += d_inv[i] * d_grad_real[layer_id]['x'][i, :]

                d_w_g = 0
                d_b_g = 0
                d_x_g = 0

                for j in range(n_g):
                    d_w_g += g_inv[j] * d_grad_gen[layer_id]['w'][j, :, :]
                    d_b_g += g_inv[j] * d_grad_gen[layer_id]['b'][j, :]
                    d_x_g += g_inv[j] * d_grad_gen[layer_id]['x'][j, :]

                loss_grad[layer_id] = {'w': d_w_r / n_r - d_w_g / n_g,
                                       'b': d_b_r / n_r - d_b_g / n_g,
                                       'x': d_x_r / n_r - d_x_g / n_g}

        return loss_grad
