import numpy as np
import tensorflow as tf
import sklearn.svm as svm
from utils import *


class Generator:
    a = 1.0             # weight of misclassification probability in loss function
    layers = []         # list of layers
    test_data = []      # data on which misclassification probability is being maximized
    test_labels = []    # labels --

    def __init__(self, layers_profile, initial_train_data, initial_train_labels, test_data, test_labels):
        self.train_data = initial_train_data
        self.train_labels = initial_train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.current_w = np.zeros(len(test_data[0]))

        # add layers
        for l_prof in layers_profile:
            self.layers.append(Layer(l_prof))

    def add_train_data(self, new_train_data, new_train_labels):
        self.train_data = np.append(self.train_data, new_train_data, axis=0)
        self.train_labels = np.append(self.train_labels, new_train_labels)

    def act(self, z):
        return Layer.layers_act(self.layers, z)

    def loss(self, discriminator, z):
        d_generated = self.act(z)
        '''d_union = np.append(self.train_data, d_generated, axis=0)
        l_generated = np.array([-1.]*len(d_generated))                  # todo: how to generate labels?
        l_union = np.append(self.train_labels, l_generated)

        # get parameters of SVM
        svc = svm.SVC(kernel='linear', C=self.a).fit(d_union, l_union)
        w = svc.coef_[0]        # normal vector
        b = svc.intercept_[0]   # intercept

        # calculate approximation to the probability of misclassification
        prob_approx = 0
        n_test = len(self.test_labels)
        for i in range(n_test):
            prob_approx += max(self.test_labels[i]*(w.dot(self.test_data[i])+b), -1)
        prob_approx /= n_test'''

        return np.mean(np.log(1.-discriminator.act(d_generated)))  # +self.a*prob_approx

    def loss_grad(self, discriminator, z):
        g_z = self.act(z)
        n_z = len(z)
        n_in_g_z = len(g_z[0])
        n_layers = len(self.layers)

        # gradient of G(z)
        g_grad = Layer.layers_grad_multidim(self.layers, z)
        mult = np.zeros(shape=(n_z, n_in_g_z))
        d_g_z = -1. / (1. - discriminator.act(g_z))
        dis_grad_x = Layer.layers_grad(discriminator.layers, g_z)[0]['x']
        for i in range(n_z):
            mult[i, :] = d_g_z[i] * dis_grad_x[i, :]

        grad = {}
        for layer_id in range(n_layers):
            if self.layers[layer_id].type_ != 'linear':
                d_w = 0
                d_b = 0
                d_x = 0
                for i in range(n_z):
                    d_w += mult[i, :] @ g_grad[layer_id]['w'][i, :, :]
                    d_b += mult[i, :] @ g_grad[layer_id]['b'][i, :]
                    d_x += mult[i, :] @ g_grad[layer_id]['x'][i, :]

                grad[layer_id] = {'w': d_w/n_z, 'b': d_b/n_z, 'x': d_x/n_z}

        return grad
