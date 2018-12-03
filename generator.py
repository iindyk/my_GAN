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

        if discriminator.act(d_generated)[0] == 1.:
            return 1000
        return np.mean(np.log(1-discriminator.act(d_generated)))  # +self.a*prob_approx

    def loss_grad(self, layer_id, discriminator, z):    # todo
        if self.layers[layer_id].type_ != 'linear':
            raise Exception('call of a gradient for a non-linear layer')
        # gradient of G(z)
        g_grad = Layer.layers_grad(self.layers, layer_id, z)
        g_z = self.act(z)
        mult = (-1./(1-discriminator.act(g_z))) * \
            Layer.layers_grad(discriminator.layers, len(discriminator.layers)-1, g_z)['x']
        # todo: 1/n ?
        return {'w': np.sum(mult*g_grad['w']), 'b': np.sum(mult*g_grad['b']), 'x': np.sum(mult*g_grad['x'])}
