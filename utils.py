import numpy as np


class Layer:
    type_ = None
    params = {}

    def __init__(self, type_, input_shape=None, output_shape=None):
        if type_ not in ('linear', 'arctan', 'ReLu', 'tanh', 'sigmoid'):
            raise Exception('unknown layer type')

        if type_ == 'linear' and (input_shape is None or output_shape is None):
            raise Exception('no input or output shape given for linear layer')

        if type_ != 'linear' and (input_shape is not None or output_shape is not None):
            raise Warning('input or output shape given for non-linear layer: ignored')

        # set parameters
        if type_ == 'linear':
            self.params = {'w': np.random.normal(size=(input_shape, output_shape)),
                           'b': np.random.normal(size=output_shape)}
        self.type_ = type_

    def act(self, x):
        if self.type_ == 'linear':
            return x.dot(self.params['w'])+self.params['b']
        elif self.type_ == 'arctan':
            return np.arctan(x)
        elif self.type_ == 'ReLu':
            return np.maximum(x, 0)
        elif self.type_ == 'tanh':
            return np.tanh(x)
        elif self.type_ == 'sigmoid':
            return 1./(1.+np.exp(-x))

    def gradient(self, x):
        if self.type_ == 'linear':
            return self.params['w']
        elif self.type_ == 'arctan':
            return 1./(1.+x**2)
        elif self.type_ == 'ReLu':
            return 1.*(x > 0)
        elif self.type_ == 'tanh':
            return 1.-np.tanh(x)**2
        elif self.type_ == 'sigmoid':
            return 1./(1.+np.exp(-x))-1./(1.+np.exp(-x))**2

    @staticmethod
    def layers_act(layers, x):
        # sequentially apply layers
        ret = np.copy(x)
        for layer in layers:
            ret = layer.act(ret)
        return ret
