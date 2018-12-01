import numpy as np


class Layer:
    type_ = None
    params = {}

    def __init__(self, type_, input_shape=None, output_shape=None):
        if type_ not in ('linear', 'arctan', 'ReLu', 'tanh'):
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
