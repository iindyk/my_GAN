import numpy as np


class Layer:
    type_ = None
    params = {}

    def __init__(self, profile):
        type_ = profile['type']
        if type_ not in ('linear', 'arctan', 'ReLu', 'tanh', 'sigmoid'):
            raise Exception('unknown layer type')

        if type_ == 'linear' and ('in' not in profile or 'out' not in profile):
            raise Exception('no input or output shape given for linear layer')

        if type_ != 'linear' and ('in' in profile or 'out' in profile):
            raise Warning('input or output shape given for non-linear layer: ignored')

        # set parameters
        if type_ == 'linear':
            input_shape = profile['in']
            output_shape = profile['out']
            self.params = {'w': np.random.normal(scale=1./np.sqrt(input_shape/2.), size=(input_shape, output_shape)),
                           'b': np.random.normal(scale=1./np.sqrt(input_shape/2.), size=output_shape)}
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

    def gradient(self, x, var='x'):   # (wx+b)'_w = x, (wx+b)'_b = 1, (wx+b)'_x = w
        if self.type_ == 'linear':
            if var == 'w':  # todo: check
                return x
            elif var == 'b':
                return np.ones_like(self.params['b'])
            elif var == 'x':
                return self.params['w']
            else:
                raise Exception('var for a gradient is not recognized')
        elif self.type_ == 'arctan':
            if var != 'x': raise Exception('var for a non-linear layer is not recognized')
            return 1./(1.+x**2)
        elif self.type_ == 'ReLu':
            if var != 'x': raise Exception('var for a non-linear layer is not recognized')
            return 1.*(x > 0)
        elif self.type_ == 'tanh':
            if var != 'x': raise Exception('var for a non-linear layer is not recognized')
            return 1.-np.tanh(x)**2
        elif self.type_ == 'sigmoid':
            if var != 'x': raise Exception('var for a non-linear layer is not recognized')
            return 1./(1.+np.exp(-x))-1./(1.+np.exp(-x))**2

    @staticmethod
    def layers_act(layers, x):
        # sequentially apply layers
        ret = np.copy(x)
        for layer in layers:
            ret = layer.act(ret)
        return ret

    @staticmethod
    def layers_grad(layers, layer_id, x):
        trunc_act = Layer.layers_act(layers[:layer_id], x)
        n_layers = len(layers)
        mult = 1
        for curr_layer_id in range(layer_id + 1, n_layers):
            trunc_act = layers[curr_layer_id - 1].act(trunc_act)
            mult *= layers[curr_layer_id].gradient(trunc_act, var='x')

        grad_x = layers[layer_id].gradient(trunc_act, var='x')
        if layers[layer_id].type_ == 'linear':
            grad_w = layers[layer_id].gradient(trunc_act, var='w')
            grad_b = layers[layer_id].gradient(trunc_act, var='b')
            return {'w': mult * grad_w, 'b': mult * grad_b, 'x': mult * grad_x}
        else:
            return {'x': mult * grad_x}


# deep copy of a gradient dictionary
def grad_deep_copy(gradient):
    gradient_copy = {}
    for key, dict_val in gradient.items():
        gradient_copy[key] = {}
        for key_inner, val in dict_val.items():
            gradient_copy[key][key_inner] = np.copy(val)

    return gradient_copy