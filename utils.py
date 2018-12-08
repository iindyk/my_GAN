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
            return x @ self.params['w']+self.params['b']
        elif self.type_ == 'arctan':
            return np.arctan(x)
        elif self.type_ == 'ReLu':
            return np.maximum(x, 0)
        elif self.type_ == 'tanh':
            return np.tanh(x)
        elif self.type_ == 'sigmoid':
            return 1./(1.+np.exp(-x))

    def gradient(self, x, var='x'):   # (wx+b)'_w = x.T, (wx+b)'_b = 1, (wx+b)'_x = w
        assert np.ndim(x) == 1
        if self.type_ == 'linear':
            if var == 'w':  # todo: check
                n_in, n_out = np.shape(self.params['w'])
                ret = np.zeros(shape=(n_out, n_in, n_out))
                for i in range(n_out):
                    ret[i, :, i] = x
                return np.reshape(ret, newshape=(n_out, -1))
            elif var == 'b':
                return np.ones_like(self.params['b'])
            elif var == 'x':
                return self.params['w'].T
            else:
                raise Exception('var for a gradient is not recognized')
        elif self.type_ == 'arctan':
            if var != 'x': raise Exception('var for a non-linear layer is not recognized')
            return np.diag(1./(1.+x**2))
        elif self.type_ == 'ReLu':
            if var != 'x': raise Exception('var for a non-linear layer is not recognized')
            return np.diag(1.*(x > 0))
        elif self.type_ == 'tanh':
            if var != 'x': raise Exception('var for a non-linear layer is not recognized')
            return np.diag(1.-np.tanh(x)**2)
        elif self.type_ == 'sigmoid':
            if var != 'x': raise Exception('var for a non-linear layer is not recognized')
            return np.diag(1./(1.+np.exp(-x))-1./(1.+np.exp(-x))**2)

    @staticmethod
    def layers_act(layers, x):
        # sequentially apply layers
        ret = np.copy(x)
        for layer in layers:
            ret = layer.act(ret)
        return ret

    @staticmethod
    def layers_grad(layers, layer_id, x):
        n, m = np.shape(x)
        lin = (layers[layer_id].type_ == 'linear')
        # find grad for each input. Output must be a number!!!
        assert np.ndim(Layer.layers_act(layers, x[0])) == 1
        # gradient with respect to x: (n, n, m)
        # gradient with respect to w: (n, w_in, w_out)
        # gradient with respect to b: (n, w_out, 1)

        n_layers = len(layers)
        grad_x = np.zeros(shape=(n, n, m))
        if lin:
            w_in, w_out = np.shape(layers[layer_id].params['w'])
            grad_w = np.zeros(shape=(n, w_in, w_out))
            grad_b = np.zeros(shape=(n, w_out, 1))
        for i in range(n):
            trunc_act = Layer.layers_act(layers[:layer_id], x[i, :])
            g_x = layers[layer_id].gradient(trunc_act, var='x')
            mult = np.identity(np.shape(g_x)[0])
            for curr_layer_id in range(layer_id + 1, n_layers):
                trunc_act = layers[curr_layer_id - 1].act(trunc_act)
                mult = layers[curr_layer_id].gradient(trunc_act, var='x') @ mult
            grad_x[i, :, :] = mult @ g_x
            if lin:
                grad_w[i, :, :] = np.reshape(mult @ layers[layer_id].gradient(trunc_act, var='w'), (w_in, w_out))
                grad_b[i, :, :] = mult @ layers[layer_id].gradient(trunc_act, var='b')

        if lin:
            return {'w': grad_w, 'b': grad_b, 'x': grad_x}
        else:
            return {'x': grad_x}


# deep copy of a gradient dictionary
def grad_deep_copy(gradient):
    gradient_copy = {}
    for key, dict_val in gradient.items():
        gradient_copy[key] = {}
        for key_inner, val in dict_val.items():
            gradient_copy[key][key_inner] = np.copy(val)

    return gradient_copy
