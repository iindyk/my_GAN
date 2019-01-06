import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm, variance_scaling_initializer
import memory_profiler
import datetime as dt


class Layer:
    type_ = None
    params = {}
    n_in = None
    n_out = None

    def __init__(self, profile):
        type_ = profile['type']
        if type_ not in ('linear', 'arctan', 'ReLu', 'tanh', 'sigmoid'):
            raise Exception('unknown layer type')

        if 'in' not in profile or 'out' not in profile:
            raise Exception('no input or output shape given')

        # set parameters
        self.n_in = profile['in']
        self.n_out = profile['out']
        self.type_ = type_
        if type_ == 'linear':
            self.params = {'w': np.random.normal(scale=1. / np.sqrt(self.n_in / 2.), size=(self.n_in, self.n_out)),
                           'b': np.random.normal(scale=1. / np.sqrt(self.n_in / 2.), size=self.n_out)}

    def act(self, x):
        if self.type_ == 'linear':
            return x @ self.params['w'] + self.params['b']
        elif self.type_ == 'arctan':
            return np.arctan(x)
        elif self.type_ == 'ReLu':
            return np.maximum(x, 0)
        elif self.type_ == 'tanh':
            return np.tanh(x)
        elif self.type_ == 'sigmoid':
            return np.exp(x) / (np.exp(x) + 1.)

    def gradient(self, x, var='x'):
        assert np.ndim(x) == 1
        if self.type_ == 'linear':
            if var == 'w':  # todo: check
                ret = np.zeros(shape=(self.n_out, self.n_in, self.n_out))
                for i in range(self.n_out):
                    ret[i, :, i] = x
                return np.reshape(ret, newshape=(self.n_out, -1))
            elif var == 'b':
                return np.diag(np.ones_like(self.params['b']))
            elif var == 'x':
                return self.params['w'].T
            else:
                raise Exception('var for a gradient is not recognized')
        elif self.type_ == 'arctan':
            if var != 'x': raise Exception('var for a non-linear layer is not recognized')
            return np.diag(1. / (1. + x ** 2))
        elif self.type_ == 'ReLu':
            if var != 'x': raise Exception('var for a non-linear layer is not recognized')
            return np.diag(1. * (x > 0))
        elif self.type_ == 'tanh':
            if var != 'x': raise Exception('var for a non-linear layer is not recognized')
            return np.diag(1. - np.tanh(x) ** 2)
        elif self.type_ == 'sigmoid':
            if var != 'x': raise Exception('var for a non-linear layer is not recognized')
            return np.diag(1. / (1. + np.exp(-x)) - 1. / (1. + np.exp(-x)) ** 2)

    @staticmethod
    def layers_act(layers, x):
        # sequentially apply layers
        ret = np.copy(x)
        for layer in layers:
            ret = layer.act(ret)
        return ret

    @staticmethod
    def _layers_grad(layers, layer_id, x):
        n, m = np.shape(x)
        lin = (layers[layer_id].type_ == 'linear')
        # find grad for each input. Output must be a number!!!
        assert np.ndim(Layer.layers_act(layers, x[0])) == 1
        # gradient with respect to x: (n, n_in, 1)
        # gradient with respect to w: (n, w_in, w_out)
        # gradient with respect to b: (n, w_out, 1)

        n_layers = len(layers)
        # get dimension of x_in for layers[layer_id]

        grad_x = np.zeros(shape=(n, layers[layer_id].n_in))
        if lin:
            grad_w = np.zeros(shape=(n, layers[layer_id].n_in, layers[layer_id].n_out))
            grad_b = np.zeros(shape=(n, layers[layer_id].n_out))
        for i in range(n):
            trunc_act = Layer.layers_act(layers[:layer_id], x[i, :])
            g_x = layers[layer_id].gradient(trunc_act, var='x')
            mult = np.identity(np.shape(g_x)[0])
            for curr_layer_id in range(layer_id + 1, n_layers):
                trunc_act = layers[curr_layer_id - 1].act(trunc_act)
                mult = layers[curr_layer_id].gradient(trunc_act, var='x') @ mult
            grad_x[i, :] = mult @ g_x
            if lin:
                grad_w[i, :, :] = np.reshape(mult @ layers[layer_id].gradient(trunc_act, var='w'),
                                             (layers[layer_id].n_in, layers[layer_id].n_out))
                grad_b[i, :] = mult @ layers[layer_id].gradient(trunc_act, var='b')

        if lin:
            return {'w': grad_w, 'b': grad_b, 'x': grad_x}
        else:
            return {'x': grad_x}

    @staticmethod
    def layers_grad(layers, x):
        n, m = np.shape(x)
        n_layers = len(layers)
        # find grad for each input. Output must be a number!!!
        assert layers[n_layers - 1].n_out == 1
        # gradient with respect to x: (n, n_in)
        # gradient with respect to w: (n, w_in, w_out)
        # gradient with respect to b: (n, w_out)

        # initial gradient
        last_grad = np.zeros(shape=(n, layers[n_layers - 1].n_in, layers[n_layers - 1].n_in))
        for i in range(n):
            last_grad[i, :, :] = np.identity(layers[n_layers - 1].n_in)
        grad = {n_layers: {'x': last_grad}}

        for layer_id in reversed(range(n_layers)):
            grad_x = np.zeros(shape=(n, layers[layer_id].n_in))
            if layers[layer_id].type_ == 'linear':
                grad_w = np.zeros(shape=(n, layers[layer_id].n_in, layers[layer_id].n_out))
                grad_b = np.zeros(shape=(n, layers[layer_id].n_out))
                for i in range(n):
                    trunc_act = Layer.layers_act(layers[:layer_id], x[i, :])
                    grad_x[i, :] = grad[layer_id + 1]['x'][i, :] @ layers[layer_id].gradient(trunc_act, var='x')
                    grad_w[i, :, :] = np.reshape(
                        grad[layer_id + 1]['x'][i, :] @ layers[layer_id].gradient(trunc_act, var='w'),
                        (layers[layer_id].n_in, layers[layer_id].n_out))
                    grad_b[i, :] = grad[layer_id + 1]['x'][i, :] @ layers[layer_id].gradient(trunc_act, var='b')
                grad[layer_id] = {'w': grad_w, 'b': grad_b, 'x': grad_x}
            else:
                for i in range(n):
                    trunc_act = Layer.layers_act(layers[:layer_id], x[i, :])
                    grad_x[i, :] = grad[layer_id + 1]['x'][i, :] @ layers[layer_id].gradient(trunc_act, var='x')
                grad[layer_id] = {'x': grad_x}
        return grad

    @staticmethod
    # @memory_profiler.profile
    def layers_grad_multidim(layers, x):
        n, m = np.shape(x)
        n_layers = len(layers)
        # find grad for each input.
        n_out = layers[n_layers - 1].n_out
        # gradient with respect to x: (n, n_out n_in)
        # gradient with respect to w: (n, n_out, w_in, w_out)
        # gradient with respect to b: (n, n_out, w_out)

        # initial gradient
        last_grad = np.zeros(shape=(n, layers[n_layers - 1].n_in, layers[n_layers - 1].n_in))
        for i in range(n):
            last_grad[i, :, :] = np.identity(layers[n_layers - 1].n_in)
        grad = {n_layers: {'x': last_grad}}

        for layer_id in reversed(range(n_layers)):
            grad_x = np.zeros(shape=(n, n_out, layers[layer_id].n_in))
            if layers[layer_id].type_ == 'linear':
                grad_w = np.zeros(shape=(n, n_out, layers[layer_id].n_in, layers[layer_id].n_out))
                grad_b = np.zeros(shape=(n, n_out, layers[layer_id].n_out))
                for i in range(n):
                    trunc_act = Layer.layers_act(layers[:layer_id], x[i, :])
                    grad_x[i, :, :] = grad[layer_id + 1]['x'][i, :] @ layers[layer_id].gradient(trunc_act, var='x')
                    for j in range(n_out):
                        grad_w[i, j, :, :] = np.reshape(
                            grad[layer_id + 1]['x'][i, j, :] @ layers[layer_id].gradient(trunc_act, var='w'),
                            (layers[layer_id].n_in, layers[layer_id].n_out))
                    grad_b[i, :, :] = grad[layer_id + 1]['x'][i, :] @ layers[layer_id].gradient(trunc_act, var='b')
                grad[layer_id] = {'w': grad_w, 'b': grad_b, 'x': grad_x}
            else:
                for i in range(n):
                    trunc_act = Layer.layers_act(layers[:layer_id], x[i, :])
                    grad_x[i, :, :] = grad[layer_id + 1]['x'][i, :] @ layers[layer_id].gradient(trunc_act, var='x')
                grad[layer_id] = {'x': grad_x}
        return grad


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()

    return tf.concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def fully_connect(input_, output_size, scope=None, with_w=False,
                  initializer=variance_scaling_initializer()):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 initializer=initializer)
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def batch_normal(inp, scope="scope" , reuse=False):
    return batch_norm(inp, epsilon=1e-5, decay=0.9 , scale=True, scope=scope , reuse = reuse , updates_collections=None)


def conv2d_1(input_, output_dim,
           k_h=3, k_w=3, d_h=2, d_w=2,
           name="conv2d"):
    with tf.variable_scope(name):

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer= variance_scaling_initializer())
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv, w


def de_conv(input_, output_shape,
             k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, name="deconv2d",
             with_w=False, initializer = variance_scaling_initializer()):

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer = initializer)
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])
        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def jacobian(y, x):
    with tf.name_scope("jacob"):
        print(dt.datetime.now().strftime("%m-%d %H:%M"), ': Jacobian construction started')
        shape = tf.shape(y)
        jac_list = []
        y_list = tf.unstack(tf.reshape(y, [-1]))
        n = len(y_list)
        tmp_progress = 0
        for i in range(n):
            jac_list.append(tf.gradients(y_list[i], x))
            if tmp_progress > 0.1:
                print(dt.datetime.now().strftime("%m-%d %H:%M"), ': ', int(100*i/n), '%')
                tmp_progress = 0
            tmp_progress += 1./n

        print(dt.datetime.now().strftime("%m-%d %H:%M"), ': Jacobian construction finished!')
        jac = tf.stack(jac_list)
        return tf.reshape(jac, shape=tf.concat([shape, len(x)], axis=0))