from utils import *


class Generator:

    def __init__(self, alpha, batch_size, y_dim, output_size, channel,
                 initial_x_train=None, initial_y_train=None, x_test=None, y_test=None):
        self.batch_size = batch_size
        self.y_dim = y_dim
        self.output_size = output_size
        self.channel = channel
        #self.initial_x_train = np.reshape(initial_x_train, newshape=(len(initial_y_train), 784))
        #self.initial_y_train = [(1. if initial_y_train[j, 0] == 1. else -1.) for j in range(len(initial_y_train))]
        #self.x_test = np.reshape(x_test, newshape=(len(x_test), 784))
        #self.y_test = [(1. if y_test[j, 0] == 1. else -1.) for j in range(len(y_test))]
        self.prob_approx = 0.
        self.a = 1.
        self.alpha = alpha
        self.output_height = 64
        self.output_width = 64
        self.c_dim = 3
        self.gfc_dim = 1024
        self.gf_dim = 64

    def act(self, z, y, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
            s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = tf.concat([z, y], 1)

            h0 = tf.nn.relu(
                batch_normal(linear(z, self.gfc_dim, scope='gen_h0_lin'), scope='gen_bn0'))
            h0 = tf.concat([h0, y], 1)

            h1 = tf.nn.relu(batch_normal(
                linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'gen_h1_lin'), scope='gen_bn1'))
            h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(batch_normal(deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='gen_h2'),
                                         scope='gen_bn2'))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(
                deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='gen_h3'))

    def adv_obj_and_grad(self, d_generated, labels):
        # todo
        return None
