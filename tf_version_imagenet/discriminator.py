from utils import *


class Discriminator:

    def __init__(self, batch_size, y_dim, c_dim):
        self.batch_size = batch_size
        self.y_dim = y_dim
        self.c_dim = c_dim      # color scale of image
        self.df_dim = 64
        self.dfc_dim = 1024

    def act(self, x, y, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(x, yb)

            _conv1, _ = conv2d_1(x, self.c_dim + self.y_dim, name='dis_h0_conv')
            h0 = tf.nn.leaky_relu(_conv1)
            h0 = conv_cond_concat(h0, yb)

            h1 = tf.nn.leaky_relu(batch_normal(conv2d(h0, self.df_dim + self.y_dim), scope='dis_h1_conv'))
            h1 = tf.reshape(h1, [self.batch_size, -1])
            h1 = tf.concat([h1, y], 1)

            h2 = tf.nn.leaky_relu(batch_normal(linear(h1, self.dfc_dim, scope='dis_h2_lin')))
            h2 = tf.concat([h2, y], 1)

            h3 = linear(h2, 1, 'dis_h3_lin')

            return tf.nn.sigmoid(h3), h3
