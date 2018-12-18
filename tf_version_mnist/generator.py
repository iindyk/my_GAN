from utils import *


class Generator:

    def act(self, z, batch_size, z_dim, reuse=False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            g_dim = 64  # Number of filters of first layer of generator
            c_dim = 1  # Color dimension of output (MNIST is grayscale, so c_dim = 1 for us)
            s = 28  # Output size of the image
            s2, s4, s8, s16 = int(s / 2), int(s / 4), int(s / 8), int(
                s / 16)  # We want to slowly upscale the image, so these values will help
            # make that change gradual.

            h0 = tf.reshape(z, [batch_size, s16 + 1, s16 + 1, 25])
            h0 = tf.nn.relu(h0)
            # Dimensions of h0 = batch_size x 2 x 2 x 25

            # First DeConv Layer
            output1_shape = [batch_size, s8, s8, g_dim * 4]
            w_conv1 = tf.get_variable('g_wconv1', [5, 5, output1_shape[-1], int(h0.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv1 = tf.get_variable('g_bconv1', [output1_shape[-1]], initializer=tf.constant_initializer(.1))
            h_conv1 = tf.nn.conv2d_transpose(h0, w_conv1, output_shape=output1_shape,
                                             strides=[1, 2, 2, 1], padding='SAME') + b_conv1
            h_conv1 = tf.contrib.layers.batch_norm(inputs=h_conv1, center=True, scale=True, is_training=True, scope="g_bn1")
            h_conv1 = tf.nn.relu(h_conv1)
            # Dimensions of h_conv1 = batch_size x 3 x 3 x 256

            # Second DeConv Layer
            output2_shape = [batch_size, s4 - 1, s4 - 1, g_dim * 2]
            w_conv2 = tf.get_variable('g_wconv2', [5, 5, output2_shape[-1], int(h_conv1.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv2 = tf.get_variable('g_bconv2', [output2_shape[-1]], initializer=tf.constant_initializer(.1))
            h_conv2 = tf.nn.conv2d_transpose(h_conv1, w_conv2, output_shape=output2_shape,
                                             strides=[1, 2, 2, 1], padding='SAME') + b_conv2
            h_conv2 = tf.contrib.layers.batch_norm(inputs=h_conv2, center=True, scale=True, is_training=True, scope="g_bn2")
            h_conv2 = tf.nn.relu(h_conv2)
            # Dimensions of h_conv2 = batch_size x 6 x 6 x 128

            # Third DeConv Layer
            output3_shape = [batch_size, s2 - 2, s2 - 2, g_dim * 1]
            w_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(h_conv2.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
            h_conv3 = tf.nn.conv2d_transpose(h_conv2, w_conv3, output_shape=output3_shape,
                                             strides=[1, 2, 2, 1], padding='SAME') + b_conv3
            h_conv3 = tf.contrib.layers.batch_norm(inputs=h_conv3, center=True, scale=True, is_training=True, scope="g_bn3")
            h_conv3 = tf.nn.relu(h_conv3)
            # Dimensions of h_conv3 = batch_size x 12 x 12 x 64

            # Fourth DeConv Layer
            output4_shape = [batch_size, s, s, c_dim]
            w_conv4 = tf.get_variable('g_wconv4', [5, 5, output4_shape[-1], int(h_conv3.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv4 = tf.get_variable('g_bconv4', [output4_shape[-1]], initializer=tf.constant_initializer(.1))
            h_conv4 = tf.nn.conv2d_transpose(h_conv3, w_conv4, output_shape=output4_shape,
                                             strides=[1, 2, 2, 1], padding='VALID') + b_conv4
            h_conv4 = tf.nn.tanh(h_conv4)
            # Dimensions of h_conv4 = batch_size x 28 x 28 x 1

        return h_conv4