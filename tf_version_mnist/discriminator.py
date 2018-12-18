from utils import *


class Discriminator:

    def act(self, x, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # First Conv and Pool Layers
            w_conv1 = tf.get_variable('d_wconv1', [5, 5, 1, 8],
                                      initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv1 = tf.get_variable('d_bconv1', [8], initializer=tf.constant_initializer(0))
            h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
            h_pool1 = avg_pool_2x2(h_conv1)

            # Second Conv and Pool Layers
            w_conv2 = tf.get_variable('d_wconv2', [5, 5, 8, 16],
                                      initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv2 = tf.get_variable('d_bconv2', [16], initializer=tf.constant_initializer(0))
            h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
            h_pool2 = avg_pool_2x2(h_conv2)

            # First Fully Connected Layer
            w_fc1 = tf.get_variable('d_wfc1', [7 * 7 * 16, 32],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_fc1 = tf.get_variable('d_bfc1', [32], initializer=tf.constant_initializer(0))
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 16])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

            # Second Fully Connected Layer
            w_fc2 = tf.get_variable('d_wfc2', [32, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_fc2 = tf.get_variable('d_bfc2', [1], initializer=tf.constant_initializer(0))

            # Final Layer
            y_conv = (tf.matmul(h_fc1, w_fc2) + b_fc2)
        return y_conv

