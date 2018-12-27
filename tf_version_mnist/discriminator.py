from utils import *
from tensorflow.contrib.layers.python.layers import xavier_initializer


class Discriminator0:

    def act(self, x, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # First Conv and Pool Layers
            w_conv1 = tf.get_variable('d_wconv1', [5, 5, 1, 8],
                                      initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv1 = tf.get_variable('d_bconv1', [8], initializer=tf.constant_initializer(0))
            h_conv1 = tf.nn.leaky_relu(conv2d(x, w_conv1) + b_conv1)
            h_pool1 = avg_pool_2x2(h_conv1)

            # Second Conv and Pool Layers
            w_conv2 = tf.get_variable('d_wconv2', [5, 5, 8, 16],
                                      initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_conv2 = tf.get_variable('d_bconv2', [16], initializer=tf.constant_initializer(0))
            h_conv2 = tf.nn.leaky_relu(conv2d(h_pool1, w_conv2) + b_conv2)
            h_pool2 = avg_pool_2x2(h_conv2)

            # First Fully Connected Layer
            w_fc1 = tf.get_variable('d_wfc1', [7 * 7 * 16, 32],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_fc1 = tf.get_variable('d_bfc1', [32], initializer=tf.constant_initializer(0))
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 16])
            h_fc1 = tf.nn.leaky_relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

            # Second Fully Connected Layer
            w_fc2 = tf.get_variable('d_wfc2', [32, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
            b_fc2 = tf.get_variable('d_bfc2', [1], initializer=tf.constant_initializer(0))

            # Final Layer
            y_conv = (tf.matmul(h_fc1, w_fc2) + b_fc2)
        return y_conv


class Discriminator1:

    def __init__(self, batch_size, y_dim):
        self.batch_size = batch_size
        self.y_dim = y_dim

    def act(self, x, y, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # mnist data's shape is (28 , 28 , 1)
            yb = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])
            # concat
            concat_data = conv_cond_concat(x, yb)

            conv1, w1 = conv2d_1(concat_data, output_dim=10, name='dis_conv1')
            tf.add_to_collection('weight_1', w1)

            conv1 = tf.nn.leaky_relu(conv1)
            conv1 = conv_cond_concat(conv1, yb)
            tf.add_to_collection('ac_1', conv1)

            conv2, w2 = conv2d_1(conv1, output_dim=64, name='dis_conv2')
            tf.add_to_collection('weight_2', w2)

            conv2 = tf.nn.leaky_relu(batch_normal(conv2, scope='dis_bn1'))
            tf.add_to_collection('ac_2', conv2)

            conv2 = tf.reshape(conv2, [self.batch_size, -1])
            conv2 = tf.concat([conv2, y], 1)

            f1 = tf.nn.leaky_relu(
                batch_normal(fully_connect(conv2, output_size=1024, scope='dis_fully1'), scope='dis_bn2', reuse=reuse))
            f1 = tf.concat([f1, y], 1)

            out = fully_connect(f1, output_size=1, scope='dis_fully2', initializer=xavier_initializer())

            return tf.nn.sigmoid(out), out

