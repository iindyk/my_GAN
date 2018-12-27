from utils import *
import sklearn.svm as svm
from tensorflow.contrib.layers.python.layers import xavier_initializer


class Generator0:
    a = 1.
    alpha = 1.

    def __init__(self, initial_x_train=None, initial_y_train=None, x_test=None, y_test=None):
        self.initial_x_train = np.reshape(initial_x_train, newshape=(len(initial_y_train), 784))
        self.initial_y_train = initial_y_train
        self.x_test = np.reshape(x_test, newshape=(len(x_test), 784))
        self.y_test = y_test
        self.prob_approx = 0.

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
            h0 = tf.nn.leaky_relu(h0)
            # Dimensions of h0 = batch_size x 2 x 2 x 25

            # First DeConv Layer
            output1_shape = [batch_size, s8, s8, g_dim * 4]
            w_conv1 = tf.get_variable('g_wconv1', [5, 5, output1_shape[-1], int(h0.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv1 = tf.get_variable('g_bconv1', [output1_shape[-1]], initializer=tf.constant_initializer(.1))
            h_conv1 = tf.nn.conv2d_transpose(h0, w_conv1, output_shape=output1_shape,
                                             strides=[1, 2, 2, 1], padding='SAME') + b_conv1
            h_conv1 = tf.contrib.layers.batch_norm(inputs=h_conv1, center=True, scale=True, is_training=True, scope="g_bn1")
            h_conv1 = tf.nn.leaky_relu(h_conv1)
            # Dimensions of h_conv1 = batch_size x 3 x 3 x 256

            # Second DeConv Layer
            output2_shape = [batch_size, s4 - 1, s4 - 1, g_dim * 2]
            w_conv2 = tf.get_variable('g_wconv2', [5, 5, output2_shape[-1], int(h_conv1.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv2 = tf.get_variable('g_bconv2', [output2_shape[-1]], initializer=tf.constant_initializer(.1))
            h_conv2 = tf.nn.conv2d_transpose(h_conv1, w_conv2, output_shape=output2_shape,
                                             strides=[1, 2, 2, 1], padding='SAME') + b_conv2
            h_conv2 = tf.contrib.layers.batch_norm(inputs=h_conv2, center=True, scale=True, is_training=True, scope="g_bn2")
            h_conv2 = tf.nn.leaky_relu(h_conv2)
            # Dimensions of h_conv2 = batch_size x 6 x 6 x 128

            # Third DeConv Layer
            output3_shape = [batch_size, s2 - 2, s2 - 2, g_dim * 1]
            w_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(h_conv2.get_shape()[-1])],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
            h_conv3 = tf.nn.conv2d_transpose(h_conv2, w_conv3, output_shape=output3_shape,
                                             strides=[1, 2, 2, 1], padding='SAME') + b_conv3
            h_conv3 = tf.contrib.layers.batch_norm(inputs=h_conv3, center=True, scale=True, is_training=True, scope="g_bn3")
            h_conv3 = tf.nn.leaky_relu(h_conv3)
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

    def adv_obj_and_grad(self, d_generated):
        d_gen = np.reshape(d_generated, newshape=(len(d_generated), 784))
        n_gen, m_gen = np.shape(d_gen)
        n_t = len(self.y_test)

        # returns the test prediction accuracy and its gradient w.r.t. poisoning data
        d_union = np.append(d_gen, self.initial_x_train, axis=0)
        l_generated = np.array([-1.] * n_gen)  # todo: how to generate labels?
        l_union = np.append(l_generated, self.initial_y_train)

        # get parameters of SVM
        svc = svm.SVC(kernel='linear', C=self.a).fit(d_union, l_union)
        w = svc.coef_[0]  # normal vector
        b = svc.intercept_[0]  # intercept

        # calculate approximation to the probability of misclassification
        prob_approx = 0
        n_test = len(self.y_test)
        for i in range(n_test):
            prob_approx += max(self.y_test[i] * (w.dot(self.x_test[i]) + b), -1)
        prob_approx /= n_test

        # construct extended dual variables vector
        n_union = len(l_union)
        l_ext = np.zeros(n_union)
        tmp_i = 0
        for i in range(n_union):
            if i in svc.support_:
                l_ext[i] = svc.dual_coef_[0, tmp_i] * l_union[i]
                tmp_i += 1
        l = l_ext[:n_gen]

        # get approximate gradient of w
        dw_dh = np.array([[l[i] * l_union[i] for j in range(m_gen)] for i in range(n_gen)])

        # get approximate gradient of b
        # 1: find point on the margin's boundary
        idx = 0
        for i in range(n_gen):
            if 0.001 < l[i] < 0.999:
                idx = i
                break
        db_dh = np.array([np.multiply(dw_dh[j, :], d_union[idx]) for j in range(n_gen)])
        cost = 0.

        obj_grad_val = np.zeros(shape=(n_gen, m_gen))
        for k in range(n_t):
            bin_ = self.y_test[k]/n_t if self.y_test[k] * (np.dot(w, self.x_test[k]) + b) > -1 else 0.0
            cost += max(self.y_test[k] * (np.dot(w, self.x_test[k]) + b), -1)
            if bin_ != 0:
                for i in range(n_gen):
                    obj_grad_val[i, :] += (np.multiply(dw_dh[i, :], self.x_test[k, :]) + db_dh[i, :]) * bin_

        self.prob_approx = cost/n_t
        return np.reshape(np.float32(obj_grad_val), newshape=np.shape(d_generated))


class Generator1:
    def __init__(self, batch_size, y_dim, output_size, channel):
        self.batch_size = batch_size
        self.y_dim = y_dim
        self.output_size = output_size
        self.channel = channel

    def act(self, z, y):
        with tf.variable_scope('generator') as scope:
            yb = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])
            z = tf.concat([z, y], 1)
            c1, c2 = int(self.output_size / 4), int(self.output_size / 2)

            # 10 stand for the num of labels
            d1 = tf.nn.relu(batch_normal(fully_connect(z, output_size=1024, scope='gen_fully'), scope='gen_bn1'))

            d1 = tf.concat([d1, y], 1)

            d2 = tf.nn.relu(
                batch_normal(fully_connect(d1, output_size=7 * 7 * 2 * 64, scope='gen_fully2'), scope='gen_bn2'))

            d2 = tf.reshape(d2, [self.batch_size, c1, c1, 64 * 2])
            d2 = conv_cond_concat(d2, yb)

            d3 = tf.nn.relu(batch_normal(de_conv(d2, output_shape=[self.batch_size, c2, c2, 128], name='gen_deconv1'),
                                         scope='gen_bn3'))

            d3 = conv_cond_concat(d3, yb)

            d4 = de_conv(d3, output_shape=[self.batch_size, self.output_size, self.output_size, self.channel],
                         name='gen_deconv2', initializer=xavier_initializer())

            return tf.nn.sigmoid(d4)

