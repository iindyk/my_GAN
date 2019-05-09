from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from tf_cnn_mnist.ops import *
from tf_cnn_mnist.utils import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class DCGAN(object):
    def __init__(self, sess, run_opts, alpha, input_height=108, input_width=108, crop=True,
                 batch_size=64, sample_num=64, output_height=64, output_width=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None, labels=None):
        """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
        self.sess = sess
        self.crop = crop
        self.labels = labels
        self.alpha = alpha
        self.run_opts = run_opts

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        # if not self.y_dim:
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        # if not self.y_dim:
        self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        if self.dataset_name == 'mnist':
            self.data_X, self.data_y, self.test_data, self.test_labels = self.load_mnist()
            self.c_dim = self.data_X[0].shape[-1]
        else:
            self.data = glob(os.path.join("./data", self.dataset_name, self.input_fname_pattern))

            self.celeba_labels = []
            with open("./list_attr_celeba.txt") as celeba_labels_list:
                # Skip header
                next(celeba_labels_list)
                next(celeba_labels_list)
                for celeba_label_line in celeba_labels_list:
                    # Prepare label list
                    image_label = celeba_label_line.split()
                    image_label = image_label[1:]
                    image_label = map(int, image_label)
                    image_label = [0 if label == -1 else 1 for label in image_label]

                    self.celeba_labels.append(image_label)

            print(self.data[0])
            print(self.celeba_labels)

            imreadImg = imread(self.data[0])
            if len(imreadImg.shape) >= 3:  # check if image is a non-grayscale image by checking channel number
                self.c_dim = imread(self.data[0]).shape[-1]
            else:
                self.c_dim = 1

        self.grayscale = (self.c_dim == 1)

        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        else:
            self.y = None

        #self.y_c = tf.placeholder(tf.float32, [int(self.batch_size*1.5), self.y_dim], name='y_c')

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')

        #self.x_c = tf.placeholder(tf.float32, [int(self.batch_size*1.5)] + image_dims, name='x_c')

        inputs = self.inputs

        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.G = self.generator(self.z, self.y)
        #self.C_train = self.classifier(tf.concat([self.G, self.x_c], axis=0))
        self.C_train = self.classifier(self.G)
        self.C_test = self.classifier(self.test_data)
        self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)
        self.sampler = self.sampler(self.z, self.y)
        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

        #self.c_loss_train = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.C_train, tf.concat([self.y, self.y_c], axis=0)))
        self.c_loss_train = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.C_train, self.y))
        self.c_loss_test = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.C_test, self.test_labels))
        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]


        self.dl_dc_dxi_mixed = tf.hessians(self.c_loss_train, self.c_sv)[0]

        self.dlt_dc = tf.gradients(self.c_loss_test, self.c_sv)[0][:self._n3]
        self.dl_dc_dc = self.dl_dc_dxi_mixed[:self._n3, :self._n3]+tf.eye(self._n3)*1e-3
        self.dl_dc_dxi = self.dl_dc_dxi_mixed[:self._n3, self._n3:]


        # define custom part of adversary's loss as tensor
        self.c_optim = tf.train.AdamOptimizer(0.001).minimize(self.c_loss_train, var_list=self.c_sv)

        dc_dxi = tf.linalg.solve(self.dl_dc_dc, self.dl_dc_dxi)

        #dc_dxi = tf.stop_gradient(tf.cond(tf.abs(tf.linalg.det(self.dl_dc_dc)) > 1e-10,
        #                                  lambda: tf.linalg.solve(self.dl_dc_dc, self.dl_dc_dxi),
        #                                  lambda: tf.matmul(tf.ones((self._n3, self._n3)), self.dl_dc_dxi)))
        self.cust_adv_grad = -tf.stop_gradient(tf.math.l2_normalize(tf.matmul(tf.expand_dims(self.dlt_dc, 0), dc_dxi)))


        self.saver = tf.train.Saver()

    def train(self, config):
        #todo: pretrain G
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)

        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        new_g_vars = []
        g_accumulation = []
        new_g_accumulation = []
        n_g_vars = len(self.g_vars)
        g_grad = tf.gradients(ys=self.g_loss, xs=self.g_vars)
        g_adv_grad = tf.gradients(ys=tf.matmul(self.cust_adv_grad, tf.reshape(self.G, (-1, 1))), xs=self.g_vars)
        for i in range(n_g_vars):
            g_accumulation.append(tf.get_variable('accum_g' + str(i), shape=g_grad[i].get_shape(), trainable=False))
            new_g_accumulation.append(
                g_accumulation[i].assign(config.beta1 * g_accumulation[i] + (1. - config.beta1) * (g_grad[i]+
                                         self.alpha*g_adv_grad[i])))
            new_g_vars.append(self.g_vars[i].assign(self.g_vars[i] - config.learning_rate * g_accumulation[i]))

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                    self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))
        if config.dataset == 'mnist':
            sample_inputs = self.data_X[0:self.sample_num]
            sample_labels = self.data_y[0:self.sample_num]
        else:
            sample_files = self.data[0:self.sample_num]
            sample = [
                get_image(sample_file,
                          input_height=self.input_height,
                          input_width=self.input_width,
                          resize_height=self.output_height,
                          resize_width=self.output_width,
                          crop=self.crop,
                          grayscale=self.grayscale) for sample_file in sample_files]
            if self.grayscale:
                sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                sample_inputs = np.array(sample).astype(np.float32)
            sample_labels = self.celeba_labels[0:self.sample_num]

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in range(config.epoch):
            if config.dataset == 'mnist':
                batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
            else:
                self.data = glob(os.path.join(
                    "./data", config.dataset, self.input_fname_pattern))
                batch_idxs = min(len(self.data), config.train_size) // config.batch_size

            for idx in range(0, batch_idxs):
                batch_images = self.data_X[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_labels = self.data_y[idx * config.batch_size:(idx + 1) * config.batch_size]

                tr_indices = np.random.randint(0, len(self.data_y), size=int(self.batch_size*1.5))
                x_tr = self.data_X[tr_indices]
                y_tr = self.data_y[tr_indices]

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)

                # Include labels in all datasets
                if config.dataset == 'mnist' or config.dataset == 'celebA':
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={
                                                       self.inputs: batch_images,
                                                       self.z: batch_z,
                                                       self.y: batch_labels,
                                                   }, options=self.run_opts)
                    self.writer.add_summary(summary_str, counter)

                    if counter > config.g_pretrain:
                        # Update C
                        for i in range(100):
                            _, _ = self.sess.run([self.c_optim, self.c_loss_train], feed_dict={
                                                self.z: batch_z,
                                                self.y: batch_labels,
                                                #self.x_c: x_tr,
                                                #self.y_c: y_tr,
                                                }, options=self.run_opts)

                        # adversarial optimization
                        _, _, _, summary_str = self.sess.run([new_g_vars, new_g_accumulation, self.c_loss_train, self.g_sum],
                                                             feed_dict={
                                                                self.z: batch_z,
                                                                self.y: batch_labels,
                                                                #self.x_c: x_tr,
                                                                #self.y_c: y_tr,
                                                            }, options=self.run_opts)
                        self.writer.add_summary(summary_str, counter)

                    # Update G network twice: experimental, not in the paper
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={
                                                          self.z: batch_z,
                                                          self.y: batch_labels,
                                                      }, options=self.run_opts)
                    self.writer.add_summary(summary_str, counter)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={
                                                       self.z: batch_z,
                                                       self.y: batch_labels,
                                                   }, options=self.run_opts)
                    self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({
                        self.z: batch_z,
                        self.y: batch_labels
                    })
                    errD_real = self.d_loss_real.eval({
                        self.inputs: batch_images,
                        self.y: batch_labels
                    })
                    errG = self.g_loss.eval({
                        self.z: batch_z,
                        self.y: batch_labels
                    })

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 20) == 2:
                    if config.dataset == 'mnist' or config.dataset == 'celebA':
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                                self.y: sample_labels,
                            }, options=self.run_opts
                        )
                        save_images(samples, (4, 6),
                                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 50) == 2:
                    self.save(config.checkpoint_dir, counter)
                    print('checkpoint saved')

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if self.y_dim:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = lrelu(conv2d(x, self.df_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2 + self.y_dim, name='d_h1_conv')))
                h1 = conv_cond_concat(h1, yb)

                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4 + self.y_dim, name='d_h2_conv')))
                h2 = conv_cond_concat(h2, yb)

                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8 + self.y_dim, name='d_h3_conv')))
                h3 = conv_cond_concat(h3, yb)

                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

                return tf.nn.sigmoid(h4), h4

            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
                h1 = tf.reshape(h1, [self.batch_size, -1])
                h1 = concat([h1, y], 1)

                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
                h2 = concat([h2, y], 1)

                h3 = linear(h2, 1, 'd_h3_lin')

                return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            if self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(
                    z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

                self.h0 = tf.reshape(
                    self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(self.h0))
                h0 = conv_cond_concat(h0, yb)

                self.h1, self.h1_w, self.h1_b = deconv2d(
                    h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))
                h1 = conv_cond_concat(h1, yb)

                h2, self.h2_w, self.h2_b = deconv2d(
                    h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))
                h2 = conv_cond_concat(h2, yb)

                h3, self.h3_w, self.h3_b = deconv2d(
                    h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))
                h3 = conv_cond_concat(h3, yb)

                h4, self.h4_w, self.h4_b = deconv2d(
                    h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

                return tf.nn.tanh(h4)

            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(
                    self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin')))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
                                                    [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(
                    deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            if self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                # project `z` and reshape
                h0 = tf.reshape(
                    linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
                    [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(h0, train=False))
                h0 = conv_cond_concat(h0, yb)

                h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))
                h1 = conv_cond_concat(h1, yb)

                h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))
                h2 = conv_cond_concat(h2, yb)

                h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))
                h3 = conv_cond_concat(h3, yb)

                h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

                return tf.nn.tanh(h4)
            else:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
                s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

                # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
                h0 = concat([h0, y], 1)

                h1 = tf.nn.relu(self.g_bn1(
                    linear(h0, self.gf_dim * 2 * s_h4 * s_w4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(
                    deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def classifier(self, images):
        with tf.variable_scope("classifier", reuse=tf.AUTO_REUSE) as scope:
            #scope.reuse_variables()
            n_hidden_1 = 8  # 1st layer number of neurons
            n_hidden_2 = 8  # 2nd layer number of neurons
            n_input = 784  # MNIST data input (img shape: 28*28)
            n_classes = 3  # MNIST total classes (0-9 digits)
            # create a supervector containing all variables
            self._n = (n_input+1)*n_hidden_1+(n_hidden_1+1)*n_hidden_2+(n_hidden_2+1)*n_classes+\
                      self.batch_size*self.input_height**2
            _n1 = (n_input+1)*n_hidden_1
            _n2 = (n_input+1)*n_hidden_1+(n_hidden_1+1)*n_hidden_2
            self._n3 = (n_input+1)*n_hidden_1+(n_hidden_1+1)*n_hidden_2+(n_hidden_2+1)*n_classes
            self.c_sv = tf.get_variable('c_sv', [self._n], initializer=tf.truncated_normal_initializer(0.02))

            # Store layers weight & bias
            weights = {
                'h1': tf.reshape(self.c_sv[:_n1-n_hidden_1], shape=[n_input, n_hidden_1]),
                'h2': tf.reshape(self.c_sv[_n1:_n2-n_hidden_2], shape=[n_hidden_1, n_hidden_2]),
                'out': tf.reshape(self.c_sv[_n2:self._n3-n_classes], shape=[n_hidden_2, n_classes])
            }
            biases = {
                'b1': self.c_sv[_n1-n_hidden_1:_n1],
                'b2': self.c_sv[_n2-n_hidden_2:_n2],
                'out': self.c_sv[self._n3-n_classes:self._n3]
            }

            # Create model
            # Hidden fully connected layer with 256 neurons
            tf.assign(self.c_sv[self._n3:], tf.reshape(images[:self.batch_size], [-1]))
            layer_1 = tf.add(tf.matmul(tf.concat([tf.reshape(self.c_sv[self._n3:], (-1, n_input)),
                                                 tf.reshape(images[self.batch_size:], (-1, n_input))], axis=0),
                                       weights['h1']), biases['b1'])
            # Hidden fully connected layer with 256 neurons
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            # Output fully connected layer with a neuron for each class
            out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
            return out_layer

    def load_mnist(self, labels=None):
        if labels:
            assert len(labels) == self.y_dim
        data_dir = os.path.join("./data", self.dataset_name)

        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        #X = np.concatenate((trX, teX), axis=0)
        #y = np.concatenate((trY, teY), axis=0).astype(np.int)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(trX)
        np.random.seed(seed)
        np.random.shuffle(trY)

        final_x = []
        final_y = []
        for i, label in enumerate(trY):
            for j in range(self.y_dim):
                if label == self.labels[j]:
                    one_hot = np.zeros(self.y_dim)
                    one_hot[j] = 1.
                    final_x.append(trX[i])
                    final_y.append(one_hot)

        final_x_te = []
        final_y_te = []
        for i, label in enumerate(teY):
            for j in range(self.y_dim):
                if label == self.labels[j]:
                    one_hot = np.zeros(self.y_dim)
                    one_hot[j] = 1.
                    final_x_te.append(teX[i])
                    final_y_te.append(one_hot)

        return np.array(final_x) / 127.5 - 1., np.array(final_y), np.array(final_x_te).astype(np.float32)/127.5-1., \
               np.array(final_y_te).astype(np.float32)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
