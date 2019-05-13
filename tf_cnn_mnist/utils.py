"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
from sklearn.metrics import accuracy_score
import tf_cnn_mnist.classifier as cl

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width,
                     resize_height, resize_width, crop)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if images.shape[3] in (3, 4):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    print('sample saved')
    return scipy.misc.imsave(path, (255.-merge(images, size)))


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(
        x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(
            image, input_height, input_width,
            resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
    return (images + 1.) / 2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append(
                        {"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2 ** (int(layer_idx) + 2), 2 ** (int(layer_idx) + 2),
                   W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'", "").split()))


def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration)


def visualize(sess, dcgan, config, option):
    image_frame_dim = int(math.ceil(config.batch_size ** .5))
    if option == 0:
        z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
        save_images(samples, [image_frame_dim, image_frame_dim],
                    './samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    elif option == 1:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in range(dcgan.z_dim):
            print(" [*] %d" % idx)
            z_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_dim))
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            if config.dataset == "mnist":
                y = np.random.choice(3, config.batch_size)
                y_one_hot = np.zeros((config.batch_size, 3))
                y_one_hot[np.arange(config.batch_size), y] = 1

                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
            else:
                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

            save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))
    elif option == 2:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in [random.randint(0, dcgan.z_dim - 1) for _ in range(dcgan.z_dim)]:
            print(" [*] %d" % idx)
            z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
            z_sample = np.tile(z, (config.batch_size, 1))
            # z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            if config.dataset == "mnist":
                y = np.random.choice(10, config.batch_size)
                y_one_hot = np.zeros((config.batch_size, 10))
                y_one_hot[np.arange(config.batch_size), y] = 1

                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
            else:
                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

            try:
                make_gif(samples, './samples/test_gif_%s.gif' % (idx))
            except:
                save_images(samples, [image_frame_dim, image_frame_dim],
                            './samples/test_%s.png' % strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    elif option == 3:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in range(dcgan.z_dim):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            make_gif(samples, './samples/test_gif_%s.gif' % (idx))
    elif option == 4:
        image_set = []
        values = np.arange(0, 1, 1. / config.batch_size)

        for idx in range(dcgan.z_dim):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

            image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
            make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

        new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
                         for idx in range(64) + range(63, -1, -1)]
        make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)

    # Sample conditional celebA
    elif option == 5:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in range(dcgan.z_dim):
            print(" [*] %d" % idx)
            z_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_dim))
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            y = np.random.choice(40, config.batch_size)
            y_one_hot = np.zeros((config.batch_size, 40))
            # y_one_hot[np.arange(config.batch_size), y] = 1

            # Use input vector from first image
            y_one_hot[0][1] = 1
            y_one_hot[0][2] = 1
            y_one_hot[0][11] = 1
            y_one_hot[0][18] = 1
            y_one_hot[0][19] = 1
            y_one_hot[0][21] = 1
            y_one_hot[0][24] = 1
            y_one_hot[0][27] = 1
            y_one_hot[0][31] = 1
            y_one_hot[0][32] = 1
            y_one_hot[0][34] = 1
            y_one_hot[0][36] = 1
            y_one_hot[0][39] = 1

            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})

            save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))

    # save groups by 4 tests
    elif option == 6:
        values = np.arange(0, 1, 1. / config.batch_size)
        for num in range(config.generate_test_images):
            for idx in range(dcgan.z_dim):
                print(" [*] %d" % idx)
                z_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_dim))
                for kdx, z in enumerate(z_sample):
                    z[idx] = values[kdx]

                y = np.zeros(config.batch_size, dtype=np.int)
                y[config.batch_size//3: 2*config.batch_size//3] = 1
                y[2*config.batch_size//3] = 2
                y_one_hot = np.zeros((config.batch_size, 3))
                y_one_hot[np.arange(config.batch_size), y] = 1

                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})

                for i in range(config.batch_size):
                    save_images(np.reshape(samples[i], newshape=(1, 28, 28, 1)), [1, 1],
                                './samples/test/test_%s-%s_%s.png' % (y[i]+7, idx, num))

    elif option == 7:
        n_trials = 100
        n_t = 100
        sample_from_orig = False  # sample generated data from original
        data_shift = 800
        validation_crit_val = -2.
        skip_validation = False
        gen_share = 0.4  # % of training set to be generated
        n_orig = int(n_t * (1 - gen_share))

        dcgan.data_X, dcgan.data_y, dcgan.test_data, dcgan.test_labels = dcgan.load_mnist()
        print(len(dcgan.test_labels))

        errs = []
        false_neg = []
        errs_cramer = []
        false_neg_cramer = []
        false_pos_cramer = []
        errs_sd = []
        false_neg_sd = []
        false_pos_sd = []

        # false positive rate calculation
        false_pos = 0
        _dt_tmp = np.reshape(np.append(np.reshape(dcgan.data_X[data_shift:n_orig + data_shift], newshape=(-1, 784)),
                            np.zeros((3*config.batch_size - n_orig, 784)), axis=0), newshape=(-1, 28, 28, 1))
        _lb_tmp = np.append(dcgan.data_y[data_shift:n_orig + data_shift],
                            [[0., 0., 1.]]*(3*config.batch_size - n_orig), axis=0)
        x_placeholder = tf.placeholder("float", shape=[config.batch_size, 28, 28, 1])
        y_placeholder = tf.placeholder("float", shape=[config.batch_size, 3])

        x_c_placeholder = tf.placeholder("float", shape=[None, 28, 28, 1])
        y_c_placeholder = tf.placeholder("float", shape=[None, 3])
        c_train = cl.c(x_c_placeholder)
        t_vars = tf.trainable_variables()
        c_vars = [var for var in t_vars if 'cl_' in var.name]
        c_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=c_train, labels=y_c_placeholder)
        optimizer = tf.train.AdamOptimizer(0.001)
        c_optim = optimizer.minimize(c_loss, var_list=c_vars)

        for c_var in c_vars:
            sess.run(c_var.initializer)
        for opt_var in optimizer.variables():
            sess.run(opt_var.initializer)

        d_x = dcgan.discriminator(x_placeholder, y=y_placeholder, reuse=True)

        _, val1 = sess.run(d_x, feed_dict={
            x_placeholder: _dt_tmp[:config.batch_size],
            y_placeholder: _lb_tmp[:config.batch_size]})
        _, val2 = sess.run(d_x, feed_dict={
            x_placeholder: _dt_tmp[config.batch_size:2*config.batch_size],
            y_placeholder: _lb_tmp[config.batch_size:2*config.batch_size]})
        _, val3 = sess.run(d_x, feed_dict={
            x_placeholder: _dt_tmp[2*config.batch_size:3*config.batch_size],
            y_placeholder: _lb_tmp[2*config.batch_size:3*config.batch_size]})
        val = np.append(np.append(val1, val2, axis=0), val3, axis=0)

        for k in range(n_orig):
            if val[k, 0] < validation_crit_val:
                false_pos += 1

        for trial in range(n_trials):
            indices = np.random.randint(low=int(n_t * (1 - gen_share)) + data_shift, high=len(dcgan.data_y),
                                        size=int(n_t * gen_share))
            additional_y_train = dcgan.data_y[indices]
            if sample_from_orig:
                additional_x_train = dcgan.data_X[indices]
            else:
                additional_x_train = np.empty((0, 28, 28, 1), np.float32)
                for j in range((int(n_t * gen_share)) // config.batch_size + 1):
                    if (j + 1) * config.batch_size <= (int(n_t * gen_share)):
                        y_batch = additional_y_train[j * config.batch_size:(j + 1) * config.batch_size]
                    else:
                        y_batch = np.append(additional_y_train[j * config.batch_size:],
                                            [[0., 0., 1.]] * ((j + 1) * config.batch_size - int(n_t * gen_share)),
                                            axis=0)

                    z_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_dim))
                    sample = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_batch})

                    additional_x_train = np.append(additional_x_train,
                                                   np.reshape(sample, newshape=(-1, 28, 28, 1)), axis=0)

            train_data_tmp = np.append(dcgan.data_X[data_shift:int(n_t * (1 - gen_share)) + data_shift],
                                        additional_x_train[:int(n_t * gen_share)], axis=0)
            train_labels_tmp = np.append(dcgan.data_y[data_shift:int(n_t * (1 - gen_share)) + data_shift],
                                         additional_y_train[:int(n_t * gen_share)], axis=0)

            train_data = []
            train_labels = []

            false_neg.append(0)
            false_neg_cramer.append(0)
            false_pos_cramer.append(0)
            false_neg_sd.append(0)
            false_pos_sd.append(0)

            # validation
            for j in range(n_t // config.batch_size + 1):
                if (j + 1) * config.batch_size <= n_t:
                    x_batch = train_data_tmp[j * config.batch_size:(j + 1) * config.batch_size]
                    y_batch = train_labels_tmp[j * config.batch_size:(j + 1) * config.batch_size]
                else:
                    x_batch = np.append(train_data_tmp[j * config.batch_size:], np.zeros(((j + 1) * config.batch_size - n_t, 28, 28, 1)),
                                        axis=0)
                    y_batch = np.append(train_labels_tmp[j * config.batch_size:], [[0., 0., 1.]] * ((j + 1) * config.batch_size - n_t),
                                        axis=0)

                # calculate statistics values
                _, stat_vals = sess.run(d_x, feed_dict={
                    x_placeholder: x_batch,
                    y_placeholder: y_batch})
                for k in range(config.batch_size):
                    if k + j * config.batch_size < n_t:
                        was_generated = (k + j * config.batch_size >= int(n_t * (1 - gen_share))) and not sample_from_orig
                        validation_success = stat_vals[k, 0] > validation_crit_val or skip_validation
                        if validation_success:
                            train_data.append(x_batch[k])
                            train_labels.append(y_batch[k])

                        if validation_success and was_generated:
                            false_neg[trial] += 1
            # validation for SD and Cramer
            val_cramer = cl.cramer_test(train_data_tmp, train_labels_tmp, dcgan.data_X[:1000], dcgan.data_y[:1000])
            val_sd = cl.sd_val_success(train_data_tmp, train_labels_tmp)
            train_data_cramer = []
            train_labels_cramer = []
            train_data_sd = []
            train_labels_sd = []
            for i in range(n_t):
                if val_cramer[i]:
                    train_data_cramer.append(train_data_tmp[i])
                    train_labels_cramer.append(train_labels_tmp[i])
                    if i > n_orig:
                        false_neg_cramer[trial] += 1
                elif i < n_orig:
                    false_pos_cramer[trial] += 1

                if val_sd[i]:
                    train_data_sd.append(train_data_tmp[i])
                    train_labels_sd.append(train_labels_tmp[i])
                    if i > n_orig:
                        false_neg_sd[trial] += 1
                elif i < n_orig:
                    false_pos_sd[trial] += 1

            # calculate classification error
            for c_var in c_vars:
                sess.run(c_var.initializer)
            for opt_var in optimizer.variables():
                sess.run(opt_var.initializer)
            for i in range(500):
                _ = sess.run(c_optim, feed_dict={
                    x_c_placeholder: train_data,
                    y_c_placeholder: train_labels
                })
            predicted_labels = sess.run(c_train, feed_dict={
                x_c_placeholder: dcgan.test_data
            })

            err = 1-accuracy_score(np.argmax(dcgan.test_labels, axis=1), np.argmax(predicted_labels, axis=1))
            errs.append(err)

            # calculate classification error: Cramer
            for c_var in c_vars:
                sess.run(c_var.initializer)
            for opt_var in optimizer.variables():
                sess.run(opt_var.initializer)
            for i in range(500):
                _ = sess.run(c_optim, feed_dict={
                    x_c_placeholder: train_data_cramer,
                    y_c_placeholder: train_labels_cramer
                })
            predicted_labels = sess.run(c_train, feed_dict={
                x_c_placeholder: dcgan.test_data
            })

            err = 1 - accuracy_score(np.argmax(dcgan.test_labels, axis=1), np.argmax(predicted_labels, axis=1))
            errs_cramer.append(err)

            # calculate classification error: SD
            for c_var in c_vars:
                sess.run(c_var.initializer)
            for opt_var in optimizer.variables():
                sess.run(opt_var.initializer)
            for i in range(500):
                _ = sess.run(c_optim, feed_dict={
                    x_c_placeholder: train_data_sd,
                    y_c_placeholder: train_labels_sd
                })
            predicted_labels = sess.run(c_train, feed_dict={
                x_c_placeholder: dcgan.test_data
            })

            err = 1 - accuracy_score(np.argmax(dcgan.test_labels, axis=1), np.argmax(predicted_labels, axis=1))
            errs_sd.append(err)

        print('error=', np.mean(errs) * 100, '+-', (np.std(errs) * 1.96 / np.sqrt(n_trials)) * 100)

        if not skip_validation:
            print('false negative=', (np.mean(false_neg) / (n_t * gen_share)) * 100,
                '+-', (np.std(false_neg) * 100 / (n_t * gen_share)) * 1.96 / np.sqrt(n_trials))
            print('false positive=', false_pos / n_orig * 100)
        print('Cramer:')
        print('Cramer error=', np.mean(errs_cramer) * 100, '+-', (np.std(errs_cramer) * 1.96 / np.sqrt(n_trials)) * 100)

        if not skip_validation:
            print('Cramer false negative=', (np.mean(false_neg_cramer) / (n_t * gen_share)) * 100,
                  '+-', (np.std(false_neg_cramer) * 100 / (n_t * gen_share)) * 1.96 / np.sqrt(n_trials))
            print('Cramer false positive=', (np.mean(false_pos_cramer) / n_orig) * 100,
                  '+-', (np.std(false_pos_cramer) * 100 / n_orig) * 1.96 / np.sqrt(n_trials))
        print('SD:')
        print('SD error=', np.mean(errs_sd) * 100, '+-', (np.std(errs_sd) * 1.96 / np.sqrt(n_trials)) * 100)

        if not skip_validation:
            print('SD false negative=', (np.mean(false_neg_sd) / (n_t * gen_share)) * 100,
                  '+-', (np.std(false_neg_sd) * 100 / (n_t * gen_share)) * 1.96 / np.sqrt(n_trials))
            print('SD false positive=', (np.mean(false_pos_sd) / n_orig) * 100,
                  '+-', (np.std(false_pos_sd) * 100 / n_orig) * 1.96 / np.sqrt(n_trials))


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w


# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, int(1E+8)))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
