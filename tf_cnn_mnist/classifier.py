import tensorflow as tf
from statsmodels import robust
import numpy as np


def c(images):

    # Network Parameters
    n_hidden_1 = 256 # 1st layer number of neurons
    n_hidden_2 = 8 # 2nd layer number of neurons
    n_input = 784 # MNIST data input (img shape: 28*28)
    n_classes = 3 # MNIST total classes (0-9 digits)

    # Store layers weight & bias
    weights = {
        'h1': tf.get_variable('cl_h1', [n_input, n_hidden_1], initializer=tf.truncated_normal_initializer(0.02)),
        'h2': tf.get_variable('cl_h2', [n_hidden_1, n_hidden_2], initializer=tf.truncated_normal_initializer(0.02)),
        'out': tf.get_variable('cl_out', [n_hidden_2, n_classes], initializer=tf.truncated_normal_initializer(0.02))
    }
    biases = {
        'b1': tf.get_variable('cl_b1', [n_hidden_1], initializer=tf.truncated_normal_initializer(0.02)),
        'b2': tf.get_variable('cl_b2', [n_hidden_2], initializer=tf.truncated_normal_initializer(0.02)),
        'out': tf.get_variable('cl_bout', [n_classes], initializer=tf.truncated_normal_initializer(0.02))
    }

    # Create model
    def multilayer_perceptron(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(tf.reshape(x, shape=(-1, 784)), weights['h1']), biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    return multilayer_perceptron(images)


def sd_val_success(images, labels):
    kappa = 0.99
    p = 50
    _images = np.reshape(images, newshape=(-1, 784))
    n = len(images)
    # construct P
    directions = []
    for i in range(p):
        # take direction between 2 random points in the training set
        indices = np.random.randint(low=0, high=n, size=2)
        new_dir = _images[indices[0]] - _images[indices[1]]
        norm_ = np.linalg.norm(new_dir)
        if norm_ > 1e-5:
            new_dir /= norm_
        directions.append(new_dir)

    directions = np.array(directions)

    # separate training set
    train_dataset_7 = []
    indices_7 = []
    train_dataset_8 = []
    indices_8 = []
    train_dataset_9 = []
    indices_9 = []

    for i in range(n):
        if labels[i, 0] == 1.:
            train_dataset_7.append(_images[i])
            indices_7.append(i)
        elif labels[i, 1] == 1.:
            train_dataset_8.append(_images[i])
            indices_8.append(i)
        else:
            train_dataset_9.append(_images[i])
            indices_9.append(i)

    train_dataset_7 = np.array(train_dataset_7)
    n_7 = len(train_dataset_7)
    n_7_refined = int(np.floor(n_7 * kappa))
    train_dataset_8 = np.array(train_dataset_8)
    n_8 = len(train_dataset_8)
    n_8_refined = int(np.floor(n_8 * kappa))
    train_dataset_9 = np.array(train_dataset_9)
    n_9 = len(train_dataset_9)
    n_9_refined = int(np.floor(n_9 * kappa))

    # calculate SD outlyingness for 7
    sd_7 = np.zeros(n_7)
    for i in range(n_7):
        for a in directions:
            sd = abs(a @ train_dataset_7[i] - np.median(train_dataset_7 @ a)) / robust.scale.mad(
                train_dataset_7 @ a)
            if sd > sd_7[i]:
                sd_7[i] = sd

    # calculate SD outlyingness for 8
    sd_8 = np.zeros(n_8)
    for i in range(n_8):
        for a in directions:
            sd = abs(a @ train_dataset_8[i] - np.median(train_dataset_8 @ a)) / robust.scale.mad(
                train_dataset_8 @ a)
            if sd > sd_8[i]:
                sd_8[i] = sd

    # calculate SD outlyingness for 9
    sd_9 = np.zeros(n_9)
    for i in range(n_9):
        for a in directions:
            sd = abs(a @ train_dataset_9[i] - np.median(train_dataset_9 @ a)) / robust.scale.mad(
                train_dataset_9 @ a)
            if sd > sd_9[i]:
                sd_9[i] = sd

    indices_refined_7 = np.array(indices_7)[sd_7.argsort()[:n_7_refined]]
    indices_refined_8 = np.array(indices_8)[sd_8.argsort()[:n_8_refined]]
    indices_refined_9 = np.array(indices_9)[sd_9.argsort()[:n_9_refined]]

    validation_success = []
    for i in range(n):
        validation_success.append((i in indices_refined_7) or (i in indices_refined_8)
                                  or (i in indices_refined_9))

    return validation_success


part_stat_7 = 0
part_stat_8 = 0
part_stat_9 = 0
valid_7_ind = []
valid_8_ind = []
valid_9_ind = []


def cramer_test(images, labels, valid_set, valid_indices):
    crit_val = 28
    global part_stat_7, part_stat_8, part_stat_9, valid_7_ind, valid_8_ind, valid_9_ind
    if part_stat_7 == 0:
        for i in range(len(valid_indices)):
            if valid_indices[i, 0] == 1:
                valid_7_ind.append(i)
            elif valid_indices[i, 1] == 1:
                valid_8_ind.append(i)
            else:
                valid_9_ind.append(i)
        for i in valid_7_ind:
            for j in valid_7_ind:
                part_stat_7 += np.linalg.norm(valid_set[i] - valid_set[j])
        part_stat_7 /= 2 * (len(valid_7_ind) ** 2)
        for i in valid_8_ind:
            for j in valid_8_ind:
                part_stat_8 += np.linalg.norm(valid_set[i] - valid_set[j])
        part_stat_8 /= 2 * (len(valid_8_ind) ** 2)
        for i in valid_9_ind:
            for j in valid_9_ind:
                part_stat_9 += np.linalg.norm(valid_set[i] - valid_set[j])
        part_stat_9 /= 2 * (len(valid_9_ind) ** 2)

    validation_succes = []
    for k in range(len(labels)):
        if labels[k, 0] == 1:
            # 7
            test_stat_7 = part_stat_7
            for j in valid_7_ind:
                test_stat_7 += np.linalg.norm(images[k] - valid_set[j]) / (len(valid_7_ind))
            test_stat_7 *= len(valid_7_ind) / (1 + len(valid_7_ind))
            validation_succes.append(test_stat_7 < crit_val)
        elif labels[k, 1] == 1:
            # 8
            test_stat_8 = part_stat_8
            for j in valid_8_ind:
                test_stat_8 += np.linalg.norm(images[k] - valid_set[j]) / (len(valid_8_ind))
            test_stat_8 *= len(valid_8_ind) / (1 + len(valid_8_ind))
            validation_succes.append(test_stat_8 < crit_val)
        else:
            # 9
            test_stat_9 = part_stat_9
            for j in valid_9_ind:
                test_stat_9 += np.linalg.norm(images[k] - valid_set[j]) / (len(valid_9_ind))
            test_stat_9 *= len(valid_9_ind) / (1 + len(valid_9_ind))
            validation_succes.append(test_stat_9 < crit_val)

    return validation_succes