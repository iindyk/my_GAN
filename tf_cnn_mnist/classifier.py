import tensorflow as tf
from statsmodels import robust
from sklearn.metrics import accuracy_score
import numpy as np
import sklearn.svm as svm


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
    kappa = 0.98
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
    crit_val = 30.5
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
        if len(valid_9_ind)!=0:
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


def roni_val_c(sess, args, images, labels, valid_set, valid_labels, train_set, train_labels):
    # fit train set
    x_c_placeholder, y_c_placeholder, c_train = args
    c_loss_train = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=c_train, labels=y_c_placeholder))
    t_vars = tf.trainable_variables()
    c_vars = [var for var in t_vars if 'cl_' in var.name]
    optimizer = tf.train.AdamOptimizer(0.001)
    c_optim = optimizer.minimize(c_loss_train, var_list=c_vars)
    for c_var in c_vars:
        sess.run(c_var.initializer)
    for opt_var in optimizer.variables():
        sess.run(opt_var.initializer)

    for i in range(1000):
        _ = sess.run(c_optim, feed_dict={
            x_c_placeholder: train_set,
            y_c_placeholder: train_labels
        })
    predicted_labels = sess.run(c_train, feed_dict={
        x_c_placeholder: valid_set
    })
    err = 1 - accuracy_score(np.argmax(valid_labels, axis=1), np.argmax(predicted_labels, axis=1))
    new_train_set = np.array(train_set)
    new_train_labels = np.array(train_labels)
    valid_success = []
    for new_image, new_label in zip(images, labels):
        _new_train_set = np.append(new_train_set, np.reshape(new_image, newshape=[1, 28, 28, 1]), axis=0)
        _new_train_labels = np.append(new_train_labels, np.reshape(new_label, newshape=[1, 3]), axis=0)
        for i in range(10):
            _ = sess.run(c_optim, feed_dict={
                x_c_placeholder: new_train_set,
                y_c_placeholder: new_train_labels
            })
        predicted_labels = sess.run(c_train, feed_dict={
            x_c_placeholder: valid_set
        })
        new_err = 1 - accuracy_score(np.argmax(valid_labels, axis=1), np.argmax(predicted_labels, axis=1))
        valid_success.append(new_err <= err)
        if new_err <= err:
            new_train_set = _new_train_set
            new_train_labels = _new_train_labels
        err = new_err
    return valid_success


def roni_val_svm(images, labels, valid_set, valid_labels, train_set, train_labels):
    # fit svm
    svc = svm.LinearSVC(loss='hinge').fit(train_set, train_labels)
    err = 1 - svc.score(valid_set, valid_labels)
    new_train_set = np.array(train_set)
    new_train_labels = np.array(train_labels)
    valid_success = []
    for new_image, new_label in zip(images, labels):
        _new_train_set = np.append(new_train_set, np.reshape(new_image, newshape=[1, 784]), axis=0)
        _new_train_labels = np.append(new_train_labels, np.reshape(new_label, newshape=[1]), axis=0)
        svc.fit(_new_train_set, _new_train_labels)
        new_err = 1 - svc.score(valid_set, valid_labels)
        valid_success.append(new_err <= err)
        if new_err <= err:
            new_train_set = _new_train_set
            new_train_labels = _new_train_labels
        err = new_err
    return valid_success
