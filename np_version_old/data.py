import numpy as np
import tensorflow as tf


def get_toy_data(n, m):
    random_flips = 0.1
    dataset = np.random.uniform(0, 1, (n, m))
    labels = []
    for i in range(n):
        if sum(dataset[i, :]) > 0.5*m:
            labels.append(1)
        else:
            labels.append(-1)
    # random attack
    indices = np.random.randint(n, size=int(random_flips*n))
    for i in indices:
        if labels[i] == 1:
            labels[i] = -1
        else:
            labels[i] = 1
    return dataset[:int(0.5*n), :], labels[:int(0.5*n)], dataset[int(0.5*n):, :], labels[int(0.5*n):]


def get_mnist_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    return np.reshape(x_train, newshape=(len(y_train), 784)), y_train, \
        np.reshape(x_test, newshape=(len(y_test), 784)), y_test
