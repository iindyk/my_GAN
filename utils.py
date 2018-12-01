import numpy as np


def linear(w, x, b):
    return w*x+b


def relu(x):
    return np.maximum(x, 0)
