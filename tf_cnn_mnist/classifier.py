import tensorflow as tf


def c(images):

    # Network Parameters
    n_hidden_1 = 256 # 1st layer number of neurons
    n_hidden_2 = 256 # 2nd layer number of neurons
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