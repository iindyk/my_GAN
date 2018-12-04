import numpy as np
from generator import *
from discriminator import *
from data import *
import matplotlib.pyplot as plt


nit = 10000
nit_dis = 100
batch_size = 128
step = 1e-3


train_data, train_labels, test_data, test_labels = get_mnist_data()  # get data
n_train = len(train_labels)
n, m = np.shape(train_data)

# generator and discriminator profiles
gen_layers_profile = [{'type': 'linear', 'in': batch_size, 'out': 64},
                      {'type': 'ReLu'},
                      {'type': 'linear', 'in': 64, 'out': 128},
                      {'type': 'ReLu'},
                      {'type': 'linear', 'in': 128, 'out': 256},
                      {'type': 'ReLu'},
                      {'type': 'linear', 'in': 256, 'out': m},
                      {'type': 'tanh'}]

dis_layers_profile = [{'type': 'linear', 'in': m, 'out': 128},
                      {'type': 'ReLu'},
                      {'type': 'linear', 'in': 128, 'out': 64},
                      {'type': 'ReLu'},
                      {'type': 'linear', 'in': 64, 'out': 1},
                      {'type': 'sigmoid'}]

# initialize
generator = Generator(gen_layers_profile, train_data, train_labels, test_data, test_labels)
discriminator = Discriminator(dis_layers_profile)

n_gen_layers = len(generator.layers)
n_dis_layers = len(discriminator.layers)

dis_losses = []
gen_losses = []

d_real = []
d_generated = []

for i in range(nit):
    for j in range(nit_dis):
        z = np.random.normal(size=batch_size)
        d_real = train_data[np.random.randint(n_train, size=batch_size)]
        d_generated = generator.act(z)

        # make gradient descent step for each linear layer parameters for discriminator
        dis_gradients = {}

        # calculate discriminator gradients for current state
        for layer_id in range(n_dis_layers):
            if discriminator.layers[layer_id] == 'linear':
                dis_gradients[layer_id] = discriminator.loss_grad(layer_id, d_real, d_generated)

        # perform gradient descent for gradient
        for layer_id in range(n_dis_layers):
            if discriminator.layers[layer_id] == 'linear':
                discriminator.layers[layer_id].params['w'] += step*dis_gradients[layer_id]['w']
                discriminator.layers[layer_id].params['b'] += step*dis_gradients[layer_id]['b']

    # make gradient descent step for each linear layer parameters for generator
    z = np.random.normal(size=batch_size)
    gen_gradients = {}

    # calculate generator gradients in current state
    for layer_id in range(n_gen_layers):
        if generator.layers[layer_id] == 'linear':
            gen_gradients[layer_id] = generator.loss_grad(layer_id, discriminator, z)

    # perform gradient descent for generator
    for layer_id in range(n_gen_layers):
        if generator.layers[layer_id] == 'linear':
            generator.layers[layer_id].params['w'] -= step * gen_gradients[layer_id]['w']
            generator.layers[layer_id].params['b'] -= step * gen_gradients[layer_id]['b']

    # append losses for graphing
    dis_losses.append(discriminator.loss(d_real, d_generated))
    gen_losses.append(generator.loss(discriminator, z))

# graphing
_, ax = plt.subplots()
iterations = np.arange(nit)
ax.plot(iterations, gen_losses, '-', label='generator loss')
ax.plot(iterations, dis_losses, '-', label='discriminator loss')
plt.legend(loc='upper left')
plt.show()
