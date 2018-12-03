import numpy as np
from generator import *
from discriminator import *
from data import *
import matplotlib.pyplot as plt


n = 200
m = 20
nit = 100
nit_dis = 10
sample_size = 10
step = 1e-4
gen_layers_profile = [{'type': 'linear', 'in': sample_size, 'out': 64},
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

train_data, train_labels, test_data, test_labels = get_toy_data(n, m)   # get data
n_train = len(train_labels)

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
        z = np.random.normal(size=sample_size)
        d_real = train_data[np.random.randint(n_train, size=sample_size)]
        d_generated = generator.act(z)

        # make gradient descent step for each linear layer parameters for discriminator

        for layer_id in range(n_dis_layers):
            if discriminator.layers[layer_id] == 'linear':
                discriminator.layers[layer_id].params['w'] -= \
                    step*discriminator.loss_grad(layer_id, d_real, d_generated)
                discriminator.layers[layer_id].params['b'] -= \
                    step*discriminator.loss_grad(layer_id, d_real, d_generated)

    # make gradient descent step for each linear layer parameters for generator
    z = np.random.normal(size=sample_size)
    for layer_id in range(n_gen_layers):
        if generator.layers[layer_id] == 'linear':
            gen_grad = generator.loss_grad(layer_id, discriminator, z)
            generator.layers[layer_id].params['w'] -= \
                step*gen_grad['w']
            generator.layers[layer_id].params['b'] -= \
                step*gen_grad['b']

    # append losses for graphing
    dis_losses.append(discriminator.loss(d_real, d_generated))
    gen_losses.append(generator.loss(discriminator, z))

# graphing
_, ax = plt.subplots()
iterations = np.arange(nit)
ax.plot(iterations, gen_losses, '-', label='generator loss')
ax.plot(iterations, dis_losses, '-', label='discriminator loss')