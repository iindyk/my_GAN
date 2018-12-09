import numpy as np
from generator import *
from discriminator import *
from data import *
import matplotlib.pyplot as plt

# training parameters
nit = 1000
nit_dis = 1
batch_size = 32
step = .01
momentum_alpha = .5
noise_dim = 1

# data fetch
train_data, train_labels, test_data, test_labels = get_toy_data(10000, 3)
n, m = np.shape(train_data)
for i in range(n):
    train_data[i, 0] = train_data[i, 2]
    train_data[i, 1] = train_data[i, 2]


# generator and discriminator profiles
gen_layers_profile = [{'type': 'linear', 'in': noise_dim, 'out': 128},
                      {'type': 'ReLu', 'in': 128, 'out': 128},
                      {'type': 'linear', 'in': 128, 'out': m},
                      {'type': 'sigmoid', 'in': m, 'out': m}]

dis_layers_profile = [{'type': 'linear', 'in': m, 'out': 240},
                      {'type': 'ReLu', 'in': 240, 'out': 240},
                      {'type': 'linear', 'in': 240, 'out': 1},
                      {'type': 'sigmoid', 'in': 1, 'out': 1}]

# initialize
generator = Generator(gen_layers_profile, train_data, train_labels, test_data, test_labels)
discriminator = Discriminator(dis_layers_profile)

n_gen_layers = len(generator.layers)
n_dis_layers = len(discriminator.layers)

dis_losses = []
gen_losses = []

d_real = []
d_generated = []

# initialize generator and discriminator old gradients
dis_updates_old = {}
for layer_id in range(n_dis_layers):
    if discriminator.layers[layer_id].type_ == 'linear':
        dis_updates_old[layer_id] = {'w': 0., 'b': 0., 'x': 0.}

gen_updates_old = {}
for layer_id in range(n_gen_layers):
    if generator.layers[layer_id].type_ == 'linear':
        gen_updates_old[layer_id] = {'w': 0., 'b': 0., 'x': 0.}

# training
for i in range(nit):
    for j in range(nit_dis):
        z = np.random.normal(scale=1./np.sqrt(m/2.), size=(batch_size, noise_dim))
        d_real = train_data[np.random.randint(n, size=batch_size)]
        d_generated = generator.act(z)

        # make gradient descent step for each linear layer parameters for discriminator
        # calculate discriminator gradients for current state
        dis_gradients = discriminator.loss_grad(d_real, d_generated)

        # perform gradient ascent for discriminator
        for layer_id in range(n_dis_layers):
            if discriminator.layers[layer_id].type_ == 'linear':
                # momentum learning rule
                upd_w = step*dis_gradients[layer_id]['w'] + \
                    step*momentum_alpha*dis_updates_old[layer_id]['w']
                upd_b = step*dis_gradients[layer_id]['b'] + \
                    step*momentum_alpha*dis_updates_old[layer_id]['b']
                dis_updates_old[layer_id] = {}
                discriminator.layers[layer_id].params['w'] += upd_w
                dis_updates_old[layer_id]['w'] = upd_w
                discriminator.layers[layer_id].params['b'] += upd_b
                dis_updates_old[layer_id]['b'] = upd_b

    # make gradient descent step for each linear layer parameters for generator
    z = np.random.normal(scale=1./np.sqrt(m/2.), size=(batch_size, noise_dim))

    # calculate generator gradients in current state
    gen_gradients = generator.loss_grad(discriminator, z)

    # perform gradient descent for generator
    for layer_id in range(n_gen_layers):
        if generator.layers[layer_id].type_ == 'linear':
            # momentum learning rule
            upd_w = -step * gen_gradients[layer_id]['w'] + \
                step * momentum_alpha * gen_updates_old[layer_id]['w']
            upd_b = -step * gen_gradients[layer_id]['b'] + \
                step * momentum_alpha * gen_updates_old[layer_id]['b']
            gen_updates_old[layer_id] = {}
            generator.layers[layer_id].params['w'] += upd_w
            gen_updates_old[layer_id]['w'] = upd_w
            generator.layers[layer_id].params['b'] += upd_b
            gen_updates_old[layer_id]['b'] = upd_b

    # append losses for graphing
    gl = generator.loss(discriminator, z)
    dl = discriminator.loss(d_real, d_generated)
    dis_losses.append(dl)
    gen_losses.append(gl)

    if i % 100 == 0:
        print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

# graphing
_, ax = plt.subplots()
iterations = np.arange(nit)
ax.plot(iterations, gen_losses, '-', label='generator loss')
ax.plot(iterations, dis_losses, '-', label='discriminator loss')
plt.legend(loc='lower right')
plt.show()
