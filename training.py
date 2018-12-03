import numpy as np
from generator import *
from discriminator import *


nit = 1000
data = []   # todo: get data
nit_dis = 100
sample_size = 100
step = 1e-4

# initialize
generator = Generator([],[],[],[],[])   # todo: fill
discriminator = Discriminator([])

n_gen_layers = len(generator.layers)
n_dis_layers = len(discriminator.layers)

for i in range(nit):
    for j in range(nit_dis):
        z = np.random.normal(size=sample_size)
        d_real = data[np.random.randint(len(data), size=sample_size)]

        # make gradient descent step for each linear layer parameters for discriminator

        for layer_id in range(n_dis_layers):
            if discriminator.layers[layer_id] == 'linear':
                discriminator.layers[layer_id].params['w'] -= \
                    step*discriminator.loss_grad(layer_id, d_real, generator.act(z))
                discriminator.layers[layer_id].params['b'] -= \
                    step*discriminator.loss_grad(layer_id, d_real, generator.act(z))

    # make gradient descent step for each linear layer parameters for generator
    z = np.random.normal(size=sample_size)
    for layer_id in range(n_gen_layers):
        if generator.layers[layer_id] == 'linear':
            gen_grad = generator.loss_grad(layer_id, discriminator, z)
            generator.layers[layer_id].params['w'] -= \
                step*gen_grad['w']
            generator.layers[layer_id].params['b'] -= \
                step*gen_grad['b']

# todo: graphing