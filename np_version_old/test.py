from np_version_old.discriminator import *
from np_version_old.data import *
import matplotlib.pyplot as plt

# training parameters
nit = 10000
batch_size = 32
step = .01
momentum_alpha = .5

# data fetch
train_data, train_labels, test_data, test_labels = get_toy_data(1000, 2)
n, m = np.shape(train_data)

for i in range(n):
    train_data[i, 0] = train_data[i, 1]


# discriminator profile
dis_layers_profile = [{'type': 'linear', 'in': m, 'out': 240},
                      {'type': 'ReLu', 'in': 240, 'out': 240},
                      {'type': 'linear', 'in': 240, 'out': 1},
                      {'type': 'sigmoid', 'in': 1, 'out': 1}]

# initialize
discriminator = Discriminator(dis_layers_profile)

n_dis_layers = len(discriminator.layers)

dis_losses = []

d_real = []
d_generated = []

# initialize discriminator old gradients
dis_updates_old = {}
for layer_id in range(n_dis_layers):
    if discriminator.layers[layer_id].type_ == 'linear':
        dis_updates_old[layer_id] = {'w': 0., 'b': 0.}

# training
for i in range(nit):
    d_real = train_data[np.random.randint(n, size=batch_size)]
    d_generated = np.random.normal(scale=1./np.sqrt(m/2.), size=(batch_size, m))

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

    # append losses for graphing
    dl = discriminator.loss(d_real, d_generated)
    dis_losses.append(dl)

    if i % 100 == 0:
        print('Step %i: Discriminator Loss: %f' % (i, dl))

# graphing
_, ax = plt.subplots()
iterations = np.arange(nit)
ax.plot(iterations, dis_losses, '-', label='discriminator loss')
plt.legend(loc='upper left')
plt.show()
