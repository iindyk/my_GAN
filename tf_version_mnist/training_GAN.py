from tf_version_mnist.discriminator import *
from tf_version_mnist.generator import *
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime as dt


nit = 20000
kit_discriminator = 1
display_step = 500
save_image_step = 1000
learning_rate = 0.02
momentum = 0.3
z_dim = 100
batch_size = 32
save_model = True

(x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.mnist.load_data()
x_train_all, x_test_all = x_train_all/255., x_test_all/255.
x_train_all, x_test_all = x_train_all-np.mean(x_train_all), x_test_all-np.mean(x_test_all)

x_train = []
y_train = []
# take only images of 0 and 9
for i in range(len(y_train_all)):
    if y_train_all[i] == 7:
        x_train.append(x_train_all[i])
        y_train.append(1.)
    elif y_train_all[i] == 1:
        x_train.append(x_train_all[i])
        y_train.append(-1.)

x_train = np.array(x_train)
n = len(x_train)

# placeholder for input images to the discriminator
x_placeholder = tf.placeholder("float", shape=[None, 28, 28, 1])
# placeholder for input noise vectors to the generator
z_placeholder = tf.placeholder(tf.float32, [None, z_dim])

discriminator = Discriminator0()
generator = Generator0(initial_x_train=x_train[:100, :, :], initial_y_train=y_train[:100],
                      x_test=x_train[:1000, :, :], y_test=y_train[:1000])

# d_x will hold discriminator prediction probabilities for the real MNIST images
d_x = discriminator.act(x_placeholder)
# g_z holds the generated images
g_z = generator.act(z_placeholder, batch_size, z_dim)
# d_g holds discriminator prediction probabilities for generated images
d_g = discriminator.act(g_z, reuse=True)

# generator and discriminator losses
# ensure forward compatibility: function needs to have logits and labels args explicitly used
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_g, labels=tf.ones_like(d_g)))
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_x, labels=tf.ones_like(d_x)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_g, labels=tf.zeros_like(d_g)))
d_loss = d_loss_real + d_loss_fake

# get discriminator and generator variables lists
train_vars = tf.trainable_variables()
d_vars = [var for var in train_vars if 'd_' in var.name]
g_vars = [var for var in train_vars if 'g_' in var.name]

n_d_vars = len(d_vars)
n_g_vars = len(g_vars)

# get discriminator and generator gradients lists
d_grad = tf.gradients(xs=d_vars, ys=d_loss)
g_grad = tf.gradients(xs=g_vars, ys=g_loss)

# gradient descent step description
new_d_vars = []
new_g_vars = []
d_accumulation = []
new_d_accumulation = []
g_accumulation = []
new_g_accumulation = []
for i in range(n_d_vars):
    d_accumulation.append(tf.get_variable('accum_d'+str(i), shape=d_grad[i].get_shape(), trainable=False))
    new_d_accumulation.append(d_accumulation[i].assign(momentum * d_accumulation[i] + (1.-momentum)*d_grad[i]))
    new_d_vars.append(d_vars[i].assign(d_vars[i] - learning_rate * d_accumulation[i]))

for i in range(n_g_vars):
    g_accumulation.append(tf.get_variable('accum_g' + str(i), shape=g_grad[i].get_shape(), trainable=False))
    new_g_accumulation.append(g_accumulation[i].assign(momentum * g_accumulation[i] + (1.-momentum) * g_grad[i]))
    new_g_vars.append(g_vars[i].assign(g_vars[i] - learning_rate * g_accumulation[i]))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# run simultaneous gradient descent
with tf.Session() as sess:
    sess.run(init)
    d_losses = []
    g_losses = []
    for epoch in range(nit):
        d_loss_val = 0
        for k in range(kit_discriminator):
            # sample noise and real data
            z_batch = np.random.normal(0., 1., size=[batch_size, z_dim])
            real_image_batch = np.reshape(x_train[np.random.randint(n, size=batch_size)], [batch_size, 28, 28, 1])

            # make gradient descent step for discriminator
            _, _, d_loss_val = sess.run([new_d_vars, new_d_accumulation, d_loss],
                                        feed_dict={z_placeholder: z_batch, x_placeholder: real_image_batch})

        z_batch = np.random.normal(0., 1., size=[batch_size, z_dim])
        # make gradient descent step for generator
        _, _, g_loss_val = sess.run([new_g_vars, new_g_accumulation, g_loss], feed_dict={z_placeholder: z_batch})

        # memorize losses for graphing
        d_losses.append(d_loss_val)
        g_losses.append(g_loss_val)

        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), " discriminator loss=", "{:.9f}".format(d_loss_val),
                  " generator loss=", "{:.9f}".format(g_loss_val))

        if (epoch + 1) % save_image_step == 0:
            sample_image = generator.act(z_placeholder, 1, z_dim, reuse=True)
            z_batch = np.random.normal(0., 1., size=[1, z_dim])
            temp = sess.run(sample_image, feed_dict={z_placeholder: z_batch})
            img_array = temp.squeeze()
            plt.imsave('/home/iindyk/PycharmProjects/my_GAN/images/generated'+str(epoch)+'.jpeg',
                       img_array, cmap='gray_r')
    if save_model:
        # Save model to file
        time = dt.datetime.now().strftime("%m-%d-%H:%M")
        save_path = saver.save(sess, '/home/iindyk/PycharmProjects/my_GAN/saved_models/model'+time+'.ckpt')
        print("Model saved in path: %s" % save_path)

# show optimization progress
_, ax = plt.subplots()
iter_arr = np.arange(nit)
ax.plot(iter_arr, g_losses, '-', label='generator loss')
ax.plot(iter_arr, d_losses, '-', label='discriminator loss')
plt.legend(loc='lower right')

plt.show()
