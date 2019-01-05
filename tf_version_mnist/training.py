from tf_version_mnist.discriminator import *
from tf_version_mnist.generator import *
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime as dt
import os


nit = 1000
kit_discriminator = 1
display_step = 100
save_image_step = 500
learning_rate = 0.02
momentum = 0.2
z_dim = 100
batch_size = 64
save_model = True
save_dir = '/home/iindyk/PycharmProjects/my_GAN/saved_models_my_GAN/'
y_dim = 2
channel = 1     # todo

(x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.mnist.load_data()
x_train_all, x_test_all = x_train_all/255., x_test_all/255.
x_train_all, x_test_all = x_train_all-np.mean(x_train_all), x_test_all-np.mean(x_test_all)

x_train = []
y_train = []
# take only images of 0 and 9
for i in range(len(y_train_all)):
    if y_train_all[i] == 7:
        x_train.append(x_train_all[i])
        y_train.append([1., 0])
    elif y_train_all[i] == 1:
        x_train.append(x_train_all[i])
        y_train.append([0., 1.])

x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
n = len(x_train)

# placeholder for input images to the discriminator
x_placeholder = tf.placeholder("float", shape=[batch_size, 28, 28, 1])
y_placeholder = tf.placeholder("float", shape=[batch_size, y_dim])
# placeholder for input noise vectors to the generator
z_placeholder = tf.placeholder(tf.float32, [None, z_dim])

discriminator = Discriminator1(batch_size, y_dim)
generator = Generator1(batch_size, y_dim, 28, channel, initial_x_train=x_train[:100, :, :], initial_y_train=y_train[:100],
                       x_test=x_train[:1000, :, :], y_test=y_train[:1000])

# d_x will hold discriminator prediction probabilities for the real MNIST images
_, d_x = discriminator.act(x_placeholder, y_placeholder)
# g_z holds the generated images
g_z = generator.act(z_placeholder, y_placeholder)
# d_g holds discriminator prediction probabilities for generated images
_, d_g = discriminator.act(g_z, y_placeholder, reuse=True)

# generator and discriminator losses
# ensure forward compatibility: function needs to have logits and labels args explicitly used
g_loss_p1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_g, labels=tf.ones_like(d_g)))
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_x, labels=tf.ones_like(d_x)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_g, labels=tf.zeros_like(d_g)))
d_loss = d_loss_real + d_loss_fake

# get discriminator and generator variables lists
train_vars = tf.trainable_variables()
d_vars = [var for var in train_vars if 'dis' in var.name]
g_vars = [var for var in train_vars if 'gen' in var.name]

n_d_vars = len(d_vars)
n_g_vars = len(g_vars)

# get discriminator and generator gradients lists
d_grad = tf.gradients(xs=d_vars, ys=d_loss)
g_grad_p1 = tf.gradients(xs=g_vars, ys=g_loss_p1)

g_grad_p2_1 = tf.py_func(generator.adv_obj_and_grad, [g_z, y_placeholder], tf.float32)

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

g_jacob_p2 = []
for j in range(batch_size):
    print(j)
    g_jacob_p2.append([])
    for k in range(generator.output_size):
        g_jacob_p2[j].append([])
        for l in range(generator.output_size):
            g_jacob_p2[j][k].append(tf.gradients(xs=g_vars, ys=g_z[j, k, l]))
g_jacob_p2 = tf.convert_to_tensor(g_jacob_p2, dtype=tf.float32)
g_grad_p2 = []
for i in range(n_g_vars):
    print(i)
    g_grad_p2.append(tf.matmul(tf.reshape(g_grad_p2_1, [-1]), tf.reshape(g_jacob_p2[:, :, :, i]), transpose_a=True))
    g_accumulation.append(tf.get_variable('accum_g' + str(i), shape=g_grad_p1[i].get_shape(), trainable=False))
    new_g_accumulation.append(g_accumulation[i].assign(momentum * g_accumulation[i] +
                              (1.-momentum) * (g_grad_p1[i]+generator.alpha*g_grad_p2[i])))
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
        # sample noise and real data
        z_batch = np.random.uniform(-1, 1, size=[batch_size, z_dim])
        perm = np.random.randint(n, size=batch_size)
        real_image_batch = np.reshape(x_train[perm], [batch_size, 28, 28, 1])
        real_labels_batch = y_train[perm]

        # make gradient descent step for discriminator
        _, _, d_loss_val = sess.run([new_d_vars, new_d_accumulation, d_loss],
                                    feed_dict={z_placeholder: z_batch, x_placeholder: real_image_batch,
                                               y_placeholder: real_labels_batch})

        # make gradient descent step for generator
        _, _, g_loss_p1_val = sess.run([new_g_vars, new_g_accumulation, g_loss_p1],
                                       feed_dict={z_placeholder: z_batch, y_placeholder: real_labels_batch})

        # memorize losses for graphing
        d_losses.append(d_loss_val)
        g_losses.append(g_loss_p1_val+generator.alpha*generator.prob_approx)

        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), " discriminator loss=", "{:.9f}".format(d_loss_val),
                  " generator loss=", "{:.9f}".format(g_loss_p1_val+generator.alpha*generator.prob_approx))

        if (epoch + 1) % save_image_step == 0:
            sample_image = generator.act(z_placeholder, y_placeholder, reuse=True)
            temp = sess.run(sample_image, feed_dict={z_placeholder: z_batch, y_placeholder: real_labels_batch})
            img_array = temp[0, :, :, 0]
            plt.imsave('/home/iindyk/PycharmProjects/my_GAN/images/generated'+str(epoch)+'.jpeg',
                       img_array, cmap='gray_r')
    if save_model:
        # Save model
        time = dt.datetime.now().strftime("%m-%d_%H:%M")
        os.mkdir(save_dir + time)
        config_file = open(save_dir+time+'/config.txt', 'w+')
        config_file.write('nit='+str(nit)+'\nkit_discriminator='+str(kit_discriminator)+'\nlearning rate='+str(learning_rate)+
                          '\nmomentum='+str(momentum)+'\nz_dim='+str(z_dim)+'\nbatch size='+str(batch_size) +
                          '\ngenerator alpha='+str(generator.alpha)+'\ngenerator a='+str(generator.a))
        config_file.close()
        os.mkdir(save_dir + time + '/generated_images')
        save_path = saver.save(sess, save_dir+time+'/model.ckpt')
        print("Model saved in path: %s" % save_path)

# show optimization progress
_, ax = plt.subplots()
iter_arr = np.arange(nit)
ax.plot(iter_arr, g_losses, '-', label='generator loss')
ax.plot(iter_arr, d_losses, '-', label='discriminator loss')
plt.legend(loc='lower right')

plt.show()
