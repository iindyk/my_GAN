from tf_version_imagenet.discriminator import *
from tf_version_imagenet.generator import *
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime as dt
import os
import resource
import pickle


nit = 3000              # number of training iterations
display_step = 500      # progress display step
save_image_step = 500   # generated image save step (in save_dir folder)
learning_rate = 0.02    # gradient descent learning step
momentum = 0.2          # parameter for momentum learning rule
z_dim = 100             # generator input dimension
batch_size = 64         # size of training sample used for training
im_dim = 64             # one-side dimension of square image
save_model = True       # if save_model generator and discriminator will be saved in save_dir folder
save_dir = '/home/iindyk/PycharmProjects/my_GAN/saved_models_DCGAN_imagenet/'
y_dim = 3               # number of classes used for training
channel = 3             # number of channels of image
labels_to_use = [1, 10, 999]

# setting max heap size limit
rsrc = resource.RLIMIT_DATA
_, hard = resource.getrlimit(rsrc)
resource.setrlimit(rsrc, ((1024**3)*10, hard))
soft, hard = resource.getrlimit(rsrc)
print('Soft RAM limit set to:', soft/(1024**3), 'GB')

# clearing computation graph
tf.reset_default_graph()

# uploading data
#(x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.cifar10.load_data()
#x_train_all, x_test_all = x_train_all/255., x_test_all/255.
#x_train_all, x_test_all = x_train_all-np.mean(x_train_all), x_test_all-np.mean(x_test_all)

f_read1 = open("/home/iindyk/PycharmProjects/my_GAN/ImageNet/data.p", "rb")
exist_data = pickle.load(f_read1)
f_read1.close()
x_train_all = exist_data['train_data']
y_train_all = exist_data['train_labels']

# take only images of digits from labels_to_use
x_train = []
y_train = []
for i in range(len(y_train_all)):
    if y_train_all[i] == labels_to_use[0]:
        x_train.append(np.swapaxes(np.reshape(np.array(x_train_all[i]), newshape=(64, 64, 3), order='F'), 0, 1)/255.)
        y_train.append([1., 0., 0.])
    elif y_train_all[i] == labels_to_use[1]:
        x_train.append(np.swapaxes(np.reshape(np.array(x_train_all[i]), newshape=(64, 64, 3), order='F'), 0, 1)/255.)
        y_train.append([0., 1., 0.])
    elif y_train_all[i] == labels_to_use[2]:
        x_train.append(np.swapaxes(np.reshape(np.array(x_train_all[i]), newshape=(64, 64, 3), order='F'), 0, 1)/255.)
        y_train.append([0., 0., 1.])

#x_test = []
#y_test = []
#for i in range(len(y_test_all)):
#    if y_test_all[i] == labels_to_use[0]:
#        x_test.append(x_test_all[i])
#        y_test.append([1., 0., 0.])
#    elif y_test_all[i] == labels_to_use[1]:
#        x_test.append(x_test_all[i])
#        y_test.append([0., 1., 0.])
#    elif y_test_all[i] == labels_to_use[2]:
#        x_test.append(x_test_all[i])
#        y_test.append([0., 0., 1.])


x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
n = len(x_train)

# placeholder for input images to the discriminator
x_placeholder = tf.placeholder("float", shape=[batch_size, im_dim, im_dim, channel])
y_placeholder = tf.placeholder("float", shape=[batch_size, y_dim])
# placeholder for input noise vectors to the generator
z_placeholder = tf.placeholder(tf.float32, [None, z_dim])

discriminator = Discriminator(batch_size, y_dim)
generator = Generator(0, batch_size, y_dim, im_dim, channel, initial_x_train=x_train[:100, :, :],
                       initial_y_train=y_train[:100], x_test=x_train[:1000, :, :], y_test=y_train[:1000])

# d_x will hold discriminator prediction probabilities for the real MNIST images
_, d_x = discriminator.act(x_placeholder, y_placeholder)
# g_z holds the generated images
g_z = generator.act(z_placeholder, y_placeholder)
# d_g holds discriminator prediction probabilities for generated images
_, d_g = discriminator.act(g_z, y_placeholder, reuse=True)

# generator and discriminator losses
# ensure forward compatibility: function needs to have logits and labels args explicitly used
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_g, labels=tf.ones_like(d_g)))
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
        # sample noise and real data
        z_batch = np.random.uniform(-1, 1, size=[batch_size, z_dim])
        perm = np.random.randint(n, size=batch_size)
        real_image_batch = np.reshape(x_train[perm], [batch_size, im_dim, im_dim, channel])
        real_labels_batch = y_train[perm]

        # make gradient descent step for discriminator
        _, _, d_loss_val = sess.run([new_d_vars, new_d_accumulation, d_loss],
                                    feed_dict={z_placeholder: z_batch, x_placeholder: real_image_batch,
                                               y_placeholder: real_labels_batch})

        # make gradient descent step for generator
        _, _, g_loss_val = sess.run([new_g_vars, new_g_accumulation, g_loss],
                                    feed_dict={z_placeholder: z_batch, y_placeholder: real_labels_batch})

        # memorize losses for graphing
        d_losses.append(d_loss_val)
        g_losses.append(g_loss_val)

        if (epoch + 1) % display_step == 0:
            print(dt.datetime.now().strftime("%H:%M"), "Epoch:", '%04d' % (epoch + 1),
                  " discriminator loss=", "{:.9f}".format(d_loss_val),
                  " generator loss=", "{:.9f}".format(g_loss_val))

        if (epoch + 1) % save_image_step == 0:
            sample_image = generator.act(z_placeholder, y_placeholder, reuse=True)
            temp = sess.run(sample_image, feed_dict={z_placeholder: z_batch, y_placeholder: real_labels_batch})
            img_array = temp[0, :, :, :]
            plt.imsave('/home/iindyk/PycharmProjects/my_GAN/images/generated'+str(epoch+1)+'.jpeg',
                       img_array)
    if save_model:
        # Save model
        time = dt.datetime.now().strftime("%m-%d_%H:%M")
        os.mkdir(save_dir + time)
        config_file = open(save_dir+time+'/config.txt', 'w+')
        config_file.write('nit='+str(nit)+'\nlearning rate='+str(learning_rate)+
                          '\nmomentum='+str(momentum)+'\nz_dim='+str(z_dim)+'\nbatch size='+str(batch_size))
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
