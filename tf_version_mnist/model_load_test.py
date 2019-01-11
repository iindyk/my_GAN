from tf_version_mnist.discriminator import *
from tf_version_mnist.generator import *
import tensorflow as tf
import matplotlib.pyplot as plt


z_dim = 100
batch_size = 64
generate_batches = 2
y_dim = 2
channel = 1
im_dim = 28
model_to_load = '01-10_19:28'
model_path = '/home/iindyk/PycharmProjects/my_GAN/saved_models_CGAN/' + model_to_load + '/model.ckpt'
generated_images_path = '/home/iindyk/PycharmProjects/my_GAN/saved_models_CGAN/' + model_to_load + '/generated_images'
gen_alpha = 1.
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
x_placeholder = tf.placeholder("float", shape=[batch_size, im_dim, im_dim, channel])
y_placeholder = tf.placeholder("float", shape=[batch_size, y_dim])
# placeholder for input noise vectors to the generator
z_placeholder = tf.placeholder(tf.float32, [None, z_dim])

discriminator = Discriminator1(batch_size, y_dim)
generator = Generator1(gen_alpha, batch_size, y_dim, im_dim, channel, initial_x_train=x_train[:100, :, :], initial_y_train=y_train[:100],
                       x_test=x_train[:1000, :, :], y_test=y_train[:1000])

# d_x will hold discriminator prediction probabilities for the real MNIST images
_, d_x = discriminator.act(x_placeholder, y_placeholder)
# g_z holds the generated images
g_z = generator.act(z_placeholder, y_placeholder)
# d_g holds discriminator prediction probabilities for generated images
_, d_g = discriminator.act(g_z, y_placeholder, reuse=True)

saver = tf.train.Saver()
with tf.Session() as sess:
    # restore model from file
    saver.restore(sess, model_path)
    print("Model restored.")
    sample_image = generator.act(z_placeholder, y_placeholder, reuse=True)
    for i in range(generate_batches):
        z_batch = np.random.uniform(-1, 1, size=[batch_size, z_dim])
        y_batch = y_train[np.random.randint(n, size=batch_size)]
        temp = sess.run(sample_image, feed_dict={z_placeholder: z_batch, y_placeholder: y_batch})
        for j in range(batch_size):
            img_array = temp[j, :, :, 0]
            label = '7' if y_batch[j, 0] == 1. else '1'
            plt.imsave(generated_images_path + '/' + label + '_batch' + str(i) + 'im' + str(j) + '.jpeg', img_array,
                       cmap='gray_r')
