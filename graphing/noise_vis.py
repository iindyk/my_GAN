import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


noise_norms = [3., 13.]
im_dir = '/home/iindyk/PycharmProjects/my_GAN/images/for_graphs/'

(x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.mnist.load_data()
x_train_all, x_test_all = x_train_all/255., x_test_all/255.
x_train_all, x_test_all = x_train_all-np.mean(x_train_all), x_test_all-np.mean(x_test_all)

for i in range(len(y_train_all)):
    if y_train_all[i] == 0:
        break
im0 = x_train_all[i, :, :]
noise1 = np.random.uniform(low=-1., high=1., size=(28, 28))
noise1 = (noise1/np.linalg.norm(noise1))*(noise_norms[0])
im1 = x_train_all[i]+noise1

noise2 = np.random.uniform(low=-1., high=1., size=(28, 28))
noise2 = (noise2/np.linalg.norm(noise2))*(noise_norms[1])
im2 = x_train_all[i]+noise2


def show_images(images, cols=1, titles=None):
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        plt.imshow(image, cmap='gray_r')
        # Turn off tick labels
        a.set_yticklabels([])
        a.set_xticklabels([])
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        # border width
        a.spines['top'].set_linewidth(2)
        a.spines['right'].set_linewidth(2)
        a.spines['bottom'].set_linewidth(2)
        a.spines['left'].set_linewidth(2)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


show_images([im0, im1, im2])


