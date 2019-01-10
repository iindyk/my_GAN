import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


im_dir = '/home/iindyk/PycharmProjects/my_GAN/images/for_graphs/1vs0/'
labels = ['0', '1']
alphas = ['0.05', '0.5', '0.75', '1.0', '10.0']
im_list = []
for l in labels:
    for a in alphas:
        im_list.append(Image.open(im_dir+a+'_'+l+'.jpg'))
show_images(im_list, cols=2)


