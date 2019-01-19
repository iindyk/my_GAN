import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def show_images(images, cols=1, titles=None):

    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        plt.gray()
        plt.imshow(image)
        a.set_title(title)
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


im_dir = '/home/iindyk/PycharmProjects/my_GAN/images/for_graphs/5vs6/'
labels = ['5', '6']
alphas = ['0.0', '0.25', '0.5', '0.75', '1.0', '10.0']
im_list = []
titles = []
for l in labels:
    for a in alphas:
        im_list.append(Image.open(im_dir+a+'_'+l+'.jpg'))
        titles.append(l+'s, alpha='+a)
show_images(im_list, cols=2, titles=['']*len(im_list))


