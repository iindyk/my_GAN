import matplotlib.pyplot as plt


im_dir = '/home/iindyk/PycharmProjects/my_GAN/images/for_graphs/1vs0/'
alphas = ['0.0', '0.05', '0.5', '1.0', '10.0']
n_r = 2
n_c = len(alphas)


fig, axs = plt.subplots(n_r, n_c)
fig.suptitle('Multiple images')

images = []
for i in range(n_r):
    for j in range(n_c):
        images.append(axs[i, j].imshow(data, cmap=cmap))
        axs[i, j].label_outer()


plt.show()
