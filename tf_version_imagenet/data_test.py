import pickle
import matplotlib.pyplot as plt
import numpy as np

f_read1 = open("/home/iindyk/PycharmProjects/my_GAN/ImageNet/data.p", "rb")
exist_data = pickle.load(f_read1)
f_read1.close()
train_data = exist_data['train_data']
train_labels = exist_data['train_labels']

n = len(train_labels)

for i in range(n):
    img = np.swapaxes(np.reshape(np.array(train_data[i]), newshape=(64, 64, 3), order='F'), 0, 1)
    plt.imsave('/home/iindyk/PycharmProjects/my_GAN/ImageNet/' + str(train_labels[i]) + '/'+str(i)+'.JPEG', img)
