import pickle
import matplotlib.pyplot as plt
import numpy as np

f_read1 = open("/home/iindyk/PycharmProjects/my_GAN/ImageNet/data.p", "rb")
exist_data = pickle.load(f_read1)
f_read1.close()
train_data = exist_data['train_data']
train_labels = exist_data['train_labels']

img = np.swapaxes(np.reshape(np.array(train_data[0]), newshape=(64, 64, 3), order='F'), 0, 1)
plt.imsave('/home/iindyk/PycharmProjects/my_GAN/0img'+str(train_labels[0])+'.jpeg', img)

img = np.swapaxes(np.reshape(np.array(train_data[3]), newshape=(64, 64, 3), order='F'), 0, 1)
plt.imsave('/home/iindyk/PycharmProjects/my_GAN/1img'+str(train_labels[3])+'.jpeg', img)

img = np.swapaxes(np.reshape(np.array(train_data[2]), newshape=(64, 64, 3), order='F'), 0, 1)
plt.imsave('/home/iindyk/PycharmProjects/my_GAN/2img'+str(train_labels[2])+'.jpeg', img)