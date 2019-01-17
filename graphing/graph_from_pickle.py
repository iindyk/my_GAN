import pickle
import matplotlib.pyplot as plt
import numpy as np

data_pickle = open("/home/iindyk/PycharmProjects/my_GAN/results.pickle", "rb")
data_dict = pickle.load(data_pickle)

_, ax = plt.subplots()
n_batches = 100
for key, val in data_dict.items():
    ax.plot(np.arange(n_batches), val, '-')
    print(key)

plt.show()