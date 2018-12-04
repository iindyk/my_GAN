from data import *


train_data, train_labels, test_data, test_labels = get_mnist_data()

print(np.shape(train_data))
print(np.shape(train_labels))