from tf_version_mnist.discriminator import *
from tf_version_mnist.generator import *
import tensorflow as tf
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle


n_trials = 1000             # number of trials
n_t = 1000                  # total number of training points
z_dim = 100                 # generator input dimension
batch_size = 64             # size of input batch
n_batches = 100             # number of generated batches = number of real training batches
y_dim = 2                   # number of classes
channel = 1                 # number of channels, MNIST is grayscale
im_dim = 28                 # dimension of 1 side of image
gen_share = 0.3             # % of training set to be generated
sample_from_orig = True     # sample generated data from original
validation_crit_val = 0.7
skip_validation = True
labels_to_use = [0, 1]
model_to_load = '01-15_19:17_10.0'
model_path = '/home/iindyk/PycharmProjects/my_GAN/saved_models_my_GAN/' + model_to_load + '/model.ckpt'


(x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.mnist.load_data()
x_train_all, x_test_all = x_train_all/255., x_test_all/255.
x_train_all, x_test_all = x_train_all-np.mean(x_train_all), x_test_all-np.mean(x_test_all)

# take only images of digits from labels_to_use
x_train = []
y_train = []

n_orig = int(n_t*(1-gen_share))     # beware of index out of bounds

for i in range(len(y_train_all)):
    if y_train_all[i] == labels_to_use[0]:
        x_train.append(x_train_all[i])
        y_train.append(1)
    elif y_train_all[i] == labels_to_use[1]:
        x_train.append(x_train_all[i])
        y_train.append(-1)

x_test = []
y_test = []
for i in range(len(y_test_all)):
    if y_test_all[i] == labels_to_use[0]:
        x_test.append(x_test_all[i])
        y_test.append(1)
    elif y_test_all[i] == labels_to_use[1]:
        x_test.append(x_test_all[i])
        y_test.append(-1)

x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
x_test = np.reshape(np.array(x_test, dtype=np.float32), newshape=(-1, 784))
y_test = np.array(y_test, dtype=np.float32)

errs = []
for trial in range(n_trials):
    if sample_from_orig:
        indices = np.random.randint(low=int(n_t*(1-gen_share)), high=len(y_train), size=int(n_t*gen_share))
        additional_x_train = np.reshape(x_train[indices, :, :], (-1, 784))
        additional_y_train = y_train[indices]
    else:
        # todo
        pass

    train_data = np.append(np.reshape(x_train[:int(n_t*(1-gen_share))], newshape=(-1, 784)), additional_x_train, axis=0)
    train_labels = np.append(y_train[:int(n_t*(1-gen_share))], additional_y_train)
    svc = svm.SVC(kernel='linear').fit(train_data, train_labels)
    errs.append(1 - accuracy_score(y_test, svc.predict(x_test)))

print('mean=', np.mean(errs))
print('stdev=', np.std(errs))

