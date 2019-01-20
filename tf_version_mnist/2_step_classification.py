from tf_version_mnist.discriminator import *
from tf_version_mnist.generator import *
import tensorflow as tf
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle


n_trials = 100             # number of trials
n_t = 100                  # total number of training points
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
labels_to_use = [5, 6]
model_to_load = '01-19_21:50_1.0'
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
        y_train.append([1., 0])
    elif y_train_all[i] == labels_to_use[1]:
        x_train.append(x_train_all[i])
        y_train.append([0., 1.])

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

# placeholder for input images to the discriminator
x_placeholder = tf.placeholder("float", shape=[batch_size, im_dim, im_dim, channel])
y_placeholder = tf.placeholder("float", shape=[batch_size, y_dim])
# placeholder for input noise vectors to the generator
z_placeholder = tf.placeholder(tf.float32, [None, z_dim])

discriminator = Discriminator1(batch_size, y_dim)
generator = Generator1(None, batch_size, y_dim, im_dim, channel,
                       initial_x_train=[], initial_y_train=[], x_test=[], y_test=[])

# d_x will hold discriminator prediction probabilities
_, d_x = discriminator.act(x_placeholder, y_placeholder)
# g_z holds the generated images
g_z = generator.act(z_placeholder, y_placeholder)

saver = tf.train.Saver()
with tf.Session() as sess:
    # restore model from file
    saver.restore(sess, model_path)
    print("Model restored.")
    false_pos = 0
    false_neg = 0

    errs = []
    for trial in range(n_trials):
        indices = np.random.randint(low=int(n_t * (1 - gen_share)), high=len(y_train), size=int(n_t * gen_share))
        additional_y_train = y_train[indices, :]
        if sample_from_orig:
            additional_x_train = np.reshape(x_train[indices, :, :], (-1, 784))
        else:
            # todo
            additional_x_train = np.empty((0, 784), np.float32)
            for j in range((int(n_t * gen_share))//batch_size+1):
                if (j+1)*batch_size <= (int(n_t * gen_share)):
                    y_batch = additional_y_train[j*batch_size:(j+1)*batch_size]
                else:
                    y_batch = np.append(additional_y_train[j*batch_size:], [[0., 1.]]*((j+1)*batch_size-int(n_t * gen_share)),
                                        axis=0)

                z_batch = np.random.uniform(-1, 1, size=[batch_size, z_dim])
                additional_x_train = np.append(additional_x_train,
                                               np.reshape(sess.run(g_z, feed_dict={z_placeholder: z_batch,
                                                                                   y_placeholder: y_batch}),
                                                          newshape=(-1, 784)), axis=0)

        train_data = np.append(np.reshape(x_train[:int(n_t*(1-gen_share))], newshape=(-1, 784)),
                               additional_x_train[:int(n_t * gen_share)], axis=0)
        train_labels = np.append(y_train[:int(n_t*(1-gen_share))], additional_y_train[:int(n_t * gen_share)], axis=0)
        train_labels = [(1. if train_labels[k, 0] == 1. else -1.) for k in range(len(train_labels))]
        svc = svm.LinearSVC(loss='hinge').fit(train_data, train_labels)
        errs.append(1 - svc.score(x_test, y_test))


print('mean=', np.mean(errs))
print('stdev=', np.std(errs))

