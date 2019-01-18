from tf_version_mnist.discriminator import *
from tf_version_mnist.generator import *
import tensorflow as tf
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle


z_dim = 100                 # generator input dimension
batch_size = 64             # size of input batch
n_batches = 100             # number of generated batches = number of real training batches
y_dim = 2                   # number of classes
channel = 1                 # number of channels, MNIST is grayscale
im_dim = 28                 # dimension of 1 side of image
gen_multiplier = 101        # 1/gen_multiplier share of training data will be generated
validation_crit_val = 0.7
skip_validation = True
labels_to_use = [0, 1]
gen_alpha = 10.0
model_to_load = '01-15_19:17_10.0'
model_path = '/home/iindyk/PycharmProjects/my_GAN/saved_models_my_GAN/' + model_to_load + '/model.ckpt'
generated_images_path = '/home/iindyk/PycharmProjects/my_GAN/saved_models_my_GAN/' + model_to_load + '/generated_images'
(x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.mnist.load_data()
x_train_all, x_test_all = x_train_all/255., x_test_all/255.
x_train_all, x_test_all = x_train_all-np.mean(x_train_all), x_test_all-np.mean(x_test_all)

# take only images of digits from labels_to_use
x_train = []
y_train = []
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
        y_test.append([1., 0])
    elif y_test_all[i] == labels_to_use[1]:
        x_test.append(x_test_all[i])
        y_test.append([0., 1.])
    #if i >= 999:                            # todo
    #    break


x_train = np.array(x_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
x_test = np.array(x_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)
y_test_1d = [(1. if y_test[j, 0] == 1. else -1.) for j in range(len(y_test))]
n = len(x_train)

# placeholder for input images to the discriminator
x_placeholder = tf.placeholder("float", shape=[batch_size, im_dim, im_dim, channel])
y_placeholder = tf.placeholder("float", shape=[batch_size, y_dim])
# placeholder for input noise vectors to the generator
z_placeholder = tf.placeholder(tf.float32, [None, z_dim])

discriminator = Discriminator1(batch_size, y_dim)
generator = Generator1(None, batch_size, y_dim, im_dim, channel,
                       initial_x_train=x_train[:100], initial_y_train=y_train[:100], x_test=x_test, y_test=y_test)

# d_x will hold discriminator prediction probabilities
_, d_x = discriminator.act(x_placeholder, y_placeholder)
# g_z holds the generated images
g_z = generator.act(z_placeholder, y_placeholder)

online_training_data = np.reshape(x_train[:10], newshape=(10, 784))
online_training_labels = y_train[:10]
errs = []
saver = tf.train.Saver()
with tf.Session() as sess:
    # restore model from file
    saver.restore(sess, model_path)
    print("Model restored.")
    false_pos = 0
    false_neg = 0
    for i in range(n_batches):
        online_training_labels_1d = [(1. if online_training_labels[j, 0] == 1. else -1.)
                                     for j in range(len(online_training_labels))]
        svc = svm.SVC(kernel='linear').fit(online_training_data, online_training_labels_1d)
        errs.append(1 - accuracy_score(y_test_1d, svc.predict(np.reshape(x_test, newshape=(len(x_test), 784)))))

        y_batch = y_train[100+i*batch_size: 100+(i+1)*batch_size]
        data_was_generated = i % gen_multiplier == 0
        if data_was_generated:
            z_batch = np.random.uniform(-1, 1, size=[batch_size, z_dim])
            x_batch = sess.run(g_z, feed_dict={z_placeholder: z_batch, y_placeholder: y_batch})
        else:
            x_batch = x_train[100+i*batch_size: 100+(i+1)*batch_size]

        # validation
        stat_vals = sess.run(d_x, feed_dict={x_placeholder: np.reshape(x_batch, newshape=[batch_size, im_dim, im_dim, channel]),
                                             y_placeholder: y_batch})
        accepted = 0
        for j in range(batch_size):
            validation_success = stat_vals[j, 0] < validation_crit_val or skip_validation
            if validation_success:
                online_training_data = np.append(online_training_data, np.reshape(x_batch[j], newshape=(1, 784)), axis=0)
                online_training_labels = np.append(online_training_labels, [y_batch[j]], axis=0)
                accepted += 1
                if data_was_generated:
                    false_neg += 1
            elif not data_was_generated:
                false_pos += 1
        print('Batch', i, ': data was generated =', data_was_generated, '; accepted images =', accepted)

print('false_positive='+str(false_pos/(n_batches*batch_size))+';false_negative='+str(false_neg/(n_batches*batch_size)))

# update pickle
data_pickle_read = open("/home/iindyk/PycharmProjects/my_GAN/results.pickle", "rb")
data_dict = pickle.load(data_pickle_read)
data_pickle_read.close()
new_key = 'gen_alpha='+str(gen_alpha)+';gen_multiplier='+str(gen_multiplier) + \
          ';validation_crit_val='+str(validation_crit_val)+';labels_to_use='+str(labels_to_use) + \
          ';skip_validation='+str(skip_validation)+';false_positive='+str(false_pos/(n_batches*batch_size)) + \
          ';false_negative='+str(false_neg/(n_batches*batch_size))
data_dict[new_key] = errs
data_pickle_write = open("/home/iindyk/PycharmProjects/my_GAN/results.pickle", "wb")
pickle.dump(data_dict, data_pickle_write)
data_pickle_write.close()

_, ax = plt.subplots()
ax.plot(np.arange(n_batches), errs, '-', label='test prediction errors')
plt.show()


