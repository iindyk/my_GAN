from tf_version_mnist.discriminator import *
from tf_version_mnist.generator import *
import tensorflow as tf
import sklearn.svm as svm
import matplotlib.pyplot as plt


n_trials = 100             # number of trials
n_t = 100                   # total number of training points
z_dim = 100                 # generator input dimension
batch_size = 64             # size of input batch
y_dim = 2                   # number of classes
channel = 1                 # number of channels, MNIST is grayscale
im_dim = 28                 # dimension of 1 side of image
gen_share = 0.4             # % of training set to be generated
noise_norm = 5.           # norm of a random noise
data_shift = 0
validation_crit_val = 3.17
skip_validation = False
labels_to_use = [0, 1]
model_to_load = '01-15_18:47_10.0'
model_path = '/home/iindyk/PycharmProjects/my_GAN/saved_models_my_GAN/' + model_to_load + '/model.ckpt'


(x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.mnist.load_data()
x_train_all, x_test_all = x_train_all/255., x_test_all/255.
x_train_all, x_test_all = x_train_all-np.mean(x_train_all), x_test_all-np.mean(x_test_all)

# take only images of digits from labels_to_use
x_train = []
y_train = []

n_orig = int(n_t*(1-gen_share))

for i in range(len(y_train_all)):
    if y_train_all[i] == labels_to_use[0]:
        x_train.append(x_train_all[i])
        y_train.append([1., 0.])
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

    errs = []
    false_pos = []
    false_neg = []
    for trial in range(n_trials):
        indices = np.random.randint(low=int(n_t * (1 - gen_share))+data_shift, high=len(y_train), size=int(n_t * gen_share))
        additional_y_train = y_train[indices, :]
        noise = np.random.uniform(low=-1., high=1., size=(int(n_t * gen_share), 784))
        # normalize noise
        noise = np.array([(noise[i]/np.linalg.norm(noise[i]))*noise_norm for i in range(len(noise))])
        #noise = (noise/np.linalg.norm(noise))*(int(n_t * gen_share)*noise_norm)
        additional_x_train = np.reshape(x_train[indices, :, :], (-1, 784)) + noise

        train_data_tmp = np.append(np.reshape(x_train[data_shift:int(n_t*(1-gen_share))+data_shift], newshape=(-1, 784)),
                                   additional_x_train[:int(n_t * gen_share)], axis=0)
        train_labels_tmp = np.append(y_train[data_shift:int(n_t*(1-gen_share))+data_shift],
                                     additional_y_train[:int(n_t * gen_share)], axis=0)

        train_data = []
        train_labels = []

        false_pos.append(0)
        false_neg.append(0)

        # validation
        for j in range(n_t//batch_size+1):
            if (j+1)*batch_size <= n_t:
                x_batch = train_data_tmp[j*batch_size:(j+1)*batch_size]
                y_batch = train_labels_tmp[j*batch_size:(j+1)*batch_size]
            else:
                x_batch = np.append(train_data_tmp[j*batch_size:], np.zeros(((j+1)*batch_size-n_t, 784)), axis=0)
                y_batch = np.append(train_labels_tmp[j*batch_size:], [[0., 1.]]*((j+1)*batch_size-n_t), axis=0)

            # calculate statistics values
            stat_vals = sess.run(d_x, feed_dict={
                x_placeholder: np.reshape(x_batch, newshape=[batch_size, im_dim, im_dim, channel]),
                y_placeholder: y_batch})
            for k in range(batch_size):
                if k+j*batch_size < n_t:
                    was_generated = (k+j*batch_size > int(n_t*(1-gen_share)))
                    validation_success = stat_vals[k, 0] > validation_crit_val or skip_validation
                    if validation_success:
                        train_data.append(np.reshape(x_batch[k], newshape=784))
                        train_labels.append(1. if y_batch[k, 0] == 1. else -1.)

                    if validation_success and was_generated:
                        false_neg[trial] += 1
                    if not validation_success and not was_generated:
                        false_pos[trial] += 1

        svc = svm.LinearSVC(loss='hinge').fit(train_data, train_labels)
        errs.append(1 - svc.score(x_test, y_test))

# show noisy images
fig = plt.figure(figsize=(2, 2))
columns = 2
rows = 2
for i in range(1, columns*rows + 1):
    img = np.reshape(additional_x_train[-i-1], newshape=(28, 28))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap='gray_r')
plt.show()

print('error=', np.mean(errs)*100, '+-', (np.std(errs)*1.96/np.sqrt(n_trials))*100)

if not skip_validation:
    print('false negative=', (np.mean(false_neg)/(n_t*gen_share))*100,
          '+-', (np.std(false_neg)*100/(n_t*gen_share))*1.96/np.sqrt(n_trials))
    print('false positive=', (np.mean(false_pos)/(n_t*(1-gen_share)))*100,
          '+-', (np.std(false_pos)*100/(n_t*(1-gen_share)))*1.96/np.sqrt(n_trials))

