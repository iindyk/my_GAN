from tf_version_mnist.discriminator import *
from tf_version_mnist.generator import *
import tensorflow as tf
import sklearn.svm as svm
import tf_cnn_mnist.classifier as cl


n_trials = 1000             # number of trials
n_t = 100                   # total number of training points
z_dim = 100                 # generator input dimension
batch_size = 64             # size of input batch
y_dim = 2                   # number of classes
channel = 1                 # number of channels, MNIST is grayscale
im_dim = 28                 # dimension of 1 side of image
gen_share = 0.4             # % of training set to be generated
sample_from_orig = False    # sample generated data from original
data_shift = 0
validation_crit_val = 4.
skip_validation = False
labels_to_use = [5, 6]
model_to_load = '01-16_19:25_0.5'
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
    false_neg = []
    errs_cramer = []
    false_neg_cramer = []
    false_pos_cramer = []
    errs_sd = []
    false_neg_sd = []
    false_pos_sd = []
    errs_roni = []
    false_neg_roni = []
    false_pos_roni = []

    # false positive rate calculation
    false_pos = 0
    n_leftover = int(np.ceil(n_orig / batch_size) * batch_size - n_orig)
    if n_leftover != 0:
        _dt_tmp = np.append(np.reshape(x_train[data_shift:n_orig + data_shift], newshape=(-1, 784)),
                            np.zeros((batch_size - n_orig, 784)), axis=0)
        _lb_tmp = np.append(y_train[data_shift:n_orig + data_shift], [[0., 1.]] * (batch_size - n_orig), axis=0)
    else:
        _dt_tmp = np.reshape(x_train[data_shift:n_orig + data_shift], newshape=(-1, 784))
        _lb_tmp = y_train[data_shift:n_orig + data_shift]

    n_runs = int(np.ceil(n_orig / batch_size))
    val = np.empty((0, 1))
    for k in range(n_runs):
        _, val_new = sess.run(d_x, feed_dict={
            x_placeholder: np.reshape(_dt_tmp[batch_size * k:batch_size * (k + 1)],
                                      newshape=[batch_size, im_dim, im_dim, channel]),
            y_placeholder: _lb_tmp[batch_size * k:batch_size * (k + 1)]})
        val = np.append(val, val_new, axis=0)

    for k in range(n_orig):
        if val[k, 0] <= validation_crit_val:
            false_pos += 1

    for trial in range(n_trials):
        indices = np.random.randint(low=int(n_t * (1 - gen_share))+data_shift, high=len(y_train), size=int(n_t * gen_share))
        additional_y_train = y_train[indices, :]
        if sample_from_orig:
            additional_x_train = np.reshape(x_train[indices, :, :], (-1, 784))
        else:
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

        train_data_tmp = np.append(np.reshape(x_train[data_shift:int(n_t*(1-gen_share))+data_shift], newshape=(-1, 784)),
                                   additional_x_train[:int(n_t * gen_share)], axis=0)
        train_labels_tmp = np.append(y_train[data_shift:int(n_t*(1-gen_share))+data_shift],
                                     additional_y_train[:int(n_t * gen_share)], axis=0)

        train_data = []
        train_labels = []

        false_neg.append(0)
        false_neg_cramer.append(0)
        false_pos_cramer.append(0)
        false_neg_sd.append(0)
        false_pos_sd.append(0)
        false_neg_roni.append(0)
        false_pos_roni.append(0)

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
                    was_generated = (k+j*batch_size >= int(n_t*(1-gen_share))) and not sample_from_orig
                    validation_success = stat_vals[k, 0] > validation_crit_val or skip_validation
                    if validation_success:
                        train_data.append(np.reshape(x_batch[k], newshape=784))
                        train_labels.append(1. if y_batch[k, 0] == 1. else -1.)

                    if validation_success and was_generated:
                        false_neg[trial] += 1

        # validation for SD, Cramer and RONI
        val_cramer = cl.cramer_test(train_data_tmp, train_labels_tmp, np.reshape(x_train[:1000], (-1, 784)), y_train[:1000])
        val_sd = cl.sd_val_success(train_data_tmp, train_labels_tmp)
        val_roni = cl.roni_val_svm(train_data_tmp, train_labels_tmp, train_data[:100], train_labels[:100],
                                    train_data_tmp[:n_orig], train_labels_tmp[:n_orig])
        train_data_cramer = []
        train_labels_cramer = []
        train_data_sd = []
        train_labels_sd = []
        train_data_roni = []
        train_labels_roni = []
        for i in range(n_t):
            if val_cramer[i]:
                train_data_cramer.append(train_data_tmp[i])
                train_labels_cramer.append(1. if train_labels_tmp[i, 0] == 1. else -1.)
                if i > n_orig:
                    false_neg_cramer[trial] += 1
            elif i < n_orig:
                false_pos_cramer[trial] += 1

            if val_sd[i]:
                train_data_sd.append(train_data_tmp[i])
                train_labels_sd.append(1. if train_labels_tmp[i, 0] == 1. else -1.)
                if i > n_orig:
                    false_neg_sd[trial] += 1
            elif i < n_orig:
                false_pos_sd[trial] += 1

            if val_roni[i]:
                train_data_roni.append(train_data_tmp[i])
                train_labels_roni.append(1. if train_labels_tmp[i, 0] == 1. else -1.)
                if i > n_orig:
                    false_neg_roni[trial] += 1
            elif i < n_orig:
                false_pos_roni[trial] += 1

        svc_sd = svm.LinearSVC(loss='hinge').fit(train_data_sd, train_labels_sd)
        errs_sd.append(1-svc_sd.score(x_test, y_test))

        svc_cramer = svm.LinearSVC(loss='hinge').fit(train_data_cramer, train_labels_cramer)
        errs_cramer.append(1-svc_cramer.score(x_test, y_test))

        svc_roni = svm.LinearSVC(loss='hinge').fit(train_data_roni, train_labels_roni)
        errs_roni.append(1 - svc_roni.score(x_test, y_test))

        svc = svm.LinearSVC(loss='hinge').fit(train_data, train_labels)
        errs.append(1 - svc.score(x_test, y_test))


print('error=', np.mean(errs)*100, '+-', (np.std(errs)*1.96/np.sqrt(n_trials))*100)

if not skip_validation:
    print('false negative=', (np.mean(false_neg)/(n_t*gen_share))*100,
          '+-', (np.std(false_neg)*100/(n_t*gen_share))*1.96/np.sqrt(n_trials))
    print('false positive=', false_pos / n_orig * 100)

print('Cramer:')
print('Cramer error=', np.mean(errs_cramer) * 100, '+-', (np.std(errs_cramer) * 1.96 / np.sqrt(n_trials)) * 100)

if not skip_validation:
    print('Cramer false negative=', (np.mean(false_neg_cramer) / (n_t * gen_share)) * 100,
                  '+-', (np.std(false_neg_cramer) * 100 / (n_t * gen_share)) * 1.96 / np.sqrt(n_trials))
    print('Cramer false positive=', (np.mean(false_pos_cramer) / n_orig) * 100,
                  '+-', (np.std(false_pos_cramer) * 100 / n_orig) * 1.96 / np.sqrt(n_trials))
print('SD:')
print('SD error=', np.mean(errs_sd) * 100, '+-', (np.std(errs_sd) * 1.96 / np.sqrt(n_trials)) * 100)

if not skip_validation:
    print('SD false negative=', (np.mean(false_neg_sd) / (n_t * gen_share)) * 100,
                  '+-', (np.std(false_neg_sd) * 100 / (n_t * gen_share)) * 1.96 / np.sqrt(n_trials))
    print('SD false positive=', (np.mean(false_pos_sd) / n_orig) * 100,
                  '+-', (np.std(false_pos_sd) * 100 / n_orig) * 1.96 / np.sqrt(n_trials))

print('RONI:')
print('RONI error=', np.mean(errs_roni) * 100, '+-', (np.std(errs_roni) * 1.96 / np.sqrt(n_trials)) * 100)

if not skip_validation:
    print('SD false negative=', (np.mean(false_neg_roni) / (n_t * gen_share)) * 100,
                  '+-', (np.std(false_neg_roni) * 100 / (n_t * gen_share)) * 1.96 / np.sqrt(n_trials))
    print('SD false positive=', (np.mean(false_pos_roni) / n_orig) * 100,
                  '+-', (np.std(false_pos_roni) * 100 / n_orig) * 1.96 / np.sqrt(n_trials))

