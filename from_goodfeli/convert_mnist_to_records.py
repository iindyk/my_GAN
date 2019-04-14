from scipy.misc import imread
from random import shuffle
import time

import tensorflow as tf
from glob import glob
from from_goodfeli.utils import get_image, colorize
import numpy as np
# this is based on tensorflow tutorial code
# https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/how_tos/reading_data/convert_to_records.py
# TODO: it is probably very wasteful to store these images as raw numpy
# strings, because that is not compressed at all.
# i am only doing that because it is what the tensorflow tutorial does.
# should probably figure out how to store them as JPEG.

IMSIZE = 28
labels_to_use = [7, 8, 9]


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(argv):
    # uploading data
    (x_train_all, y_train_all), (x_test_all, y_test_all) = tf.keras.datasets.mnist.load_data()

    # take only images of digits from labels_to_use
    x_train = []
    y_train = []
    for i in range(len(y_train_all)):
        if y_train_all[i] == labels_to_use[0]:
            x_train.append(x_train_all[i])
            y_train.append(0)
        elif y_train_all[i] == labels_to_use[1]:
            x_train.append(x_train_all[i])
            y_train.append(1)
        elif y_train_all[i] == labels_to_use[2]:
            x_train.append(x_train_all[i])
            y_train.append(2)
    outfile = '/home/iindyk/PycharmProjects/my_GAN/MNIST/MNIST_train_labeled_' + str(IMSIZE) + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(outfile)

    for i in range(len(y_train)):
        image = x_train[i]
        image = np.reshape(image, newshape=(IMSIZE, IMSIZE, 1))
        image = image.astype('uint8')
        image_raw = image.tostring()
        label = y_train[i]
        if i % 1 == 0:
            print(i, '\t',label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(IMSIZE),
            'width': _int64_feature(IMSIZE),
            'depth': _int64_feature(3),
            'image_raw': _bytes_feature(image_raw),
            'label': _int64_feature(label)
            }))
        writer.write(example.SerializeToString())

    writer.close()


if __name__ == "__main__":
    tf.app.run()

