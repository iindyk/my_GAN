import os
import numpy as np
import tensorflow as tf

from from_goodfeli.model import DCGAN
from from_goodfeli.discriminator import discriminator
from from_goodfeli.build_model import build_model, build_model_single_gpu
from from_goodfeli.train import train
from from_goodfeli.generator import Generator
from from_goodfeli.utils import pp, visualize, to_json

flags = tf.app.flags
# flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("discriminator_learning_rate", 0.0004, "Learning rate of for adam [0.0004]")
flags.DEFINE_float("generator_learning_rate", 0.0004, "Learning rate of for adam [0.0004]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", 18065, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108,"The size of image to use (will be center cropped) [108] "
                     "(This one does not make any sense, it is not the size of the image presented to the model)")
flags.DEFINE_integer("image_width", 28, "The width of the images presented to the model")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)) as sess:
        # if FLAGS.dataset == 'mnist':
        #    assert False
        dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                      sample_size=128,
                      z_dim=8192,
                      d_label_smooth=.25,
                      generator_target_prob=.75 / 2.,
                      out_stddev=.075,
                      out_init_b=- .45,
                      image_shape=[FLAGS.image_width, FLAGS.image_width, 1],
                      dataset_name=FLAGS.dataset, is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir,
                      generator=Generator(),
                      train_func=train, discriminator_func=discriminator,
                      build_model_func=build_model, config=FLAGS,
                      devices=["GPU:0"]  # , ", "gpu:1", "gpu:2", "gpu:3", gpu:4"]
                      )

        if FLAGS.is_train:
            print("TRAINING")
            dcgan.train(FLAGS)
            print("DONE TRAINING")
        else:
            dcgan.load(FLAGS.checkpoint_dir)

        OPTION = 2
        #
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        #
        visualize(sess, dcgan, FLAGS, OPTION)


if __name__ == '__main__':
    tf.app.run()
