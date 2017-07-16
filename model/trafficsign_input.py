########################################################################
#
# A Convolutional Neural Network binary classifier implementation designed
# to work with a custom road traffic sign data set.
#
# **This module handles data inputs, batching and augmentation**
#
# Implemented in Python 3.5, TF v1.1, CuDNN 5.1
#
# Ryan Halliburton 2017
#
# This project is available at the following 2 repositories:
# 1. https://github.com/animaala/adversarial_ML
# 2. https://gitlab.ncl.ac.uk/securitylab/adversarial_ML.git
#
########################################################################

import tensorflow as tf
import numpy as np
import os

########################################################################

# Various constants for describing the data set

# location of the data set: a TFRecord file
DATA_PATH = "../dataset/traffic_sign/"

# Width and height of each image. (pixels)
WIDTH = 72
HEIGHT = 72

# number of classes is 2 (go and stop)
NUM_CLASSES = 2

# Number of channels in each image, 3 channels: Red, Green, Blue.
NUM_CHANNELS = 3

# batch size for training/validating network
BATCH_SIZE = 64


def get_image(path_to_file, adversarial=False, name="get_image"):
    """Preprocessing function which gets an image from the location pointed to by 'filename',
    decodes it, converts dtype to float and sets the label to a one hot tensor.
    :param path_to_file: The location of the TFRecord file
    :param adversarial: Boolean: Is the image for an adversarial example?
    :param name: The name_scope of the function call
    :return: An image tensor and a one hot label
    """
    if not os.path.isfile(path_to_file):
        raise ValueError("path_to_file is not a file")
    with tf.name_scope(name):
        if adversarial:
            # hard coding the label for now, we're creating adversarial examples for the stop class, i.e class 2
            label = 2

            # The match_filenames_once() function saves the list of files matching pattern "filename".
            # e.g "./images/*.jpg" to retrieve all .jpg images in the directory.
            filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(path_to_file))
            # Read a whole JPEG file
            reader = tf.WholeFileReader()
            # Read a whole file from the queue, the first returned value in the tuple is the
            # filename which we are ignoring.
            _, image_buffer = reader.read(filename_queue)
        else:
            # convert filename to a queue for an input pipeline.
            filename_queue = tf.train.string_input_producer([path_to_file], num_epochs=None)
            # object to read records
            reader = tf.TFRecordReader()
            # read the full set of features for a single example
            key, example = reader.read(filename_queue)

            # parse the full example into its' component features.
            features = tf.parse_single_example(
                example,
                features={
                    'image/height': tf.FixedLenFeature([], tf.int64),
                    'image/width': tf.FixedLenFeature([], tf.int64),
                    'image/colorspace': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
                    'image/channels': tf.FixedLenFeature([], tf.int64),
                    'image/class/label': tf.FixedLenFeature([], tf.int64),
                    'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
                    'image/format': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
                    'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
                    'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
                })
            # now we are going to manipulate the label and image features
            label = features['image/class/label']
            image_buffer = features['image/encoded']

        # Decode the jpeg
        # name_scope effects ops
        with tf.name_scope('decode_jpeg', [image_buffer], None):
            # decode turns tensor of type string. 0-D the JPEG encoded image
            # to tensor of type uint8. 3-D with shape [height, width, channels]
            image = tf.image.decode_jpeg(image_buffer, channels=3, name="decode")
            image = tf.image.convert_image_dtype(image, dtype=tf.float32, name="convert_dtype")

        image.set_shape([HEIGHT, WIDTH, NUM_CHANNELS])

        # re-define label as a "one-hot" vector (if '1' vector will be [1,0] else [0,1])
        label = tf.stack(tf.one_hot(label - 1, NUM_CLASSES), name="one_hot")

        return image, label


def distort_colour(image):
    """Distort the colour of the image. Each colour distortion is non-commutative and thus ordering of the
    colour ops matters. This effectively expands the data set to an unlimited number of examples.
    :param image: Tensor containing a single image
    :return: Tensor containing the colour distorted image
    """
    ordering = np.random.random_integers(11)
    with tf.name_scope("distort_colour"):
        if ordering == 1:
            pass
        elif ordering == 2:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=0.12)
            image = tf.image.random_saturation(image, lower=0.3, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.1)
        elif ordering == 3:
            image = tf.image.random_brightness(image, max_delta=0.12)
            image = tf.image.random_saturation(image, lower=0.3, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif ordering == 4:
            image = tf.image.random_saturation(image, lower=0.3, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=0.12)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif ordering == 5:
            image = tf.image.random_hue(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.3, upper=1.5)
        elif ordering == 6:
            image = tf.image.random_brightness(image, max_delta=0.12)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif ordering == 7:
            image = tf.image.random_saturation(image, lower=0.3, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=0.12)
        elif ordering == 8:
            image = tf.image.random_brightness(image, max_delta=0.12)
        elif ordering == 9:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif ordering == 10:
            image = tf.image.random_saturation(image, lower=0.3, upper=1.5)
        elif ordering == 11:
            image = tf.image.random_hue(image, max_delta=0.1)
    return image


def create_batch(image, label):
    # train.shuffle_batch creates batches by randomly shuffling tensors. Adds to the current graph:
    # 1: A shuffling queue into which tensors from the tensors arg are enqueued.
    # 2: A dequeue_many operation to create batches from the queue.
    # 3: A QueueRunner to QUEUE_RUNNER collection, to enqueue the tensors from tensors arg.
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=BATCH_SIZE,
        capacity=200,
        min_after_dequeue=20)
    return image_batch, label_batch



