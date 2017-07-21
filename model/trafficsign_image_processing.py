########################################################################
#
# A Convolutional Neural Network binary classifier implementation designed
# to work with a custom road traffic sign data set.
#
# **Read, preprocess image data and construct batches of images for evaluation**
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
import random

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



def parse_tfrecord_file(path_to_file):
    """Parses a TFRecord file containing the training/test example images.
    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:
      image/height: 72
      image/width: 72
      image/colorspace: 'RGB'
      image/channels: 3
      image/class/label: 1
      image/class/text: 'go'
      image/format: 'JPEG'
      image/filename: 'go (97).jpeg'
      image/encoded: <JPEG encoded string>
    Args:
      example_serialized: scalar Tensor tf.string containing a serialized
        Example protocol buffer.
    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      text: Tensor tf.string containing the human-readable label.
    """
    if not os.path.isfile(path_to_file):
        raise ValueError("File does not exist")
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
            'image/height':tf.FixedLenFeature([], tf.int64),
            'image/width':tf.FixedLenFeature([], tf.int64),
            'image/colorspace':tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/channels':tf.FixedLenFeature([], tf.int64),
            'image/class/label':tf.FixedLenFeature([], tf.int64),
            'image/class/text':tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/format':tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/filename':tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/encoded':tf.FixedLenFeature([], dtype=tf.string, default_value='')
        })
    label = features['image/class/label']
    # re-define label as a "one-hot" vector (if "go" aka '1' vector will be [1,0] else [0,1])
    label = tf.stack(tf.one_hot(label-1, NUM_CLASSES), name="one_hot_label")

    return features['image/encoded'], label, features['image/class/text']


def parse_single_image():
    """Parses a single JPEG image located in the train/stop/ directory.
    This is hardcoded in.
    Returns:
        String literal
    """
    file_name = random.choice(os.listdir(DATA_PATH+"train/stop/"))
    image_data = tf.gfile.FastGFile(DATA_PATH+"train/stop/"+file_name, 'rb').read()
    label = 2
    return image_data, label


def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.
    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for name_scope.
    Returns:
      3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(values=[image_buffer], name=scope,
                       default_name='decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        # After this point, all image pixels reside in (0,1)
        image.set_shape([HEIGHT, WIDTH, NUM_CHANNELS])
        return image


def convert_image_to_float(image, scope=None):
    """
    Args:
      image: 3-D integer tensor [height, width, channels]
      scope: scope of the function
    Returns:
      The image tensor with values between (0,1)
    """
    with tf.name_scope(values=[image], name=scope,
                       default_name='convert_jpeg_to_float'):
        # After this point, all image pixels reside in (0,1)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image

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



def image_batch(path_to_file):
    """Construct input batch for Traffic sign evaluation
    :param path: path to file
    :returns:
        image_batch: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        label_batch: Labels. 2D tensor of [batch_size, NUM_CLASSES]
    """
    if not os.path.isfile(path_to_file):
        raise ValueError("File does not exist")
    image_buffer, label, text = parse_tfrecord_file(path_to_file)
    image = decode_jpeg(image_buffer)
    image = convert_image_to_float(image)
    image_batch, label_batch = create_batch(image, label)
    return image_batch, label_batch



def distorted_image_batch(path_to_file):
    """Construct distorted input batch for Traffic sign evaluation
    :param path: path to file
    :returns:
        image_batch: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        label_batch: Labels. 2D tensor of [batch_size, NUM_CLASSES]
    """
    if not os.path.isfile(path_to_file):
        raise ValueError("File does not exist")
    image_buffer, label, text = parse_tfrecord_file(path_to_file)
    image = decode_jpeg(image_buffer)
    image = convert_image_to_float(image)
    image = distort_colour(image)
    image_batch, label_batch = create_batch(image, label)
    return image_batch, label_batch



def random_stop_image():
    """Get a random 'Stop' class image from the training set.
    :returns:
        image: 3-D Tensor: A random image from the Stop class
        label: The accompanying label
    """
    image_buffer, label = parse_single_image()
    image = decode_jpeg(image_buffer)
    return image, label



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
