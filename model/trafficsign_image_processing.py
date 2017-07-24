########################################################################
#
# A Convolutional Neural Network binary classifier implementation
# designed to work with a custom road traffic sign data set.
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
"""Read and preprocess image data and construct batches of images for evaluation.

    -- Provide processed image data for a network:
    image_batch: Generate batches of evaluation examples of images.
    distorted_image_batch: Generate batches of training examples of images.
    create_batch: Construct batches from provided examples.

    -- Data processing:
    parse_tfrecord_file: Parses a TFRecord file containing serialized Example proto's.
    parse_single_image: Parses a single JPEG image into bytes.

    -- Image decoding:
    decode_jpeg: Decode a JPEG encoded string into a 3-D float32 Tensor.

    -- Image preprocessing:
    convert_image_to_float: Converts an image to float32.
    distort_colour: Distort the color in one image for training.
    get_stop_image: Get one stop image for generating adversarial example.
"""
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

########################################################################


def image_batch(file_path):
    """Generate batches of Traffic_Sign images for evaluation.

    Use this function as the inputs for evaluating a network.

    Args:
      file_path: Relative path to the TFRecord file containing the serialized
      example data set.
    Returns:
      image_batch: Images. 4D tensor with shape -> [BATCH_SIZE, HEIGHT, WIDTH, NUM_CHANNELS].
      label_batch: Labels. 2D tensor with shape -> [BATCH_SIZE, NUM_CLASSES]
    Raises:
      ValueError: if user provided an incorrect file location.
    """
    if not os.path.isfile(file_path):
        raise ValueError("File not found")
    image_buffer, label, text = parse_tfrecord_file(file_path)
    image = decode_jpeg(image_buffer)
    image = convert_image_to_float(image)
    image_batch, label_batch = create_batch(image, label)
    return image_batch, label_batch


def distorted_image_batch(file_path):
    """Generate batches of Traffic_Sign images with a chance for distortion.

    Use this function as the inputs for training a network.

    Args:
      file_path: Relative path to the TFRecord file containing the serialized
      example data set.
    Returns:
      image_batch: Images. 4D tensor with shape -> [BATCH_SIZE, HEIGHT, WIDTH, NUM_CHANNELS].
      label_batch: Labels. 2D tensor with shape -> [BATCH_SIZE, NUM_CLASSES]
    Raises:
      ValueError: if user provided an incorrect file location.
    """
    if not os.path.isfile(file_path):
        raise ValueError("File not found")
    image_buffer, label, text = parse_tfrecord_file(file_path)
    image = decode_jpeg(image_buffer)
    image = convert_image_to_float(image)
    image = distort_colour(image)
    image_batch, label_batch = create_batch(image, label)
    return image_batch, label_batch


def decode_jpeg(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.

    Args:
      image_buffer: scalar string Tensor.
      scope: Optional scope for name_scope.
    Returns:
      3-D int Tensor with values ranging from [0, 255).
    """
    with tf.name_scope(values=[image_buffer], name=scope,
                       default_name='decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        # Set the shape statically
        image.set_shape([HEIGHT, WIDTH, NUM_CHANNELS])
        return image


def convert_image_to_float(image, scope=None):
    """Converts an image tensor to float32.

    Args:
      image: integer image tensor [height, width, channels]
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
    """Distort the colour of the image.

    Each distortion is non-commutative thus ordering of the colour ops matters.
    This effectively expands the data set to an unlimited number of examples.

    Args:
      image: Tensor containing single image.
    Returns:
      Colour distorted image
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


def get_stop_image(file_name):
    """Get a random 'Stop' class image from the training set.

    Args:
      file_name: The file name of the image from the "/validation/stop/" directory.
      e.g. "stop (286).jpg"
    Returns:
      image: 3-D int Tensor with values ranging from [0, 255).
      label: The accompanying label for the image
    """
    image_buffer, label = parse_single_image(file_name)
    image = decode_jpeg(image_buffer)
    return image, label


def create_batch(image, label):
    """Construct batches of shuffled images and labels.

    train.shuffle_batch creates batches by randomly shuffling tensors. Adds to the current graph:
    1. Constructs a shuffling queue into which the image and label are enqueued.
    2. A dequeue_many operation to create batches from the queue.
    3. A QueueRunner to QUEUE_RUNNER collection, to enqueue the tensors from tensors arg.
    Args:
      image: 3-D image tensor.
      label: 1-hot representation of the label.
    Returns:
      4-D image tensor with shape -> [BATCH_SIZE, HEIGHT, WIDTH, NUM_CHANNELS]
      2-D label tensor with shape -> [BATCH_SIZE, 2]
    """
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=BATCH_SIZE,
        capacity=200,
        min_after_dequeue=20)
    return image_batch, label_batch



def parse_tfrecord_file(file_path):
    """Parses a TFRecord file containing serialized Example proto's containing images.

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
      file_path: Relative path to the TFRecord file containing the serialized
      example data set.
    Returns:
      image_buffer: Tensor tf.string containing the contents of a JPEG file.
      label: Tensor tf.int32 containing the label.
      text: Tensor tf.string containing the human-readable label.
    Raises:
      ValueError: if user provided an incorrect file location.
    """
    if not os.path.isfile(file_path):
        raise ValueError("File not found")
    # convert filename to a queue for an input pipeline.
    filename_queue = tf.train.string_input_producer([file_path], num_epochs=None)
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


def parse_single_image(file_name):
    """Parses a single JPEG image located at "/dataset/traffic_sign/validation/stop/"
    This is a low level function. If you need a single image from the 'Stop' class
    use get_stop_image()

    Args:
      file_name: The file name of the image from the "/validation/stop/" directory.
      e.g. "stop (286).jpg"
    Returns:
      image_data: Byte array: the encoded JPEG.
      label: The class number, int32
    """
    image_data = tf.gfile.FastGFile(DATA_PATH+"validation/stop/"+file_name, 'rb').read()
    label = 2
    return image_data, label


