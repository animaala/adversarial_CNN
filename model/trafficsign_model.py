########################################################################
#
# A Convolutional Neural Network binary classifier implementation designed
# to work with a custom road traffic sign data set.
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
"""Representation of Convolution network.

    -- Visualisations and summaries:
    _visualise_kernel: Scale the weights and log a summary for Tensorboard.
    _activation_summary: Logs histogram of activations & measures sparsity of activations.

    -- Model representation:
    _hidden_layer: Create one hidden convolution layer within the network.
    inference: Representation for the network. Returns the logits tensor for classification.
    loss: Computes either mean cross entropy loss or element wise.
    accuracy: Computes accuracy of classification.
    train: Optimisation operation. Uses ADAM optimizer.
"""
import tensorflow as tf
import trafficsign_image_processing as input

########################################################################

# Various constants for describing the data set

# location of the data set: a TFRecord file
DATA_PATH = input.DATA_PATH

# Width and height of each image. (pixels)
WIDTH = input.WIDTH
HEIGHT = input.HEIGHT

# number of classes is 2 (go and stop)
NUM_CLASSES = input.NUM_CLASSES

# Number of channels in each image, 3 channels: Red, Green, Blue.
NUM_CHANNELS = input.NUM_CHANNELS

# batch size for training/validating network
BATCH_SIZE = input.BATCH_SIZE

########################################################################


def _visualise_kernel(W):
    """Helper to create visualisations for the kernel weights within Tensorboard.

    Args:
      W: 4-D Weight tensor to visualize.
    Returns:
      Nothing.
    """
    # scale weights to [0 1], type is still float
    x_min = tf.reduce_min(W)
    x_max = tf.reduce_max(W)
    W_0_to_1 = (W - x_min) / (x_max - x_min)

    # to tf.image_summary format [batch_size, height, width, channels]
    kernel_transposed = tf.transpose(W_0_to_1, [3, 0, 1, 2])
    # this will display random 3 filters from the 128 in conv1
    tf.summary.image('conv1', kernel_transposed, 3)


def _hidden_layer(images, kernel, bias, pkeep):
    """Helper that creates one hidden layer in the model.

    Args:
      images: 4-D Tensor: The input images or output of previous layer.
      kernel: 4-D Tensor: The weights to convolute with the input.
      bias: 1-D Tensor: A small bias to add to the convolution.
      pkeep: Float: Probability to drop a neuron.
    Returns:
      4-D float tensor: result of the convolution, pooling and dropout. Normalised with ReLU
    """
    conv = tf.nn.relu(tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME') + bias)
    # 3x3 pooling area with stride of 3 reduces the image by 2/3 = 24x24 images after max_pool
    pool = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding="SAME")
    return tf.nn.dropout(pool, pkeep)


def _activation_summary(x):
    """Helper to create activation summaries.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
      x: Tensor
    Return:
      Nothing
    """
    tf.summary.histogram('activations', x)
    tf.summary.scalar('sparsity', tf.nn.zero_fraction(x))


def inference(images, pkeep):
    """Builds the Traffic_Sign model.
    Note this model outputs the logits, NOT softmax. If softmax is needed apply it
    to the logits after the function call.

    There are 2 convolution layers with kernel shape:
    [filter_height, filter_width, in_channels, out_channels].
    And 2 fully connected layers, one to bring all the activation maps together
    (outputs of all the filters) and one final softmax layer to predict the class.

    Args:
      images: 4-D Tensor of images with shape -> [BATCH_SIZE, HEIGHT, WIDTH, NUM_CHANNELS]
      pkeep: Float: Dropout probability e.g. 0.5
    Returns:
      logits: the unscaled log probabilities
    """
    with tf.name_scope("inference"):
        # 2 convolutional layers with their channel counts, and a
        # fully connected layer (the last layer has 2 softmax neurons for "stop" and "go")
        J = 128   # 1st convolutional layer output channels
        K = 172   # 2nd convolutional layer output channels
        N = 1536  # fully connected layer

        # weights / kernels
        # e.g. W1 = 7x7 patch, 3 input channel, J output channels
        W1 = tf.Variable(tf.truncated_normal([7, 7, NUM_CHANNELS, J], stddev=0.1))
        W2 = tf.Variable(tf.truncated_normal([5, 5, J, K], stddev=0.1))
        W3 = tf.Variable(tf.truncated_normal([8 * 8 * K, N], stddev=0.1))
        W4 = tf.Variable(tf.truncated_normal([N, NUM_CLASSES], stddev=0.1))

        # biases
        B1 = tf.Variable(tf.constant(0.1, tf.float32, [J]))
        B2 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
        B3 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
        B4 = tf.Variable(tf.constant(0.1, tf.float32, [NUM_CLASSES]))

        _visualise_kernel(W1)

        # First conv layer, 72x72 images
        Y1 = _hidden_layer(images, W1, B1, pkeep)
        _activation_summary(Y1)

        # Second conv layer, 24x24 images
        Y2 = _hidden_layer(Y1, W2, B2, pkeep)
        _activation_summary(Y2)

        # First FC layer, 8x8 images
        YY = tf.reshape(Y2, shape=[-1, 8 * 8 * K])
        Y3 = tf.nn.relu(tf.matmul(YY, W3) + B3)
        _activation_summary(Y3)

        # Softmax layer (we don't return softmax, returns the logits)
        YY4 = tf.nn.dropout(Y3, pkeep)
        logits = tf.matmul(YY4, W4) + B4
        _activation_summary(logits)
        return logits


def loss(logits, Y_, mean=True):
    """Computes cross entropy loss on the unscaled logits from model and labels.
    Normalised for batches of BATCH_SIZE images. Adds summary for "cross_entropy".

    Args:
      logits: Unscaled log probabilities from inference().
      Y_: Placeholder which will contain one-hot label tensor with shape -> [BATCH_SIZE, 2].
      mean: bool: The user can request the whole loss tensor or the mean cross_entropy.
    Returns:
      Loss tensor of type float.
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_)
    if mean:
        cross_entropy = tf.reduce_mean(cross_entropy, name="cross_entropy") * BATCH_SIZE
        tf.summary.scalar("x-ent", cross_entropy)
    return cross_entropy


def accuracy(logits, Y_):
    """Computes the accuracy of predicted classes against the ground truth.

    Args:
      logits: unscaled Logits (output of inference).
      Y_: one-hot label tensor
    Returns:
      accuracy: The mean of an element wise accuracy comparison.
    """
    Y = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy


def train(loss, lr):
    """Train the Traffic_Sign model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.

    Args:
      loss: cross entropy loss
      lr: float Learning rate
    Returns:
        Op which computes and applies gradients
    """
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(loss)
    train_step = opt.apply_gradients(grads)
    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    return train_step
