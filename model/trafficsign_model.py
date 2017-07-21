########################################################################
#
# A Convolutional Neural Network binary classifier implementation designed
# to work with a custom road traffic sign data set.
#
# **Representation, create, train and optimize a network. Summary functions**
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


def train(X, Y_, pkeep, lr):
    """Performs one training step with a batch of images and labels. Returns ops
    train_step and accuracy.
    :param X: 4-D tensor. Batch of images of shape [BATCH_SIZE, 72, 72, 3]
    :param Y_: 2-D tensor. Batch of labels of shape [BATCH_SIZE, 2]
    :param pkeep: Float. Probability of dropout. i.e. 0.5
    :param lr: Float. Learning rate. i.e. 0.0008
    :returns:
        train_step: op to run the training step
        accuracy: op to calculate the accuracy
    """
    # Build a Graph that computes the logits predictions from the inference model.
    logits = inference(X, pkeep)
    # compute cross entropy loss on the logits
    l = loss(logits, Y_)
    # get accuracy of logits with the ground truth
    acc = accuracy(logits, Y_)
    # Build a Graph that trains the model with one batch of examples and updates the model parameters.
    train_step = optimize(l, lr)
    return train_step, acc



def loss(logits, Y_):
    """Computes cross entropy loss on the unscaled logits from model
    normalised for batches of BATCH_SIZE images.
    Add summary for cross entropy.
    :param logits: Logits from inference().
    :param Y_: one-hot label tensor
    :return: Loss tensor of type float
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy, name="cross_entropy") * BATCH_SIZE
    tf.summary.scalar("x-ent", cross_entropy)
    return cross_entropy



def accuracy(logits, Y_):
    """Computes the accuracy of predictions.
    :param logits: Logits from inference().
    :param Y_: one-hot label tensor
    :return:
    """
    Y = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy



def optimize(loss, lr):
    """Returns an operation that applies the gradients from ADAM
    :param loss: cross entropy loss
    :param lr: Learning rate
    :return: Op which applies gradients
    """
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(loss)
    train_step = opt.apply_gradients(grads)
    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    return train_step



def inference(images, pkeep):
    """The model:
    2 conv layers, kernel shape [filter_height, filter_width, in_channels, out_channels]
    and 2 fully connected layers, one to bring all the activation maps together (outputs
    of all the filters) and one final softmax layer to predict the class.
    :param images: 4-D Tensor of images [batch, height, width, channels]
    :param pkeep: Dropout probability
    :return: logits
    """
    with tf.name_scope("the_model"):
        # 2 convolutional layers with their channel counts, and a
        # fully connected layer (the last layer has 2 softmax neurons for "stop" and "go")
        J = 128   # 1st convolutional layer output channels
        K = 172   # 2nd convolutional layer output channels
        N = 1536  # fully connected layer

        # weights / kernels
        # 7x7 patch, 3 input channel, J output channels
        W1 = tf.Variable(tf.truncated_normal([7, 7, NUM_CHANNELS, J], stddev=0.1))
        W2 = tf.Variable(tf.truncated_normal([5, 5, J, K], stddev=0.1))
        W3 = tf.Variable(tf.truncated_normal([8 * 8 * K, N], stddev=0.1))
        W4 = tf.Variable(tf.truncated_normal([N, NUM_CLASSES], stddev=0.1))

        # biases
        B1 = tf.Variable(tf.constant(0.1, tf.float32, [J]))
        B2 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
        B3 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
        B4 = tf.Variable(tf.constant(0.1, tf.float32, [NUM_CLASSES]))

        _visualize_kernel(W1)

        # First conv layer, 72x72 images
        Y1 = _conv_layer(images, W1, B1, pkeep)
        _activation_summary(Y1)

        # Second conv layer, 24x24 images
        Y2 = _conv_layer(Y1, W2, B2, pkeep)
        _activation_summary(Y2)

        # First FC layer, 8x8 images
        YY = tf.reshape(Y2, shape=[-1, 8 * 8 * K])
        Y3 = tf.nn.relu(tf.matmul(YY, W3) + B3)
        _activation_summary(Y3)

        # Softmax layer, although we don't return softmax, return logits which can be softmax'd if needed
        YY4 = tf.nn.dropout(Y3, pkeep)
        Ylogits = tf.matmul(YY4, W4) + B4
        _activation_summary(Ylogits)
        return Ylogits


########################################################################
# Below we have private helper functions which shouldn't be used outside of this module


def _conv_layer(input, kernel, bias, pkeep):
    """Creates one layer of convolution within the model.
    :param input: 4-D Tensor: The images or output of previous layer.
    :param kernel: 4-D Tensor: The weights to convolute with the input
    :param bias: 1-D Tensor: A small bias to add to the convolution
    :param pkeep: Float: Probability to drop a neuron
    :return: The result of the convolution, pooling and dropout. Normalised with ReLU
    """
    conv = tf.nn.relu(tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding='SAME') + bias)
    # 3x3 pooling area with stride of 3 reduces the image by 2/3 = 24x24 images after max_pool
    pool = tf.nn.max_pool(conv, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding="SAME")
    return tf.nn.dropout(pool, pkeep)


def _visualize_kernel(W):
    """Creates visualised kernels within Tensorboard
    :param W: 4-D Tensor to visualize
    :return: Nothing
    """
    # scale weights to [0 1], type is still float
    x_min = tf.reduce_min(W)
    x_max = tf.reduce_max(W)
    W_0_to_1 = (W - x_min) / (x_max - x_min)

    # to tf.image_summary format [batch_size, height, width, channels]
    kernel_transposed = tf.transpose(W_0_to_1, [3, 0, 1, 2])

    # this will display random 3 filters from the 128 in conv1
    tf.summary.image('conv1', kernel_transposed, 3)


def _activation_summary(x):
    """Helper to create activation summaries. Creates a summary that provides a histogram
    of activations. Creates a summary that measures the sparsity of activations.
    :param x: Tensor
    :return: Nothing
    """
    tf.summary.histogram('activations', x)
    tf.summary.scalar('sparsity', tf.nn.zero_fraction(x))
