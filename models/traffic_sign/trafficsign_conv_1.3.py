########################################################################
#
# A Convolutional Neural Network binary classifier implementation designed
# to work with a custom road traffic sign data set.
#
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
from tensorflow.python.platform import tf_logging as logging

print("Tensorflow version " + tf.__version__)
logging.set_verbosity(logging.INFO)

########################################################################
# neural network structure:
#
# · · · · · · · · ·       (input data, 1-deep)                 X   [batch, 72, 72, 3]
# @ @ @ @ @ @ @ @ @    -- conv. layer 7x7x3=>128 stride 1      W1  [7, 7, 3, 128]         B1 [128]
#   :::::::::::::      -- max pool 3x3 stride 3                Y1  [batch, 24, 24, 128]
#   @ @ @ @ @ @ @      -- conv. layer 5x5x128=>172 stride 1    W2  [5, 5, 128, 172]       B2 [172]
#     :::::::::        -- max pool 3x3 stride 3                Y2  [batch, 8, 8, 172]
#                                               => reshaped to YY  [batch, 8*8*172]
#     \x/x\x\x/ ✞      -- fully connected layer (relu+dropout) W3  [8*8*172, 1536]        B3 [1536]
#      · · · ·                                                 Y3  [batch, 1536]
#      \x/x\x/         -- fully connected layer (softmax)      W4  [1536, 2]              B4 [2]
#       · · ·                                                   Y  [batch, 2]
#
########################################################################

# Various constants for describing the data set

# location of the data set: a TFRecord file
DATA_PATH = "../../dataset/traffic_sign/"

# number of classes is 2 (go and stop)
NUM_CLASSES = 2

# Width and height of each image. (pixels)
WIDTH = 72
HEIGHT = 72

# Number of channels in each image, 3 channels: Red, Green, Blue.
NUM_CHANNELS = 3

# batch size for training/validating network
BATCH_SIZE = 64

########################################################################


def get_image(filename, adversarial=False, name="get_image"):
    """Preprocessing function which gets an image from the location pointed to by 'filename',
    decodes it, converts dtype to float and sets the label to a one hot tensor.
    :param filename: The location of the TFRecord file
    :param adversarial: Boolean: Is the image for an adversarial example?
    :param name: The name_scope of the function call
    :return: An image tensor and a one hot label
    """
    with tf.name_scope(name):
        if adversarial:
            # hard coding the label for now, we're creating adversarial examples for the stop class, i.e class 2
            label = 2

            # The match_filenames_once() function saves the list of files matching pattern "filename".
            # e.g "./images/*.jpg" to retrieve all .jpg images in the directory.
            filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(filename), num_epochs=None)
            # Read a whole JPEG file
            reader = tf.WholeFileReader()
            # Read a whole file from the queue, the first returned value in the tuple is the
            # filename which we are ignoring.
            _, image_buffer = reader.read(filename_queue)

        else:
            # convert filename to a queue for an input pipeline.
            filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
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
    """Distort the colour of the image.
    Each colour distortion is non-commutative and thus ordering of the colour ops
    matters. This effectively expands the data set to an unlimited number of examples
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


def visualize_kernel(W):
    with tf.variable_scope('kernel_visualisation'):
        # scale weights to [0 1], type is still float
        x_min = tf.reduce_min(W)
        x_max = tf.reduce_max(W)
        W_0_to_1 = (W - x_min) / (x_max - x_min)

        # to tf.image_summary format [batch_size, height, width, channels]
        kernel_transposed = tf.transpose(W_0_to_1, [3, 0, 1, 2])

        # this will display random 3 filters from the 128 in conv1
        tf.summary.image('conv1', kernel_transposed, 3)


def create_batch(image, label):
    # associate "label_batch" and "image_batch" objects with a randomly selected batch of labels and images respectively
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


# Placeholders for data we will populate later
with tf.name_scope("inputs"):
    # input X: 72*72*3 pixel images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, NUM_CHANNELS], name="images")
    # similarly, we have a placeholder for true outputs (obtained from labels)
    Y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="labels")
    # variable learning rate
    lr = tf.placeholder(tf.float32, name="learning_rate")
    # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.5 at training time
    pkeep = tf.placeholder(tf.float32, name="dropout_prob")
    tf.summary.image("image", X, 3)



def model():
    """The convolutional model: 2 conv layers with kernel shape [filter_height, filter_width, in_channels, out_channels]
    and 2 fully connected layers, one to bring all the activation maps together (outputs of all the filters) and one
    final softmax layer to predict the class
    :return: The predictions Y and the logits
    """
    with tf.variable_scope("the_model"):
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
        tf.summary.histogram("W1", W1)
        tf.summary.histogram("W4", W4)

        # biases
        B1 = tf.Variable(tf.constant(0.1, tf.float32, [J]))
        B2 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
        B3 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
        B4 = tf.Variable(tf.constant(0.1, tf.float32, [NUM_CLASSES]))
        tf.summary.histogram("B1", B1)
        tf.summary.histogram("B4", B4)

        visualize_kernel(W1)

        with tf.name_scope("first_layer"):
            # 72x72 images
            Y1r = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + B1)
            # 3x3 pooling area with stride of 3 reduces the image by 2/3 = 24x24 images after max_pool
            Y1p = tf.nn.max_pool(Y1r, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding="SAME")
            Y1 = tf.nn.dropout(Y1p, pkeep)

        with tf.name_scope("second_layer"):
            Y2r = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, 1, 1, 1], padding='SAME') + B2)
            # 3x3 pooling area with stride 3 reduces image by 2/3 = 8x8
            Y2p = tf.nn.max_pool(Y2r, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding="SAME")
            Y2 = tf.nn.dropout(Y2p, pkeep)

        with tf.name_scope("fc_layer"):
            YY = tf.reshape(Y2, shape=[-1, 8 * 8 * K])
            Y3 = tf.nn.relu(tf.matmul(YY, W3) + B3)

            YY4 = tf.nn.dropout(Y3, pkeep)
            Ylogits = tf.matmul(YY4, W4) + B4
            Y = tf.nn.softmax(Ylogits)

        return Y, Ylogits

Y, Ylogits = model()

# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 64 images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
with tf.name_scope("x-ent"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy) * 64
    tf.summary.scalar("x-ent", cross_entropy)

# accuracy of the trained model, between 0 (worst) and 1 (best)
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

# training step, the learning rate is a placeholder
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# interactive session allows interleaving of building and running steps
sess = tf.InteractiveSession()
# init
sess.run(tf.global_variables_initializer())

# start the threads used for reading files
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

writer = tf.summary.FileWriter("./traffic_graph/main", sess.graph)
summary = tf.summary.merge_all()

# start training
nSteps = 200
for i in range(nSteps):

    batch_xs, batch_ys = sess.run([imageBatch, labelBatch])

    # train_step is the backpropagation step. Running this op allows the network to learn the distribution of the data
    sess.run([train_step], feed_dict={X: batch_xs, Y_: batch_ys, lr: 0.0008, pkeep: 0.5})

    if (i + 1) % 50 == 0:  # work out training accuracy and log summary info for tensorboard
        train_acc, s = sess.run([accuracy, summary], feed_dict={X: batch_xs, Y_: batch_ys, lr: 0.0008, pkeep: 0.5})
        writer.add_summary(s, i)
        print("step %d, training accuracy %g" % (i + 1, train_acc))

    if (i + 1) % 100 == 0:  # then perform validation

        # get a validation batch and calculate the accuracy of the predictions
        # note accuracy.eval() is equivalent to sess.run(accuracy)
        vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
        test_acc = accuracy.eval(feed_dict={X: vbatch_xs, Y_: vbatch_ys, lr: 0.0008, pkeep: 1.0})

        print("step %d, test accuracy %g" % (i + 1, test_acc))


# The network has now been trained. So lets move on to crafting adversarial examples.
print("{} steps have been completed. Beginning construction of adversarial example.".format(nSteps))

# Use spare CPU cycles to retrieve 1 (or more) images.
with tf.device('/cpu:0'):
    label, image = get_image("./adversarial_image/*.jpg", adversarial=True, name="get_adversarial_img")



with tf.name_scope("adversarial_inputs"):
    image = tf.reshape(image, (1, 72, 72, 3), "adversarial_reshape")
    tf.summary.image("adversarial_example", image, 1)

# finalise
writer.close()
coord.request_stop()
coord.join(threads)

