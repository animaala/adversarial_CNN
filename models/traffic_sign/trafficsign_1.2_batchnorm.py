########################################################################
#
# Have some documentation here
#
# Implemented in Python 3.5, TF v1.1, CuDNN 5.1
#
# Ryan Halliburton 2017
########################################################################

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

print("Tensorflow version " + tf.__version__)
logging.set_verbosity(logging.INFO)

########################################################################

# Various constants for describing the data set

# number of classes is 2 (go and stop)
NUM_CLASSES = 2

# Width and height of each image. (pixels)
WIDTH = 72
HEIGHT = 72

# Number of channels in each image, 3 channels: Red, Green, Blue.
NUM_CHANNELS = 3

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


# Function to read a single image from input file
def get_image(filename, name="get_image"):
    with tf.name_scope(name):
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

        # re-define label as a "one-hot" vector
        # it will be [0,1] or [1,0] here.
        # This approach can easily be extended to more classes
        label = tf.stack(tf.one_hot(label - 1, NUM_CLASSES), name="one_hot")

        return label, image


# "label" and "image" are associated with corresponding feature from a single example in the training data file
# at this point label is one hot vector. If label = 1 then [1,0]... if label = 2 then [0,1]
# (and yes that's opposite to binary!)
label, image = get_image("../../dataset/traffic_sign/train-00000-of-00001")


# and similarly for the validation data
vlabel, vimage = get_image("../../dataset/traffic_sign/validation-00000-of-00001")


# associate "label_batch" and "image_batch" objects with a randomly selected batch of labels and images respectively
# train.shuffle_batch creates batches by randomly shuffling tensors. Adds to the current graph:
# 1: A shuffling queue into which tensors from the tensors arg are enqueued.
# 2: A dequeue_many operation to create batches from the queue.
# 3: A QueueRunner to QUEUE_RUNNER collection, to enqueue the tensors from tensors arg.
with tf.name_scope("shuffle_batch"):
    imageBatch, labelBatch = tf.train.shuffle_batch(
        [image, label],
        batch_size=64,
        capacity=220,
        min_after_dequeue=60)

    # and similarly for the validation data
    vimageBatch, vlabelBatch = tf.train.shuffle_batch(
        [vimage, vlabel],
        batch_size=64,
        capacity=220,
        min_after_dequeue=15)


# Placeholders for data we will populate later
with tf.name_scope("inputs"):
    # input X: 72*72*3 pixel images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, NUM_CHANNELS], name="images")
    # similarly, we have a placeholder for true outputs (obtained from labels)
    Y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="labels")
    # variable learning rate
    lr = tf.placeholder(tf.float32, name="learning_rate")
    # test flag for batch norm
    tst = tf.placeholder(tf.bool)
    iter = tf.placeholder(tf.int32)
    # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
    pkeep = tf.placeholder(tf.float32, name="dropout_prob")
    pkeep_conv = tf.placeholder(tf.float32)
    tf.summary.image("input", X, 4)


def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_everages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_everages


def no_batchnorm(Ylogits, is_test, iteration, offset):
    return Ylogits, tf.no_op()


def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])
    return noiseshape


def conv_layer(input, kernel, scope_name):
    """
    Represents one layer of convolution on some input image.
    Creates and initializes wieghts in the shape of 'kernel_shape', this is the filter we pass over the image.
    :param input: 4-D Tensor of images with shape [batch, in_height, in_width, in_channels]
    :param kernel_shape: 4-D Tensor with shape [filter_height, filter_width, in_channels, out_channels]
    :param scope_name: The variable scope name for this function call
    :return: A Tensor of type tf.float32.
    """
    with tf.variable_scope(scope_name):
        conv = tf.nn.conv2d(input, kernel, strides=[1, 1, 1, 1], padding="SAME")
        tf.summary.histogram("kernel", kernel)
        return conv


def model():
    """
    The convolutional model: 3 conv layers with kernel shape [filter_height, filter_width, in_channels, out_channels]
    and 2 fully connected layers, one to bring all the activation maps together (outputs of all the filters) and one
    final layer to predict a class
    :return: The predictions Y and the logits
    """
    with tf.variable_scope("the_model"):
        # three convolutional layers with their channel counts, and a
        # fully connected layer (the last layer has 2 softmax neurons for "stop" and "go")
        J = 64   # first convolutional layer output depth
        K = 128  # second convolutional layer output depth
        L = 256  # third convolutional layer
        M = 384  # fourth convolutional layer
        N = 2048  # fully connected layer

        # weights / kernels
        # 6x6 patch, 3 input channel, J output channels
        W1 = tf.Variable(tf.truncated_normal([6, 6, NUM_CHANNELS, J], stddev=0.1))
        W2 = tf.Variable(tf.truncated_normal([5, 5, J, K], stddev=0.1))
        W3 = tf.Variable(tf.truncated_normal([4, 4, K, L], stddev=0.1))
        W4 = tf.Variable(tf.truncated_normal([3, 3, L, M], stddev=0.1))
        W5 = tf.Variable(tf.truncated_normal([5 * 5 * M, N], stddev=0.1))
        W6 = tf.Variable(tf.truncated_normal([N, NUM_CLASSES], stddev=0.1))

        # biases
        B1 = tf.Variable(tf.constant(0.1, tf.float32, [J]))
        B2 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
        B3 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
        B4 = tf.Variable(tf.constant(0.1, tf.float32, [M]))
        B5 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
        B6 = tf.Variable(tf.constant(0.1, tf.float32, [2]))

        with tf.name_scope("first_layer"):
            # 72x72 images
            Y1l = conv_layer(X, W1, "conv1")
            # 36x36 images after max_pool
            Y1p = tf.nn.max_pool(Y1l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            Y1bn, update_ema1 = batchnorm(Y1p, tst, iter, B1, convolutional=True)
            Y1r = tf.nn.relu(Y1bn)
            Y1 = tf.nn.dropout(Y1r, pkeep_conv, compatible_convolutional_noise_shape(Y1r))

        with tf.name_scope("second_layer"):
            Y2l = conv_layer(Y1, W2, "conv2")
            # 18x18 images after max_pool
            Y2p = tf.nn.max_pool(Y2l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            Y2bn, update_ema2 = batchnorm(Y2p, tst, iter, B2, convolutional=True)
            Y2r = tf.nn.relu(Y2bn)
            Y2 = tf.nn.dropout(Y2r, pkeep_conv, compatible_convolutional_noise_shape(Y2r))

        with tf.name_scope("third_layer"):
            Y3l = conv_layer(Y2, W3, "conv3")
            # 9x9 images after max_pool
            Y3p = tf.nn.max_pool(Y3l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            Y3bn, update_ema3 = batchnorm(Y3p, tst, iter, B3, convolutional=True)
            Y3r = tf.nn.relu(Y3bn)
            Y3 = tf.nn.dropout(Y3r, pkeep_conv, compatible_convolutional_noise_shape(Y3r))

        with tf.name_scope("fourth_layer"):
            Y4l = conv_layer(Y3, W4, "conv4")
            # 5x5 images after max_pool
            Y4p = tf.nn.max_pool(Y4l, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            Y4bn, update_ema4 = batchnorm(Y4p, tst, iter, B4, convolutional=True)
            Y4r = tf.nn.relu(Y4bn)
            Y4 = tf.nn.dropout(Y4r, pkeep_conv, compatible_convolutional_noise_shape(Y4r))

        with tf.name_scope("fc_layer"):
            YY = tf.reshape(Y4, shape=[-1, 5 * 5 * M])
            Y5l = tf.matmul(YY, W5)
            print(Y5l)
            Y5bn, update_ema5 = batchnorm(Y5l, tst, iter, B5)
            Y5r = tf.nn.relu(Y5bn)
            Y5 = tf.nn.dropout(Y5r, pkeep)
            Ylogits = tf.matmul(Y5, W6) + B6
            Y = tf.nn.softmax(Ylogits)

            update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4, update_ema5)

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

writer = tf.summary.FileWriter("./traffic_graph/1.1", sess.graph)
merged_summary = tf.summary.merge_all()

# start training
nSteps = 2000
for i in range(nSteps):

    # get a batch of images
    batch_xs, batch_ys = sess.run([imageBatch, labelBatch])
    # Run the training step with a feed of images
    if i % 5 == 0:
        s = sess.run(merged_summary, feed_dict={X: batch_xs, Y_: batch_ys, lr: 0.003, pkeep: 0.5})
        writer.add_summary(s, i)
 #   train_step.run(feed_dict={X: batch_xs, Y_: batch_ys, lr: 0.01, pkeep: 0.5})

    if (i + 1) % 100 == 0:  # then perform validation

        # get a validation batch
        vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
        train_accuracy = accuracy.eval(feed_dict={X: vbatch_xs, Y_: vbatch_ys, lr: 0.003, pkeep: 1.0})

        print("step %d, training accuracy %g" % (i + 1, train_accuracy))



# finalise
coord.request_stop()
coord.join(threads)