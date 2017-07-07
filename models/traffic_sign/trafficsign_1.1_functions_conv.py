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
    # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
    keep_prob = tf.placeholder(tf.float32, name="dropout_prob")


def conv_layer(input, kernel_shape, scope_name):
    """
    Represents one layer of convolution on some input image.
    Creates and initializes wieghts in the shape of 'kernel_shape', this is the filter we pass over the image.
    Biases dimensions are taken from the last dimension of the weights, i.e. the out_channels value.
    :param input: 4-D Tensor of images with shape [batch, in_height, in_width, in_channels]
    :param kernel_shape: 4-D Tensor with shape [filter_height, filter_width, in_channels, out_channels]
    :param scope_name: The variable scope name for this function call
    :return: A Tensor of type tf.float32. The input image with half the pixels. e.g if the input images were
    72x72, the output images will be 36x36
    """
    with tf.variable_scope(scope_name):
        w = tf.get_variable("weights", kernel_shape, tf.float32,
                            tf.truncated_normal_initializer(mean=0.03, stddev=0.01))
        bias_shape = w.get_shape().as_list()[-1]
        b = tf.get_variable("biases", bias_shape, tf.float32,
                            tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu((conv + b))
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, weights_shape, scope_name):
    """
    Represents a fully connected layer.
    :param input: 2-D Tensor of images with shape [batch, flattened_image]
    :param weights_shape: The weights to use to perform the matrix multiplication
    :param scope_name: The variable scope name for this function call
    :return: The result of the matrix multiplication normalised with ReLU.
    """
    with tf.variable_scope(scope_name):
        w = tf.get_variable("weights", weights_shape, tf.float32,
                            tf.truncated_normal_initializer(0.03, 0.01))
        n = w.get_shape().as_list()[-1]
        b = tf.get_variable("biases", [n])
        return tf.nn.relu(tf.matmul(input, w) + b)


def model():
    """
    The convolutional model: 3 conv layers with kernel shape [filter_height, filter_width, in_channels, out_channels]
    and 2 fully connected layers, one to bring all the activation maps together (outputs of all the filters) and one
    final layer to predict a class
    :return: The predictions Y and the logits
    """
    # three convolutional layers with their channel counts, and a
    # fully connected layer (the last layer has 2 softmax neurons for "stop" and "go")
    K = 24  # first convolutional layer output depth
    L = 48  # second convolutional layer output depth
    M = 64  # third convolutional layer
    N = 200  # fully connected layer
    conv1 = conv_layer(X, [5, 5, NUM_CHANNELS, K], "conv1")
    conv2 = conv_layer(conv1, [5, 5, K, L], "conv2")
    conv3 = conv_layer(conv2, [3, 3, L, M], "conv3")
    flattened = tf.reshape(conv3, shape=[-1, 9 * 9 * M])
    fc_layer1 = fc_layer(flattened, [9 * 9 * M, N], "fc1")
    dropped = tf.nn.dropout(fc_layer1, keep_prob)
    logits = fc_layer(dropped, [N, NUM_CLASSES], "fc2")
    Y = tf.nn.softmax(logits)
    return Y, logits


Y, Ylogits = model()


# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 50 images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
with tf.name_scope("loss_function"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy) * 50

# accuracy of the trained model, between 0 (worst) and 1 (best)
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
with tf.name_scope("optimization"):
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# interactive session allows interleaving of building and running steps
sess = tf.InteractiveSession()
# init
sess.run(tf.global_variables_initializer())

# start the threads used for reading files
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

writer = tf.summary.FileWriter("./traffic_graph", sess.graph)

# start training
nSteps = 1000
for i in range(nSteps):

    # get a batch of images
    batch_xs, batch_ys = sess.run([imageBatch, labelBatch])
    # Run the training step with a feed of images
    train_step.run(feed_dict={X: batch_xs, Y_: batch_ys, lr: 0.01, keep_prob: 0.5})
#    p.eval()

    if (i + 1) % 100 == 0:  # then perform validation

        # get a validation batch
        vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
        train_accuracy = accuracy.eval(feed_dict={X: vbatch_xs, Y_: vbatch_ys, lr: 0.01, keep_prob: 1.0})

        print("step %d, training accuracy %g" % (i + 1, train_accuracy))



# finalise
coord.request_stop()
coord.join(threads)

