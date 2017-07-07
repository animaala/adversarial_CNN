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


# interactive session allows interleaving of building and running steps
sess = tf.InteractiveSession()

# "label" and "image" are associated with corresponding feature from a single example in the training data file
# at this point label is one hot vector. If label = 1 then [1,0]... if label = 2 then [0,1]
# (and yes that's opposite to binary!)
label, image = get_image("../../dataset/traffic_sign/train-00000-of-00001")

#p = tf.Print(image, [72, 72, 3])

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


with tf.name_scope("inputs"):
    # input X: 72*72*3 pixel images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, NUM_CHANNELS], name="images")
    # similarly, we have a placeholder for true outputs (obtained from labels)
    Y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="labels")
    # variable learning rate
    lr = tf.placeholder(tf.float32, name="learning_rate")
    # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
    keep_prob = tf.placeholder(tf.float32, name="dropout_prob")



# three convolutional layers with their channel counts, and a
# fully connected layer (the last layer has 2 softmax neurons for "stop" and "go")
K = 24  # first convolutional layer output depth
L = 48  # second convolutional layer output depth
M = 64  # third convolutional layer
N = 200  # fully connected layer


with tf.name_scope("weights_biases"):
    # create first set of weights aka kernel with 5x5 spacial convolution, depth of filter=NUM_CHANNELS, K output channels
    # I've left in arg names for W1 for clarity for the reader
    W1 = tf.get_variable(name='W1', shape=[5, 5, NUM_CHANNELS, K], dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(mean=0.03, stddev=0.01))

    B1 = tf.get_variable('B1', [K], tf.float32, tf.constant_initializer(0.01))
    W2 = tf.get_variable('W2', [5, 5, K, L], tf.float32, tf.truncated_normal_initializer(0.03, 0.01))
    B2 = tf.get_variable('B2', [L], tf.float32, tf.constant_initializer(0.01))
    W3 = tf.get_variable('W3', [3, 3, L, M], tf.float32, tf.truncated_normal_initializer(0.03, 0.01))
    B3 = tf.get_variable('B3', [M], tf.float32, tf.constant_initializer(0.01))
    # 9x9 below is related to the max-pooling operations defined in "The Model" below, in short by the time this kernel
    # is used to image is 9x9 pixels
    W4 = tf.get_variable('W4', [9 * 9 * M, N], tf.float32, tf.truncated_normal_initializer(0.03, 0.01))
    B4 = tf.get_variable('B4', [N], tf.float32, tf.constant_initializer(0.01))
    W5 = tf.get_variable('W5', [N, NUM_CLASSES], tf.float32, tf.truncated_normal_initializer(0.03, 0.01))
    B5 = tf.get_variable('B5', [NUM_CLASSES], tf.float32, tf.constant_initializer(0.01))


with tf.name_scope("the_model"):
    with tf.name_scope("first_layer"):
        # First convolutional layer + ReLU
        Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + B1)
        # Max-Pooling layer.. shrinks image to 36x36 pixels
        pool1 = tf.nn.max_pool(Y1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    with tf.name_scope("second_layer"):
        # Second convolutional layer + ReLU
        Y2 = tf.nn.relu(tf.nn.conv2d(pool1, W2, strides=[1, 1, 1, 1], padding='SAME') + B2)
        # Max-Pooling layer.. shrinks image to 18x18 pixels
        pool2 = tf.nn.max_pool(Y2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    with tf.name_scope("third_layer"):
        # Third convolutional layer + ReLU
        Y3 = tf.nn.relu(tf.nn.conv2d(pool2, W3, strides=[1, 1, 1, 1], padding='SAME') + B3)
        # Max-Pooling layer.. shrinks image to 9x9 pixels
        pool3 = tf.nn.max_pool(Y3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    with tf.name_scope("fully_connected_layer"):
        # reshape the output from the third convolution for the fully connected layer
        YY = tf.reshape(pool3, shape=[-1, 9 * 9 * M])
        Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
        YY4 = tf.nn.dropout(Y4, keep_prob)
        Ylogits = tf.matmul(YY4, W5) + B5
        Y = tf.nn.softmax(Ylogits)


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

