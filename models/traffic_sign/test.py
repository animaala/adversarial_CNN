########################################################################
#
# Have some documentation here
#
# Implemented in Python 3.5, TF v1.1, CuDNN 5.1
#
# Ryan Halliburton 2017
########################################################################

import tensorflow as tf
# import numpy as np
# import sys
# import os
#
# print("Tensorflow version " + tf.__version__)
#
# ########################################################################
#
# Various constants for describing the data set

# number of classes is 2 (go and stop)
NUM_CLASSES = 2

# Width and height of each image. (pixels)
WIDTH = 72
HEIGHT = 72

# Number of channels in each image, 3 channels: Red, Green, Blue.
NUM_CHANNELS = 3


# Function to read a single image from input file
def get_image(filename):
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
            'image/class/label': tf.FixedLenFeature([1], tf.int64),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/format': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
        })

    # now we are going to manipulate the label and image features

    label = features['image/class/label']
    image_buffer = features['image/encoded']
    file_name = features['image/filename']
    h = features['image/height']
    w = features['image/width']



    # Decode the jpeg
    # name_scope effects ops
    with tf.name_scope('decode_jpeg', [image_buffer], None):
        # decode turns tensor of type string. 0-D the JPEG encoded image
        # to tensor of type uint8. 3-D with shape [height, width, channels]
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)


    image.set_shape([HEIGHT, WIDTH, NUM_CHANNELS])

    # re-define label as a "one-hot" vector
    # it will be [0,1] or [1,0] here.
    # This approach can easily be extended to more classes
    label = tf.stack(tf.one_hot(label - 1, NUM_CLASSES))

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
imageBatch, labelBatch = tf.train.shuffle_batch(
    [image, label],
    batch_size=50,
    capacity=220,
    min_after_dequeue=60)


# and similarly for the validation data
vimageBatch, vlabelBatch = tf.train.shuffle_batch(
    [vimage, vlabel], batch_size=30,
    capacity=40,
    min_after_dequeue=15)


# input X: 72*72*3 pixel images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, NUM_CHANNELS])
# similarly, we have a placeholder for true outputs (obtained from labels)
Y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])


print("Running Simple Model y=Wx+b")  # initialise weights and biases to zero
# W maps input to output so is of size: (number of pixels) * (Number of Classes)
W = tf.Variable(tf.zeros([WIDTH * HEIGHT, NUM_CLASSES]))
# b is vector which has a size corresponding to number of classes
b = tf.Variable(tf.zeros([NUM_CLASSES]))

# define output calc (for each class) y = softmax(Wx+b)
# softmax gives probability distribution across all classes
y = tf.nn.softmax(tf.matmul(X, W) + b)


# measure of error of our model
# this needs to be minimised by adjusting W and b
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# define training step which minimises cross entropy
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# argmax gives index of highest entry in vector (1st axis of 1D tensor)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# get mean of all entries in correct prediction, the higher the better
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# init
sess.run(tf.global_variables_initializer())

# start the threads used for reading files
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# start training
nSteps = 1000
for i in range(nSteps):

    # get a batch of images
    batch_xs, batch_ys = sess.run([imageBatch, labelBatch])
    # Run the training step with a feed of images
    train_step.run(feed_dict={X: batch_xs, Y_: batch_ys})


    if (i + 1) % 200 == 0:  # then perform validation

        # get a validation batch
        vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
        train_accuracy = accuracy.eval(feed_dict={X: vbatch_xs, Y_: vbatch_ys})

        print("step %d, training accuracy %g" % (i + 1, train_accuracy))



# finalise
coord.request_stop()
coord.join(threads)


















#number of classes is 2 (go and stop)
# NUM_CLASSES = 2
#
# # Width and height of each image. (pixels)
# WIDTH = 72
# HEIGHT = 72
#
# # Number of channels in each image, 3 channels: Red, Green, Blue.
# NUM_CHANNELS = 3
#
# sess = tf.InteractiveSession()
#
# filename = "../../dataset/traffic_sign/train-00000-of-00001"
#
# # Function to read a single image from input file
#
# # convert filename to a queue for an input pipeline.
# filenameQ = tf.train.string_input_producer([filename], num_epochs=None)
#
# #print(filenameQ.eval())
#
# # object to read records
# recordReader = tf.TFRecordReader()
#
# # read the full set of features for a single example
# key, fullExample = recordReader.read(filenameQ)
#
# #print(fullExample.eval())
#
# # parse the full example into its' component features.
# features = tf.parse_single_example(
#     fullExample,
#     features={
#         'image/height': tf.FixedLenFeature([], tf.int64),
#         'image/width': tf.FixedLenFeature([], tf.int64),
#         'image/colorspace': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
#         'image/channels': tf.FixedLenFeature([], tf.int64),
#         'image/class/label': tf.FixedLenFeature([], tf.int64),
#         'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
#         'image/format': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
#         'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
#         'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
#     })
#
# # now we are going to manipulate the label and image features
#
# label = features['image/class/label']
# image_buffer = features['image/encoded']
#
# print(image_buffer)
# # Decode the jpeg
# # name_scope effects ops
# with tf.name_scope('decode_jpeg'):
#     # decode turns tensor of type string. 0-D the JPEG encoded image
#     # to tensor of type uint8. 3-D with shape [height, width, channels]
#     image = tf.image.decode_jpeg(image_buffer, channels=3)
#
#
# image = tf.reshape(image, [HEIGHT, WIDTH, NUM_CHANNELS])
# image = tf.to_float(image, "ToFloat")
#
# # re-define label as a "one-hot" vector
# # it will be [0,1] or [1,0] here.
# # This approach can easily be extended to more classes
# label = tf.one_hot(label - 1, NUM_CLASSES, dtype=tf.int64)


#
# a = tf.get_variable("a", initializer=tf.constant(2))
# a = tf.one_hot(a - 1, 2, dtype=tf.int64)
# print(a)
#
# B1 = tf.Variable(tf.ones([4])/10)
# B2 = tf.get_variable('bias', shape=[4], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
#W1 = tf.get_variable('W1', [2,2], tf.float32, tf.truncated_normal_initializer(0.03, 0.01))

# W4 = tf.Variable(tf.truncated_normal([7 * 7, 200], stddev=0.1))
# #W5 = tf.get_variable('W5', [2, 5], tf.float32, tf.truncated_normal_initializer(0.03, 0.01))
# print(W4)
# print(image_buffer)
# print(image)
# #print(W5)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(W4.eval())
#    print("====================+++++++++++++++++++++++++========================")
#    print(W5.eval())

# sess = tf.InteractiveSession()
# tensor = tf.truncated_normal([6, 6, 3], 1, 0.2)
# print(tensor.eval())
#
# i = 0
# k = 0
# filename = "../../dataset/traffic_sign/validation-00000-of-00001"
# for serialized_example in tf.python_io.tf_record_iterator(filename):
#     example = tf.train.Example()
#     example.ParseFromString(serialized_example)
#
#     print(type(example))
#
#     # traverse the Example format to get data
#     image_buffer = example.features.feature['image/encoded'].bytes_list.value
#     label = example.features.feature['image/class/label'].int64_list.value[0]
#
#     print(image_buffer)
#
#     # Decode the jpeg
#     # name_scope effects ops
#     with tf.name_scope('decode_jpeg'):
#         # decode turns tensor of type string. 0-D the JPEG encoded image
#         # to tensor of type uint8. 3-D with shape [height, width, channels]
#         image = tf.image.decode_jpeg(image_buffer, channels=3)
#
# a = tf.truncated_normal([3, 2, 2], 1, 0.1)
# tf.Print(a)
#
# print(k)

