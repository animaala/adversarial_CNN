import tensorflow as tf



X = tf.range(1, 16908289, 1, tf.float32)
X = tf.reshape(X, [1536, 8*8*172])
W1 = tf.Variable(tf.range(1, 16908289, 1, tf.float32))
W1 = tf.reshape(W1, [8*8*172, 1536])

YY = tf.nn.relu(tf.matmul(X, W1))

with tf.Session() as sess:
    print(sess.run(tf.shape(YY)))