import tensorflow as tf


# def foo(input, channels, name="conv", reuse=False):
#     with tf.name_scope(name):
#         v1 = tf.get_variable("var1", [1], dtype=tf.float32)




with tf.variable_scope("my_scope"):
    v1 = tf.get_variable("var1", [2, 2, 3, 4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(0.2, 0.1))
    v2 = tf.Variable(1, name="var2", dtype=tf.float32)



# print(v1.name)  # var1:0
# print(v2.name)  # my_scope/var2:0


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(v1))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
print(a.eval())
print(b.eval())
z = tf.nn.relu(tf.matmul(a, b))
print(z.eval())
