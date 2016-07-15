import tensorflow as tf
import numpy as np

data = np.loadtxt('train2.txt', dtype='float32')
x_data = data[:, :3]
y_data = data[:, 3:]

print(x_data)
print(y_data)

X = tf.placeholder('float', [None, 3])
Y = tf.placeholder('float', [None, 3])


W = tf.Variable(tf.random_uniform([3, 3], -0.01, 0.01))

hypothesis = tf.nn.softmax(tf.matmul(X, W))

learning_rate = 0.3



cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}),
                  sess.run(W), '\n')

    a = sess.run(hypothesis, feed_dict={X:[[1, 11, 7]]})
    print(a, sess.run(tf.arg_max(a, 1)), '\n')

    b = sess.run(hypothesis, feed_dict={X: [[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
    print(b, sess.run(tf.arg_max(b, 1)))