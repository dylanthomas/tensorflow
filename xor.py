import tensorflow as tf
import numpy as np


data = np.loadtxt('xor_data.txt', unpack=True, dtype='float32')

x_data = data[0:-1].T
y_data = data[-1].T.reshape((-1, 1))


#x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
#y_data = [[0], [1], [1], [0]]
print(x_data, y_data)


x = tf.placeholder(dtype=tf.float32, name='x')
y_ = tf.placeholder(dtype=tf.float32, name='y_')

#W1 = tf.Variable(tf.random_uniform([2, 2], -1., 1.))
#W2 = tf.Variable(tf.random_uniform([2, 1], -1., 1.))

W1 = tf.Variable(tf.random_uniform([2, 5], -1., 1.), name='W1')
W2 = tf.Variable(tf.random_uniform([5, 4], -1., 1.), name='W2')
W3 = tf.Variable(tf.random_uniform([4, 1], -1., 1.), name='W3')

b1 = tf.Variable(tf.random_uniform([5], -1., 1.), name='b1')
b2 = tf.Variable(tf.random_uniform([4], -1., 1.), name='b2')
b3 = tf.Variable(tf.random_uniform([1], -1., 1.), name='b3')


w1_hist = tf.histogram_summary('W1', W1)
w2_hist = tf.histogram_summary('W2', W2)
w3_hist = tf.histogram_summary('W3', W3)
b1_hist = tf.histogram_summary('b1', b1)
b2_hist = tf.histogram_summary('b2', b2)
b3_hist = tf.histogram_summary('b3', b3)




with tf.name_scope('L1') as scope:
    L1 = tf.sigmoid(tf.matmul(x, W1) + b1)

with tf.name_scope('L2') as scope:
    L2 = tf.sigmoid(tf.matmul(L1, W2) + b2)

with tf.name_scope('y') as scope:
    y = tf.sigmoid(tf.matmul(L2, W3) + b3)
    y_hist = tf.histogram_summary('y', y)

with tf.name_scope('cost') as scope:
    cost = -tf.reduce_mean(y_*tf.log(y) + ( 1. - y_)*tf.log(1. - y))
    cost_sum = tf.scalar_summary('cost', cost)

with tf.name_scope('train') as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('./logs/xor_logs', sess.graph_def)
    for step in range(100001):
        sess.run(optimizer, feed_dict={x: x_data, y_: y_data})
        if step % 20000 == 0:
            summary = sess.run(merged, feed_dict={x: x_data, y_: y_data})
            writer.add_summary(summary, step)
            print(step)
 #           print(step, sess.run([y, cost], feed_dict={x: x_data, y_: y_data}), '\n',
 #                 sess.run(W1), '\n', sess.run(b1), '\n',
 #                 sess.run(W2), '\n', sess.run(b2), '\n',
 #                 sess.run(W3), '\n', sess.run(b3), '\n')

    correct_predictions = tf.equal(y_, tf.floor(y + 0.5))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    print('Results: \n')
    print(sess.run([y, tf.floor(y + 0.5), correct_predictions, accuracy],
                   feed_dict={x: x_data, y_: y_data}))

    print(accuracy.eval(feed_dict={x: x_data, y_: y_data}))








