import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


xy_data = np.loadtxt('xy_data.txt', unpack=True, dtype='float32')

x_data = xy_data[:-1]
y_data = xy_data[-1]

print(x_data)
print(y_data)

W = tf.Variable(tf.random_uniform([1, 3], -5.0, 5.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = tf.matmul(W, X)
cost = tf.reduce_mean(tf.square(hypothesis - Y))


a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)


init = tf.initialize_all_variables()


sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W, feed_dict={X: x_data, Y: y_data}))



