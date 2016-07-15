import tensorflow as tf
import timeit
from xavier_init import xavier_init

# Mnist data import
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 64
display_step = 1
dropout_rate = tf.placeholder(tf.float32)


# MLP Weights and biases
W1 = tf.get_variable('W1', shape=[784, 256], initializer=xavier_init(784, 256))
W2 = tf.get_variable('W2', shape=[256, 256], initializer=xavier_init(256, 256))
W3 = tf.get_variable('W3', shape=[256, 256], initializer=xavier_init(256, 256))
W4 = tf.get_variable('W4', shape=[256, 256], initializer=xavier_init(256, 256))
W5 = tf.get_variable('W5', shape=[256, 10], initializer=xavier_init(256, 10))


b1 = tf.Variable(tf.random_uniform([256], -1., 1.), name='b1')
b2 = tf.Variable(tf.random_uniform([256], -1., 1.), name='b2')
b3 = tf.Variable(tf.random_uniform([256], -1., 1.), name='b3')
b4 = tf.Variable(tf.random_uniform([256], -1., 1.), name='b4')
b5 = tf.Variable(tf.random_uniform([10], -1., 1.), name='b5')


w1_hist = tf.histogram_summary('W1', W1)
w2_hist = tf.histogram_summary('W2', W2)
w3_hist = tf.histogram_summary('W3', W3)
w4_hist = tf.histogram_summary('W4', W2)
w5_hist = tf.histogram_summary('W5', W3)

b1_hist = tf.histogram_summary('b1', b1)
b2_hist = tf.histogram_summary('b2', b2)
b3_hist = tf.histogram_summary('b3', b3)
b4_hist = tf.histogram_summary('b4', b4)
b5_hist = tf.histogram_summary('b5', b5)


# Construct model
with tf.name_scope('L1') as scope:
    _L1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    L1 = tf.nn.dropout(_L1, dropout_rate)

with tf.name_scope('L2') as scope:
    _L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
    L2 = tf.nn.dropout(_L2, dropout_rate)

with tf.name_scope('L3') as scope:
    _L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
    L3 = tf.nn.dropout(_L3, dropout_rate)

with tf.name_scope('L4') as scope:
    _L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
    L4 = tf.nn.dropout(_L4, dropout_rate)

with tf.name_scope('y') as scope:
    y = tf.add(tf.matmul(L4, W5), b5)
    y_hist = tf.histogram_summary('y', y)

# Loss and optimizer
with tf.name_scope('cost') as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    cost_sum = tf.scalar_summary('cost', cost)
#    avg_cost = tf.Variable(0.)
#    cost_add = avg_cost + (cost/num_batches)
#    cost_update = avg_cost.assign(cost_add)
#    avg_cost_sum = tf.scalar_summary('avg_cost', avg_cost)
with tf.name_scope('train') as scope:
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# Initialize all variables
init = tf.initialize_all_variables()


# Training cycles
tic = timeit.default_timer()
with tf.Session() as sess:
    sess.run(init)
    # initialize tensorboard
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('./logs/mnist_logs10', sess.graph)

    for epoch in range(training_epochs):
#        sess.run(avg_cost.assign(0))
        avg_cost2 = 0.
        num_batches = int(mnist.train.num_examples / batch_size)
         # Loop over all batches
        for i in range(num_batches):
            batch = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1],
                                            dropout_rate: .7})
            avg_cost2 += sess.run(cost, feed_dict={x: batch[0], y_: batch[1],
                                                   dropout_rate: .7}) / num_batches

        # Display logs per every epoch
        if epoch % display_step == 0:
            print(epoch + 1)
            print('Epoch: %04d \n' % (epoch + 1), 'cost= {:.9f} \n'.format(avg_cost2))
            summary = sess.run(merged, feed_dict={x: mnist.validation.images,
                                                  y_: mnist.validation.labels, dropout_rate: 1.})
            writer.add_summary(summary, epoch)

    toc = timeit.default_timer()

    print('Done in %.3f seconds \n' % (toc - tic))

    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, dropout_rate: 1.}))


