import tensorflow as tf
import input_data
import timeit


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def model(x, W_conv1, W_conv2, W_conv3, W_conv4, W_conv5, W_fc1, W_fc2,
          b_conv1, b_conv2, b_conv3, b_conv4, b_conv5, b_fc1, b_fc2, keep_prob):

    # Reshape x to a 4d tensor
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # Construct model
    with tf.name_scope('conv1') as scope:
        _h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_conv1 = tf.nn.dropout(_h_conv1, keep_prob)
        # (?, 28, 28, 1) -> (?, 28, 28, 32)

    with tf.name_scope('conv2') as scope:
        _h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
        h_conv2 = tf.nn.dropout(_h_conv2, keep_prob)
        # (?, 28, 28, 32) -> (?, 28, 28, 32)

    with tf.name_scope('conv3_res') as scope:
        _h_conv3 = conv2d(h_conv2, W_conv3) + b_conv3
        _h_res1 = tf.nn.relu(_h_conv3 + h_conv1)
        h_res1 = tf.nn.dropout(_h_res1, keep_prob)
        # (?, 28, 28, 32) -> (?, 28, 28, 32)

    with tf.name_scope('conv4') as scope:
        h_conv4 = tf.nn.relu(conv2d(h_res1, W_conv4) + b_conv4)
        _h_pool4 = max_pool_2x2(h_conv4)
        h_pool4 = tf.nn.dropout(_h_pool4, keep_prob)
        # (?, 28, 28, 32) -> (?, 28, 28, 64) -> (?, 14, 14, 64)

    with tf.name_scope('conv5') as scope:
        h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
        _h_pool5 = max_pool_2x2(h_conv5)
        h_pool5 = tf.nn.dropout(_h_pool5, keep_prob)
        h_pool5_flat = tf.reshape(h_pool5, [-1, W_fc1.get_shape().as_list()[0]])
        # (?, 14, 14, 64) -> (?, 14, 14, 64) -> (?, 7, 7, 64) -> (?, 3136)

    with tf.name_scope('fc1') as scope:
        h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        # (?, 3136) -> (?, 1024)

    with tf.name_scope('fc2') as scope:
        h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        # (?, 1024) -> (?, 10)


    return h_fc2


# Import data and reshape
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# Set up parameters and variables
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')
keep_prob = tf.placeholder(tf.float32)
learning_rate = 0.0005
reg_strength = 1e-4
training_epochs = 2
batch_size = 64
display_step = 1
restore = False


# Define weights and biases
W_conv1 = weight_variable([3, 3, 1, 32], name='W_conv1')
W_conv2 = weight_variable([3, 3, 32, 32], name='W_conv2')
W_conv3 = weight_variable([3, 3, 32, 32], name='W_conv3')
W_conv4 = weight_variable([3, 3, 32, 64], name='W_conv4')
W_conv5 = weight_variable([3, 3, 64, 64], name='W_conv5')
W_fc1 = weight_variable([7 * 7 * 64, 1024], name='W_fc1')
W_fc2 = weight_variable([1024, 10], name='W_fc2')

b_conv1 = bias_variable([32], name='b_conv1')
b_conv2 = bias_variable([32], name='b_conv2')
b_conv3 = bias_variable([32], name='b_conv3')
b_conv4 = bias_variable([64], name='b_conv4')
b_conv5 = bias_variable([64], name='b_conv5')
b_fc1 = bias_variable([1024], name='b_fc1')
b_fc2 = bias_variable([10], name='b_fc2')


# Evaluate unnormalized log probabilities
with tf.name_scope('model_out') as scope:
    y = model(x, W_conv1, W_conv2, W_conv3, W_conv4, W_conv5, W_fc1, W_fc2,
              b_conv1, b_conv2, b_conv3, b_conv4, b_conv5, b_fc1, b_fc2, keep_prob)

# L2 regularization
with tf.name_scope('l2_regularization') as scope:
    l2reg = tf.reduce_sum(tf.square(W_conv1)) + tf.reduce_sum(tf.square(W_conv2)) + \
            tf.reduce_sum(tf.square(W_conv3)) + tf.reduce_sum(tf.square(W_conv4)) + \
            tf.reduce_sum(tf.square(W_conv5)) + tf.reduce_sum(tf.square(W_fc1)) + \
            tf.reduce_sum(tf.square(W_fc2))

# Loss and optimizer
with tf.name_scope('cost') as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_)) + \
           reg_strength*l2reg
    cost_sum = tf.scalar_summary('cost', cost)

with tf.name_scope('train') as scope:
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#    train_step = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost)


# Calculate accuracy
with tf.name_scope('accuracy') as scope:
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_sum = tf.scalar_summary('accuracy', accuracy)

# Initialize all variables
init = tf.initialize_all_variables()

# Add ops to save and restore all the variables
saver = tf.train.Saver()


restore = True
# Training cycles
with tf.Session() as sess:

    if restore == True:
        saver.restore(sess, './ckpt/resNet/model.ckpt')
        print('Model restored.')
    else:
        sess.run(init)

    # initialize tensorboard
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('./logs/resNet_logs', sess.graph)

    for epoch in range(training_epochs):
        avg_cost = 0.
        num_batches = int(mnist.train.num_examples / batch_size)
         # Loop over all batches
        tic = timeit.default_timer()
        for i in range(num_batches):
            batch = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1],
                                            keep_prob: .5})
            avg_cost += sess.run(cost, feed_dict={x: batch[0], y_: batch[1],
                                                   keep_prob: .5}) / num_batches
        toc = timeit.default_timer()
        # Display logs per every epoch
        if epoch % display_step == 0:
            print(epoch + 1)
            print('Epoch: %04d \n' % (epoch + 1), 'cost= {:.9f} \n'.format(avg_cost))
            summary = sess.run(merged, feed_dict={x: mnist.validation.images,
                                                  y_: mnist.validation.labels, keep_prob: 1.})
            writer.add_summary(summary, epoch)
            print('Done in %.3f seconds \n' % (toc - tic))

    # Save the variables to disk
    save_path = saver.save(sess, './ckpt/resNet/model.ckpt')
    print('Model save in file: %s' % save_path)

    # Calculate accuracy and cost with the test set
    accuracy_test_eval = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                  y_: mnist.test.labels, keep_prob: 1.})
    print(accuracy_test_eval)
    summary = sess.run(merged, feed_dict={x: mnist.test.images,
                                          y_: mnist.test.labels, keep_prob: 1.})
    writer.add_summary(summary)








