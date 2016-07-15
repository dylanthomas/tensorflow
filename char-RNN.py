import tensorflow as tf
#from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

# electroencephalographically
char_rdic = ['e', 'l', 'c', 't', 'r', 'o', 'n', 'p', 'h', 'a', 'g', 'i', 'y'] # id -> char
char_dic = {w: i for i, w in enumerate(char_rdic)} # char -> id

sample = [char_dic[c] for c in 'electroencephalographically'] # to index

# Configuration
char_vocab_size = len(char_dic)
rnn_size = char_vocab_size # 1 hot coding
time_step_size = 26 # 'electroencephalographicall' -> predict 'lectroencephalographically'
num_batch = 1 # one sample

# Data setup
x_data = np.zeros((len(sample) - 1, char_vocab_size), dtype='f4')
x_data[np.arange(26), sample[:-1]] = 1.


print(sample[:-1])
print(x_data)




# RNN model
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
state = tf.zeros([num_batch, rnn_cell.state_size])
X_split = tf.split(0, time_step_size, x_data)
outputs, state = tf.nn.rnn(rnn_cell, X_split, state)

# logits : list of 2D Tensors of shape [batch_size x num_decoder symbols]
# targets : list of 1D batch-sized int32 Tensor of the same length as logits
# weights : list of 1 D batch-sized float-Tensors of the same length as logits

logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
targets = tf.reshape(sample[1:], [-1])
weights = tf.ones([time_step_size*num_batch])

loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / num_batch
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

#Launch the graph in a session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(100):
        sess.run(train_op)
        result = sess.run(tf.arg_max(logits, 1))
        print(i)
        print(result)
        print([char_rdic[t] for t in result])
    print(logits)
    print(targets)
    print(weights)