import numpy as np

# electroencephalographically
char_rdic = ['e', 'l', 'c', 't', 'r', 'o', 'n', 'p', 'h', 'a', 'g', 'i', 'y'] # id -> char
char_dic = {w: i for i, w in enumerate(char_rdic)} # char -> id

sample = [char_dic[c] for c in 'electroencephalographically'] # to index

# Configuration
char_vocab_size = len(char_dic)
rnn_size = char_vocab_size # 1 hot coding
time_step_size = 26 # 'electroencephalographicall' -> predict 'lectroencephalographically'
batch_size = 1 # one sample

# Data setup
x_data = np.zeros((len(sample) - 1, char_vocab_size))

x_data[np.arange(26), sample[:-1]] = 1.

print(sample[:-1])
print(x_data)

