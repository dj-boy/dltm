'''
Created on 2018年9月14日
用tensorflow实施 min-char-rnn
@author: Allen
'''

import tensorflow as tf
import numpy as np

def getBatch(data, batch_size, seq_length, voc_size, p):
    """
    data是一个list结构，每个元素是一个字符
    p指示从哪个位置开始抽取一个batch的数据
    """
    inputs =  []
    targets = []
    for i in range(batch_size):
        one_input = data[p : p+seq_length]
        one_target = data[p+1 : p+seq_length+1]
        inputs.append(oneHot(one_input, vocab_size))
        targets.append(oneHot(one_target, vocab_size))
#         p += seq_length+1
        p += 1
    return inputs, targets, p
    
def oneHot(input, voc_size):
    """
            对一个输入向量，进行one-hot编码，返回的是一个矩阵。其每一行是一个one-hot编码的字符向量
    """
    t_input = np.zeros([len(input),voc_size])
    
    for i in range(len(input)):
        ix = char_to_ix[input[i]]
        t_input[i][ix] = 1 
    return t_input

# data
data = open('t1.txt', 'r').read().lower()# should be simple plain text file
chars = list(set(data))
data = data.replace('\n',' ')
data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# parameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 40 # number of steps to unroll the RNN for
learning_rate = 0.007
epoch_size = 10
batch_size = 1000 # minibatch中的训练数据条数
init_state = np.random.randn(batch_size, hidden_size)
mode = 2

#graph
hstate = tf.placeholder(dtype=tf.float32, shape=[None, hidden_size])
input = tf.placeholder(dtype=tf.float32, shape=[None, seq_length, vocab_size])
target = tf.placeholder(dtype=tf.float32, shape=[None, seq_length,vocab_size])

weights = {
    'hidden': tf.Variable(tf.random_normal([vocab_size, hidden_size])*0.01), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([hidden_size, vocab_size])*0.01)
}
biases = {
    'hidden': tf.Variable(tf.zeros([hidden_size])),
    'out': tf.Variable(tf.zeros([vocab_size]))
}
X1 = tf.transpose(input, [1, 0, 2])
X2 = tf.reshape(X1, [-1, vocab_size])
X3 = tf.nn.xw_plus_b(X2, weights['hidden'], biases['hidden'])
X  = tf.split(value=X3, axis=0, num_or_size_splits=seq_length)

if mode == 1:
    cell = tf.contrib.rnn.BasicRNNCell(hidden_size)
    # rnn的outputs是一个list结构，一个元素对应一个时间步的输出。state是最后一个时间步的状态
    outputs, state = tf.nn.static_rnn(cell, X, initial_state=hstate, dtype=tf.float32)
elif mode==2:
    cell = tf.contrib.rnn.UGRNNCell(hidden_size,forget_bias=0.8)
    outputs, state = tf.nn.static_rnn (cell, X, 
                                        initial_state=hstate, dtype=tf.float32)


# 重新安排outputs到一个二维结构[batch_size*seq_length, hidden_size]
# 连续的steps行对应着模型的seq_length个时间步的输出
new_output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
#计算每个time step的输出的loss
Y = tf.nn.xw_plus_b(new_output, weights['out'], biases['out'])
T = tf.reshape(target, [-1, vocab_size])
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y, labels=T))

optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars)
# train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# prediction
pred = tf.argmax(tf.nn.softmax(Y), 1)
# Initializing the variables
init = tf.global_variables_initializer()

data_ex = []
for _ in range(epoch_size):
    data_ex.extend(data)
    
# Launch the graph, train model
with tf.Session() as sess:
    sess.run(init)
    p = 0
    k = 1
    while p + batch_size < len(data_ex)-seq_length:
        init_state = np.random.randn(batch_size, hidden_size)
        inputs, targets, p = getBatch(data_ex, batch_size, 
                                      seq_length, vocab_size, p)
        feed_dict={hstate:init_state, input:inputs, target:targets}
        _, _, loss_x, pred_s = sess.run(
                [train_op, state, loss, pred], feed_dict)

        if(k%10==0):
            print("%s,%s,%s"%(len(data_ex),p,loss_x))
            slist = np.argmax(inputs[0],1)
            rs =[ix_to_char[idx] for idx in slist]
            print(''.join(rs))
            rs =[ix_to_char[idx] for idx in pred_s[0:seq_length]]
            print(''.join(rs))
        k += 1
    
    # validation
#     p=10
#     s = data_ex[p : p+seq_length]
#     print(''.join(s))
#     inputs, _, _ = getBatch(data_ex, 1, seq_length, vocab_size, p)
#     feed_dict[input]=inputs
#     t = sess.run([pred], feed_dict)
#     rs =[ix_to_char[idx] for idx in t[0]]
#     print(''.join(rs))
     
        
    
