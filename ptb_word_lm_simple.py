'''
Created on 2018年9月9日
这段程序有几个特点：
（1）所有的数据只参与一次训练，因此没有设定epoch_size。
(2) 设定词向量的大小和隐层的unit数一样，这样就少一个输入的全连接层，也没有embedding_size参数

@author: Allen
'''
import tensorflow as tf
import reader

def getBatch(data, batch_size, num_steps, p):
    '''
            返回shape=[batch_size, num_steps]的数据集
    '''
    inputs = []
    target = []
    for i in range(0, batch_size):
        inputs.append(data[i+p : i+p+num_steps])
        target.append(data[i+p+1 : i+p+num_steps+1])
    p += batch_size
    return(inputs, target, p) 

# parameters
hidden_size   = 1000   # 隐层unit的个数，对应了词向量的长度
keep_prob     = 0.35
num_steps     = 50
batch_size    = 100
vocab_size    = 10000   # 只选择vacab_size个词
is_training   = True
learning_rate = 0.8
lr_decay      = 0.5
max_epoch     = 1
max_grad_norm = 5
max_max_epoch = 13
data_path="D:/qjt/data/simple-examples/data"

# data
raw_data = reader.ptb_raw_data(data_path) 
train_data, valid_data, test_data, _, id2word= raw_data

# graph
input_x = tf.placeholder(tf.int32, [None, num_steps], name="input_x")
targets = tf.placeholder(tf.int32, [None, num_steps], name="targets")
lookup_table = tf.Variable(
                tf.random_uniform([vocab_size, hidden_size], -1.0, 1.0))
inputs = tf.nn.embedding_lookup(lookup_table, input_x)

if is_training and keep_prob < 1:
    inputs = tf.nn.dropout(inputs, keep_prob)

cell1_ = tf.contrib.rnn.BasicLSTMCell(
          hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=False)

cell1 = tf.contrib.rnn.DropoutWrapper(cell1_, output_keep_prob=keep_prob)
cell2_ = tf.contrib.rnn.BasicLSTMCell(
          hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=False)

cell2 = tf.contrib.rnn.DropoutWrapper(cell2_, output_keep_prob=keep_prob)
cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2], state_is_tuple=True)

initial_state = cell.zero_state(batch_size, tf.float32)
inputs = tf.unstack(inputs, num=num_steps, axis=1)
outputs, state = tf.nn.static_rnn(cell, inputs,
                                      initial_state=initial_state)
output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])

softmax_w = tf.get_variable(
        "softmax_w", [hidden_size, vocab_size], dtype=tf.float32)
softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)

logits_ = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
logits = tf.reshape(logits_, [batch_size, num_steps, vocab_size])

# slice part of logits for valid
slice_ = tf.slice(logits, begin=[0,0,0], size=[1,num_steps,vocab_size])
pres_ = tf.reshape(slice_, [-1,vocab_size])
valid = tf.argmax(tf.nn.softmax(pres_), 1)

t_ = tf.slice(targets, begin=[0,0], size=[1,num_steps])
t_valid = tf.reshape(t_,[num_steps])

# Use the contrib sequence loss and average over the batches
loss = tf.contrib.seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([batch_size, num_steps], dtype=tf.float32),
        average_across_timesteps=True,
        average_across_batch=True)

# Update the cost
cost = tf.reduce_sum(loss)
final_state = state

tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),max_grad_norm)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.apply_gradients(zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())

fetches = {
      "cost": cost,
      "final_state": final_state,
      "train_op": train_op,
      "valid": valid,
      "t_valid": t_valid
}
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
p = 0
while p<len(train_data)-batch_size:
    i_inputs, i_targets, p = getBatch(train_data, batch_size, num_steps, p)
    feed_dict = {input_x:i_inputs, targets:i_targets}
    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    print(cost)
    state = vals["final_state"]
    t_valid = vals["t_valid"]
    s_ = [id2word[key] for key in t_valid]
    s = ' '.join(s_)
    print(s)
    
    valid = vals["valid"]
    s_ = [id2word[key] for key in valid]
    s = ' '.join(s_)
    print(s)
    
    
    
