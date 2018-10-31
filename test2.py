# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 08:59:27 2018

@author: Allen
"""
import tensorflow as tf
import data_helpers
from preprocessor import vocabulary
import numpy as np 

#1. 准备数据集
x_text, y = data_helpers.load_data_and_labels()
voc = vocabulary(x_text)
x = np.array(voc.data)

np.random.seed(10)
random_index = np.random.permutation(len(y))
x_random = x[random_index]
y_random = y[random_index]

x_train, x_test = x_random[:-1000],x_random[-1000:]
y_train, y_test = y_random[:-1000],y_random[-1000:]

sequence_len = x.shape[1]
voc_size = len(voc.vocab)
#2. 超参数
num_class = 2
embed_size = 100
filter_sizes = [3,4,5]
filter_num = 128
batch_size= 50
num_epochs = 1

#3. 构建数据流图
#（1）placeholder
x_input = tf.placeholder(dtype=tf.int32
                         , shape=[None, sequence_len])
y_input = tf.placeholder(dtype=tf.int32
                         , shape=[None, num_class ])
keep_prob = tf.placeholder(tf.float32)

#print(x_input)
#print(y_input)
#（2）查找表
C = tf.Variable(tf.truncated_normal(
        shape=[voc_size, embed_size]))
embeds = tf.nn.embedding_lookup(C, x_input)
embeds_extends = tf.expand_dims(embeds, -1)
#print(embeds_extends)

#（3）搭建深度网络
pool_list = []
for filter_size in filter_sizes:

    W_filter = tf.Variable(tf.truncated_normal(
        shape=[filter_size,embed_size,1,filter_num]))

    conv = tf.nn.conv2d(input=embeds_extends,
             filter=W_filter,
             strides=[1,1,1,1],
             padding="VALID")
    h = tf.nn.relu(conv)
    pool = tf.nn.max_pool(h,
            ksize=[1,sequence_len-filter_size+1,1,1 ],
            strides=[1,1,1,1],
            padding="VALID")
#    print(pool)
    pool_list.append(pool)

h_pool=tf.concat(values=pool_list,axis=3)
h_pool2 = tf.reshape(h_pool, 
                shape=[-1,filter_num*len(filter_sizes)])

h_dropout = tf.nn.dropout(h_pool2, keep_prob)

W_out = tf.Variable(tf.truncated_normal(
        shape=[filter_num*len(filter_sizes),num_class]))
b_out = tf.Variable(tf.zeros(num_class))

out = tf.matmul(h_dropout,W_out)+b_out

predictions = tf.argmax(out,1)
y_ = tf.argmax(y_input,1)
corrects = tf.equal(predictions,y_)
accu = tf.reduce_mean(tf.cast(corrects, "float"))

#print(out)
#4. 损失函数
loss =tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(
        logits=out,
        labels=y_input))
#5. 优化器
train = tf.train.AdamOptimizer(1e-4).minimize(loss)
#6. 其他操作

#7. 创建session，训练模型
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

batches = data_helpers.batch_iter(list(zip(x_train, y_train)),
                                  batch_size, num_epochs)
for batch in batches:
    x_batch, y_batch = zip(*batch)
    feed_dict={x_input:x_batch,
                y_input:y_batch,
                keep_prob:0.5}
    _, m_loss,m_accu = sess.run([train,loss,accu], feed_dict=feed_dict)
    print("loss:%s, accurary:%s"%(m_loss,m_accu))
     
# check in test
feed_dict={x_input:x_test,
                y_input:y_test,
                keep_prob:1}    
m_accu = sess.run([accu], feed_dict=feed_dict)
print("In test: accurary:%s"%(m_accu))
     