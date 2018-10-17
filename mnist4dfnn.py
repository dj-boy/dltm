'''
Created on 2018年10月8日
构建深度前馈神经网络为mnist任务
@author: Allen
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 超参数
m1 = 300
m2 = 1000
m3 = 1800
m4 = 60
g = 1 # gain factor

interval = np.sqrt(6/(m1+784))
W1 = tf.Variable(g*tf.random_uniform(shape=[784, m1],minval=-interval,maxval=interval))
b1 = tf.Variable(tf.zeros([m1]))
layer1 = tf.nn.relu(tf.matmul(x, W1)+ b1)

interval = np.sqrt(6/(m1+m2))
W2 = tf.Variable(g*tf.random_uniform(shape=[m1, m2],minval=-interval,maxval=interval))
b2 = tf.Variable(tf.zeros([m2]))
layer2 = tf.nn.relu(tf.matmul(layer1, W2)+ b2)

interval = np.sqrt(6/(m3+m2))
W3 = tf.Variable(g*tf.random_uniform(shape=[m2, m3],minval=-interval,maxval=interval))
b3 = tf.Variable(tf.zeros([m3]))
layer3 = tf.nn.relu(tf.matmul(layer2, W3)+ b3)

interval = np.sqrt(6/(m3+m4))
W4 = tf.Variable(g*tf.random_uniform(shape=[m3, m4],minval=-interval,maxval=interval))
b4 = tf.Variable(tf.zeros([m4]))
layer4 = tf.nn.relu(tf.matmul(layer3, W4)+ b4)

interval = np.sqrt(6/(10+m4))
W_output = tf.Variable(g*tf.random_uniform(shape=[m4, 10],minval=-interval,maxval=interval))
b_output = tf.Variable(tf.zeros([10]))
y = tf.matmul(layer4 , W_output) + b_output

cross_entropy = tf.losses.softmax_cross_entropy(logits=y, onehot_labels=y_)
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
# train_step = tf.train.RMSPropOptimizer(0.001).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(50000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
    print(loss)
# define operation for evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()
