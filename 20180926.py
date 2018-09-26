'''
Created on 2018年9月26日

@author: Allen
'''

import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True) 

inputs = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros(shape=[784, 10]))
b = tf.Variable(tf.zeros( shape=[10]))
logits =  tf.nn.softmax(tf.matmul(inputs, W)+b)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for _ in range(1,1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    feed_dict = {inputs:batch_xs, y:batch_ys}
    _, mloss = sess.run([train_step, loss],feed_dict=feed_dict)
    print(mloss)

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={inputs: mnist.test.images, y: mnist.test.labels}))

