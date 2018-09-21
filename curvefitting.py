# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 20:39:45 2018
建立一个两层的神经网络进行函数拟合。可以进行两种试验。
没加入噪声的数据，神经网络拟合完美。
产生的数据加入了噪声，神经网络也几乎可以克服噪声的干扰。
@author: Allen
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

h1_size = 15
h2_size = 10

x = np.arange(-3, 3, 0.1)
y1 = np.sin(x)
# 加入噪声
y = y1 + np.random.randn(len(y1))*0.1

# build graph
X = tf.Variable(x, dtype=tf.float32, trainable=False)
inputs = tf.reshape(X, [-1, 1] )
Y = tf.Variable(y, dtype=tf.float32, trainable=False)
target = tf.reshape(Y, [-1, 1] )

W1 = tf.Variable(tf.random_normal([1, h1_size])*0.01) 
W2 = tf.Variable(tf.random_normal([h1_size, h2_size])*0.01)
W3 = tf.Variable(tf.random_normal([h2_size,1])*0.01)

b1 = tf.Variable(tf.zeros([h1_size]))
b2 = tf.Variable(tf.zeros([h2_size]))
b3 = tf.Variable(tf.zeros([1]))

h1 = tf.nn.sigmoid(tf.nn.xw_plus_b(inputs, W1, b1))
h2 = tf.nn.xw_plus_b(h1, W2, b2)
model = tf.nn.xw_plus_b(h2, W3, b3)

#predication
pred = model

loss=tf.reduce_mean(tf.square(model-target))
#step = tf.Variable(0, trainable=False)
#rate = tf.train.exponential_decay(0.15, step, 1, 0.9999)
#optimizer = tf.train.AdamOptimizer(rate)
#train_op = optimizer.minimize(loss)

optimizer = tf.train.AdamOptimizer(0.009)
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars)

init_op=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init_op)
    
for step in range(1,80000):
    _,val, y2, t =sess.run([train_op,loss, model, target])
    if step % 1000 == 0:
        print("error:%s"%(val))

plt.plot(x,y,'g', label='noise')
plt.plot(x,y1,'k', label='sin')
plt.plot(x,y2,'r', label='fit')
plt.legend()
