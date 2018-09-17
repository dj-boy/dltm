# -*- coding: utf-8 -*-

"""
Created on Tue Aug 29 01:45:35 2017
train a controversy predictive modle with ANN
It has two layers:
  (1) input layer has three nodes
  (2) first layer has seven nodes
  (3) output layer has one node
@author: Allen
"""
import numpy 
import tensorflow as tf
from scipy.stats import logistic
    
x1 = []
z1 = []

# read train file
file = 'data/controversy.txt'
with open( file) as f:
    for line in f.readlines ():
        line = line.strip ()
        if line.startswith('//') or line=="":
            continue
        mlist = line.split (" ")
        d = list(map(float, mlist))
        x1.append(d[0:len(d)-1])
        z1.append(d[len(d)-1])
        
#print(x)
#print(z)
x2=numpy.array(x1)   # input
z=numpy.array([z1]) # output

x = x2.transpose()
#%%
size=len(z)     # the number of instances
idim=3          # the dim of input
wsize=7         # the number of nodes in hidden layer

with tf.Graph().as_default():
    iph=tf.placeholder(tf.float32, [idim, None]) # input
    lph=tf.placeholder(tf.float32, [1, None])    # label
    w1=tf.Variable(tf.random_uniform((wsize,idim), 0, 1, dtype=tf.float32))
    w2=tf.Variable(tf.random_uniform((1, wsize), 0, 1, dtype=tf.float32))
    b1 = tf.Variable(tf.zeros([wsize,1]))
    b2 = tf.Variable(tf.zeros([1]))
    
    hidden1=tf.nn.sigmoid(tf.matmul(w1, iph)+b1)
    modle  =tf.matmul(w2, hidden1)+b2
    
    loss=tf.reduce_mean(tf.square(modle-lph))
    step = tf.Variable(0, trainable=False)
    rate = tf.train.exponential_decay(0.15, step, 1, 0.9999)
    optimizer = tf.train.AdamOptimizer(rate)
    
    train_op = optimizer.minimize(loss)
        
    
    init_op = tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init_op)
    
    for step in range(1,80000):
        feed_dict={iph:x,lph:z} 
        _,val=sess.run([train_op,loss],feed_dict=feed_dict)
        if step % 1000 == 0:
            print(val)

#    W1=sess.run(w1)
#    print(W1)
#    W2=sess.run(w2)
#    print(W2)
#    B1=sess.run(b1)
#    print(B1)
#    B2=sess.run(b2)
#    print(B2)

#%%
t = numpy.matmul(W1, x)+B1
hidden = logistic.cdf(t)
out = numpy.matmul(W2,hidden)+B2
#%%
val = numpy.mean(abs(out-z))
print(['logistic:',val])

#print(z)
#print(out)