# -*- coding: utf-8 -*-

"""
Created on Tue Aug 29 01:45:35 2017
训练一个两层的神经网络，用于预测争议性评分
被注释掉的代码是用Logistics回归做的。
可以对比两个模型的性能
@author: Allen
"""
import numpy 
import tensorflow as tf
#from scipy.stats import logistic
    
x1 = []
z1 = []

# read train file
file = 'D:/qjt/paper/emotionalSN/exp1710/controversy.txt'
with open( file) as f:
    for line in f.readlines ():
        line = line.strip ()
        if line.startswith('//') or line=="":
            continue
        mlist = line.split (" ")
        d = list(map(float, mlist))
        x1.append(d[0:len(d)-1])
        z1.append(d[len(d)-1])
        
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
        
    
    init_op=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init_op)
    
    for step in range(1,80000):
        feed_dict={iph:x,lph:z} 
        _,val=sess.run([train_op,loss],feed_dict=feed_dict)
        if step % 1000 == 0:
            print("error:%s"%(val))


#%%
#t = numpy.matmul(W1, x)+B1
#hidden = logistic.cdf(t)
#out = numpy.matmul(W2,hidden)+B2
#val = numpy.mean(abs(out-z))
#print(['logistic error:',val])
