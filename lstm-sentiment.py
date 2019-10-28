# -*- coding: utf-8 -*-
"""
实现第9章的LSTM文本情感分析。没有用论文的cell，而是tensorflow提供的basic cell
数据集没有使用第9章中的imdb电影评论数据集，而是使用了第7章CNN用于文本分类中的
sentence polarity dataset v1.0 
Created on Sat Oct 26 23:05:32 2019

@author: Allen
"""
import tensorflow as tf
import numpy as np
import data_helpers
from tensorflow.contrib import learn

# Parameters
embedding_dim=128       # Dimensionality of character embedding
keep_prob=0.5           # Dropout keep probability 
l2_reg_lambda=0         # L2 regularizaion lambda
batch_size=64           # Batch Size
num_epochs=30          # Number of training epochs
evaluate_every=100      # Evaluate model on dev set after this many steps
num_classes=2           # the number of output layer
hidden_size = 1000
initial_state = np.random.randn(1,hidden_size)

# data
x_text, y = data_helpers.load_data_and_labels()

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Graph: build lstm-sentiment model
tf.reset_default_graph()
time_step=x_train.shape[1]

input_x = tf.placeholder(tf.int32, [None, time_step], name="input_x")
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
hstate = tf.placeholder(dtype=tf.float32, shape=[None, hidden_size])

vocab_size=len(vocab_processor.vocabulary_)
C = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0), name="lookuptable")
inputs = tf.nn.embedding_lookup(C, input_x)

cell = tf.contrib.rnn.BasicLSTMCell(
          hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=False)


inputs = tf.unstack(inputs, num=time_step, axis=1)
outputs, state = tf.nn.static_rnn(cell, inputs, dtype=tf.float32)
#mean pooling
np_output=np.array(outputs)
mean_output=sum(np_output)/time_step

# logistic regression
W = tf.Variable(tf.random_uniform([hidden_size, num_classes], -1.0, 1.0))
b = tf.Variable(tf.zeros([num_classes]))
scores = tf.matmul(mean_output, W) + b 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y))
optimizer = tf.train.AdamOptimizer(0.001)
train_op = optimizer.minimize(loss)

preds=tf.argmax(tf.nn.softmax(scores),1)
correct_predictions = tf.equal(preds, tf.argmax(input_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


sess = tf.Session()
sess.run(tf.initialize_all_variables())
res=sess.run(C)

 # Generate batches
batches = data_helpers.batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)
k=0
for batch in batches:
    x_batch, y_batch = zip(*batch)
    feed_dict = {
        input_x: x_batch,
        input_y: y_batch,
        dropout_keep_prob: keep_prob,
        hstate:initial_state
    }
    _, loss_val,acc = sess.run(
                [train_op, loss, accuracy],
                feed_dict)
    if(k%1000==0):
        print("loss {:g}, acc {:g}".format(loss_val,acc))
    k=k+1

# validation
feed_dict = {
        input_x: x_dev,
        input_y: y_dev,
        dropout_keep_prob: 1
}
acc_val = sess.run([accuracy],feed_dict)
print("Accuracy in dev: ",acc_val)

