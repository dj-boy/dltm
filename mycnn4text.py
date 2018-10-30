# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 09:48:03 2018

@author: Allen
"""
import tensorflow as tf
import numpy as np
import data_helpers
from tensorflow.contrib import learn

# Parameters
embedding_dim=128       # Dimensionality of character embedding
filter_sizes=[3,4,5]    # Comma-separated filter sizes
num_filters=128         # Number of filters per filter size
keep_prob=0.5           # Dropout keep probability 
l2_reg_lambda=0         # L2 regularizaion lambda
batch_size=64           # Batch Size
num_epochs=200          # Number of training epochs
evaluate_every=100      # Evaluate model on dev set after this many steps
checkpoint_every=100    # Save model after this many steps
num_classes=2           # the number of output layer

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
# TODO: This is very crude, should use cross-validation
x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# build cnn model
sequence_length=x_train.shape[1]

input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

vocab_size=len(vocab_processor.vocabulary_)
C = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0), name="lookuptable")
embedded_chars = tf.nn.embedding_lookup(C, input_x)
embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    # Convolution Layer
    filter_shape = [filter_size, embedding_dim, 1, num_filters]
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
    conv = tf.nn.conv2d(
        embedded_chars_expanded,
        W,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
        
    # Apply nonlinearity
    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        
    # Maxpooling over the outputs
    pooled = tf.nn.max_pool(
        h,
        ksize=[1, sequence_length - filter_size + 1, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name="pool")
    pooled_outputs.append(pooled)

# Combine all the pooled features
num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(axis=3, values=pooled_outputs)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

# Add dropout
h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

# output layer
W_output_shape=[num_filters_total, num_classes]
W_output = tf.Variable(tf.truncated_normal(W_output_shape, stddev=0.1))
b_output = tf.Variable(tf.constant(0.1, shape=[num_classes]))

# Final (unnormalized) scores and predictions
scores = tf.nn.xw_plus_b(h_drop, W_output, b_output, name="scores")
predictions = tf.argmax(scores, 1, name="predictions")
losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
l2_loss = tf.nn.l2_loss(W_output)
l2_loss += tf.nn.l2_loss(b_output)

loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

optimizer = tf.train.AdamOptimizer(0.0001)
grads_and_vars = optimizer.compute_gradients(loss)
global_step = tf.Variable(0, name="global_step", trainable=False)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


loss_summary = tf.summary.scalar("loss", loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
res=sess.run(C)
print("C:",res[0,1])
 # Generate batches
batches = data_helpers.batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)

for batch in batches:
    x_batch, y_batch = zip(*batch)
    feed_dict = {
        input_x: x_batch,
        input_y: y_batch,
        dropout_keep_prob: keep_prob
    }
    _, step, summaries, loss_val,acc_val = sess.run(
                [train_op, global_step, loss_summary, loss, accuracy],
                feed_dict)
    print("step {}, loss {:g}, acc {:g}".format(step, loss_val, acc_val))
    current_step = tf.train.global_step(sess, global_step)


# validation
feed_dict = {
        input_x: x_dev,
        input_y: y_dev,
        dropout_keep_prob: 1
}
acc_val = sess.run([accuracy],feed_dict)
print("Accuracy in dev: ",acc_val)
