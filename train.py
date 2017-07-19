from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

from tensorflow.python.ops.nn import dropout as drop
from util.cnn import conv_layer as conv
from util.cnn import conv_relu_layer as conv_relu
from util.cnn import fc_layer as fc
from util.cnn import fc_relu_layer as fc_relu
from models import vgg_net
from models.processing_tools import *

import sys
import os; os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import numpy as np

from util import data_reader
from util import loss

################################################################################
# Parameters
################################################################################

# Model Params
N = 10

# Training Params
pos_loss_mult = 1.
neg_loss_mult = 1.

start_lr = 0.01
lr_decay_step = 10000
lr_decay_rate = 0.1
weight_decay = 0.0005
momentum = 0.9
max_iter = 25000

fix_convnet = True
vgg_dropout = False
mlp_dropout = False
vgg_lr_mult = 1.

# Data Params
data_folder = '../trainbatch/'
data_prefix = 'train_det'

# Snapshot Params
snapshot = 5000
snapshot_file = './tfmodel/referit_fc8_det_iter_%d.tfmodel'

################################################################################
# The model
################################################################################

# Inputs
imcrop_batch = tf.placeholder(tf.float32, [N, 224, 224, 3])
label_batch = tf.placeholder(tf.float32, [N, 5])

feat_vis = vgg_net.vgg_prob(imcrop_batch, 'vgg_local', apply_dropout=vgg_dropout)
#probs = tf.nn.softmax(feat_vis)

################################################################################
# Collect trainable variables, regularized variables and learning rates
################################################################################
train_var_list = [var for var in tf.trainable_variables()]
print('Collecting variables to train:')
for var in train_var_list: print('\t%s' % var.name)
print('Done.')

# Add regularization to weight matrices (excluding bias)
reg_var_list = [var for var in tf.trainable_variables()
                if (var in train_var_list) and
                (var.name[-9:-2] == 'weights' or var.name[-8:-2] == 'Matrix')]
print('Collecting variables for regularization:')
for var in reg_var_list: print('\t%s' % var.name)
print('Done.')

# Collect learning rate for trainable variables
var_lr_mult = {var: (vgg_lr_mult if var.name.startswith('vgg_local') else 1.0)
               for var in train_var_list}
print('Variable learning rate multiplication:')
for var in train_var_list:
    print('\t%s: %f' % (var.name, var_lr_mult[var]))
print('Done.')

################################################################################
# Loss function and accuracy
################################################################################

cls_loss = loss.classfication_loss(label_batch, feat_vis)
reg_loss = loss.l2_regularization_loss(reg_var_list, weight_decay)
total_loss = cls_loss + reg_loss


################################################################################
# Solver
################################################################################

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(start_lr, global_step, lr_decay_step,
    lr_decay_rate, staircase=True)
solver = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
# Compute gradients
grads_and_vars = solver.compute_gradients(total_loss, var_list=train_var_list)
# Apply learning rate multiplication to gradients
grads_and_vars = [((g if var_lr_mult[v] == 1 else tf.mul(var_lr_mult[v], g)), v)
                  for g, v in grads_and_vars]
# Apply gradients
train_step = solver.apply_gradients(grads_and_vars, global_step=global_step)

################################################################################
# Initialize parameters and load data
################################################################################

# Load data
reader = data_reader.DataReader(data_folder, data_prefix)

snapshot_saver = tf.train.Saver()
sess = tf.Session()

# Run Initialization operations
sess.run(tf.initialize_all_variables())

################################################################################
# Optimization loop
################################################################################

cls_loss_avg = 0
avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg = 0, 0, 0
decay = 0.99

# Run optimization
for n_iter in range(max_iter):
    # Read one batch
    batch = reader.read_batch()
    imcrop_val = batch['imcrop_batch'].astype(np.float32) - vgg_net.channel_mean
    label_val = batch['label_batch'].astype(np.float32)

    # Forward and Backward pass
    scores_val, cls_loss_val, _, lr_val = sess.run([feat_vis, cls_loss, train_step, learning_rate],
        feed_dict={
            imcrop_batch    : imcrop_val,
            label_batch     : label_val
        })
    cls_loss_avg = decay*cls_loss_avg + (1-decay)*cls_loss_val
    print('\titer = %d, cls_loss (cur) = %f, cls_loss (avg) = %f, lr = %f'
        % (n_iter, cls_loss_val, cls_loss_avg, lr_val))

    # Accuracy
    accuracy_all, accuracy_pos, accuracy_neg = vgg_net.compute_accuracy(scores_val, label_val)
    avg_accuracy_all = decay*avg_accuracy_all + (1-decay)*accuracy_all
    avg_accuracy_pos = decay*avg_accuracy_pos + (1-decay)*accuracy_pos
    avg_accuracy_neg = decay*avg_accuracy_neg + (1-decay)*accuracy_neg
    print('\titer = %d, accuracy (cur) = %f (all), %f (pos), %f (neg)'
          % (n_iter, accuracy_all, accuracy_pos, accuracy_neg))
    print('\titer = %d, accuracy (avg) = %f (all), %f (pos), %f (neg)'
          % (n_iter, avg_accuracy_all, avg_accuracy_pos, avg_accuracy_neg))

    # Save snapshot
    if (n_iter+1) % snapshot == 0 or (n_iter+1) == max_iter:
        snapshot_saver.save(sess, snapshot_file % (n_iter+1))
        print('snapshot saved to ' + snapshot_file % (n_iter+1))

print('Optimization done.')
sess.close()
