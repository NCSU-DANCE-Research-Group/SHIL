"""

Auto Encoder for Anomaly Detection
Builds an auto-encoder with TensorFlow to compress application's system call freq vectors to a
lower latent space and then reconstruct them.

2 layers:
- input layer
- hidden layer 1
- hidden layer 2
- output layer
- sigmoid activation


######
Basic AutoEncoder Tutorial Reference

Builds a 2 layer auto-encoder with TensorFlow to compress MNIST dataset's handwritten digit vectors to a
lower latent space and then reconstruct them.

Consists of: input layer, hidden layer 1, hidden layer 2, output layer,
with neurons, all of which use sigmoid activation

References:
    Aymeric Damien

    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.


"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import sys
import pandas as pd
import csv

# command line args
# sys.argv[0]: shaped_filename

##########################
# PREPROCESS INPUT FILES
##########################

# Import syscall vector data
basedir = './'
# 'C:/Users/Olufogorehan/PycharmProjects/vidhyaexample/'
# train data
#shaped_filename = basedir+'shaped-input/activemq/activemq-1_freqvector.csv'
shaped_filename = sys.argv[1]
print(shaped_filename)
print()
# read file
data = pd.read_csv(shaped_filename, delimiter=',')
'''
# timestamp column
timestamps = data.ix[:, 0]
'''
# headings row
headings = data.columns.values
# print(headings)
# headings row without timestamp
syscalls = headings[1:]


# standardize data (counts)
scaler = StandardScaler()
dataset_train = scaler.fit_transform(data.ix[:, 1:])
# shape
rows, columns = dataset_train.shape
print(dataset_train.shape)


# Threshold could be set based on the training process
# init, (able to be changed during training)
# anomaly_threshold = 3.00;
# manual_threshold is just a flag, change anomaly_threshold
manual_threshold = 1

# Saver() prep: saving the model
model_save_dir = basedir+'model/'
model_name = 'tomcat'  # tomcat activemq


#####################
# AUTOENCODER START
#####################

# Training Parameters
learning_rate = 0.01
batch_size = 256
# 1170/6 = 195
# 1170/9 = 130
# 1170/15 = 78
# 1170/18 = 65
# epochs
num_steps = 2000

# output batch loss every display_step
display_step = 250
record_step = 50
# display_step_test = 400
# examples_to_show = 10

# Network Parameters
# defined in initial training, restored below

# Construct model
# defined in initial training, reconstructed below


##################
# START TRAINING
##################

num_batches = int(rows/batch_size)
cost_summary = []

# Start a new TF session
with tf.Session() as sess:
    # LOAD/RESTORE TRAINED MODEL
    # restore network
    saver = tf.train.import_meta_graph(model_save_dir + model_name + '.meta')
    # load parameters/variables
    saver.restore(sess, tf.train.latest_checkpoint(model_save_dir))

    # re define variables & operations
    graph = tf.get_default_graph()

    # tf Graph input
    X = graph.get_tensor_by_name('X:0')
    # Re-construct model
    # name gotten from initial training
    encoder_op = graph.get_tensor_by_name('Sigmoid_1:0')
    decoder_op = graph.get_tensor_by_name('Sigmoid_3:0')

    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = X

    # Define loss and optimizer, minimize the squared error
    loss = graph.get_tensor_by_name('loss:0')
    # does not work, apparently has 0 output?
    # optimizer = graph.get_tensor_by_name('optimizer:0')
    # use collection for op (good if I didnt implement the op myself)
    optimizer = tf.get_collection('optimizer')[0]

    # other
    final_loss = graph.get_tensor_by_name('final_loss:0')
    # print(final_loss)
    # Training
    for i in range(1, num_steps+1):

        for j in range(num_batches):
            # Prepare Data
            # Get the next batch (of MNIST data - only images are needed, not labels)

            # random order
            # batch_x, _ = dataset_train.next_batch(batch_size)
            #batch_x = next_batch(batch_size, dataset_train.values)

            # in order
            batch_start = j * batch_size
            batch_end = (j + 1) * batch_size
            batch_x = dataset_train[batch_start:batch_end, :]
            batch_y = sess.run(decoder_op, feed_dict={X: batch_x})

            # Run optimization op (backprop) and cost op (to get loss value)
            _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
            # Display logs per batch
            # print('Step %i: Minibatch Loss: %f' % (i, l))

        l = sess.run(loss, feed_dict={X: dataset_train})
        # update anomaly threshold if needed
        if not manual_threshold:
            anomaly_threshold = l

        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Total Loss: %f' % (i, l))

        # record step for graph (different from display?)
        if i % record_step == 0 or i == 1:
            cost_summary.append({'epoch': i, 'cost': l})

    # print training cost summary
    f, ax1 = plt.subplots(1, 1, figsize=(10, 4))
    ax1.plot(list(map(lambda x: x['epoch'], cost_summary)), list(
        map(lambda x: x['cost'], cost_summary)))
    ax1.set_title('Cost')
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.savefig('figures/updatecost.png', bbox_inches='tight')
    plt.show(block=False)

    ###############
    # PRINT STATS
    ###############

    print()
    print("FINAL LOSS from prev training %f" % final_loss.eval())
    print()
    print()
    print("FINAL LOSS %f" % l)
    print()

    # save final loss to variable
    final_loss_update = tf.assign(final_loss, l)
    sess.run(final_loss_update)

    # check
    #print("FINAL LOSS variable %f" % final_loss.eval())
    # print()

    # see tf graph
    '''
    print([n.name for n in tf.get_default_graph().as_graph_def().node])
    # OR
    graph = tf.get_default_graph()
    list_of_tuples = [op.values() for op in graph.get_operations()]
    print(list_of_tuples)
    '''

    # SAVE TRAINED MODEL
    # Saver() instance, empty Saver argument saves all variables
    # save_relative_paths=False allows saving to a specific folder
    saver = tf.train.Saver(save_relative_paths=True)
    saver.save(sess, model_save_dir+model_name)

    # END
    file_writer = tf.summary.FileWriter(basedir+'/log', sess.graph)
    sess.close()

# show graphs finally
# plt.show()
plt.close()


# next (random) batch function
def next_batch(size, data_array):
    '''
    Gets next random batch of data specified by size

    :return: size amount of random samples of data array (labels not needed/implemented)
    '''
    # shuffle data using indexes
    idx = np.arange(0, len(data_array))
    np.random.shuffle(idx)
    # trim index element to size amount
    idx = idx[:size]
    data_shuffled = data_array[idx]

    # label needed?
    #labels_shuffled = labels[idx]
    # reshape to column vector
    #labels_shuffled = np.asarray(labels_shuffled.values.reshape(len(labels_shuffled), 1))

    # return data_shuffled, labels_shuffled
    return data_shuffled
