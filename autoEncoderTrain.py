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

import csv
import pandas as pd
import sys
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tensorflow import set_random_seed
import tensorflow as tf
import random as rn
import numpy as np
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'


rn.seed(1)
np.random.seed(1)
set_random_seed(1)


# command line args
# sys.argv[1]: shaped_filename


##########################
# PREPROCESS INPUT FILES
##########################

# Import syscall vector data
basedir = './'
# 'C:/Users/Olufogorehan/PycharmProjects/vidhyaexample/'
# train data
#shaped_filename = basedir+'/shaped-input/activemq/activemq-3_freqvector.csv'
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
dataset_train = scaler.fit_transform(data.iloc[:, 1:])
# store the fitted scaler to a pickle file
train_app = shaped_filename.split("/")[-1].split(".csv")[0].split("_")[0]
with open('{}.pkl'.format(train_app), 'wb') as f:
    pickle.dump(scaler, f)
# shape
rows, columns = dataset_train.shape
print(dataset_train.shape)

# Threshold could be set based on the training process
# init, (able to be changed during training)
# anomaly_threshold = 3.00;
# manual_threshold is just a flag, change anomaly_threshold
manual_threshold = 1


# Saver() prep
model_save_dir = basedir+'model/'
model_name = 'tomcat'  # tomcat activemq


#####################
# AUTOENCODER START
#####################

# Training Parameters
learning_rate = 0.001
batch_size = 384  # org 256, tried 128
# 1170/6 = 195
# 1170/9 = 130
# 1170/15 = 78
# 1170/18 = 65
# epochs
num_steps = 10  # 2000, 2

# output batch loss every display_step
display_step = 250
record_step = 50
# display_step_test = 400
# examples_to_show = 10

# Network Parameters
# how to choose? previously, 32 and 16
num_hidden_1 = 278  # 1st layer num features = 256
num_hidden_2 = 70  # 2nd layer num features (the latent dim, half?) =  128
num_input = columns  # syscall data input (syscall data shape) = 68*1169=79492

# tf Graph input
X = tf.placeholder("float", [None, num_input], name="X")
# reference for final loss
final_loss = tf.Variable(0.0, name="final_loss")

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder


def encoder(x):
    # Encoder Hidden layer with relu, not sigmoid, activation #1, name=encoder_out
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with relu, not sigmoid, activation #2, name=encoder_out
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with relu, not sigmoid, activation #1, name=decoder_in
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2,  name=decoder_out
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
# for reference after I save the model
#print('encoder tf name: ' + encoder_op.name)
#print('decoder tf name: ' + decoder_op.name)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2), name="loss")
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
# put op in collection for saving reference later, (good if I didnt implement the op myself)
tf.add_to_collection('optimizer', optimizer)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


##################
# START TRAINING
##################

num_batches = int(rows/batch_size)
cost_summary = []

# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

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

        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Total Loss: %f' % (i, l))

        # record step for graph (different from display?)
        if i % record_step == 0 or i == 1:
            cost_summary.append({'epoch': i, 'cost': l})

    # print training cost summary
    # f, ax1 = plt.subplots(1, 1, figsize=(10, 4))
    # ax1.plot(list(map(lambda x: x['epoch'], cost_summary)), list(map(lambda x: x['cost'], cost_summary)))
    # ax1.set_title('Cost')
    # plt.ylabel('MSE')
    # plt.xlabel('Epochs')
    # #plt.savefig('figures/traincost.png', bbox_inches='tight')
    # plt.show(block=False)

    ###############
    # PRINT STATS
    ###############

    print()
    print("FINAL LOSS %f" % l)
    print()

    # save final loss to variable
    final_loss_op = final_loss.assign(l)
    sess.run(final_loss_op)
    #print('final_loss tf name: ' + final_loss.name)
    #print('final_loss_op tf name: ' + final_loss_op.name)
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
    #file_writer = tf.summary.FileWriter(basedir+'/log', sess.graph)
    sess.close()

# show graphs finally
# plt.show()
# plt.close()
