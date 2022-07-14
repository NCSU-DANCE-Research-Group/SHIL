"""
Authors: Yuhang Lin

Generate threshold and save it to a text file before testing.
"""

from exportCSV import exportCSV
import csv
import pandas as pd
import sys
from baseline import prepare
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

# from keras import backend as k
# config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
# allow_soft_placement=True, device_count = {'CPU': 1})
# sess = tf.Session(graph=tf.get_default_graph(),config=config)
# k.set_session(sess)


def get_stats(loss_summary):
    # original: 90, tried: 90, 95, 99
    threshold = np.percentile(loss_summary, 95)
#     with open(f"{train_app}_all.txt", "w") as fout:
#         fout.write(str(loss_summary))
    with open("{}.txt".format(train_app), "w") as fout:
        fout.write("{}\n".format(threshold))
    return threshold


def get_threshold():
    # Saver() prep
    model_save_dir = basedir+'model/'
    model_name = 'tomcat'
    # record loss
    loss_summary = []

    ########################
    # AUTOENCODER START
    ########################

    # Testing Parameters
    display_step_test = 256
    batch_size_test = 384

    # Start a new TF session
    with tf.Session() as sess:

        # LOAD/RESTORE TRAINED MODEL
        # restore network
        saver = tf.train.import_meta_graph(model_save_dir+model_name+'.meta')
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

        # Define loss and optimizer, minimize the squared error
        loss = graph.get_tensor_by_name('loss:0')

        # other
        final_loss = graph.get_tensor_by_name('final_loss:0')

        # Testing
        test_summary = []
        num_batches_test = int(rows / batch_size_test)
        # testing for threshold
        total_loss = average_loss = 0

        for j in range(num_batches_test+1):
            # Prepare Data
            # in order
            batch_start = j * batch_size_test
            batch_end = (j + 1) * batch_size_test
            # last batch
            if batch_end > rows:
                batch_end = rows
            batch_x = dataset_train[batch_start:batch_end, :]
            if len(batch_x) == 0:
                continue
            # Encode and decode the batch
            g_pred = sess.run(decoder_op, feed_dict={X: batch_x})
            g_true = batch_x

            # Get loss (for each sample in batch)
            # arg: keepdims=True would keep it a column vector. row/list better for later processing
            l = sess.run(tf.reduce_mean(tf.pow(g_true - g_pred, 2), 1))
            # if batch_start % display_step_test == 0 or batch_start == 1:
            print('Test Step %i: Step Loss: %f' % (batch_start, l[0]))
            loss_summary.extend(l)

        # END
        sess.close()
    return get_stats(loss_summary)

# command line args
# sys.argv[1]: shaped filename


basedir = './'
#shaped_filename = basedir+'shaped-input/tomcat/tomcat-2_freqvector.csv'
shaped_filename = sys.argv[1]
train_app = shaped_filename.split("/")[-1].split(".csv")[0].split("_")[0]

# Training info needed for standardizing
print(shaped_filename)
# read file
data = pd.read_csv(shaped_filename, delimiter=',')
# headings row
headings = data.columns.values
# print(headings)
# headings row without timestamp
syscalls = headings[1:]

# standardize data (counts)
scaler = StandardScaler()
dataset_train = scaler.fit_transform(data.iloc[:, 1:])
# shape
rows, columns = dataset_train.shape
print(dataset_train.shape)

anomaly_threshold = get_threshold()  # update threshold
