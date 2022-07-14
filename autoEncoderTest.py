"""
Authors: Fogo Tunde-Onadele, Yuhang Lin

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

import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] ='0'

import numpy as np
import random as rn
import tensorflow as tf

rn.seed(1)
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from baseline import prepare

import sys
import pandas as pd
import csv

from exportCSV import exportCSV

# command line args
# sys.argv[1]: application_name
# sys.argv[2]: train_file_num
# sys.argv[3]: test_file_num

########################
# PREPROCESS INPUT FILES
########################

# Import syscall vector data
basedir = './'
train_app = sys.argv[1]
train_num = sys.argv[2]
test_num = sys.argv[3]
test_app = train_app

# Training info needed for standardizing
#shaped_filename = basedir+'shaped-input/tomcat/tomcat-2_freqvector.csv'
shaped_filename = basedir+'shaped-transformed/{}/{}-{}_freqvector.csv'.format(train_app, train_app, train_num)
print(shaped_filename)
# read file
data = pd.read_csv(shaped_filename, delimiter=',')
'''
# timestamp column
timestamps = data.ix[:, 0]
'''
# headings row
headings = data.columns.values
#print(headings)
# headings row without timestamp
syscalls = headings[1:]


# Test data
#shaped_filename_test = basedir+'shaped-input/tomcat/tomcat-2_freqvector_test.csv'
shaped_filename_test = basedir+'shaped-transformed/{}/{}-{}_freqvector_test.csv'.format(test_app, test_app, test_num)
print(shaped_filename_test)
print()
# read file
data_test = pd.read_csv(shaped_filename_test, delimiter=',')

# headings row
headings_test = data_test.columns.values
#print(headings_test)
# headings row without timestamp
syscalls_test = headings_test[1:]

# standardize data (counts)
scaler = StandardScaler()
dataset_train = scaler.fit_transform(data.iloc[:,1:])
# shape
rows, columns = dataset_train.shape
print(dataset_train.shape)

dataset_test = scaler.transform(data_test.iloc[:,1:])
# shape
rows_test, columns_test = dataset_test.shape
print(dataset_test.shape)

# Saver() prep
model_save_dir = basedir+'models/{}/{}/'.format(train_app, train_num)
model_name = 'tomcat'

# prep for evaluation statistics later
app_time = prepare()
times = app_time[test_app][int(test_num) - 1]
print(times)
anomalous_sample_start = int(times[0])
anomalous_sample_shell = int(times[1])
anomalous_sample_stop = int(times[2])
sample_rate = 0.1 #seconds
# init, (able to be changed during training)
with open("{}{}-{}.txt".format(model_save_dir, train_app, train_num)) as fin:
    line = fin.readline().strip()
    anomaly_threshold = float (line)
    print("Threshold is: {}".format(anomaly_threshold))
# manual_threshold is just a flag, change anomaly_threshold
manual_threshold = 1

# pred_labels is autoencoder's prediction
pred_labels = np.zeros(rows_test)

# truth_labels is actual truth value
# 1: let abnormal period be all abnormal samples after exploit?
truth_labels = np.zeros(rows_test)
truth_labels[anomalous_sample_start:].fill(1)
# 2: let abnormal period be the abnormal samples during exploit command window?
truth_labels_window = np.zeros(rows_test)
truth_labels_window[anomalous_sample_start:anomalous_sample_shell + 1].fill(1)
# 3: let truth agree with pred for abnormal period?

# record loss
loss_summary = []

########################
# AUTOENCODER START
########################

# Testing Parameters
display_step_test = 256
batch_size_test = 256

tf.reset_default_graph()

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

    # Prediction
    # y_pred = decoder_op
    # Targets (Labels) are the input data.
    # y_true = X

    # Define loss and optimizer, minimize the squared error
    loss = graph.get_tensor_by_name('loss:0')
    # apparently has 0 output
    # optimizer = graph.get_tensor_by_name('optimizer:0')

    # other
    final_loss = graph.get_tensor_by_name('final_loss:0')

    # use appropriate anomaly threshold
    if not manual_threshold:
        anomaly_threshold = final_loss

    # Testing
    test_summary = []
    num_batches_test = int(rows_test / batch_size_test)
    # testing for threshold
    total_loss = 0
    average_loss = 0

    for j in range(num_batches_test + 1):
        # Prepare Data
        # in order
        batch_start = j * batch_size_test
        batch_end = (j + 1) * batch_size_test
        # last batch
        if batch_end > rows_test:
            batch_end = rows_test
        if batch_start >= batch_end:
            continue
        batch_x = dataset_test[batch_start:batch_end, :]
        # Encode and decode the batch
        g_pred = sess.run(decoder_op, feed_dict={X: batch_x})
        g_true = batch_x

        # Get loss (for each sample in batch)
        # arg: keepdims=True would keep it a column vector. row/list better for later processing
        l = sess.run(tf.reduce_mean(tf.pow(g_true - g_pred, 2), 1 ))
        print('Test Step %i: Step Loss: %f' % (batch_start, l[0]))

        loss_summary.extend(l)

        # Declare anomaly if loss is greater than threshold
        batch_labels = tf.cast(tf.greater(l,anomaly_threshold), tf.int64).eval()
        # batch_labels to pred_labels
        pred_labels[batch_start:batch_end] = batch_labels
        
        # for average calc
        total_loss = total_loss + tf.reduce_sum(l).eval()

    # prepare to graph reconstruction errors
    for loss_item in range(len(loss_summary)):
        test_summary.append({'rmse': loss_summary[loss_item], 'target': loss_item})
    # plot
    _, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(list(map(lambda x: x['target'], test_summary)), list(map(lambda x: x['rmse'], test_summary)))
    ax.axhline(anomaly_threshold, color='red', ls='--')
    ax.axvline(anomalous_sample_start, color='purple', ls='--', label=str(anomalous_sample_start))
    ax.axvline(anomalous_sample_shell, color='red', ls='--', label=str(anomalous_sample_shell))
    ax.axvline(anomalous_sample_stop, color='saddlebrown', ls='--', label=str(anomalous_sample_stop))
    ax.set_title('Error')
    plt.ylabel('MSE')
    plt.xlabel('Sample')
    plt.legend(loc=0)
    plt.ylim(0, anomaly_threshold + 3) # ignore some high reconstruction errors on the graph
    plt.savefig('figures/error.png', bbox_inches='tight')
    plt.show(block=False)
    
    # find lead time
    lead_sample = anomalous_sample_shell + 1
    for i in range(anomalous_sample_start, len(pred_labels)):
        if pred_labels[i] == 1:
            lead_sample = i
            print()
            print("first anomalous sample: %f" %lead_sample)
            print()
            break
    lead_time = sample_rate*(anomalous_sample_shell-lead_sample)
    is_detected = 1
    if lead_time < 0:
        lead_time = 0
        is_detected = 0
    
    # graph pred_labels
    # print()
    # print("pred labels")
    # print(*pred_labels, sep = ", ") 
    # print()
    # plot
    _, ax2 = plt.subplots(1, 1, figsize=(10, 4))
    ax2.plot(list(map(lambda x: x['target'], test_summary)), pred_labels)
    # ax2.axhline(anomaly_threshold, color='red', ls='--')
    ax2.axvline(anomalous_sample_start, color='purple', ls='--', label=str(anomalous_sample_start))
    ax2.axvline(anomalous_sample_shell, color='red', ls='--', label=str(anomalous_sample_shell))
    ax2.axvline(anomalous_sample_stop, color='saddlebrown', ls='--', label=str(anomalous_sample_stop))
    ax2.set_title('PRED')
    plt.ylabel('Prediction')
    plt.xlabel('Sample')
    plt.legend(loc=0)
    plt.savefig('figures/prediction.png', bbox_inches='tight')
    plt.show(block=False)

    ########################
    # PRINT STATS
    ########################

    average_loss = total_loss/rows_test

    # let truth agree with pred for abnormal period?
    '''
    for j in truth_labels:
        if anomalous_sample_start <= j <= anomalous_sample_stop:
            truth_labels[j] = pred_labels[j]
    '''
    # normal vs abnormal
    tn, fp, fn, tp = confusion_matrix(truth_labels, pred_labels).ravel()
    # Accuracy
    acc = (tp+tn)/(tp+fp+fn+tn)
    # FPR, TPR
    fpr = fp/(fp + tn)
    tpr = tp/(tp + fn)
    
    # first and last detected time samples
    if tp + fp == 0:
        first_alarm = -1
        last_alarm = -1
    else:
        first_alarm = pred_labels.tolist().index(1)
        last_alarm =  len(pred_labels) - 1 - pred_labels[::-1].tolist().index(1)  

    print()
    print("FINAL LOSS from prev training %f" % final_loss.eval())
    print()
    print("AVERAGE LOSS DURING TEST %f" % average_loss)
    print()
    print("ANOMALY THRESHOLD USED %f" % anomaly_threshold)
    print("# ANOMALIES PRESENT (trace samples)")
    print("\t whole exploit period %d" % (anomalous_sample_stop - anomalous_sample_start + 1))
    #print("\t anomalies found during exploit period %d" % np.count_nonzero(truth_labels))
    print("# ANOMALIES DETECTED  %d" % np.count_nonzero(pred_labels))
    print()
    print("FP  %d" % fp)
    print("TP  %d" % tp)
    print("FN  %d" % fn)
    print("TN  %d" % tn)
    print()
    print("ACC  %f" % acc)
    print("FPR  %f" % fpr)
    print("TPR  %f" % tpr)
    print()
    print("LEAD TIME: %f seconds" % lead_time)
    print()
    print("FIRST ALARM: sample %d, LAST ALARM: sample %d" % (first_alarm, last_alarm))
    print()
    
    # inside vs outside exploit window
    tn_window, fp_window, fn_window, tp_window = confusion_matrix(truth_labels_window, pred_labels).ravel()
    print("# ANOMALIES DETECTED (in window) %d" % tp_window)
    
    # END
    #file_writer = tf.summary.FileWriter(basedir+'/log', sess.graph)
    data = [train_app, fpr * 100, fp, tp, fn, tn, lead_time, first_alarm, last_alarm, is_detected * 100]
    exportCSV(data, f"individual-{test_num}.csv")
    exportCSV(loss_summary, f"recon_errors-{test_num}.csv")
    sess.close()

# show graphs finally
plt.show()

# find max MSE, avoiding peaks: find max under 2*std dev from mean.
mse_mean = np.mean(loss_summary)
mse_stddev = np.std(loss_summary)
print("MSE Mean: \t%.4f" % mse_mean)
print("MSE Std. Dev.: \t%.4f" % mse_stddev)
peak_limit = mse_mean + 2 * mse_stddev
print("MSE Peak Limit: %.4f" % peak_limit)
threshold_max = 0
for loss_item in range(len(loss_summary)):
    if (loss_summary[loss_item] < peak_limit and loss_summary[loss_item] > threshold_max):
        threshold_max = loss_summary[loss_item]
        threshold_max_index = loss_item
print("Max MSE of smaller values: %.5f, at sample: %d" % (threshold_max, threshold_max_index))
