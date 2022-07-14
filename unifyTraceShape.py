"""
Author: Fogo Tunde-Onadele

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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import sys
import pandas as pd
import csv


# command line args
# sys.argv[0]: raw_filenames (comma separated)
# sys.argv[1]: shaped_filenames (comma separated, matching same order as raw_filenames)

########################
# PREPROCESS INPUT FILES
########################

# Import syscall vector data
basedir = './'
#'C:/Users/Olufogorehan/PycharmProjects/vidhyaexample/'


raw_filenames = sys.argv[1].split(",")
'''
raw_filenames = [
    # train
    #basedir+'raw-input/activemq/activemq-1_freqvector.csv',
    #basedir+'raw-input/activemq/activemq-2_freqvector.csv',
    #basedir+'raw-input/activemq/activemq-3_freqvector.csv',
    basedir+'raw-input/tomcat/tomcat-c1-freq.csv',
    basedir+'raw-input/tomcat/tomcat-c2-freq.csv',
    basedir+'raw-input/tomcat/tomcat-c3-freq.csv',

    # test
    #basedir+'/raw-input/activemq/activemq-1_freqvector_test.csv',
    #basedir+'/raw-input/activemq/activemq-2_freqvector_test.csv',
    #basedir+'/raw-input/activemq/activemq-3_freqvector_test.csv'
    basedir+'raw-input/tomcat/tomcat-c1-freq-test.csv',
    basedir+'raw-input/tomcat/tomcat-c2-freq-test.csv',
    basedir+'raw-input/tomcat/tomcat-c3-freq-test.csv'
    ]
'''

print(raw_filenames)
print()

# in order corresponding to raw_filenames
'''
out_filenames = [
    # train
    #basedir+'shaped-input/activemq/activemq-1_freqvector.csv',
    #basedir+'shaped-input/activemq/activemq-2_freqvector.csv',
    #basedir+'shaped-input/activemq/activemq-3_freqvector.csv',
    basedir+'shaped-input/tomcat/tomcat-1_freqvector.csv',
    basedir+'shaped-input/tomcat/tomcat-2_freqvector.csv',
    basedir+'shaped-input/tomcat/tomcat-3_freqvector.csv',

    # test
    #basedir+'/shaped-input/activemq/activemq-1_freqvector_test.csv',
    #basedir+'/shaped-input/activemq/activemq-2_freqvector_test.csv',
    #basedir+'/shaped-input/activemq/activemq-3_freqvector_test.csv'
    basedir+'shaped-input/tomcat/tomcat-1_freqvector_test.csv',
    basedir+'shaped-input/tomcat/tomcat-2_freqvector_test.csv',
    basedir+'shaped-input/tomcat/tomcat-3_freqvector_test.csv'
    ]

'''
out_filenames = sys.argv[2].split(",")

print(out_filenames)


# read files to get system calls
data = []
for rf in raw_filenames:
    data.append(pd.read_csv(basedir+rf, delimiter=','))

unique_syscalls = []
for i in range(len(data)):
    # headings row
    headings = data[i].columns.values
    # print(headings_test)
    # headings row without timestamp
    syscalls = headings[1:]

    # unique_syscalls so far
    unique_syscalls = list(set(unique_syscalls + syscalls.tolist()))

# sort unique_syscalls
unique_syscalls = sorted(unique_syscalls)


# for each trace
for i in range(len(data)):
    # add 0 columns for missing calls
    for call in unique_syscalls:
        if call not in data[i].columns:
            # at column 1(after timestamps), insert column with values at 0 
            data[i].insert(1, call, 0)

    # sort trace (alphabetically)
    head_row = ['timestamp'] + sorted(unique_syscalls)
    data[i] = data[i].reindex(head_row, axis="columns")
    # equivalent line for old pandas version
    # data[i] = data[i].reindex(columns=head_row)

    # "post-processing"
    # (adjust column headings for upload to InsightFinder)
    # but not the timestamp heading
    for name_index in range(1, len(data[i].columns)):
        new_name = data[i].columns[name_index] + '[node1]:' + str(name_index)
        data[i].rename(columns={data[i].columns[name_index]: new_name}, inplace=True)
    # previous "post-processing" attempt
    # data[i].rename(columns=lambda x: x + '[node1]:' + data[i].columns.get_loc(x), inplace=True)
    # not for timestamp header
    # data[i].rename(columns={data[i].columns[0]: 'timestamp'}, inplace=True)
    
    # output to new file
    data[i].to_csv(basedir+out_filenames[i], line_terminator ='\n', index=False)
