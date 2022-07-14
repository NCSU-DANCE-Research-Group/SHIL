"""
Author: Fogo Tunde-Onadele

- concatTraining.py  
concatenates 2 (training) files, keeping timestamps of the leading file continuous.  
// command line args  
sys.argv[0]: 2 input raw_filenames (comma separated)  
sys.argv[1]: shaped_filename  
e.g: `python concatTraining.py shaped-input/tomcat/tomcat-1_freqvector.csv,shaped-input/tomcat/tomcat-2_freqvector.csv shaped-input/tomcat/tomcat-1-2-fuse_freqvector.csv`  

NOTE:
assumes step between timestamps is even
"""

from __future__ import division, print_function, absolute_import

#import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import confusion_matrix

import sys
import pandas as pd
import csv


########################
# PREPROCESS INPUT FILES
########################

# Import syscall vector data
basedir = './'
#'C:/Users/Olufogorehan/PycharmProjects/'

raw_filenames = sys.argv[1].split(",")
print(raw_filenames)
print()

out_filename = sys.argv[2].strip()
print(out_filename)

# read input files
data = []
for rf in raw_filenames:
    data.append(pd.read_csv(basedir+rf, delimiter=','))

# info from first file
mark_rows, _ = data[0].shape
mark_timestamp = data[0].ix[mark_rows-1,0]
mark_timestep = data[0].ix[mark_rows-1,0] - data[0].ix[mark_rows-2,0]

# merge
merged_data = pd.concat([data[i] for i in range(len(data))], ignore_index=True)
merged_rows, _ = merged_data.shape

# maintain timestamps
end_timestamp = mark_timestamp + mark_timestep*(merged_rows-mark_rows)
adjusted_timestamps = np.arange(mark_timestamp+mark_timestep, end_timestamp + 1, mark_timestep)
merged_data.ix[mark_rows:,0] =  adjusted_timestamps

# output to new file
merged_data.to_csv(basedir+out_filename, line_terminator ='\n', index=False)
