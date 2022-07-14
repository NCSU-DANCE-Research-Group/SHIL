#!/usr/bin/env python
# coding: utf-8

# In[1]:

from sklearn.ensemble import IsolationForest
from baseline import prepare
from exportCSV import exportCSV
import pandas as pd
import numpy as np
from count_time import count_time
from training_util import NUM_CVE, NUM_CONTAINER, USE_10MS, read_record

if USE_10MS:
    data_folder = 'shaped-transformed-10ms'
else:
    data_folder = 'shaped-transformed'

app = 'CVE-2012-1823'
test_file_num = 1
test_file = f"{data_folder}/{app}/{app}-{test_file_num}_freqvector_test.csv"
df = pd.read_csv(test_file)


# read from apps-all.txt
application_list = []
with open("data/apps-all.txt") as fin_apps:
    for line in fin_apps:
        application_list.append(line.strip())
print(f"There are {len(application_list)} applications: \n{application_list}")
app_time = prepare()



last_time = 0
for app in application_list:
    last_time = count_time(last_time, "before")
    test_file_list = [1, 2, 3, 4]
    for test_file_num in test_file_list:
        df_train = pd.DataFrame()

        for offset in range(1, 4):
            train_file_num = (test_file_num - 1) * 3 + offset
            train_file = f"{data_folder}/{app}/{app}-{train_file_num}_freqvector.csv"
            print(train_file)
            df = pd.read_csv(train_file)
            df = df.iloc[:, 1:]
            df_train = pd.concat([df_train, df], axis=0)
            print(df_train.shape)
        test_file = f"{data_folder}/{app}/{app}-{test_file_num}_freqvector_test.csv"
        print(test_file)
        df = pd.read_csv(test_file)
        times = app_time[app][test_file_num - 1]
        anomalous_start = int(times[0]) - 1
        df_test = df.iloc[anomalous_start:, 1:]
        df_train = pd.concat([df_train, df_test], axis=0)
        print(df_train.shape)
        clf = IsolationForest(contamination='auto', behaviour='new')
        clf.fit(df_train)
        prediction = clf.predict(df_test)
        detected = [app]
        non_detected = [app]
        real_outliers = [app]
        for i, val in enumerate(prediction):
            real_index = i + anomalous_start + 1
            if val == -1:  # outlier
                detected.append(real_index)
            else:
                non_detected.append(real_index)
        exportCSV(detected, f"outlier_{test_file_num}.csv")
        exportCSV(non_detected, f"nonoutlier_{test_file_num}.csv")
    last_time = count_time(last_time, "done")

