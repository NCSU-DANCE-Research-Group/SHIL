#!/usr/bin/env python
# coding: utf-8

# In[1]:


from baseline import prepare
from timeit import default_timer as timer
from autoEncoderTestOnline import test
from ApplicationClassifier import ApplicationClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from datetime import datetime
from exportCSV import exportCSV
import joblib
import pathlib
import pandas as pd
import os
import numpy as np
import pickle
from IPython.display import Image
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


def count_time(last_time, message):
    diff = timer() - last_time
    print("{}: {}".format(message, diff))
    last_time += diff
    return last_time


def evaluate(predicted, app, file_num, recon_errors, thresholds):
    """
    Get and save summary results
    """
    times = app_time[app][int(file_num) - 1]
    anomalous_sample_start = int(times[0])
    anomalous_sample_shell = int(times[1])
    anomalous_sample_stop = int(times[2])
    print(len(predicted))
    print(times)

    sample_rate = 0.1  # seconds
    truth_labels = np.zeros(len(predicted))
    truth_labels[anomalous_sample_start:].fill(1)

    lead_sample = anomalous_sample_shell + 1
    detected_samples = []
    for i in range(anomalous_sample_start, min(anomalous_sample_shell + 1, len(predicted))):
        if predicted[i] == 1:
            lead_sample = i
            print(f"first anomalous sample: {lead_sample}")
            break
    lead_time = sample_rate * (anomalous_sample_shell - lead_sample)
    is_detected = 1
    if lead_time < 0:
        is_detected = 0
        lead_time = 0
    else:
        for i in range(lead_sample, min(anomalous_sample_shell + 1, len(predicted))):
            if predicted[i] == 1:
                detected_samples.append(i)
    tn, fp, fn, tp = confusion_matrix(truth_labels, predicted).ravel()
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    data = [app, fpr * 100, tpr * 100, fp, tp,
            fn, tn, lead_time, is_detected * 100]
    # save results to a file
    exportCSV(data, "testing-res.csv")
    exportCSV(predicted, "predicted.csv")
    exportCSV(detected_samples, "detected.csv")
    exportCSV(recon_errors, "recon_errors.csv")
    exportCSV(thresholds, "thresholds.csv")


def get_data(application_list, file_num, interval=300, step=10, test_anomaly=True, evaluate_result=True, measure_time=False):
    new_data_folder = './data/new_training'
    pathlib.Path(new_data_folder).mkdir(parents=True, exist_ok=True)
    # load the saved trained models as well as pickle file of standardscaler
    with open("data/classifier/{}-200.pkl".format("randomforest"), "rb") as input_file:
        model = pickle.load(input_file)
    for index in range(len(application_list)):
        app_name = application_list[index]
        print(app_name)
        file_name = 'shaped-transformed/{}/{}-{}_freqvector_test.csv'.format(
            app_name, app_name, file_num)
        df = pd.read_csv(file_name)
        rows, columns = df.shape
        sc_index_list = [i for i in range(1, columns)]

        last_data = None
        predicted_lables = []
        recon_errors = []
        thresholds = []
        for i in range(0, rows, step):
            count = 0
            end = min(i + interval, rows)
            if last_data is not None:
                row_data = last_data[:]
                count = interval - step  # we have some old data
                # remove the last step rows from the sum
                for row in range(i - (end - start), i):
                    for j, item in enumerate(sc_index_list):
                        row_data[j] -= df.iat[row, item]
            else:
                row_data = [0] * (columns - 1)
                start = 0
            for row in range(start, end):  # one line lasts 0.1 second
                for j, item in enumerate(sc_index_list):
                    row_data[j] += df.iat[row, item]
            count += end - start
            last_data = row_data
            row_data = [t / count for t in row_data]
            # load the trained classifier, then get the results from classifier, return the label (applicationID)
            predictY = model.predict([row_data])[0]
            # test the model using data of the rolling step size's data by calling autoEncoderTest,
            # and report the found anomalies
            if test_anomaly:
                data_test = df.iloc[start: end]
                labels, errors, threshold = test(
                    predictY, data_test, get_recon_error=True)
                predicted_lables.extend(labels)
                recon_errors.extend(errors)
                thresholds.extend([threshold for _ in range(len(errors))])
            start = end  # start becomes the new end
            if end == rows:
                if evaluate_result:
                    evaluate(predicted_lables, app_name,
                             file_num, recon_errors, thresholds)
                break


# In[3]:


# read from apps-all.txt
application_list = []
with open("data/apps-all.txt") as fin_apps:
    for line in fin_apps:
        application_list.append(line.strip())

print(f"There are {len(application_list)} applications: {application_list}")
app_time = prepare()


# In[ ]:


# normal model, evaluate_result=True
for i in range(1, 5):
    sperator = ['container {}'.format(i)]
    exportCSV(sperator, "testing-res.csv")
    exportCSV(sperator, "detected.csv")
    exportCSV(sperator, "predicted.csv")
    exportCSV(sperator, "recon_errors.csv")
    exportCSV(sperator, "thresholds.csv")
    print(datetime.now())
    get_data(application_list, i, 300, evaluate_result=True)
    print(datetime.now())


# In[ ]:
