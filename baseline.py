"""
Some common functions used by pca.py, kmeans10.py, knn.py.
"""

import numpy as np
import pandas as pd
from numpy import genfromtxt
from datetime import datetime
from sklearn.metrics import confusion_matrix
from training_util import USE_10MS

def readfile(path):
    matrix = genfromtxt(path, delimiter=',', skip_header=1)
    matrix = np.delete(matrix, 0, axis=1)
    return matrix

def calnum(starttime, trigtime, successtime, completetime):
    # this method is currently unused
    '''
    starttmp = starttime.split(".")
    #print(starttmp)
    start = starttmp[0].split(":")
    start.append(starttmp[1])
    trigtmp = trigtime.split(".")
    trig = trigtmp[0].split(":")
    trig.append(trigtmp[1])
    successtmp = successtime.split(".")
    success = successtmp[0].split(":")
    success.append(successtmp[1])
    #print(start)
    # trignum = (int(trig[1])-int(start[1]))*600 + (int(trig[2])-int(start[2]))*10 + (int(trig[3])-int(start[3]))
    # successnum = (int(success[1])-int(start[1]))*600 + (int(success[2])-int(start[2]))*10 + (int(success[3])-int(start[3]))
    '''
    startobj = datetime.strptime(starttime, "%H:%M:%S.%f")
    trigobj = datetime.strptime(trigtime, "%H:%M:%S.%f")
    successobj = datetime.strptime(successtime, "%H:%M:%S.%f")
    completeobj = datetime.strptime(completetime, "%H:%M:%S.%f")
    trignum = int(1 + (trigobj - startobj).total_seconds() * 10)  # 1000/100
    successnum = int(1 + (successobj - startobj).total_seconds() * 10)
    completenum = int(1 + (completeobj - startobj).total_seconds() * 10)
    return [trignum,successnum,completenum]

def fp(labels, trig, success, complete):
    total = len(labels)
    # construct truth labels
    # samples in entire abnormal period (trig to stop) assumed to be TP:
    truth_labels = np.zeros(total)
    #truth_labels[trig:complete + 1].fill(1)
    truth_labels[trig:].fill(1)
    tn, fp, fn, tp = confusion_matrix(truth_labels, labels).ravel()
    print("!!!fp number:" + str(fp))
    # should be same as 
    # cluster1 = sum(labels[0:trig-1]) + sum(labels[complete:total-1])
    # print("!!!fp number:" + str(cluster1))
    is_detected = False
    leadtime = 0
    # get lead time
    t = sum(labels[trig:success])
    if (t==0):
        print("No detection!!!")
    else:
        firstdet = list(labels[trig:success]).index(1)
        leadtime = (success - (firstdet + trig))/10
        print("!!!Lead time is: {} seconds".format(leadtime))
        is_detected = True

    # other metrics
    #acc = (tp+tn)/(tp+fp+fn+tn)
    fpr = fp/(fp + tn)
    #tpr = tp/(tp + fn)
    return fpr, leadtime, is_detected

def prepare():
    timing_file = "./data/Experiment-100ms.xlsx"
    df = pd.read_excel(f"{timing_file}", header=None)

    app = ""
    app_time = dict()
    for index, row in df.iterrows():
        if row[0] == "Application":
            app = row[1]
            app_time[app] = [[] for i in range(4)] # assuming the sample numbers start from 1
        if row[0] in ("exploit command 1 entered", "shell returned", "command returned", "all commands ended"):
            for i in range(4):
                app_time[app][i].append(row[1 + 3 * i])
    
    return app_time

# app_time = prepare()
# print(app_time)