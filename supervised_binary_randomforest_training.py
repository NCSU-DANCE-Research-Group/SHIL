#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Extra data of between the attack triggering time and the attack success time (from container 1).
"""

from email.mime import application
from count_time import count_time
# import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from baseline import prepare
import joblib
import pathlib
import numpy as np
from supervised_training import get_data_nonoutlier_positive, get_data_by_app
import os
np.random.seed(1)

# read from apps-all.txt
application_list = []
with open("data/apps-all.txt") as fin_apps:
    for line in fin_apps:
        application_list.append(line.strip())

print(f"There are {len(application_list)} applications. \n{application_list}")

# training
classifier_type = 'RF'
folder = f'./data/supervised_model/{classifier_type}'
pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
for app in application_list:
    X_train, y_train = get_data_by_app(
        app, normal_file_list=[1, 2], attack_file_list=[1], label_mode='similar', use_nonoutlier=False, similar_threshold=5)
    labels_counter = Counter(y_train[app])
    print(f"{app}, {labels_counter[0]}, {labels_counter[1]}")

    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf = clf.fit(X_train[app], y_train[app])
    # save the classifier to files
    joblib.dump(clf, f'{folder}/{app}.pkl')
