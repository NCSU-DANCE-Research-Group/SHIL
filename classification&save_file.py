#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from ApplicationClassifier import ApplicationClassifier
import DataConversion
import datetime
import numpy as np
np.random.seed(1)


# In[2]:


# read from apps-all.txt
application_list = []
with open("data/apps-all.txt") as fin_apps:
    for line in fin_apps:
        application_list.append(line.strip())

print("There are {} applications. ".format(len(application_list)))
print(application_list)


# In[3]:


# integration of classification and training
classifier = ApplicationClassifier(interval=300, n_estimators=200, algorithm='randomforest') 
true_labels, predicted = classifier.test_all(application_list, save_raw_classified=True, save_classifier_traindata=True, save_classifier_testdata=False, retrain=True)


# In[ ]:




