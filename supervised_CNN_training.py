#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Extra data of between the attack triggering time and the attack success time (from container 1).
"""

import joblib
import pathlib
import random
import numpy as np
from baseline import prepare
from count_time import count_time
from supervised_training import get_data

# In[2]:


from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from keras.regularizers import l2
from baseline import prepare
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc


# In[3]:


# read from apps-all.txt
application_list = []
with open("data/apps-all.txt") as fin_apps:
    for line in fin_apps:
        application_list.append(line.strip())

print(f"There are {len(application_list)} applications. \n{application_list}")


# In[4]:

def convert(data):
    for app in data:
        item = np.array(data[app])
        item = item.reshape(item.shape[0], item.shape[1], 1)
        data[app] = item
    return data


# In[6]:


# X_train, y_train, normal_data = get_data([1])
# print(len(X_train), len(y_train), len(normal_data))

# y_train = [1 for _ in range(len(X_train))] # label all as abnormal

# total = len(X_train)
# X_train.extend(random.sample(normal_data, total))
# y_train.extend([0 for i in range(total)])
# # 0 for normal, 1 for abnormal

# X_train = convert(X_train)


# In[7]:


def get_X_y(container):
    X, y = get_data(application_list, container)
    X = convert(X)
    return X, y


# In[8]:


X_train, y_train = get_X_y([1, 2])
# print(len(X_train), len(y_train))


# In[9]:


# X_valid, y_valid = get_X_y([2])


# # In[10]:


# X_test, y_test = get_X_y([3])


# In[11]:

classifier_type = 'CNN'
folder = f'./data/supervised_model/{classifier_type}'
pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

last_time = 0
last_time = count_time(last_time, "before training")
for app in application_list:
    # create model
    model = Sequential()
    step = 10
    # add model layers
    model.add(Conv1D(8, kernel_size=step,
              activation='relu', input_shape=(555, 1)))
    model.add(Conv1D(4, kernel_size=step, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(l=0.01)))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    # In[12]:
    model.fit(X_train[app], y_train[app], epochs=35, shuffle=True)
    # print(f"Training dataset size: {len(X_train)}")
    model.save(f'{folder}/{app}.h5')
last_time = count_time(last_time, "after training")
# In[13]:


# # testing
# y_pred = model.predict(X_test).ravel()
# y_pred_num = [0 if i < 0.5 else 1 for i in y_pred]

# print(classification_report(y_test, y_pred_num))

# cnf_matrix = confusion_matrix(y_test, y_pred_num)
# print(cnf_matrix)
# tn, fp, fn, tp = cnf_matrix.ravel()
# fpr = fp / (fp + tn) * 100
# tpr = tp / (tp + fn) * 100
# print(fpr, tpr)


# In[14]:


# fpr, tpr, _ = roc_curve(y_test, y_pred)
# roc_auc = auc(fpr, tpr)
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
# lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic of CNN')
# plt.legend(loc="lower right")
# plt.show()


# In[15]:


# In[21]:


# fpr = [0, 0.091, 0.1706, 0.2432, 0.3332, 0.6992,1]
# tpr = [0, 0.1067, 0.1916, 0.2718, 0.3723, 0.7494, 1]
# roc_auc = auc(fpr, tpr)
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange', lw=lw, marker='D', label='ROC curve (area = %0.2f)' % roc_auc)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic of CNN')
# plt.legend(loc="lower right")
# plt.show()


# In[ ]:
