#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
from keras.models import load_model
from count_time import count_time
from baseline import prepare
import pandas as pd
from datetime import datetime
from exportCSV import exportCSV
from supervised_testing import evaluate


# In[2]:

parser = argparse.ArgumentParser("simple_example")
parser.add_argument(
    "minconf", nargs='?', help="Minimum required confidence.", type=float, default=0.5)
args = parser.parse_args()
print(f"Required confidence: {args.minconf}")


# In[3]:


def get_data(application_list, file_num, test_anomaly=True, evaluate_result=True, measure_time=False):
    total_lines = 0
    # load the saved trained models as well as pickle file of standardscaler
    for index in range(len(application_list)):
        app_name = application_list[index]
        print(app_name)
        file_name = 'shaped-transformed/{}/{}-{}_freqvector_test.csv'.format(
            app_name, app_name, file_num)
        df = pd.read_csv(file_name)
        num_row, num_col = df.shape
        total_lines += num_row
        predicted_labels = []
        # test the model using data of the rolling step size's data by calling autoEncoderTest,
        # and report the found anomalies
        if test_anomaly:
            row_raw = df.iloc[:, 1:]
            rows, columns = row_raw.shape
            row = row_raw.values.reshape(rows, columns, 1)
            # load model
            model = load_model(f'./data/supervised_model/CNN/{app_name}.h5')
            # # summarize model
            # model.summary()
            model._make_predict_function()
            prediction = model.predict(row).ravel()
            labels = [0 if confidence <
                      args.minconf else 1 for confidence in prediction]
            predicted_labels = labels
        if evaluate_result:
            evaluate(app_time, predicted_labels, prediction, app_name, file_num)
    return total_lines


# In[4]:


# read from apps-all.txt
application_list = []
with open("data/apps-all.txt") as fin_apps:
    for line in fin_apps:
        application_list.append(line.strip())

print("There are {} applications. ".format(len(application_list)))
print(application_list)
app_time = prepare()


# In[5]:

last_time = 0
total_lines = 0
last_time = count_time(last_time, "testing before")
# normal model, evaluate_result=True
for i in range(1, 5):
    sperator = ['container {}'.format(i)]
    exportCSV(sperator, "testing-res.csv")
    exportCSV(sperator, "detected.csv")
    exportCSV(sperator, "predicted.csv")
    exportCSV(sperator, "probabiltity.csv") 
    print(datetime.now())
    total_lines += get_data(application_list, i, evaluate_result=True)
    print(datetime.now())
last_time = count_time(last_time, "testing done")
print(total_lines)
# In[ ]:
