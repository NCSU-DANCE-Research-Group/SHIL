#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
from count_time import count_time
from baseline import prepare
from timeit import default_timer as timer
import pandas as pd
from datetime import datetime
from exportCSV import exportCSV
import joblib
from training_util import USE_10MS, read_record, is_empty_line
from supervised_testing import evaluate

parser = argparse.ArgumentParser("simple_example")
parser.add_argument(
    "minconf", nargs='?', help="Minimum required confidence.", type=float, default=0.5)
args = parser.parse_args()
print(f"Required confidence: {args.minconf}")


if USE_10MS:
    data_folder = 'shaped-transformed-10ms'
else:
    data_folder = 'shaped-transformed'

def count_time(last_time, message):
    diff = timer() - last_time
    print(f"{message}: {diff}")
    last_time += diff
    return last_time


def get_data(application_list, file_num, test_anomaly=True, evaluate_result=True, measure_time=False):
    total_lines = 0
    # load the saved trained models as well as pickle file of standardscaler
    for index in range(len(application_list)):
        app_name = application_list[index]
        print(app_name)
        model = joblib.load(
            f'./data/supervised_model/RF/{app_name}.pkl')
        file_name = f'{data_folder}/{app_name}/{app_name}-{file_num}_freqvector_test.csv'
        df = pd.read_csv(file_name)
        num_row, num_col = df.shape
        total_lines += num_row
        predicted_labels = []
        predicted_probs = []
        # test the model using data of the rolling step size's data by calling autoEncoderTest,
        # and report the found anomalies
        # zero_vector_positve = 0
        if test_anomaly:
            df = df.iloc[:, 1:]
            proba = model.predict_proba(df)
            # count = 0
            for zero, one in proba:
                predicted_probs.append(one)
                if one <= args.minconf:
                    predicted_labels.append(0)
                else:
                    predicted_labels.append(1)
            #     row_list = df.iloc[[count]].values.tolist()[0]
            #     if is_empty_line(row_list) and predicted_labels[-1] == 1:
            #         zero_vector_positve += 1
            #     count += 1
            # print(f"zero vector positive: {zero_vector_positve}")
        if evaluate_result:
            evaluate(app_time, predicted_labels, predicted_probs, app_name, file_num)
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
    sperator = [f'container {i}']
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
