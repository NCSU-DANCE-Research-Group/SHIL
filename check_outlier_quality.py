import argparse
from count_time import count_time
from baseline import prepare
from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime
from exportCSV import exportCSV
from collections import defaultdict


NUM_CVE = 41
NUM_CONTAINER = 4


application_dict = defaultdict(int)
index = 0
with open("data/apps-all.txt") as fin_apps:
    for line in fin_apps:
        app = line.strip()
        application_dict[app] = index
        index += 1


application_list = list(application_dict.keys())
app_debug_list = application_list
app_debug_dict = application_dict
# app_debug_list = ['CVE-2016-10033', 'CVE-2017-7494', 'CVE-2017-11610', 'CVE-2015-8562',
#                   'CVE-2016-3714', 'CVE-2018-19475', 'CVE-2021-44228', 'CVE-2021-28169', 'CVE-2017-12635']
# app_debug_dict = defaultdict(int)
# for app in app_debug_list:
#     app_debug_dict[app] = -1
# with open("data/apps-all.txt") as fin_apps:
#     for line in fin_apps:
#         app = line.strip()
#         if app in app_debug_dict:
#             app_debug_dict[app] = index
#         index += 1
print(f"There are {len(application_list)} applications: \n{application_list}")

CDL_predicted_path = "./experiment/original_CDL/41CVE/AE95-41CVE"
super_predicted_path = "./experiment/hybrid_approach/cdl_with_single_clf_randomforest/label_isolationforest/isolationforest_newthreshold_start_with_triggering/Desktop/41CVE/individual_classifier/HRF-2.0"  # CDL 95%
# "./experiment/original_CDL/41CVE/AE95-41CVE"  # CDL 95%
# "./experiment/combined-supervised+unsupervised/41CVE/HRF-2.0" # combined-200%


# CDL_predicted_path = "./experiment/hybrid_approach/cdl_with_single_clf_randomforest/label_isolationforest/isolationforest_newthreshold_start_with_triggering/Desktop/40CVE/AE95/BC200%/50%"
# super_predicted_path = "./experiment/combined-supervised+unsupervised/40CVE/HRF-2.0"

app_time = prepare()


def get_data(path, is_int, keep_float, file_name):
    data = []
    rows = []
    print(f"{path}/{file_name}")
    with open(f"{path}/{file_name}") as fin:
        for line in fin:
            line = line.strip()
            if "," not in line:
                # new container list
                if len(rows):
                    data.append(rows)
                rows = []
            else:
                rows.append([float(i) for i in line.split(",")])
    data.append(rows)
    assert(len(data) == NUM_CONTAINER)
    # for i in range(len(data)):
    #     assert(len(data[i]) == NUM_CVE)
    return data


# # read in the threshold, recon errors, original prediction of CDL
# CDL_prediction = get_data(CDL_predicted_path, True, False, "predicted.csv")
# # CDL_threshold = get_data(CDL_predicted_path, False, True, "thresholds.csv")
# # CDL_recon = get_data(CDL_predicted_path, False, True, "recon_errors.csv")
# # read in all the decision made by the supervised model
# super_prediction = get_data(
#     super_predicted_path, False, False, "predicted.csv")


# find the locations of all lines with no system call
def find_all_zero_training(CVE, container):
    line_all_zero = []
    index = 0
    time = app_time[CVE][container - 1]
    attack_start = time[0]
    for i in range(1, 4):
        file_num = (container - 1) * 3 + i
        file_name = f'./shaped-transformed/{CVE}/{CVE}-{file_num}_freqvector.csv'
        with open(file_name) as fin:
            for line in fin:
                parts = line.strip().split(",")[1:]
                found_nonzero = False
                for val in parts:
                    if val != '0':
                        found_nonzero = True
                        break
                if not found_nonzero:
                    line_all_zero.append(index)
                index += 1
    return line_all_zero


def find_all_zero(CVE, container, before_attack):
    line_all_zero = []
    index = 0
    time = app_time[CVE][container - 1]
    attack_start = time[0]
    file_name = f'./shaped-transformed/{CVE}/{CVE}-{container}_freqvector_test.csv'
    with open(file_name) as fin:
        for line in fin:
            parts = line.strip().split(",")[1:]
            found_nonzero = False
            for val in parts:
                if val != '0':
                    found_nonzero = True
                    break
            if not found_nonzero:
                if before_attack and index < attack_start or not before_attack and index >= attack_start:
                    # + 1801 for checking the raw file
                    line_all_zero.append(index)
            index += 1
    return line_all_zero


def get_outlier_percentage(CVE, container, outlier_path="data/label_using_outlier"):
    counter = 0
    num_outlier = 0
    with open(f'{outlier_path}/outlier_{container}.csv') as fin:
        for line in fin:
            parts = line.strip().split(",")
            if parts[0] == CVE:
                num_outlier = len(parts) - 1
                break
    with open(f'./shaped-transformed/{CVE}/{CVE}-{container}_freqvector_test.csv') as fin:
        for line in fin:
            counter += 1
    return num_outlier / counter * 100


def get_true_outlier_percentage(app, container, outlier_path="data/label_using_outlier"):
    total = 0
    with open(f'{outlier_path}/outlier_{container}.csv') as fin:
        for line in fin:
            parts = line.strip().split(",")
            if parts[0] != app:
                continue
            times = app_time[app][container - 1]
            attack_start = int(times[0])
            attack_success = int(times[1])
            total = attack_success - attack_start + 1
            count = 0
            for val in parts[1:]:
                if attack_start <= int(val) <= attack_success:
                    count += 1
    return count / total * 100

# check the lines of zeros before or after the start of the attack
# summary = defaultdict(list)
# for app in application_list:
#     for container in range(1, 5):
#         line_all_zero = find_all_zero(app, container, before_attack=True)
#         summary[app].append(str(len(line_all_zero)))
# for app in summary:
#     line = f"{app},"
#     line += ",".join(summary[app])
#     print(line)


# # check the lines of zeros in the training data
# summary = defaultdict(list)
# for app in application_list:
#     for container in range(1, 5):
#         line_all_zero = find_all_zero_training(app, container)
#         summary[app].append(str(len(line_all_zero)))
# for app in summary:
#     line = f"{app},"
#     line += ",".join(summary[app])
#     print(line)

# # check the predictions on the lines of zeros before the start of the attack to see the impact on FPR
# app_positive = defaultdict(list)
# for app in app_debug_list:
#     app_index = app_debug_dict[app]
#     for container_index in range(0, 4):
#         total_positive = 0
#         print(f"container: {container_index + 1}")
#         all_zero_lines = find_all_zero(
#             app, container_index + 1, before_attack=True)
#         print(f"Found {len(all_zero_lines)} lines of all zeros")
#         times = app_time[app][container_index]
#         attack_start = times[0] + 1801
#         attack_end = times[2] + 1801
#         print(f"attack starts: {attack_start}, attack ends: {attack_end}")
#         # print(all_zero_lines)
#         # compare combined prediction
#         for line in all_zero_lines:
#             # check prediction
#             total_positive += super_prediction[container_index][app_index][line]
#         app_positive[app].append(int(total_positive))

# for app in app_positive:
#     line = f"{app},"
#     for container in range(0, 4):
#         num_pos = app_positive[app][container]
#         line += f'{num_pos},'
#     print(line)


# compare C3 and C4 of CVE-2016-6515
# cve_index = 0
# for container_index in [2, 3]:
#     print(f"container: {container_index + 1}")
#     all_zero_lines = find_all_zero(CVE, container_index + 1)
#     print(f"Found {len(all_zero_lines)} lines of all zeros")
#     times = app_time[CVE][container_index]
#     attack_start = times[0] + 1801
#     attack_end = times[2] + 1801
#     print(f"attack starts: {attack_start}, attack ends: {attack_end}")
#     # print(all_zero_lines)
#     # compare HML and combined prediction
#     for i in all_zero_lines:
#         # print(CDL_prediction[container_index][cve_index][i],super_prediction[container_index][cve_index][i])
#         if CDL_prediction[container_index][cve_index][i] != super_prediction[container_index][cve_index][i]:
#             print(i + 1801, CDL_prediction[container_index][cve_index]
#                   [i], super_prediction[container_index][cve_index][i])
#         # if CDL_prediction[container_index][cve_index][i] == 1 and super_prediction[container_index][cve_index][i] == 0:
#         #     print(i + 1800 + 1)

outlier_path = "data/old_outlier_detection_data/label_using_outlier_knn"
print("Get percentage of outliers within the attack window in each container of all CVEs.")
for app in application_list:
    line = f"{app},"
    for container in range(1, 5):
        percentage = get_true_outlier_percentage(app, container, outlier_path)
        line += f'{percentage},'
    print(line)


print("Get percentage of outliers in each container of all CVEs.")
for app in application_list:
    line = f"{app},"
    for container in range(1, 5):
        percentage = get_outlier_percentage(app, container, outlier_path)
        line += f'{percentage},'
    print(line)
