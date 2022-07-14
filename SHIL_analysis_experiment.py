import argparse
from count_time import count_time
from baseline import prepare
from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime
from exportCSV import exportCSV
from training_util import read_record

parser = argparse.ArgumentParser("simple_example")
parser.add_argument(
    "boundary", nargs='?', help="Minimum required confidence.", type=float, default=2.0)
args = parser.parse_args()
print(f"Required confidence: {args.boundary}")

application_list = []
with open("data/apps-all.txt") as fin_apps:
    for line in fin_apps:
        app = line.strip()
        application_list.append(app)
print(f"There are {len(application_list)} applications: \n{application_list}")
app_time = prepare()

CDL_predicted_path = "./result/CDL/AE95-41CVE"
super_predicted_path = "./result/supervisedRF/RF-0.6"

# read in the threshold, recon errors, original prediction of CDL
CDL_prediction = read_record(CDL_predicted_path, True, False, "predicted.csv")
CDL_threshold = read_record(CDL_predicted_path, False, True, "thresholds.csv")
CDL_recon = read_record(CDL_predicted_path, False, True, "recon_errors.csv")
# read in all the decision made by the supervised model
super_prediction = read_record(
    super_predicted_path, False, False, "predicted.csv")

# recalculate the result
def evaluate(predicted, app, file_num):
    """
    Get and save summary results
    """
    times = app_time[app][int(file_num) - 1]
    anomalous_sample_start = int(times[0])
    anomalous_sample_shell = int(times[1])
    anomalous_sample_stop = int(times[2])
    predicted = list(map(int, predicted))
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
    tn, fp, fn, tp = confusion_matrix(
        truth_labels, predicted, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    data = [app, fpr * 100, tpr * 100, fp, tp,
            fn, tn, lead_time, is_detected * 100]
    # save results to a file
    exportCSV(data, "testing-res.csv")
    exportCSV([app], "detected.csv")
    exportCSV(predicted, "predicted.csv")
    exportCSV(detected_samples, "detected.csv")


# find the spot for correcting the label
test_file_list = [_ for _ in range(1, 5)]  # 1, 2, 3, 4
for test_file_num in test_file_list:
    sperator = [f'container {test_file_num}']
    exportCSV(sperator, "testing-res.csv")
    exportCSV(sperator, "detected.csv")
    exportCSV(sperator, "predicted.csv")
    exportCSV(sperator, "recon_errors.csv")
    exportCSV(sperator, "thresholds.csv")
    for cve_index, app in enumerate(application_list):
        print(app)
        data_folder = 'shaped-transformed'
        test_file = f"{data_folder}/{app}/{app}-{test_file_num}_freqvector_test.csv"
        print(test_file)
        container = test_file_num - 1
        CDL_recon_curr = CDL_recon[container][cve_index]
        CDL_threshold_curr = CDL_threshold[container][cve_index]
        CDL_prediction_curr = CDL_prediction[container][cve_index]
        super_prediction_curr = super_prediction[container][cve_index]
        assert(len(CDL_recon_curr) == len(CDL_threshold_curr) ==
               len(CDL_recon_curr) == len(super_prediction_curr))
         # SHIL mode, always predict zero vector as negative
        index = 0
        with open(test_file) as fin:
            fin.readline()  # skip the header
            for line in fin:
                parts = [int(val) for val in line.strip().split(",")[1:]]
                if sum(parts) == 0:  # this line is a zero vector, we should always predict negaitve
                    CDL_prediction_curr[index] = 0
                elif CDL_prediction_curr[index] == 1:
                    ratio = CDL_recon_curr[index] / \
                        CDL_threshold_curr[index]
                    if ratio < args.boundary:
                        # print(ratio)
                        # replace the label
                        CDL_prediction_curr[index] = super_prediction_curr[index]
                index += 1
        evaluate(CDL_prediction_curr, app, test_file_num)
